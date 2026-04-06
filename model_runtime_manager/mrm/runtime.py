# runtime.py
import json
import os
from pathlib import Path
import yaml
import time
import logging
import subprocess
import asyncio
import docker
import requests
from redis import Redis
from pydantic import BaseModel, Field, field_validator as pydantic_validator
from typing import Any, Dict, Literal, List, Optional

from .config import Settings, ModelSpec
from ..model_variant_registry import get_registry, VariantRegistry
from ..backends.base import NodeCapabilities
from ..backends.vllm_backend import VLLMBackend
from ..backends.llama_cpp_backend import LlamaCppBackend
from ..backends.cpu_runtime_backend import CpuRuntimeBackend
from ..execution_selector import ExecutionSelector
from ..vram_estimator import auto_tune_config, get_available_vram_gb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MRM.runtime")


class EnsureReq(BaseModel):
    base_model: str
    profile: Optional[str] = None          # "quality" | "balanced" | "cheap"
    node_capabilities: Optional[Dict[str, Any]] = None  # optional hardware override


class StopReq(BaseModel):
    base_model: str


class RemoveReq(BaseModel):
    base_model: str


class LoraRegisterReq(BaseModel):
    base_model: str
    lora_id: str
    host_path: str


class RuntimeError409(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


def _now() -> int:
    return int(time.time())


def _redis_key_model(base_model: str) -> str:
    return f"mrm:model:{base_model}"


def _redis_key_gpu(gpu: str) -> str:
    return f"mrm:gpu:{gpu}"


def _redis_lock_key(base_model: str) -> str:
    return f"mrm:lock:{base_model}"


def _redis_key_loras(base_model: str) -> str:
    return f"mrm:loras:{base_model}"


def _redis_key_lora_path(base_model: str, lora_id: str) -> str:
    return f"mrm:lora_path:{base_model}:{lora_id}"


def _redis_retry_key(base_model: str) -> str:
    return f"mrm:retry:{base_model}"


def _redis_config_cache_key(base_model: str) -> str:
    return f"mrm:config_cache:{base_model}"


def _estimate_eta(retry_step: int) -> int:
    """Simple linear ETA heuristic: 20s base + 15s per retry step."""
    return 20 + retry_step * 15


class RegisterReq(BaseModel):
    spec: ModelSpec


CPU_PRESETS: Dict[str, Dict[str, Any]] = {
    # 7B GGUF Q4_K_M — fits in ~6 GB RAM, 4 cores, ~8-12 tok/s
    "cpu_7b_q4": dict(
        image="model-runtime-cpu:latest",   # built from cpu_runtime/Dockerfile
        port=8090,
        cpu_cores=4,
        memory_mb=6144,
        n_ctx=4096,
    ),
    # 3B GGUF Q4_K_M — fits in ~3 GB RAM, 2 cores, ~20 tok/s
    "cpu_3b_q4": dict(
        image="model-runtime-cpu:latest",
        port=8090,
        cpu_cores=2,
        memory_mb=3072,
        n_ctx=2048,
    ),
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "small_chat": dict(
        image="vllm/vllm-openai:v0.13.0",
        launch_mode="openai",
        port=8000,
        gpu_memory_utilization=0.40,
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        enable_lora=True,
        max_loras=10,
        max_lora_rank=32,
        enforce_eager=True,
        quantization=None,
        dtype="auto",
        shm_size="8gb",
        ipc_host=True,
    ),
    "7b_awq": dict(
        image="vllm/vllm-openai:v0.13.0",
        launch_mode="openai",
        port=8000,
        gpu_memory_utilization=0.90,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        enable_lora=True,
        max_loras=5,
        max_lora_rank=32,
        enforce_eager=True,
        quantization="awq",
        dtype="auto",
        shm_size="8gb",
        ipc_host=True,
    ),
}

_ALLOWED_OVERRIDES = {
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_lora",
    "max_loras",
    "max_lora_rank",
    "enforce_eager",
    "quantization",
    "dtype",
}


class PlanFromHFReq(BaseModel):
    repo_id: str
    preset: Optional[str] = None   # formerly Literal["small_chat","7b_awq"]; now optional
    gpu: str = "0"
    overrides: Dict[str, Any] = Field(default_factory=dict)

    @pydantic_validator("gpu")
    @classmethod
    def _validate_gpu(cls, v: str) -> str:
        v = str(v).strip()
        if not v.isdigit():
            raise ValueError(
                f"gpu must be an integer GPU index (e.g. '0', '1') — got '{v}'. "
                "Did you accidentally pass gpu_memory_utilization?"
            )
        return v


def _build_launch_reasoning(
    backend_name: str,
    variant_format: str,
    reason: str,
    tuning,                  # AutoTuneResult | None
    available_vram: float,
) -> list:
    """Build a human-readable list of lines explaining a backend selection."""
    lines: list = [f"GPU VRAM: {available_vram:.1f} GB available"]
    if tuning is not None:
        est = tuning.estimate.total_gb
        ratio = est / available_vram if available_vram else 0
        fit_tag = "fits" if ratio <= 0.95 else "tight fit"
        lines.append(
            f"Model {variant_format} estimated: {est:.1f} GB "
            f"({ratio * 100:.0f}% of available — {fit_tag})"
        )
        lines.append(
            f"Auto-tuned: max_model_len={tuning.max_model_len:,}, "
            f"num_seqs={tuning.num_seqs}"
        )
    else:
        lines.append("VRAM estimate unavailable (model_size_b not set in variant)")
    lines.append(f"Selected backend: {backend_name}")
    lines.append(f"Selected variant: {variant_format}  (reason: {reason})")
    return lines


def _slug(s: str) -> str:
    return (
        s.lower()
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace(":", "_")
    )


def _apply_overrides_whitelist(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return base
    for k, v in overrides.items():
        if k in _ALLOWED_OVERRIDES:
            base[k] = v
    return base


def _build_spec_from_hf(
    repo_id: str, preset: Optional[str], gpu: str, overrides: Dict[str, Any]
) -> ModelSpec:
    # None or unknown preset → fall back to small_chat defaults; overrides still apply
    if not preset or preset not in PRESETS:
        if preset and preset not in PRESETS:
            raise RuntimeError409(f"Unknown preset: {preset}")
        preset = "small_chat"

    p = dict(PRESETS[preset])
    p = _apply_overrides_whitelist(p, overrides or {})

    alias = _slug(repo_id.split("/")[-1])
    safe = _slug(repo_id)
    container_name = ("vllm_" + safe)[:55]

    return ModelSpec(
        base_model=repo_id,
        model_alias=alias,
        container_name=container_name,
        image=p["image"],
        launch_mode=p["launch_mode"],
        hf_model=repo_id,
        served_model_name=alias,
        allowed_gpus=[gpu],
        port=int(p["port"]),
        gpu_memory_utilization=float(p["gpu_memory_utilization"]),
        max_model_len=int(p["max_model_len"]),
        max_num_seqs=int(p.get("max_num_seqs", 1)),
        max_num_batched_tokens=int(p.get("max_num_batched_tokens", 1024)),
        enable_lora=bool(p.get("enable_lora", False)),
        max_loras=int(p.get("max_loras", 0)),
        max_lora_rank=int(p.get("max_lora_rank", 0)),
        enforce_eager=bool(p.get("enforce_eager", False)),
        quantization=p.get("quantization", None),
        dtype=p.get("dtype", "auto"),
        shm_size=p.get("shm_size", "8gb"),
        ipc_host=bool(p.get("ipc_host", True)),
    )


def register_from_hf(self, req: PlanFromHFReq) -> Dict[str, Any]:
    spec = _build_spec_from_hf(req.repo_id, req.preset, req.gpu, req.overrides)
    self.registry[spec.base_model] = spec
    applied = {k: v for k, v in (req.overrides or {}).items() if k in _ALLOWED_OVERRIDES}
    quant = spec.quantization
    debug: Dict[str, Any] = {
        "selected_backend": "vllm",
        "selected_variant": "manual",
        "quantization": quant,
        "auto_tuned": False,
        "max_model_len": spec.max_model_len,
        "max_num_seqs": spec.max_num_seqs,
        "estimated_vram_gb": None,
        "available_vram_gb": None,
        "reason": "manual_registration",
        "reasoning": [
            f"Registered from HuggingFace: {req.repo_id}",
            f"Backend: vllm  ·  max_model_len: {spec.max_model_len:,}  ·  "
            f"max_num_seqs: {spec.max_num_seqs}",
            f"Quantization: {quant or 'none (fp16/auto)'}",
            "VRAM estimate unavailable at registration time (model size not known).",
            "VRAM will be reported after the model is started.",
        ],
    }
    return {
        "registered": True,
        "base_model": spec.base_model,
        "container": spec.container_name,
        "preset": req.preset,
        "gpu": req.gpu,
        "applied_overrides": applied,
        "debug": debug,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic (auto-fit) registration — builds ModelSpec from VllmConfig
# ──────────────────────────────────────────────────────────────────────────────

def build_spec_from_config(
    repo_id: str,
    final_repo_id: str,         # may differ if quant alternative was selected
    config,                      # VllmConfig
    gpu: str,
    settings: "Settings",
) -> "ModelSpec":
    """
    Build a ModelSpec from a dynamically generated VllmConfig.
    Uses sensible structural defaults (image, seqs, LoRA, etc.)
    that are independent of preset logic.
    """
    alias          = _slug(final_repo_id.split("/")[-1])
    safe           = _slug(final_repo_id)
    container_name = ("vllm_" + safe)[:55]

    # Env / volumes from settings (same as default registry)
    base_env = {
        "AWS_ACCESS_KEY_ID":       os.getenv("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY":   os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_DEFAULT_REGION":      os.getenv("AWS_DEFAULT_REGION", ""),
        "HF_TOKEN":                os.getenv("HF_TOKEN", ""),
        "HUGGINGFACE_HUB_TOKEN":   os.getenv("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN":  os.getenv("HF_TOKEN", ""),
    }

    art = os.path.abspath(settings.artifacts_host_path)
    hfc = os.path.abspath(settings.hf_cache_host_path)
    volumes = {art: "/app/artifacts", hfc: "/root/.cache/huggingface"}

    return ModelSpec(
        base_model            = repo_id,          # original repo for registry key
        model_alias           = alias,
        container_name        = container_name,
        image                 = "vllm/vllm-openai:v0.13.0",
        launch_mode           = "openai",
        hf_model              = final_repo_id,    # actual model to download
        served_model_name     = alias,
        allowed_gpus          = [gpu],
        port                  = 8000,
        gpu_memory_utilization= config.gpu_memory_utilization,
        max_model_len         = config.max_model_len,
        max_num_seqs          = 1,
        max_num_batched_tokens= 1024,
        enable_lora           = True,
        max_loras             = 10,
        max_lora_rank         = 32,
        enforce_eager         = True,
        quantization          = config.quantization,
        dtype                 = config.dtype,
        shm_size              = "8gb",
        ipc_host              = True,
        volumes               = volumes,
        env                   = base_env,
    )


def _is_nvidia_runtime_error(exc: Exception) -> bool:
    """Return True when the Docker API error indicates that the nvidia container
    runtime is not installed — so we know it's safe to fall back to CPU.

    docker.errors.APIError has a custom __str__ that formats to
    "{status} Server Error for {url}: {reason}" — the original message
    passed to the constructor is only in exc.args[0] and exc.explanation.
    We check all three sources to be thorough.
    """
    import docker.errors
    # Only treat Docker API errors as nvidia-unavailable, not our own
    # RuntimeError409 (which means GPU is busy or model not found).
    if not isinstance(exc, docker.errors.APIError):
        return False

    nvidia_keywords = [
        "unknown runtime",
        "nvidia",
        "could not select device driver",
        "no such runtime",
        "runtime not found",
        "no devices found",
        "failed to initialize nvml",
        "oci runtime create failed",
    ]

    # Check str(exc), exc.explanation, and the raw constructor message (args[0])
    candidates = [str(exc).lower()]
    if getattr(exc, "explanation", None):
        candidates.append(str(exc.explanation).lower())
    if exc.args:
        candidates.append(str(exc.args[0]).lower())

    return any(kw in candidate for kw in nvidia_keywords for candidate in candidates)


class ModelRuntimeManager:
    def __init__(self, settings: Settings, variant_registry: Optional[VariantRegistry] = None):
        self.s = settings
        self.docker = docker.from_env()
        self.redis = Redis.from_url(self.s.redis_url, decode_responses=True)
        self.registry: Dict[str, ModelSpec] = self.s.load_default_registry() | (self.s.model_registry or {})

        # ── Multi-backend selection layer ──────────────────────────────
        self._variant_registry: VariantRegistry = variant_registry or get_registry()
        self._backends = [
            VLLMBackend(self),
            LlamaCppBackend(self),
            CpuRuntimeBackend(self),
        ]
        self._execution_selector = ExecutionSelector(self._backends, self._variant_registry)

    def _log_ready(self, spec: ModelSpec) -> None:
        logger.info(
            f"✅ READY: {spec.container_name} api={self._api_base(spec)} "
            f"model={spec.hf_model or spec.base_model} served={spec.served_model_name} "
            f"lora={'on' if spec.enable_lora else 'off'} max_loras={spec.max_loras} rank={spec.max_lora_rank} "
            f"max_len={spec.max_model_len} gpu_util={spec.gpu_memory_utilization}"
        )

    def _detect_node_capabilities(self, spec: ModelSpec) -> NodeCapabilities:
        """Auto-detect hardware capabilities from the node running MRM."""
        gpu_id = spec.allowed_gpus[0] if spec.allowed_gpus else "0"
        gpu_mem = self._gpu_mem(gpu_id)
        has_gpu = gpu_mem.get("total_mib", 0) > 0

        ram_mb = 0
        cpu_cores = 4
        try:
            import psutil  # type: ignore[import]
            ram_mb = psutil.virtual_memory().total // (1024 * 1024)
            cpu_cores = psutil.cpu_count(logical=False) or 4
        except Exception:
            pass

        supported_backends = []
        if has_gpu:
            supported_backends.append("vllm")
        supported_backends.extend(["llama.cpp", "cpu"])

        return NodeCapabilities(
            gpu=has_gpu,
            vram_mb=gpu_mem.get("free_mib", 0),
            ram_mb=ram_mb,
            cpu_cores=cpu_cores,
            supported_backends=supported_backends,
        )

    def lora_register(self, req: LoraRegisterReq) -> Dict[str, Any]:
        spec = self._spec(req.base_model)

        if not spec.enable_lora:
            raise RuntimeError409(f"LoRA disabled for base_model={req.base_model}")

        hp = Path(req.host_path)
        cfg = hp / "adapter_config.json"
        if not cfg.exists():
            raise RuntimeError409(f"adapter_config.json not found at host_path={req.host_path}")

        kset = _redis_key_loras(req.base_model)
        self.redis.sadd(kset, req.lora_id)
        self.redis.expire(kset, max(self.s.idle_timeout_sec * 4, 3600))

        kpath = _redis_key_lora_path(req.base_model, req.lora_id)
        self.redis.set(kpath, str(hp))
        self.redis.expire(kpath, max(self.s.idle_timeout_sec * 4, 3600))

        c = self._container_get(spec.container_name)
        running = bool(c and self._container_is_running(c))
        if running:
            self._set_state(req.base_model, {"needs_restart": "1"})

        return {
            "registered": True,
            "base_model": req.base_model,
            "lora_id": req.lora_id,
            "host_path": req.host_path,
            "running": running,
            "needs_restart": running,
            "hint": "Restart vLLM container for this base_model to load newly registered LoRAs.",
        }

    def _litellm_static_models(self) -> list[dict]:
        return [
            {
                "model_name": "all-MiniLM-L6-v2",
                "litellm_params": {
                    "model": "openai/sentence-transformers/all-MiniLM-L6-v2",
                    "api_base": "http://embeddings_api:7997",
                    "api_key": "empty",
                },
            }
        ]

    def _litellm_dynamic_models(self) -> list[dict]:
        models = []
        for _, spec in self.registry.items():
            models.append(
                {
                    "model_name": spec.served_model_name,
                    "litellm_params": {
                        "model": f"hosted_vllm/{spec.served_model_name}",
                        "api_base": f"http://{spec.container_name}:{spec.port}/v1",
                        "api_key": "none",
                    },
                }
            )
        return models

    def materialize_litellm_config(self) -> dict:
        out = {"model_list": self._litellm_static_models() + self._litellm_dynamic_models()}

        p = Path(self.s.litellm_config_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        yaml_txt = yaml.safe_dump(out, sort_keys=False)

        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(yaml_txt, encoding="utf-8")
        os.replace(tmp, p)

        self._restart_litellm_if_enabled()

        return {
            "written": True,
            "path": str(p),
            "models": [m["model_name"] for m in out["model_list"]],
        }

    def register_from_hf(self, req: PlanFromHFReq) -> Dict[str, Any]:
        return register_from_hf(self, req)

    def register(self, spec: ModelSpec) -> Dict[str, Any]:
        if not spec.base_model:
            raise RuntimeError409("base_model required")
        self.registry[spec.base_model] = spec
        return {"registered": True, "base_model": spec.base_model, "container": spec.container_name}

    def _restart_litellm_if_enabled(self) -> None:
        if not self.s.litellm_restart_on_write:
            return
        try:
            c = self.docker.containers.get(self.s.litellm_container_name)
            c.restart()
        except Exception as e:
            logger.warning(f"LiteLLM restart failed: {e}")

    def _gpu_mem(self, gpu: str) -> Dict[str, int]:
        try:
            cmd = [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, text=True).strip()
            total, used, free = [int(x.strip()) for x in out.split(",")]
            return {"total_mib": total, "used_mib": used, "free_mib": free}
        except Exception:
            return {"total_mib": 0, "used_mib": 0, "free_mib": 0}

    def _gpu_is_free(self, gpu: str) -> bool:
        if self.redis.scard(_redis_key_gpu(gpu)) != 0:
            return False
        try:
            mem = self._gpu_mem(gpu)
            return mem["used_mib"] < 1500
        except Exception:
            return True

    def _gpu_reserve(self, gpu: str, base_model: str) -> None:
        self.redis.sadd(_redis_key_gpu(gpu), base_model)
        self.redis.expire(_redis_key_gpu(gpu), max(self.s.idle_timeout_sec * 4, 3600))

    def _gpu_release(self, gpu: str, base_model: str) -> None:
        self.redis.srem(_redis_key_gpu(gpu), base_model)

    def _choose_gpu(self, spec: ModelSpec) -> str:
        for g in spec.allowed_gpus:
            if self._gpu_is_free(g):
                return g

        logger.warning(f"⚠️ No free GPU found for {spec.base_model}. Checking for zombie reservations...")

        for g in spec.allowed_gpus:
            occupier_model = self.redis.srandmember(_redis_key_gpu(g))
            if not occupier_model:
                return g

            occupier_spec = self.registry.get(occupier_model)
            if not occupier_spec:
                self._gpu_release(g, occupier_model)
                return g

            c = self._container_get(occupier_spec.container_name)
            if not c or not self._container_is_running(c):
                logger.info(f"🧟 Zombie detected on GPU {g}: {occupier_model}. Cleaning up...")
                self._gpu_release(g, occupier_model)
                self._set_state(occupier_model, {"state": "ABSENT", "gpu": ""})
                return g

        raise RuntimeError409(f"No free GPU. All allowed GPUs {spec.allowed_gpus} are busy.")

    def _spec(self, base_model: str) -> ModelSpec:
        if base_model in self.registry:
            return self.registry[base_model]
        # Fall back to alias/served_model_name reverse lookup so callers can
        # use either the HuggingFace repo ID or the short alias (e.g. the name
        # vLLM reports via /v1/models).
        for spec in self.registry.values():
            if spec.model_alias == base_model or spec.served_model_name == base_model:
                return spec
        raise RuntimeError409(f"Unknown base_model: {base_model}")

    def _health_url(self, spec: ModelSpec) -> str:
        return f"http://{spec.container_name}:{spec.port}{spec.health_path}"

    def _api_base(self, spec: ModelSpec) -> str:
        return f"http://{spec.container_name}:{spec.port}/v1"

    def _get_state(self, base_model: str) -> Dict[str, Any]:
        return self.redis.hgetall(_redis_key_model(base_model)) or {}

    def _set_state(self, base_model: str, updates: Dict[str, Any]) -> None:
        k = _redis_key_model(base_model)
        if updates:
            self.redis.hset(k, mapping={k2: str(v2) for k2, v2 in updates.items()})
        self.redis.expire(k, max(self.s.idle_timeout_sec * 4, 3600))

    def touch(self, base_model: str) -> None:
        spec = self._spec(base_model)
        base_model = spec.base_model  # normalize alias → canonical registry key
        st = self._get_state(base_model)
        self._set_state(base_model, {"last_used": _now()})
        gpu = st.get("gpu")
        if gpu:
            self.redis.expire(_redis_key_gpu(gpu), max(self.s.idle_timeout_sec * 4, 3600))

    def _acquire_lock(self, base_model: str) -> bool:
        ttl = max(self.s.health_timeout_sec + 60, 1800)
        return bool(self.redis.set(_redis_lock_key(base_model), "1", nx=True, ex=ttl))

    def _release_lock(self, base_model: str) -> None:
        self.redis.delete(_redis_lock_key(base_model))

    def _container_get(self, name: str):
        try:
            return self.docker.containers.get(name)
        except docker.errors.NotFound:
            return None

    def _container_is_running(self, c) -> bool:
        try:
            c.reload()
            return (c.status or "").lower() == "running"
        except Exception:
            return False

    def _container_start(self, c) -> None:
        c.reload()
        if (c.status or "").lower() != "running":
            c.start()

    def _container_stop(self, c, timeout: int = 30) -> None:
        try:
            c.reload()
            if (c.status or "").lower() == "running":
                c.stop(timeout=timeout)
        except Exception:
            pass

    def _container_remove_if_exists(self, name: str) -> None:
        try:
            c = self._container_get(name)
            if not c:
                return
            try:
                c.remove(force=True)
            except Exception:
                time.sleep(0.5)
                c = self._container_get(name)
                if c:
                    c.remove(force=True)
        except Exception:
            pass

    def _maybe_remove_container(self, name: str) -> None:
        if self.s.keep_failed_containers:
            return
        self._container_remove_if_exists(name)

    def _make_vllm_serve_command(self, spec: ModelSpec) -> List[str]:
        model_id = spec.hf_model or spec.base_model

        cmd = [
            model_id,
            "--host", "0.0.0.0",
            "--port", str(spec.port),
            "--served-model-name", spec.served_model_name,
            "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
            "--max-model-len", str(spec.max_model_len),
            "--max-num-seqs", str(spec.max_num_seqs),
            "--max-num-batched-tokens", str(spec.max_num_batched_tokens),
            "--trust-remote-code",
            "--dtype", spec.dtype,
        ]

        if getattr(spec, "quantization", None):
            cmd += ["--quantization", spec.quantization]

        if spec.enable_lora:
            cmd += [
                "--enable-lora",
                "--max-loras", str(spec.max_loras),
                "--max-lora-rank", str(spec.max_lora_rank),
            ]

            try:
                lora_ids = sorted(list(self.redis.smembers(_redis_key_loras(spec.base_model)) or []))
            except Exception:
                lora_ids = []

            for lid in lora_ids:
                # ✅ ВАЖЛИВО: Оскільки ми знаємо структуру, ми формуємо шлях жорстко
                cmd += ["--lora-modules", f"{lid}=/lora/{lid}"]

        if spec.enforce_eager:
            cmd += ["--enforce-eager"]

        return cmd

    def _wait_ready(self, spec: ModelSpec) -> None:
        deadline = time.time() + self.s.health_timeout_sec
        url = self._health_url(spec)
        last_err = None

        logger.info(f"⏳ Waiting for {spec.container_name} to be ready... (Timeout: {self.s.health_timeout_sec}s)")

        while time.time() < deadline:
            c = self._container_get(spec.container_name)
            if not c or not self._container_is_running(c):
                logs = "No logs available"
                if c:
                    try:
                        logs = c.logs().decode("utf-8", errors="replace")[-4000:]
                    except Exception:
                        pass
                logger.error(f"❌ Container {spec.container_name} CRASHED during startup.\nLogs:\n{logs}")
                raise RuntimeError409("Container crashed during startup. Check logs.")

            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    return
            except Exception as e:
                last_err = e

            time.sleep(self.s.health_poll_interval_sec)

        raise RuntimeError409(f"Healthcheck timeout for {spec.container_name} ({url}). Last error={last_err}")

    def _max_len_ladder(self, spec: ModelSpec) -> List[int]:
        if spec.max_model_len >= 4096:
            return [4096, 3072, 2048, 1536, 1024]
        if spec.max_model_len >= 2048:
            return [2048, 1536, 1024]
        return [spec.max_model_len]

    def _start_container_fresh(self, spec: ModelSpec, gpu: str) -> str:
        self._maybe_remove_container(spec.container_name)

        logger.info("⏳ Sleeping 10s to let GPU memory settle...")
        time.sleep(10)

        env = dict(spec.env or {})
        env["NVIDIA_VISIBLE_DEVICES"] = gpu
        env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility"
        env["VLLM_TARGET_DEVICE"] = "cuda"
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        env["NCCL_P2P_DISABLE"] = "1"
        env["UVLOOP_NO_EXTENSIONS"] = "1"

        volumes = {}
        for host_path, container_path in (spec.volumes or {}).items():
            volumes[host_path] = {"bind": container_path, "mode": "rw"}

        if spec.enable_lora:
            host_lora_path = self.s.get_lora_host_path()

            logger.info(f"🔗 Mounting LoRA dir: Host[{host_lora_path}] -> Container[/lora]")

            volumes[host_lora_path] = {"bind": "/lora", "mode": "ro"}

        image = spec.image
        command = self._make_vllm_serve_command(spec)

        host_config = {
            "runtime": "nvidia",
            "detach": True,
            "name": spec.container_name,
            "network": self.s.docker_network,
            "environment": env,
            "volumes": volumes,
            "command": command,
        }

        if spec.ipc_host:
            host_config["ipc_mode"] = "host"
        if spec.shm_size:
            host_config["shm_size"] = spec.shm_size

        try:
            img = self.docker.images.get(image)
            ep = (img.attrs.get("Config") or {}).get("Entrypoint") or []
            cmd0 = (img.attrs.get("Config") or {}).get("Cmd") or []
            logger.info("🐳 Image=%s Entrypoint=%s Cmd=%s", image, ep, cmd0)
        except Exception as e:
            logger.warning("Could not inspect image %s: %s", image, e)

        logger.info("🚀 Starting container name=%s gpu=%s image=%s", spec.container_name, gpu, image)
        logger.info("VLLM CMD (args to `vllm serve`): %s", " ".join(command))
        logger.info(
            "EQUIV: docker run --rm --gpus all --network %s --name %s %s %s",
            self.s.docker_network,
            spec.container_name,
            image,
            " ".join(command),
        )

        c = self.docker.containers.run(image, **host_config)

        logger.info(
            "🐳 Container created: name=%s id=%s image=%s gpu=%s model=%s served=%s quant=%s",
            spec.container_name,
            c.id,
            image,
            gpu,
            (spec.hf_model or spec.base_model),
            spec.served_model_name,
            (spec.quantization or "none"),
        )

        return c.id

    def _start_cpu_container(self, spec: ModelSpec) -> str:
        """
        Start a cpu_runtime container for a GGUF model.

        Key differences from vLLM:
        - No GPU device assignment or reservation
        - Uses cpu_runtime Docker image
        - Passes GGUF path and thread count via env vars
        - Memory limit enforced via Docker nano_cpus + mem_limit
        """
        self._maybe_remove_container(spec.container_name)

        env = dict(spec.env or {})
        env["CPU_RUNTIME_MODEL_PATH"] = spec.gguf_path or "/models/model.gguf"
        env["CPU_RUNTIME_MODEL_ALIAS"] = spec.served_model_name
        env["CPU_RUNTIME_N_THREADS"] = str(spec.cpu_cores)
        env["CPU_RUNTIME_N_CTX"] = str(spec.max_model_len)
        env["LOG_FORMAT"] = "json"

        volumes = {}
        for host_path, container_path in (spec.volumes or {}).items():
            volumes[host_path] = {"bind": container_path, "mode": "rw"}

        logger.info(
            "🚀 Starting CPU container name=%s image=%s threads=%s gguf=%s",
            spec.container_name,
            spec.image,
            spec.cpu_cores,
            spec.gguf_path,
        )

        c = self.docker.containers.run(
            spec.image,
            detach=True,
            name=spec.container_name,
            network=self.s.docker_network,
            environment=env,
            volumes=volumes,
            # Resource limits: no GPU, bounded CPU + RAM
            nano_cpus=int(spec.cpu_cores * 1e9),
            mem_limit=f"{spec.memory_mb}m",
        )

        logger.info("🐳 CPU container created: name=%s id=%s", spec.container_name, c.id)
        return c.id

    # -------------------------
    # Public API
    # -------------------------
    def ensure_running(
        self,
        base_model: str,
        profile: Optional[str] = None,
        node_capabilities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        spec = self._spec(base_model)
        base_model = spec.base_model  # normalize alias → canonical registry key

        # ── Multi-backend selection path ───────────────────────────────
        # Used when model variants are registered in the variant registry.
        _vr = getattr(self, "_variant_registry", None)
        if _vr is not None and _vr.has_variants(base_model):
            if node_capabilities is not None:
                node_caps = NodeCapabilities.from_dict(node_capabilities)
            else:
                node_caps = self._detect_node_capabilities(spec)

            _sel = getattr(self, "_execution_selector", None)
            result = _sel.select(base_model, node_caps, profile) if _sel else None
            if result:
                launched = result.backend.launch(base_model, result.variant, node_caps)

                # ── Build debug / explainability metadata ──────────────
                tuning = auto_tune_config(result.variant, node_caps)
                available_vram = get_available_vram_gb(node_caps)
                debug: Dict[str, Any] = {
                    "selected_backend": result.backend.name,
                    "selected_variant": result.variant.format,
                    "quantization": result.variant.quantization,
                    "auto_tuned": tuning is not None,
                    "max_model_len": tuning.max_model_len if tuning else None,
                    "max_num_seqs": tuning.num_seqs if tuning else None,
                    "estimated_vram_gb": tuning.estimate.total_gb if tuning else None,
                    "available_vram_gb": round(available_vram, 2),
                    "reason": result.reason,
                    "reasoning": _build_launch_reasoning(
                        result.backend.name,
                        result.variant.format,
                        result.reason,
                        tuning,
                        available_vram,
                    ),
                }
                launched["debug"] = debug

                # Persist key fields to Redis so status() can surface them
                self._set_state(base_model, {
                    "backend": result.backend.name,
                    "variant": result.variant.format,
                    "debug_est_vram": str(debug["estimated_vram_gb"] or ""),
                    "debug_avail_vram": str(debug["available_vram_gb"] or ""),
                    "debug_max_len": str(debug["max_model_len"] or ""),
                })
                return launched
            # Fall through to legacy path if selector finds nothing
            logger.warning(
                "ExecutionSelector found no runnable variant for %s "
                "(profile=%s). Falling back to legacy ensure path.",
                base_model, profile,
            )

        # ── Legacy path ────────────────────────────────────────────────
        # CPU models use a separate, simpler path — no GPU allocation needed
        if getattr(spec, "runtime", "gpu") == "cpu":
            return self._ensure_cpu(base_model, spec)

        try:
            return self._ensure_gpu(base_model, spec)
        except Exception as exc:
            # If the Docker nvidia runtime is missing AND a CPU fallback URL is
            # configured (LITE mode), transparently serve via cpu_runtime instead.
            fallback_url = self.s.auto_fallback_cpu_url
            if fallback_url and _is_nvidia_runtime_error(exc):
                logger.warning(
                    "GPU runtime unavailable for %s (%s). "
                    "Falling back to CPU runtime at %s",
                    base_model, exc, fallback_url,
                )
                return self._fallback_to_cpu_runtime(base_model, spec, fallback_url)
            raise

    def _fallback_to_cpu_runtime(
        self, base_model: str, spec: ModelSpec, cpu_url: str
    ) -> Dict[str, Any]:
        """Return the pre-running cpu_runtime service as the backend for this model."""
        cpu_alias = self.s.auto_fallback_cpu_alias
        api_base = f"{cpu_url.rstrip('/')}/v1"
        self._set_state(base_model, {
            "state": "READY",
            "model_alias": cpu_alias,
            "api_base": api_base,
            "runtime_type": "cpu",
            "gpu": "",
            "last_used": _now(),
        })
        return {
            "base_model": base_model,
            "model_alias": cpu_alias,
            "api_base": api_base,
            "container": "cpu_runtime",
            "gpu": "",
            "state": "READY",
            "runtime_type": "cpu",
            "fallback": True,
        }

    def _ensure_cpu(self, base_model: str, spec: ModelSpec) -> Dict[str, Any]:
        """
        Ensure a CPU (GGUF / llama.cpp) model container is running.

        No GPU reservation. No OOM fallback ladder.
        The container is lightweight enough that a single start attempt suffices.
        """
        if not self._acquire_lock(base_model):
            raise RuntimeError409("Model is being started/stopped by another request")

        try:
            st = self._get_state(base_model)
            c = self._container_get(spec.container_name)

            if c and self._container_is_running(c):
                self._set_state(base_model, {"state": "READY", "container_id": c.id, "last_used": _now()})
                return self._cpu_ensure_result(base_model, spec, c.id)

            if c and not self._container_is_running(c):
                self._set_state(base_model, {"state": "STARTING", "last_used": _now()})
                self._container_start(c)
                self._wait_ready(spec)
                self._set_state(base_model, {"state": "READY", "last_used": _now()})
                return self._cpu_ensure_result(base_model, spec, c.id)

            # Fresh start
            self._set_state(base_model, {"state": "STARTING", "last_used": _now()})
            cid = self._start_cpu_container(spec)
            self._set_state(base_model, {"container_id": cid})
            self._wait_ready(spec)
            self._log_ready(spec)
            self._set_state(base_model, {"state": "READY", "last_used": _now()})
            return self._cpu_ensure_result(base_model, spec, cid)

        except Exception as e:
            logger.error("CPU ENSURE ERROR for %s: %s", base_model, e)
            self._maybe_remove_container(spec.container_name)
            self._set_state(base_model, {"state": "ABSENT"})
            raise
        finally:
            self._release_lock(base_model)

    def _cpu_ensure_result(self, base_model: str, spec: ModelSpec, container_id: str) -> Dict[str, Any]:
        return {
            "base_model": base_model,
            "model_alias": spec.model_alias,
            "api_base": self._api_base(spec),
            "container": spec.container_name,
            "gpu": "",          # no GPU for CPU runtime
            "state": "READY",
            "runtime_type": "cpu",
        }

    # ── Learning Runtime: config cache ────────────────────────────────

    def _load_config_cache(self, base_model: str) -> Optional[Dict[str, Any]]:
        """
        Return the cached best_config for *base_model* if it has a success
        rate ≥ 70%.  Returns None if no cache, insufficient data, or unreliable.
        """
        try:
            raw = self.redis.get(_redis_config_cache_key(base_model))
            if not raw:
                return None
            data = json.loads(raw)
            success  = int(data.get("success_count", 0))
            failure  = int(data.get("failure_count", 0))
            total    = success + failure
            if total == 0:
                return None
            if success / total < 0.7:
                return None
            return data.get("best_config")
        except Exception as exc:
            logger.warning("Failed to read config cache for %s: %s", base_model, exc)
            return None

    def _update_config_cache_success(
        self, base_model: str, max_model_len: int, gpu: str
    ) -> None:
        """Record a successful launch config; persist to Redis for 30 days."""
        try:
            key  = _redis_config_cache_key(base_model)
            raw  = self.redis.get(key)
            data = json.loads(raw) if raw else {}
            data["best_config"]    = {"max_model_len": max_model_len, "gpu": gpu}
            data["success_count"]  = int(data.get("success_count", 0)) + 1
            self.redis.set(key, json.dumps(data), ex=86400 * 30)
            logger.info(
                '{"event":"config_cached","model":"%s","max_model_len":%d,'
                '"success_count":%d}',
                base_model, max_model_len, data["success_count"],
            )
        except Exception as exc:
            logger.warning("Failed to update config cache for %s: %s", base_model, exc)

    def _update_config_cache_failure(self, base_model: str) -> None:
        """Increment failure count in the config cache."""
        try:
            key  = _redis_config_cache_key(base_model)
            raw  = self.redis.get(key)
            data = json.loads(raw) if raw else {}
            data["failure_count"] = int(data.get("failure_count", 0)) + 1
            self.redis.set(key, json.dumps(data), ex=86400 * 30)
        except Exception as exc:
            logger.warning("Failed to update failure count for %s: %s", base_model, exc)

    # ── WARMING state: retry context ──────────────────────────────────

    def _set_retry_context(
        self, base_model: str, retry_step: int, max_model_len: int
    ) -> None:
        """Persist retry metadata so status() can expose ETA and step number."""
        try:
            self.redis.set(
                _redis_retry_key(base_model),
                json.dumps({
                    "retry_step":    retry_step,
                    "max_model_len": max_model_len,
                    "started_at":    _now(),
                }),
                ex=3600,
            )
            logger.info(
                '{"event":"model_warming","model":"%s","retry_step":%d,"max_model_len":%d}',
                base_model, retry_step, max_model_len,
            )
        except Exception as exc:
            logger.warning("Failed to set retry context for %s: %s", base_model, exc)

    def _clear_retry_context(self, base_model: str) -> None:
        """Remove retry metadata after success or terminal failure."""
        try:
            self.redis.delete(_redis_retry_key(base_model))
        except Exception:
            pass

    # ── GPU ensure ────────────────────────────────────────────────────

    def _ensure_gpu(self, base_model: str, spec: ModelSpec) -> Dict[str, Any]:
        """Original GPU (vLLM) ensure logic — unchanged."""
        if not self._acquire_lock(base_model):
            raise RuntimeError409("Model is being started/stopped by another request")

        gpu_reserved = False
        gpu = ""

        try:
            st = self._get_state(base_model)
            c = self._container_get(spec.container_name)

            if c and self._container_is_running(c):
                gpu = st.get("gpu", "")
                self._set_state(
                    base_model,
                    {"state": "READY", "container_id": c.id, "gpu": gpu, "last_used": _now()},
                )
                return {
                    "base_model": base_model,
                    "model_alias": spec.model_alias,
                    "api_base": self._api_base(spec),
                    "container": spec.container_name,
                    "gpu": gpu,
                    "state": "READY",
                    "runtime_type": "gpu",
                }

            if c and not self._container_is_running(c):
                gpu = st.get("gpu", "")
                if not gpu:
                    gpu = self._choose_gpu(spec)
                    self._gpu_reserve(gpu, base_model)
                    gpu_reserved = True
                    self._set_state(base_model, {"gpu": gpu})

                self._set_state(base_model, {"state": "STARTING", "last_used": _now(), "container_id": c.id})
                self._container_start(c)
                self._wait_ready(spec)
                self._log_ready(spec)
                self._set_state(base_model, {"state": "READY", "last_used": _now(), "gpu": gpu})
                return {
                    "base_model": base_model,
                    "model_alias": spec.model_alias,
                    "api_base": self._api_base(spec),
                    "container": spec.container_name,
                    "gpu": gpu,
                    "state": "READY",
                    "runtime_type": "gpu",
                }

            # Fresh start
            # ── Learning Runtime: start from cached best config if reliable ──
            full_ladder = self._max_len_ladder(spec)
            cached = self._load_config_cache(base_model)
            if cached and cached.get("max_model_len"):
                cached_len = int(cached["max_model_len"])
                if cached_len in full_ladder:
                    ladder = full_ladder[full_ladder.index(cached_len):]
                    logger.info(
                        "Config cache hit for %s: starting at max_model_len=%d",
                        base_model, cached_len,
                    )
                else:
                    ladder = full_ladder
            else:
                ladder = full_ladder

            self._set_state(base_model, {"state": "STARTING", "last_used": _now()})
            gpu = self._choose_gpu(spec)
            self._gpu_reserve(gpu, base_model)
            gpu_reserved = True

            last_exc = None

            for retry_step, L in enumerate(ladder):
                spec.max_model_len = L

                # Transition to WARMING on retry steps to distinguish
                # "first attempt" from "fallback attempt" in the UI / status API.
                if retry_step == 0:
                    self._set_state(base_model, {"trying_max_model_len": L})
                else:
                    self._set_state(base_model, {"state": "WARMING", "trying_max_model_len": L})
                    self._set_retry_context(base_model, retry_step, L)

                self._container_remove_if_exists(spec.container_name)

                try:
                    cid = self._start_container_fresh(spec, gpu)
                    self._set_state(base_model, {"container_id": cid, "gpu": gpu})
                    self._wait_ready(spec)
                    self._log_ready(spec)
                    self._set_state(base_model, {"best_max_model_len": L})
                    self._clear_retry_context(base_model)
                    self._update_config_cache_success(base_model, L, gpu)
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    logger.error("Attempt failed for max_model_len=%s: %s", L, e)
                    self._update_config_cache_failure(base_model)
                    self._container_remove_if_exists(spec.container_name)
                    time.sleep(1)

            if last_exc is not None:
                self._clear_retry_context(base_model)
                raise last_exc

            self._set_state(base_model, {"state": "READY", "last_used": _now(), "gpu": gpu})
            return {
                "base_model": base_model,
                "model_alias": spec.model_alias,
                "api_base": self._api_base(spec),
                "container": spec.container_name,
                "gpu": gpu,
                "state": "READY",
                "runtime_type": "gpu",
            }

        except Exception as e:
            logger.error("CRITICAL ENSURE ERROR for %s: %s", base_model, e)
            try:
                c = self._container_get(spec.container_name)
                if c:
                    logs = c.logs().decode("utf-8", errors="replace")[-2000:]
                    logger.error("\n🔥🔥🔥 vLLM LOGS (tail):\n%s\n🔥🔥🔥", logs)
            except Exception:
                pass

            try:
                self._maybe_remove_container(spec.container_name)
            finally:
                if gpu_reserved and gpu:
                    self._gpu_release(gpu, base_model)
                self._set_state(base_model, {"state": "ABSENT"})
            raise
        finally:
            self._release_lock(base_model)

    def stop(self, base_model: str) -> Dict[str, Any]:
        spec = self._spec(base_model)
        base_model = spec.base_model  # normalize alias → canonical registry key
        if not self._acquire_lock(base_model):
            # Allow stop to proceed on a FAILED model — the lock may be stale
            # from an ensure() that never completed (e.g. container OOM crash).
            st = self._get_state(base_model)
            if st.get("state") != "FAILED":
                raise RuntimeError409("Model locked")
            logger.warning("stop: force-clearing stale lock for FAILED model %s", base_model)
            self._release_lock(base_model)
            if not self._acquire_lock(base_model):
                raise RuntimeError409("Model locked")

        try:
            st = self._get_state(base_model)
            gpu = st.get("gpu", "")
            c = self._container_get(spec.container_name)
            if c and self._container_is_running(c):
                self._set_state(base_model, {"state": "STOPPING"})
                self._container_stop(c, timeout=30)

            if gpu:
                self._gpu_release(gpu, base_model)
            else:
                for g in spec.allowed_gpus:
                    self._gpu_release(g, base_model)

            self._set_state(base_model, {"state": "STOPPED"})
            return {"base_model": base_model, "state": "STOPPED"}
        finally:
            self._release_lock(base_model)

    def remove(self, base_model: str) -> Dict[str, Any]:
        spec = self._spec(base_model)
        base_model = spec.base_model  # normalize alias → canonical registry key
        if not self._acquire_lock(base_model):
            # Allow remove on a FAILED model — stale lock from crashed ensure().
            st = self._get_state(base_model)
            if st.get("state") != "FAILED":
                raise RuntimeError409("Model locked")
            logger.warning("remove: force-clearing stale lock for FAILED model %s", base_model)
            self._release_lock(base_model)
            if not self._acquire_lock(base_model):
                raise RuntimeError409("Model locked")

        try:
            st = self._get_state(base_model)
            gpu = st.get("gpu", "")
            self._container_remove_if_exists(spec.container_name)
            if gpu:
                self._gpu_release(gpu, base_model)
            self._set_state(base_model, {"state": "ABSENT", "container_id": "", "gpu": ""})
            return {"base_model": base_model, "state": "ABSENT"}
        finally:
            self._release_lock(base_model)

    def status(self, base_model: str) -> Dict[str, Any]:
        spec = self._spec(base_model)
        base_model = spec.base_model  # normalize alias → canonical registry key
        st = self._get_state(base_model)
        c = self._container_get(spec.container_name)
        running = bool(c and self._container_is_running(c))
        state = st.get("state", "ABSENT")
        if running and state != "READY":
            state = "READY"
        if (not running) and state not in ("ABSENT", "STOPPING", "STARTING", "STOPPED", "WARMING"):
            state = "STOPPED" if c else "ABSENT"

        active_loras = []
        try:
            # Використовуємо ту саму логіку ключів, що і при старті
            key = _redis_key_loras(base_model)
            if self.redis.exists(key):
                active_loras = sorted(list(self.redis.smembers(key)))
        except Exception as e:
            logger.error(f"Failed to read active loras: {e}")

        # ── Optional debug/explainability fields (set during ensure_running) ──
        backend  = st.get("backend", "")
        variant  = st.get("variant", "")
        debug: Optional[Dict[str, Any]] = None
        if backend:
            def _maybe_float(v: str) -> Optional[float]:
                try: return float(v) if v else None
                except (ValueError, TypeError): return None

            def _maybe_int(v: str) -> Optional[int]:
                try: return int(v) if v else None
                except (ValueError, TypeError): return None

            debug = {
                "selected_backend": backend,
                "selected_variant": variant,
                "estimated_vram_gb": _maybe_float(st.get("debug_est_vram", "")),
                "available_vram_gb": _maybe_float(st.get("debug_avail_vram", "")),
                "max_model_len":     _maybe_int(st.get("debug_max_len", "")),
            }

        out = {
            "base_model": base_model,
            "model_alias": spec.model_alias,
            "container": spec.container_name,
            "api_base": self._api_base(spec),
            "state": state,
            "running": running,
            "gpu": st.get("gpu", ""),
            "last_used": st.get("last_used", ""),
            "active_loras": active_loras,
            "active_loras_count": len(active_loras),
            # Optional — present only when backend selection was performed
            "backend": backend,
            "model_variant": variant,
        }
        if debug:
            out["debug"] = debug

        # Retry / WARMING metadata — read from mrm:retry:{model}
        retry_info = None
        try:
            retry_raw = self.redis.get(_redis_retry_key(base_model))
            if retry_raw:
                retry_info = json.loads(retry_raw)
        except Exception:
            pass
        if retry_info:
            retry_step = int(retry_info.get("retry_step", 0))
            out.update({
                "retry_step": retry_step,
                "max_model_len": int(retry_info.get("max_model_len", 0)) or None,
                "eta_sec": _estimate_eta(retry_step),
            })

        # Only write back state if we actually normalised it (don't clobber WARMING)
        if state not in ("WARMING",):
            self._set_state(base_model, {"state": state})
        return out

    def status_all(self) -> List[Dict[str, Any]]:
        return [self.status(bm) for bm in self.registry.keys()]

    def sweep_idle(self) -> List[str]:
        stopped = []
        now = _now()
        for bm, spec in self.registry.items():
            st = self._get_state(bm)
            state = st.get("state", "ABSENT")
            # Never evict models that are mid-launch or OOM-retrying
            if state in ("STARTING", "WARMING"):
                continue
            last_used = int(st.get("last_used") or "0")
            if last_used == 0:
                continue
            if now - last_used < self.s.idle_timeout_sec:
                continue

            c = self._container_get(spec.container_name)
            if not c:
                continue

            if self._container_is_running(c):
                logger.info("🧹 Sweeping idle model: %s (idle=%ss)", bm, (now - last_used))
                try:
                    self.stop(bm)
                    stopped.append(bm)
                except Exception as e:
                    logger.error("Failed to sweep %s: %s", bm, e)
        return stopped


async def idle_reaper_loop(mrm_instance: ModelRuntimeManager, interval_sec: int):
    logger.info("Starting Idle Reaper (interval=%ss)", interval_sec)
    REAPER_OP_TIMEOUT = 45.0

    while True:
        try:
            stopped = await asyncio.wait_for(asyncio.to_thread(mrm_instance.sweep_idle), timeout=REAPER_OP_TIMEOUT)
            if stopped:
                logger.info("Reaper collected idle models: %s", stopped)
        except asyncio.TimeoutError:
            logger.error("Reaper operation timed out (> %ss). Docker might be slow/hanging.", REAPER_OP_TIMEOUT)
        except Exception as e:
            logger.error("Reaper crashed: %s", e, exc_info=True)

        await asyncio.sleep(interval_sec)

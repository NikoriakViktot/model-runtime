"""
engines/vllm.py

vLLM engine adapter.

All vLLM-specific knowledge is contained here.  No vLLM flag names, env
variables, or path conventions should appear anywhere else in the codebase.

vLLM is treated as an external service: this adapter configures a container
that *runs* vLLM; it does not import or call vLLM Python internals.

Container assumptions
---------------------
The default image ``vllm/vllm-openai`` has the entrypoint::

    python -m vllm.entrypoints.openai.api_server

So ``build_command()`` returns only the model path and flags (argv), which
Docker appends to that entrypoint.  If you use a custom image without this
entrypoint, prepend ``["python", "-m", "vllm.entrypoints.openai.api_server"]``
or set ``RuntimeSpec.extra_args["entrypoint_mode"] = "full"``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from engines.base import EngineAdapter

if TYPE_CHECKING:
    pass

from core.specs import AdapterConfig, AdapterMount, FallbackSpec, RuntimeSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_IMAGE = "vllm/vllm-openai:latest"
_HF_CACHE_CONTAINER_PATH = "/root/.cache/huggingface"
_LORA_BASE_CONTAINER_PATH = "/lora"

# Extra vLLM args that are read from RuntimeSpec.extra_args
_EXTRA_BOOL_FLAGS = {"enforce_eager", "trust_remote_code", "disable_log_requests"}
_EXTRA_VALUE_FLAGS = {
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_loras",
    "max_lora_rank",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "served_model_name",
}


class VLLMAdapter(EngineAdapter):
    """
    EngineAdapter for vLLM.

    This class is the single authoritative source for how to configure a vLLM
    container.  The RuntimeManager and all other components are unaware of
    vLLM internals.
    """

    engine_type = "vllm"
    default_port = 8000

    def default_image(self) -> str:
        return _DEFAULT_IMAGE

    # ------------------------------------------------------------------
    # Container specification
    # ------------------------------------------------------------------

    def build_command(
        self,
        spec: RuntimeSpec,
        adapter_mounts: list[AdapterMount],
    ) -> list[str]:
        """
        Build argv for the vLLM OpenAI-compatible server.

        These args are appended to the container's ENTRYPOINT
        (``python -m vllm.entrypoints.openai.api_server``).
        """
        model_id = spec.hf_model

        cmd: list[str] = [
            model_id,
            "--host", "0.0.0.0",
            "--port", str(spec.container_port),
            "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
            "--dtype", spec.dtype,
        ]

        if spec.max_model_len is not None:
            cmd += ["--max-model-len", str(spec.max_model_len)]

        if spec.quantization is not None:
            cmd += ["--quantization", spec.quantization]

        # LoRA support
        if adapter_mounts:
            cmd += ["--enable-lora"]
            for mount in adapter_mounts:
                # vLLM convention: --lora-modules <id>=<container_path>
                cmd += ["--lora-modules", f"{mount.adapter_id}={mount.container_path}"]

        # Optional vLLM flags from extra_args
        for flag in _EXTRA_BOOL_FLAGS:
            if spec.extra_args.get(flag) in ("1", "true", "True", "yes"):
                cmd += [f"--{flag.replace('_', '-')}"]

        for flag in _EXTRA_VALUE_FLAGS:
            value = spec.extra_args.get(flag)
            if value is not None:
                cmd += [f"--{flag.replace('_', '-')}", str(value)]

        return cmd

    def build_env(
        self,
        spec: RuntimeSpec,
        gpu_index: str,
    ) -> dict[str, str]:
        """
        Environment variables required by vLLM and the NVIDIA runtime.
        """
        env: dict[str, str] = {
            "NVIDIA_VISIBLE_DEVICES": gpu_index,
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            # Recommended for large PyTorch allocations
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }

        if spec.hf_token:
            env["HF_TOKEN"] = spec.hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = spec.hf_token

        # Allow adding LoRA adapters to a running vLLM server via its API.
        # Only set when adapters are registered — unnecessary overhead otherwise.
        if spec.adapters:
            env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"

        # Pass through any env overrides from spec (e.g. VLLM_WORKER_MULTIPROC_METHOD)
        for key, value in spec.extra_args.items():
            if key.startswith("env_"):
                env[key[4:].upper()] = str(value)

        return env

    def build_volumes(
        self,
        spec: RuntimeSpec,
        adapter_mounts: list[AdapterMount],
    ) -> dict[str, dict]:
        """
        Volume mounts in Docker SDK format.

        The HuggingFace model cache is mounted read-only to avoid re-downloads.
        Each adapter is mounted at its own path under ``/lora/``.
        """
        volumes: dict[str, dict] = {
            spec.hf_cache_host_path: {
                "bind": _HF_CACHE_CONTAINER_PATH,
                "mode": "ro",
            },
        }

        for mount in adapter_mounts:
            volumes[mount.host_path] = {
                "bind": mount.container_path,
                "mode": mount.mode,
            }

        return volumes

    # ------------------------------------------------------------------
    # Readiness detection
    # ------------------------------------------------------------------

    def health_check_url(self, host: str, port: int) -> str:
        """vLLM exposes GET /health; returns 200 when the model is loaded."""
        return f"http://{host}:{port}/health"

    async def is_ready(self, host: str, port: int) -> bool:
        """
        Return True when vLLM responds 200 on /health.

        Never raises — connection errors return False so the caller can retry.
        """
        url = self.health_check_url(host, port)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
            return False
        except Exception as exc:
            logger.debug("vLLM health check error at %s: %r", url, exc)
            return False

    # ------------------------------------------------------------------
    # Inference routing
    # ------------------------------------------------------------------

    def inference_base_url(self, host: str, port: int) -> str:
        """vLLM serves an OpenAI-compatible API under /v1."""
        return f"http://{host}:{port}/v1"

    # ------------------------------------------------------------------
    # OOM recovery
    # ------------------------------------------------------------------

    def get_fallback_specs(self, spec: RuntimeSpec) -> list[FallbackSpec]:
        """
        Progressively reduce ``max_model_len`` to recover from CUDA OOM.

        Steps below the current configured value are returned in descending
        order so the first retry uses the largest acceptable reduction.
        Return empty when the context is already at the floor.
        """
        current = spec.max_model_len or 8192
        ladder = [4096, 3072, 2048, 1536, 1024]
        return [
            FallbackSpec(
                label=f"max_model_len={length}",
                overrides={"max_model_len": length},
            )
            for length in ladder
            if length < current
        ]

    # ------------------------------------------------------------------
    # Adapter support
    # ------------------------------------------------------------------

    def supports_adapters(self) -> bool:
        return True

    def build_adapter_mount(self, config: AdapterConfig) -> AdapterMount:
        """
        vLLM expects each LoRA adapter in its own directory under /lora/.

        The ``--lora-modules`` flag maps ``adapter_id`` → container path,
        so vLLM serves the adapter under that name in its API.
        """
        return AdapterMount(
            adapter_id=config.adapter_id,
            host_path=config.local_path,
            container_path=f"{_LORA_BASE_CONTAINER_PATH}/{config.adapter_id}",
            mode="ro",
        )

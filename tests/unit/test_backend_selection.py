"""
tests/unit/test_backend_selection.py

Tests for the multi-backend selection layer:
    - NodeCapabilities.from_dict()
    - VLLMBackend.can_run / estimate_cost
    - LlamaCppBackend.can_run / estimate_cost / launch simulation
    - CpuRuntimeBackend.can_run / estimate_cost
    - ExecutionSelector: normal selection, profile routing, fallback logic
    - ensure_running() with profile and variant registry
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from model_runtime_manager.backends.base import NodeCapabilities, Instance
from model_runtime_manager.backends.vllm_backend import VLLMBackend
from model_runtime_manager.backends.llama_cpp_backend import LlamaCppBackend
from model_runtime_manager.backends.cpu_runtime_backend import CpuRuntimeBackend
from model_runtime_manager.model_variant_registry import ModelVariant, VariantRegistry
from model_runtime_manager.execution_selector import ExecutionSelector


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_mrm():
    """Minimal ModelRuntimeManager mock — no Docker, no Redis."""
    mrm = MagicMock()
    mrm.s.auto_fallback_cpu_url = ""
    mrm.s.auto_fallback_cpu_alias = "cpu-model"
    return mrm


def _gpu_caps(vram_mb: int = 24576, ram_mb: int = 65536, cpu_cores: int = 16) -> NodeCapabilities:
    return NodeCapabilities(
        gpu=True,
        vram_mb=vram_mb,
        ram_mb=ram_mb,
        cpu_cores=cpu_cores,
        supported_backends=["vllm", "llama.cpp", "cpu"],
    )


def _cpu_caps(ram_mb: int = 32768, cpu_cores: int = 8) -> NodeCapabilities:
    return NodeCapabilities(
        gpu=False,
        vram_mb=0,
        ram_mb=ram_mb,
        cpu_cores=cpu_cores,
        supported_backends=["llama.cpp", "cpu"],
    )


def _fp16_variant(size_gb: float = 3.5) -> ModelVariant:
    return ModelVariant(
        base_model_id="test/model",
        format="fp16",
        size_gb=size_gb,
        backend_compatibility=["vllm"],
    )


def _awq_variant(size_gb: float = 2.0) -> ModelVariant:
    return ModelVariant(
        base_model_id="test/model",
        format="awq",
        quantization="awq",
        size_gb=size_gb,
        backend_compatibility=["vllm"],
    )


def _gguf_q8_variant(size_gb: float = 3.0) -> ModelVariant:
    return ModelVariant(
        base_model_id="test/model",
        format="gguf_q8",
        size_gb=size_gb,
        gguf_path="/models/model-q8.gguf",
        backend_compatibility=["llama.cpp", "cpu"],
        n_ctx=2048,
    )


def _gguf_q4_variant(size_gb: float = 1.5) -> ModelVariant:
    return ModelVariant(
        base_model_id="test/model",
        format="gguf_q4",
        size_gb=size_gb,
        gguf_path="/models/model-q4.gguf",
        backend_compatibility=["llama.cpp", "cpu"],
        n_ctx=2048,
    )


def _registry_with_all_variants() -> VariantRegistry:
    reg = VariantRegistry()
    reg.register("test/model", [
        _fp16_variant(),
        _awq_variant(),
        _gguf_q8_variant(),
        _gguf_q4_variant(),
    ])
    return reg


# ── NodeCapabilities ──────────────────────────────────────────────────────────

class TestNodeCapabilities:
    def test_from_dict_gpu(self):
        caps = NodeCapabilities.from_dict({
            "gpu": True,
            "vram_mb": 24576,
            "ram_mb": 65536,
            "cpu_cores": 16,
            "supported_backends": ["vllm", "llama.cpp", "cpu"],
        })
        assert caps.gpu is True
        assert caps.vram_mb == 24576
        assert caps.ram_mb == 65536
        assert caps.cpu_cores == 16
        assert "vllm" in caps.supported_backends

    def test_from_dict_cpu_only(self):
        caps = NodeCapabilities.from_dict({
            "gpu": False,
            "vram_mb": 0,
            "ram_mb": 32768,
            "cpu_cores": 8,
            "supported_backends": ["llama.cpp", "cpu"],
        })
        assert caps.gpu is False
        assert caps.vram_mb == 0

    def test_from_dict_defaults(self):
        caps = NodeCapabilities.from_dict({})
        assert caps.gpu is False
        assert caps.vram_mb == 0
        assert caps.cpu_cores == 4


# ── VLLMBackend ───────────────────────────────────────────────────────────────

class TestVLLMBackend:
    def setup_method(self):
        self.backend = VLLMBackend(_make_mrm())

    def test_can_run_with_sufficient_vram(self):
        variant = _fp16_variant(size_gb=3.5)  # needs ~4300 MiB
        caps = _gpu_caps(vram_mb=8192)
        assert self.backend.can_run(variant, caps) is True

    def test_cannot_run_without_gpu(self):
        variant = _fp16_variant(size_gb=3.5)
        caps = _cpu_caps()
        assert self.backend.can_run(variant, caps) is False

    def test_cannot_run_with_insufficient_vram(self):
        # fp16 3.5GB needs ~4300 MiB (with 1.2 overhead)
        variant = _fp16_variant(size_gb=3.5)
        caps = _gpu_caps(vram_mb=2048)
        assert self.backend.can_run(variant, caps) is False

    def test_cannot_run_wrong_backend_compatibility(self):
        variant = _gguf_q4_variant()  # only llama.cpp + cpu
        caps = _gpu_caps(vram_mb=24576)
        assert self.backend.can_run(variant, caps) is False

    def test_cannot_run_when_vllm_not_in_supported_backends(self):
        variant = _fp16_variant(size_gb=3.5)
        caps = NodeCapabilities(gpu=True, vram_mb=24576, ram_mb=65536, cpu_cores=16,
                                supported_backends=["llama.cpp"])
        assert self.backend.can_run(variant, caps) is False

    def test_estimate_cost_scales_with_vram_ratio(self):
        variant = _fp16_variant(size_gb=3.5)
        low_vram = _gpu_caps(vram_mb=8192)
        high_vram = _gpu_caps(vram_mb=24576)
        assert self.backend.estimate_cost(variant, low_vram) > self.backend.estimate_cost(variant, high_vram)

    def test_estimate_cost_infinite_without_vram(self):
        variant = _fp16_variant()
        caps = NodeCapabilities(gpu=True, vram_mb=0, ram_mb=65536)
        assert self.backend.estimate_cost(variant, caps) == float("inf")

    def test_metadata_flags(self):
        assert VLLMBackend.supports_flash_attention is True
        assert VLLMBackend.supports_continuous_batching is True
        assert VLLMBackend.name == "vllm"


# ── LlamaCppBackend ───────────────────────────────────────────────────────────

class TestLlamaCppBackend:
    def setup_method(self):
        self.backend = LlamaCppBackend(_make_mrm())

    def test_can_run_gguf_q4_with_sufficient_ram(self):
        # q4 1.5GB needs ~2000 MiB (1.5 * 1024 * 1.3)
        variant = _gguf_q4_variant(size_gb=1.5)
        caps = _cpu_caps(ram_mb=8192)
        assert self.backend.can_run(variant, caps) is True

    def test_can_run_gguf_q8(self):
        variant = _gguf_q8_variant(size_gb=3.0)
        caps = _cpu_caps(ram_mb=16384)
        assert self.backend.can_run(variant, caps) is True

    def test_cannot_run_fp16_format(self):
        variant = _fp16_variant()
        caps = _cpu_caps(ram_mb=65536)
        assert self.backend.can_run(variant, caps) is False

    def test_cannot_run_with_insufficient_ram(self):
        variant = _gguf_q4_variant(size_gb=7.0)  # needs ~9400 MiB
        caps = _cpu_caps(ram_mb=4096)
        assert self.backend.can_run(variant, caps) is False

    def test_cannot_run_when_not_in_supported_backends(self):
        variant = _gguf_q4_variant()
        caps = NodeCapabilities(gpu=False, vram_mb=0, ram_mb=32768,
                                supported_backends=["cpu"])
        assert self.backend.can_run(variant, caps) is False

    def test_estimate_cost_has_penalty_over_vllm(self):
        mrm = _make_mrm()
        vllm = VLLMBackend(mrm)
        llama = LlamaCppBackend(mrm)
        fp16 = _fp16_variant(size_gb=3.5)
        gguf = _gguf_q4_variant(size_gb=1.5)
        caps = _gpu_caps()
        # On a GPU node, vllm cost < llama.cpp cost
        assert vllm.estimate_cost(fp16, caps) < llama.estimate_cost(gguf, caps)

    def test_launch_simulation(self):
        """Simulate launch: _ensure_cpu returns a proper dict, backend adds fields."""
        mrm = _make_mrm()
        from model_runtime_manager.mrm.config import ModelSpec
        spec = MagicMock(spec=ModelSpec)
        spec.model_dump.return_value = {
            "base_model": "test/model",
            "model_alias": "test",
            "container_name": "cpu_test",
            "image": "model-runtime-cpu:latest",
            "launch_mode": "openai",
            "hf_model": "test/model",
            "served_model_name": "test",
            "gpu_memory_utilization": 0.6,
            "max_model_len": 2048,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 1024,
            "enable_lora": False,
            "max_loras": 0,
            "max_lora_rank": 0,
            "enforce_eager": False,
            "port": 8090,
            "allowed_gpus": [],
            "quantization": None,
            "volumes": {},
            "env": {},
            "ipc_host": False,
            "dtype": "auto",
            "shm_size": "4gb",
            "health_path": "/health",
            "runtime": "gpu",
            "gguf_path": "",
            "cpu_cores": 4,
            "memory_mb": 4096,
        }
        mrm.registry = {"test/model": spec}
        mrm._spec.return_value = spec
        mrm._ensure_cpu.return_value = {
            "base_model": "test/model",
            "model_alias": "test",
            "api_base": "http://cpu_test:8090/v1",
            "container": "cpu_test",
            "gpu": "",
            "state": "READY",
            "runtime_type": "cpu",
        }

        backend = LlamaCppBackend(mrm)
        variant = _gguf_q4_variant(size_gb=1.5)
        caps = _cpu_caps(ram_mb=8192, cpu_cores=4)

        result = backend.launch("test/model", variant, caps)

        assert result["backend"] == "llama.cpp"
        assert result["model_variant"] == "gguf_q4"
        assert result["ram_mb"] > 0
        assert result["state"] == "READY"
        mrm._ensure_cpu.assert_called_once()

    def test_launch_restores_original_spec(self):
        """Ensure that the original registry spec is restored after launch."""
        mrm = _make_mrm()
        from model_runtime_manager.mrm.config import ModelSpec
        original_spec = MagicMock(spec=ModelSpec)
        original_spec.model_dump.return_value = {
            "base_model": "test/model", "model_alias": "test",
            "container_name": "vllm_test", "image": "vllm/vllm-openai:latest",
            "launch_mode": "openai", "hf_model": "test/model",
            "served_model_name": "test", "gpu_memory_utilization": 0.6,
            "max_model_len": 2048, "max_num_seqs": 1, "max_num_batched_tokens": 1024,
            "enable_lora": False, "max_loras": 0, "max_lora_rank": 0,
            "enforce_eager": False, "port": 8000, "allowed_gpus": ["0"],
            "quantization": None, "volumes": {}, "env": {},
            "ipc_host": True, "dtype": "auto", "shm_size": "8gb",
            "health_path": "/health", "runtime": "gpu",
            "gguf_path": "", "cpu_cores": 4, "memory_mb": 6144,
        }
        mrm.registry = {"test/model": original_spec}
        mrm._spec.return_value = original_spec
        mrm._ensure_cpu.return_value = {"state": "READY", "gpu": "", "runtime_type": "cpu",
                                         "api_base": "x", "base_model": "test/model",
                                         "model_alias": "test", "container": "c"}

        backend = LlamaCppBackend(mrm)
        variant = _gguf_q4_variant()
        backend.launch("test/model", variant, _cpu_caps())

        # Original spec must be restored
        assert mrm.registry["test/model"] is original_spec


# ── CpuRuntimeBackend ─────────────────────────────────────────────────────────

class TestCpuRuntimeBackend:
    def setup_method(self):
        self.backend = CpuRuntimeBackend(_make_mrm())

    def test_can_run_with_sufficient_ram(self):
        variant = _gguf_q4_variant(size_gb=1.5)
        caps = _cpu_caps(ram_mb=8192)
        assert self.backend.can_run(variant, caps) is True

    def test_cannot_run_with_insufficient_ram(self):
        variant = _gguf_q4_variant(size_gb=10.0)
        caps = _cpu_caps(ram_mb=4096)
        assert self.backend.can_run(variant, caps) is False

    def test_cannot_run_when_not_in_supported_backends(self):
        variant = _gguf_q4_variant()
        caps = NodeCapabilities(gpu=False, ram_mb=32768, supported_backends=["llama.cpp"])
        assert self.backend.can_run(variant, caps) is False

    def test_cost_penalty_higher_than_llama_cpp(self):
        mrm = _make_mrm()
        llama = LlamaCppBackend(mrm)
        cpu = CpuRuntimeBackend(mrm)
        variant = _gguf_q4_variant(size_gb=1.5)
        caps = _cpu_caps(ram_mb=32768)
        assert cpu.estimate_cost(variant, caps) > llama.estimate_cost(variant, caps)


# ── ExecutionSelector: backend selection ─────────────────────────────────────

class TestExecutionSelectorSelection:
    def _make_selector(self, mrm=None) -> ExecutionSelector:
        if mrm is None:
            mrm = _make_mrm()
        backends = [VLLMBackend(mrm), LlamaCppBackend(mrm), CpuRuntimeBackend(mrm)]
        registry = _registry_with_all_variants()
        return ExecutionSelector(backends, registry)

    def test_selects_vllm_on_gpu_node(self):
        selector = self._make_selector()
        caps = _gpu_caps(vram_mb=24576)
        result = selector.select("test/model", caps)
        assert result is not None
        assert result.backend.name == "vllm"

    def test_selects_llama_cpp_on_cpu_node(self):
        selector = self._make_selector()
        caps = _cpu_caps(ram_mb=16384)
        result = selector.select("test/model", caps)
        assert result is not None
        assert result.backend.name == "llama.cpp"

    def test_returns_none_for_unknown_model(self):
        selector = self._make_selector()
        caps = _gpu_caps()
        result = selector.select("nonexistent/model", caps)
        assert result is None

    def test_returns_none_when_no_variant_fits(self):
        reg = VariantRegistry()
        reg.register("big/model", [
            ModelVariant("big/model", "fp16", size_gb=200.0, backend_compatibility=["vllm"]),
        ])
        backends = [VLLMBackend(_make_mrm())]
        selector = ExecutionSelector(backends, reg)
        # Tiny node — nothing fits
        caps = NodeCapabilities(gpu=True, vram_mb=1024, ram_mb=2048, supported_backends=["vllm"])
        result = selector.select("big/model", caps)
        assert result is None


# ── ExecutionSelector: profile routing ───────────────────────────────────────

class TestExecutionSelectorProfileRouting:
    def _make_selector(self) -> ExecutionSelector:
        mrm = _make_mrm()
        backends = [VLLMBackend(mrm), LlamaCppBackend(mrm), CpuRuntimeBackend(mrm)]
        registry = _registry_with_all_variants()
        return ExecutionSelector(backends, registry)

    def test_quality_profile_selects_fp16(self):
        selector = self._make_selector()
        caps = _gpu_caps(vram_mb=24576)
        result = selector.select("test/model", caps, profile="quality")
        assert result is not None
        assert result.variant.format == "fp16"
        assert result.backend.name == "vllm"

    def test_balanced_profile_prefers_awq(self):
        selector = self._make_selector()
        caps = _gpu_caps(vram_mb=8192)
        result = selector.select("test/model", caps, profile="balanced")
        assert result is not None
        assert result.variant.format in ("awq", "fp16")
        assert result.backend.name == "vllm"

    def test_cheap_profile_selects_gguf_q4(self):
        selector = self._make_selector()
        caps = _cpu_caps(ram_mb=32768)
        result = selector.select("test/model", caps, profile="cheap")
        assert result is not None
        assert result.variant.format in ("gguf_q4", "gguf_q8")
        assert result.backend.name == "llama.cpp"

    def test_unknown_profile_falls_back_to_default(self):
        """Unknown profile names are treated as no-profile (use all variants)."""
        selector = self._make_selector()
        caps = _gpu_caps(vram_mb=24576)
        result = selector.select("test/model", caps, profile="nonexistent")
        assert result is not None  # still picks something

    def test_no_profile_selects_best_backend(self):
        selector = self._make_selector()
        caps = _gpu_caps(vram_mb=24576)
        result = selector.select("test/model", caps, profile=None)
        assert result is not None
        assert result.backend.name == "vllm"


# ── ExecutionSelector: quantization fallback logic ───────────────────────────

class TestExecutionSelectorFallback:
    def test_falls_back_from_fp16_to_awq_when_vram_limited(self):
        """fp16 doesn't fit, awq does — selector picks awq."""
        # fp16 = 14GB (needs ~17000 MiB), awq = 5GB (needs ~6100 MiB)
        reg = VariantRegistry()
        reg.register("big/model", [
            ModelVariant("big/model", "fp16", size_gb=14.0, backend_compatibility=["vllm"]),
            ModelVariant("big/model", "awq", quantization="awq", size_gb=5.0,
                         backend_compatibility=["vllm"]),
        ])
        mrm = _make_mrm()
        backends = [VLLMBackend(mrm)]
        selector = ExecutionSelector(backends, reg)

        # 8GB VRAM: fp16 doesn't fit, awq does
        caps = NodeCapabilities(gpu=True, vram_mb=8192, ram_mb=65536,
                                supported_backends=["vllm"])
        result = selector.select("big/model", caps)
        assert result is not None
        assert result.variant.format == "awq"

    def test_falls_back_to_gguf_when_no_vram(self):
        """No GPU → falls back to gguf_q4 via llama.cpp."""
        mrm = _make_mrm()
        backends = [VLLMBackend(mrm), LlamaCppBackend(mrm)]
        registry = _registry_with_all_variants()
        selector = ExecutionSelector(backends, registry)

        caps = _cpu_caps(ram_mb=32768)
        result = selector.select("test/model", caps)
        assert result is not None
        assert result.variant.format in ("gguf_q4", "gguf_q8")

    def test_fallback_order_fp16_awq_gguf_q8_gguf_q4(self):
        """Fallback tries fp16→awq→gguf_q8→gguf_q4. Returns first that fits."""
        reg = VariantRegistry()
        # Only gguf_q4 fits on this tiny node
        reg.register("test/model", [
            ModelVariant("test/model", "fp16", size_gb=14.0, backend_compatibility=["vllm"]),
            ModelVariant("test/model", "awq", size_gb=7.0, backend_compatibility=["vllm"]),
            ModelVariant("test/model", "gguf_q8", size_gb=7.0, backend_compatibility=["llama.cpp"]),
            ModelVariant("test/model", "gguf_q4", size_gb=4.0, backend_compatibility=["llama.cpp"]),
        ])
        mrm = _make_mrm()
        backends = [VLLMBackend(mrm), LlamaCppBackend(mrm)]
        selector = ExecutionSelector(backends, reg)

        # Small node: no GPU, 8GB RAM → only gguf_q4 fits (4GB * 1.3 = ~5GB)
        caps = NodeCapabilities(gpu=False, vram_mb=0, ram_mb=8192,
                                supported_backends=["llama.cpp"])
        result = selector.select("test/model", caps)
        assert result is not None
        assert result.variant.format == "gguf_q4"

    def test_returns_none_when_nothing_fits_after_fallback(self):
        reg = VariantRegistry()
        reg.register("test/model", [
            ModelVariant("test/model", "fp16", size_gb=100.0, backend_compatibility=["vllm"]),
        ])
        mrm = _make_mrm()
        backends = [VLLMBackend(mrm)]
        selector = ExecutionSelector(backends, reg)

        caps = NodeCapabilities(gpu=True, vram_mb=1024, ram_mb=2048, supported_backends=["vllm"])
        result = selector.select("test/model", caps)
        assert result is None


# ── ensure_running() with variant registry ───────────────────────────────────

class TestEnsureRunningWithVariantRegistry:
    """Integration test: ensure_running() routes through ExecutionSelector when variants registered."""

    def _make_mrm_with_variants(self, variants):
        """Return a real ModelRuntimeManager skeleton with injected mocks."""
        with (
            patch("docker.from_env"),
            patch("redis.Redis.from_url"),
        ):
            from model_runtime_manager.mrm.config import Settings, ModelSpec
            from model_runtime_manager.mrm.runtime import ModelRuntimeManager

            settings = MagicMock(spec=Settings)
            settings.auto_fallback_cpu_url = ""
            settings.auto_fallback_cpu_alias = "cpu-model"
            settings.idle_timeout_sec = 900
            settings.redis_url = "redis://localhost:6379/0"
            settings.docker_network = "test_net"
            settings.keep_failed_containers = False
            settings.model_registry = {}
            settings.load_default_registry.return_value = {}

            spec = MagicMock(spec=ModelSpec)
            spec.runtime = "gpu"
            spec.model_alias = "test-model"
            spec.allowed_gpus = ["0"]

            reg = VariantRegistry()
            reg.register("test/model", variants)

            mrm = ModelRuntimeManager.__new__(ModelRuntimeManager)
            mrm.s = settings
            mrm.redis = MagicMock()
            mrm.docker = MagicMock()
            mrm.registry = {"test/model": spec}

            # Inject real variant registry + fresh backends
            from model_runtime_manager.backends.vllm_backend import VLLMBackend
            from model_runtime_manager.backends.llama_cpp_backend import LlamaCppBackend
            from model_runtime_manager.backends.cpu_runtime_backend import CpuRuntimeBackend
            from model_runtime_manager.execution_selector import ExecutionSelector

            mrm._variant_registry = reg
            mrm._backends = [VLLMBackend(mrm), LlamaCppBackend(mrm), CpuRuntimeBackend(mrm)]
            mrm._execution_selector = ExecutionSelector(mrm._backends, reg)
            return mrm, spec

    def test_ensure_routes_through_selector_for_registered_variants(self):
        variants = [_fp16_variant(size_gb=3.5)]
        mrm, spec = self._make_mrm_with_variants(variants)

        expected = {
            "base_model": "test/model", "model_alias": "test-model",
            "api_base": "http://x:8000/v1", "container": "c",
            "gpu": "0", "state": "READY", "runtime_type": "gpu",
        }
        with patch.object(mrm, "_ensure_gpu", return_value=expected) as mock_gpu:
            with patch.object(mrm, "_detect_node_capabilities",
                              return_value=_gpu_caps(vram_mb=24576)):
                with patch.object(mrm, "_set_state"):
                    result = mrm.ensure_running("test/model")

        mock_gpu.assert_called_once()
        assert result["backend"] == "vllm"
        assert result["model_variant"] == "fp16"

    def test_quality_profile_uses_vllm_fp16(self):
        variants = [_fp16_variant(size_gb=3.5), _gguf_q4_variant(size_gb=1.5)]
        mrm, spec = self._make_mrm_with_variants(variants)

        expected = {
            "base_model": "test/model", "model_alias": "test-model",
            "api_base": "http://x:8000/v1", "container": "c",
            "gpu": "0", "state": "READY", "runtime_type": "gpu",
        }
        with patch.object(mrm, "_ensure_gpu", return_value=expected):
            with patch.object(mrm, "_detect_node_capabilities",
                              return_value=_gpu_caps(vram_mb=24576)):
                with patch.object(mrm, "_set_state"):
                    result = mrm.ensure_running("test/model", profile="quality")

        assert result["model_variant"] == "fp16"

    def test_cheap_profile_uses_llama_cpp_gguf(self):
        variants = [_fp16_variant(size_gb=3.5), _gguf_q4_variant(size_gb=1.5)]
        mrm, spec = self._make_mrm_with_variants(variants)

        spec.model_dump.return_value = {
            "base_model": "test/model", "model_alias": "test-model",
            "container_name": "cpu_c", "image": "model-runtime-cpu:latest",
            "launch_mode": "openai", "hf_model": "test/model",
            "served_model_name": "test-model", "gpu_memory_utilization": 0.6,
            "max_model_len": 2048, "max_num_seqs": 1, "max_num_batched_tokens": 1024,
            "enable_lora": False, "max_loras": 0, "max_lora_rank": 0,
            "enforce_eager": False, "port": 8090, "allowed_gpus": [],
            "quantization": None, "volumes": {}, "env": {},
            "ipc_host": False, "dtype": "auto", "shm_size": "4gb",
            "health_path": "/health", "runtime": "gpu",
            "gguf_path": "", "cpu_cores": 4, "memory_mb": 4096,
        }
        expected_cpu = {
            "base_model": "test/model", "model_alias": "test-model",
            "api_base": "http://cpu_c:8090/v1", "container": "cpu_c",
            "gpu": "", "state": "READY", "runtime_type": "cpu",
        }
        with patch.object(mrm, "_ensure_cpu", return_value=expected_cpu) as mock_cpu:
            with patch.object(mrm, "_detect_node_capabilities",
                              return_value=_cpu_caps(ram_mb=16384)):
                with patch.object(mrm, "_set_state"):
                    result = mrm.ensure_running("test/model", profile="cheap")

        mock_cpu.assert_called_once()
        assert result["backend"] == "llama.cpp"
        assert result["model_variant"] == "gguf_q4"

    def test_legacy_path_used_when_no_variants_registered(self):
        """Without variant registry entries, ensure_running() takes the legacy path."""
        with (
            patch("docker.from_env"),
            patch("redis.Redis.from_url"),
        ):
            from model_runtime_manager.mrm.config import Settings, ModelSpec
            from model_runtime_manager.mrm.runtime import ModelRuntimeManager

            settings = MagicMock(spec=Settings)
            settings.auto_fallback_cpu_url = ""
            settings.idle_timeout_sec = 900
            settings.redis_url = "redis://localhost:6379/0"
            settings.docker_network = "test_net"
            settings.keep_failed_containers = False
            settings.model_registry = {}
            settings.load_default_registry.return_value = {}

            spec = MagicMock(spec=ModelSpec)
            spec.runtime = "gpu"
            spec.model_alias = "test-model"

            mrm = ModelRuntimeManager.__new__(ModelRuntimeManager)
            mrm.s = settings
            mrm.redis = MagicMock()
            mrm.docker = MagicMock()
            mrm.registry = {"test/model": spec}
            mrm._variant_registry = VariantRegistry()  # empty — no variants
            mrm._backends = []
            from model_runtime_manager.execution_selector import ExecutionSelector
            mrm._execution_selector = ExecutionSelector([], mrm._variant_registry)

            expected = {"state": "READY", "runtime_type": "gpu", "base_model": "test/model",
                        "model_alias": "test-model", "api_base": "x", "container": "c", "gpu": "0"}
            with patch.object(mrm, "_ensure_gpu", return_value=expected) as mock_gpu:
                result = mrm.ensure_running("test/model")

            mock_gpu.assert_called_once()
            assert result["state"] == "READY"

    def test_node_capabilities_override_respected(self):
        """Caller-supplied node_capabilities dict is used instead of auto-detection."""
        variants = [_fp16_variant(size_gb=3.5)]
        mrm, spec = self._make_mrm_with_variants(variants)

        expected = {
            "base_model": "test/model", "model_alias": "test-model",
            "api_base": "http://x:8000/v1", "container": "c",
            "gpu": "0", "state": "READY", "runtime_type": "gpu",
        }
        node_caps_dict = {
            "gpu": True, "vram_mb": 32768, "ram_mb": 65536,
            "cpu_cores": 16, "supported_backends": ["vllm", "llama.cpp", "cpu"],
        }
        with patch.object(mrm, "_ensure_gpu", return_value=expected):
            with patch.object(mrm, "_detect_node_capabilities") as mock_detect:
                with patch.object(mrm, "_set_state"):
                    mrm.ensure_running("test/model", node_capabilities=node_caps_dict)

        # _detect_node_capabilities must NOT be called when override is supplied
        mock_detect.assert_not_called()


# ── Quantization conflict resolution ─────────────────────────────────────────

class TestQuantizationResolution:
    """
    Tests for ExecutionSelector._resolve_quantization() via the full select() path.

    inspect_model is always mocked — no real HF network calls.
    """

    def _make_selector(self, variants) -> ExecutionSelector:
        mrm = _make_mrm()
        backends = [VLLMBackend(mrm)]
        reg = VariantRegistry()
        reg.register("test/model", variants)
        return ExecutionSelector(backends, reg)

    def _gpu_caps_large(self) -> NodeCapabilities:
        return NodeCapabilities(
            gpu=True, vram_mb=48000, ram_mb=65536, cpu_cores=16,
            supported_backends=["vllm", "llama.cpp", "cpu"],
        )

    # ------------------------------------------------------------------
    # Case 1 — CONFLICT: native=fp8, variant=awq → quantization=None
    # ------------------------------------------------------------------

    def test_conflict_clears_quantization(self):
        """Model config says fp8; variant requests awq → auto-fix to None."""
        from model_runtime_manager.model_inspector import ModelCapabilities

        variant = ModelVariant(
            "test/model", "awq",
            quantization="awq", size_gb=4.0,
            backend_compatibility=["vllm"],
        )
        selector = self._make_selector([variant])

        with patch(
            "model_runtime_manager.execution_selector.inspect_model",
            return_value=ModelCapabilities(native_quantization="fp8"),
        ):
            result = selector.select("test/model", self._gpu_caps_large())

        assert result is not None
        assert result.variant.quantization is None

    def test_conflict_emits_warning_log(self):
        """A structured warning must be logged when auto-fixing a conflict."""
        import logging
        from model_runtime_manager.model_inspector import ModelCapabilities

        variant = ModelVariant(
            "test/model", "awq",
            quantization="awq", size_gb=4.0,
            backend_compatibility=["vllm"],
        )
        selector = self._make_selector([variant])

        with patch(
            "model_runtime_manager.execution_selector.inspect_model",
            return_value=ModelCapabilities(native_quantization="fp8"),
        ):
            with patch.object(
                __import__("model_runtime_manager.execution_selector", fromlist=["logger"]).logger,
                "warning",
            ) as mock_warn:
                selector.select("test/model", self._gpu_caps_large())

        mock_warn.assert_called_once()
        logged = mock_warn.call_args[0][0]
        assert "quantization_conflict_auto_fixed" in logged

    def test_conflict_does_not_crash(self):
        """Ensure selecting a conflicting variant never raises."""
        from model_runtime_manager.model_inspector import ModelCapabilities

        variant = ModelVariant(
            "test/model", "awq",
            quantization="awq", size_gb=4.0,
            backend_compatibility=["vllm"],
        )
        selector = self._make_selector([variant])

        with patch(
            "model_runtime_manager.execution_selector.inspect_model",
            return_value=ModelCapabilities(native_quantization="fp8"),
        ):
            result = selector.select("test/model", self._gpu_caps_large())

        # Must return a result — not None, no exception
        assert result is not None

    # ------------------------------------------------------------------
    # Case 2 — COMPATIBLE: native=None, variant=awq → quantization=awq preserved
    # ------------------------------------------------------------------

    def test_no_native_quant_preserves_variant_quantization(self):
        """Model has no native quantization; variant's awq must be kept."""
        from model_runtime_manager.model_inspector import ModelCapabilities

        variant = ModelVariant(
            "test/model", "awq",
            quantization="awq", size_gb=4.0,
            backend_compatibility=["vllm"],
        )
        selector = self._make_selector([variant])

        with patch(
            "model_runtime_manager.execution_selector.inspect_model",
            return_value=ModelCapabilities(native_quantization=None),
        ):
            result = selector.select("test/model", self._gpu_caps_large())

        assert result is not None
        assert result.variant.quantization == "awq"

    def test_inspect_failure_treated_as_no_native_quant(self):
        """If inspect_model raises or returns None, variant quantization is unchanged."""
        from model_runtime_manager.model_inspector import ModelCapabilities

        variant = ModelVariant(
            "test/model", "awq",
            quantization="awq", size_gb=4.0,
            backend_compatibility=["vllm"],
        )
        selector = self._make_selector([variant])

        # Simulate fail-safe return (as if inspect_model caught an exception)
        with patch(
            "model_runtime_manager.execution_selector.inspect_model",
            return_value=ModelCapabilities(native_quantization=None),
        ):
            result = selector.select("test/model", self._gpu_caps_large())

        assert result is not None
        assert result.variant.quantization == "awq"

    # ------------------------------------------------------------------
    # Case 3 — EXACT MATCH: native=awq, variant=awq → quantization=None
    # ------------------------------------------------------------------

    def test_exact_match_clears_quantization(self):
        """Native and selected both awq → clear the flag (vLLM reads from config)."""
        from model_runtime_manager.model_inspector import ModelCapabilities

        variant = ModelVariant(
            "test/model", "awq",
            quantization="awq", size_gb=4.0,
            backend_compatibility=["vllm"],
        )
        selector = self._make_selector([variant])

        with patch(
            "model_runtime_manager.execution_selector.inspect_model",
            return_value=ModelCapabilities(native_quantization="awq"),
        ):
            result = selector.select("test/model", self._gpu_caps_large())

        assert result is not None
        assert result.variant.quantization is None

    def test_exact_match_does_not_log_conflict(self):
        """Exact match must NOT trigger the conflict warning."""
        from model_runtime_manager.model_inspector import ModelCapabilities
        import model_runtime_manager.execution_selector as sel_module

        variant = ModelVariant(
            "test/model", "awq",
            quantization="awq", size_gb=4.0,
            backend_compatibility=["vllm"],
        )
        selector = self._make_selector([variant])

        with patch(
            "model_runtime_manager.execution_selector.inspect_model",
            return_value=ModelCapabilities(native_quantization="awq"),
        ):
            with patch.object(sel_module.logger, "warning") as mock_warn:
                selector.select("test/model", self._gpu_caps_large())

        mock_warn.assert_not_called()


# ── VLLMBackend.launch() quantization sync ───────────────────────────────────

class TestVLLMBackendQuantizationSync:
    """VLLMBackend.launch() must always write variant.quantization → spec.quantization."""

    def _make_backend_with_spec(self, spec_quant):
        """Return a VLLMBackend + spec mock where spec.quantization = spec_quant."""
        mrm = _make_mrm()
        from model_runtime_manager.mrm.config import ModelSpec
        spec = MagicMock(spec=ModelSpec)
        spec.quantization = spec_quant
        mrm._spec.return_value = spec
        mrm._ensure_gpu.return_value = {
            "base_model": "test/model", "model_alias": "test",
            "api_base": "http://x:8000/v1", "container": "c",
            "gpu": "0", "state": "READY", "runtime_type": "gpu",
        }
        return VLLMBackend(mrm), spec

    def test_conflict_resolved_variant_clears_spec_quantization(self):
        """When variant.quantization=None (post auto-fix), spec.quantization must become None."""
        backend, spec = self._make_backend_with_spec(spec_quant="awq")

        variant = ModelVariant(
            "test/model", "awq",
            quantization=None,  # already resolved by ExecutionSelector
            size_gb=4.0, backend_compatibility=["vllm"],
        )
        backend.launch("test/model", variant, _gpu_caps())

        assert spec.quantization is None

    def test_valid_quantization_applied_to_spec(self):
        """When variant.quantization='awq' and no native conflict, spec gets 'awq'."""
        backend, spec = self._make_backend_with_spec(spec_quant=None)

        variant = ModelVariant(
            "test/model", "awq",
            quantization="awq", size_gb=4.0, backend_compatibility=["vllm"],
        )
        backend.launch("test/model", variant, _gpu_caps())

        assert spec.quantization == "awq"

    def test_none_quantization_variant_clears_existing_spec_quant(self):
        """spec starts with a quant; variant says None → spec must be cleared."""
        backend, spec = self._make_backend_with_spec(spec_quant="gptq")

        variant = ModelVariant(
            "test/model", "fp16",
            quantization=None, size_gb=14.0, backend_compatibility=["vllm"],
        )
        backend.launch("test/model", variant, _gpu_caps())

        assert spec.quantization is None


# ── ModelVariant.copy_with ────────────────────────────────────────────────────

class TestModelVariantCopyWith:
    def test_copy_with_overrides_quantization(self):
        v = ModelVariant("m", "awq", quantization="awq", size_gb=4.0,
                         backend_compatibility=["vllm"])
        v2 = v.copy_with(quantization=None)
        assert v2.quantization is None
        # Original unchanged
        assert v.quantization == "awq"

    def test_copy_with_preserves_other_fields(self):
        v = ModelVariant("m", "awq", quantization="awq", size_gb=4.0,
                         backend_compatibility=["vllm"], n_ctx=4096)
        v2 = v.copy_with(quantization=None)
        assert v2.base_model_id == "m"
        assert v2.format == "awq"
        assert v2.size_gb == 4.0
        assert v2.n_ctx == 4096
        assert v2.backend_compatibility == ["vllm"]

    def test_copy_with_multiple_fields(self):
        v = ModelVariant("m", "fp16", size_gb=14.0, backend_compatibility=["vllm"])
        v2 = v.copy_with(size_gb=7.0, quantization="awq")
        assert v2.size_gb == 7.0
        assert v2.quantization == "awq"
        assert v.size_gb == 14.0  # original unchanged

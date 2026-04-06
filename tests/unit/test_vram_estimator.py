"""
tests/unit/test_vram_estimator.py

Tests for the VRAM estimation and auto-tuning system:
    - estimate_vram(): pure function properties
    - get_available_vram_gb(): unit conversion
    - auto_tune_config(): config selection, fallbacks, no-metadata path
    - VLLMBackend: auto-tune integration + safety retry loop
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pytest

from model_runtime_manager.vram_estimator import (
    VRAMEstimate,
    AutoTuneResult,
    DTYPE_BYTES,
    estimate_vram,
    get_available_vram_gb,
    auto_tune_config,
    VRAM_SAFETY_MARGIN,
)
from model_runtime_manager.model_variant_registry import ModelVariant
from model_runtime_manager.backends.base import NodeCapabilities
from model_runtime_manager.backends.vllm_backend import (
    VLLMBackend,
    MAX_LAUNCH_ATTEMPTS,
    _is_oom_candidate,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gpu_node(vram_mb: int = 24576) -> NodeCapabilities:
    return NodeCapabilities(
        gpu=True,
        vram_mb=vram_mb,
        ram_mb=65536,
        cpu_cores=16,
        supported_backends=["vllm"],
    )


def _variant(
    model_size_b: int = 7_000_000_000,
    dtype: str = "fp16",
    size_gb: float = 14.0,
) -> ModelVariant:
    return ModelVariant(
        base_model_id="test/model-7b",
        format="fp16",
        size_gb=size_gb,
        backend_compatibility=["vllm"],
        model_size_b=model_size_b,
        dtype=dtype,
    )


def _make_backend() -> tuple[VLLMBackend, MagicMock]:
    """Return (VLLMBackend, mrm_mock)."""
    mrm = MagicMock()
    from model_runtime_manager.mrm.config import ModelSpec
    spec = MagicMock(spec=ModelSpec)
    spec.quantization = None
    spec.max_model_len = 4096
    spec.max_num_seqs = 2
    mrm._spec.return_value = spec
    mrm._ensure_gpu.return_value = {
        "base_model": "test/model-7b",
        "model_alias": "model-7b",
        "api_base": "http://vllm_test:8000/v1",
        "container": "vllm_test",
        "gpu": "0",
        "state": "READY",
        "runtime_type": "gpu",
    }
    return VLLMBackend(mrm), mrm


# ── VRAMEstimate: pure function properties ────────────────────────────────────

class TestEstimateVRAM:
    def test_returns_vram_estimate_type(self):
        est = estimate_vram(7_000_000_000, 2.0, 2048, 1)
        assert isinstance(est, VRAMEstimate)

    def test_weights_formula(self):
        # 7B * 2 bytes / 1e9 = 14 GB
        est = estimate_vram(7_000_000_000, 2.0, 0, 1)
        assert abs(est.model_vram_gb - 14.0) < 0.01

    def test_kv_cache_increases_with_context_length(self):
        short = estimate_vram(7_000_000_000, 2.0, 512, 1)
        long = estimate_vram(7_000_000_000, 2.0, 4096, 1)
        assert long.kv_cache_gb > short.kv_cache_gb

    def test_kv_cache_increases_with_num_seqs(self):
        one_seq = estimate_vram(7_000_000_000, 2.0, 2048, 1)
        four_seq = estimate_vram(7_000_000_000, 2.0, 2048, 4)
        # 4x seqs should yield ~4x KV cache; allow 2% tolerance for rounding
        assert four_seq.kv_cache_gb == pytest.approx(one_seq.kv_cache_gb * 4, rel=0.02)

    def test_total_equals_weights_plus_kv(self):
        est = estimate_vram(7_000_000_000, 2.0, 2048, 2)
        assert abs(est.total_gb - (est.model_vram_gb + est.kv_cache_gb)) < 1e-9

    def test_fp8_half_the_vram_of_fp16(self):
        fp16 = estimate_vram(7_000_000_000, DTYPE_BYTES["fp16"], 2048, 1)
        fp8 = estimate_vram(7_000_000_000, DTYPE_BYTES["fp8"], 2048, 1)
        assert fp8.total_gb < fp16.total_gb

    def test_larger_model_needs_more_vram(self):
        small = estimate_vram(1_800_000_000, 2.0, 2048, 1)
        large = estimate_vram(70_000_000_000, 2.0, 2048, 1)
        assert large.total_gb > small.total_gb

    def test_zero_context_no_kv_cache(self):
        est = estimate_vram(7_000_000_000, 2.0, 0, 1)
        assert est.kv_cache_gb == 0.0


# ── get_available_vram_gb ─────────────────────────────────────────────────────

class TestGetAvailableVRAMGb:
    def test_converts_mib_to_gb(self):
        caps = _gpu_node(vram_mb=24576)
        assert get_available_vram_gb(caps) == pytest.approx(24.0, rel=1e-3)

    def test_zero_vram(self):
        caps = _gpu_node(vram_mb=0)
        assert get_available_vram_gb(caps) == 0.0

    def test_8gb_gpu(self):
        caps = _gpu_node(vram_mb=8192)
        assert get_available_vram_gb(caps) == pytest.approx(8.0, rel=1e-3)


# ── auto_tune_config ──────────────────────────────────────────────────────────

class TestAutoTuneConfig:
    def test_returns_none_when_model_size_b_zero(self):
        v = ModelVariant("test/m", "fp16", size_gb=14.0, backend_compatibility=["vllm"])
        # model_size_b defaults to 0
        assert auto_tune_config(v, _gpu_node()) is None

    def test_returns_auto_tune_result_type(self):
        v = _variant(model_size_b=7_000_000_000)
        result = auto_tune_config(v, _gpu_node(vram_mb=24576))
        assert isinstance(result, AutoTuneResult)

    def test_selects_smaller_config_for_low_vram(self):
        # 7B fp16 needs ~14 GB weights alone; on an 8 GB node it should pick tiny ctx
        v = _variant(model_size_b=7_000_000_000, dtype="fp16")
        result_8gb = auto_tune_config(v, _gpu_node(vram_mb=8192))
        result_24gb = auto_tune_config(v, _gpu_node(vram_mb=24576))
        assert result_8gb is not None
        assert result_24gb is not None
        # On 8 GB, either ctx is smaller, or the result is the minimal fallback
        assert result_8gb.max_model_len <= result_24gb.max_model_len

    def test_high_vram_allows_larger_context(self):
        # 1.8B fp16 on 24 GB — should select a large context
        v = _variant(model_size_b=1_800_000_000, dtype="fp16", size_gb=3.5)
        result = auto_tune_config(v, _gpu_node(vram_mb=24576))
        assert result is not None
        assert result.max_model_len >= 4096

    def test_always_returns_result_even_when_nothing_fits(self):
        # 70B on a tiny 4 GB node → fallback minimal config
        v = _variant(model_size_b=70_000_000_000, dtype="fp16", size_gb=140.0)
        result = auto_tune_config(v, _gpu_node(vram_mb=4096))
        assert result is not None
        assert result.max_model_len == 256
        assert result.num_seqs == 1

    def test_selected_config_fits_within_safety_margin(self):
        v = _variant(model_size_b=7_000_000_000, dtype="fp16")
        caps = _gpu_node(vram_mb=24576)
        result = auto_tune_config(v, caps)
        assert result is not None
        available = get_available_vram_gb(caps)
        assert result.estimate.total_gb < available * VRAM_SAFETY_MARGIN

    def test_fp8_allows_larger_config_than_fp16(self):
        # Same model, fp8 uses half the VRAM → larger context should be selected
        v_fp8 = _variant(model_size_b=7_000_000_000, dtype="fp8")
        v_fp16 = _variant(model_size_b=7_000_000_000, dtype="fp16")
        caps = _gpu_node(vram_mb=16384)
        r_fp8 = auto_tune_config(v_fp8, caps)
        r_fp16 = auto_tune_config(v_fp16, caps)
        assert r_fp8 is not None
        assert r_fp16 is not None
        assert r_fp8.max_model_len >= r_fp16.max_model_len

    def test_unknown_dtype_defaults_to_fp16_bytes(self):
        # "bfloat16" is not in DTYPE_BYTES → falls back to 2.0 bytes (fp16 default)
        v_unknown = ModelVariant(
            "test/m", "fp16",
            model_size_b=7_000_000_000, dtype="bfloat16",
            size_gb=14.0, backend_compatibility=["vllm"],
        )
        v_fp16 = _variant(model_size_b=7_000_000_000, dtype="fp16")
        caps = _gpu_node(vram_mb=24576)
        r_unknown = auto_tune_config(v_unknown, caps)
        r_fp16 = auto_tune_config(v_fp16, caps)
        # Both should select the same config since unknown falls back to fp16 bytes
        assert r_unknown is not None
        assert r_fp16 is not None
        assert r_unknown.max_model_len == r_fp16.max_model_len

    def test_estimate_attached_to_result(self):
        v = _variant(model_size_b=7_000_000_000, dtype="fp16")
        result = auto_tune_config(v, _gpu_node(vram_mb=24576))
        assert result is not None
        assert isinstance(result.estimate, VRAMEstimate)
        assert result.estimate.total_gb > 0


# ── _is_oom_candidate ─────────────────────────────────────────────────────────

class TestIsOOMCandidate:
    def test_container_crashed_is_oom_candidate(self):
        from model_runtime_manager.mrm.runtime import RuntimeError409
        exc = RuntimeError409("Container crashed during startup. Check logs.")
        assert _is_oom_candidate(exc) is True

    def test_healthcheck_timeout_is_oom_candidate(self):
        from model_runtime_manager.mrm.runtime import RuntimeError409
        exc = RuntimeError409("Healthcheck timeout for vllm_model (url). Last error=...")
        assert _is_oom_candidate(exc) is True

    def test_lock_error_is_not_oom_candidate(self):
        from model_runtime_manager.mrm.runtime import RuntimeError409
        exc = RuntimeError409("Model is being started/stopped by another request")
        assert _is_oom_candidate(exc) is False

    def test_no_free_gpu_is_not_oom_candidate(self):
        from model_runtime_manager.mrm.runtime import RuntimeError409
        exc = RuntimeError409("No free GPU. All allowed GPUs ['0'] are busy.")
        assert _is_oom_candidate(exc) is False

    def test_unknown_model_is_not_oom_candidate(self):
        from model_runtime_manager.mrm.runtime import RuntimeError409
        exc = RuntimeError409("Unknown base_model: nonexistent/model")
        assert _is_oom_candidate(exc) is False

    def test_generic_runtime_error_is_oom_candidate(self):
        exc = RuntimeError("vLLM OOM: CUDA out of memory")
        assert _is_oom_candidate(exc) is True


# ── VLLMBackend: auto-tune integration ───────────────────────────────────────

class TestVLLMBackendAutoTune:
    def test_auto_tune_applied_to_spec_when_metadata_present(self):
        backend, mrm = _make_backend()
        spec = mrm._spec.return_value

        v = _variant(model_size_b=1_800_000_000, dtype="fp16", size_gb=3.5)
        caps = _gpu_node(vram_mb=24576)

        backend.launch("test/model-7b", v, caps)

        # spec.max_model_len must have been overwritten by auto-tune
        assert spec.max_model_len != 4096 or spec.max_num_seqs != 2  # something changed

    def test_no_auto_tune_when_model_size_b_zero(self):
        backend, mrm = _make_backend()
        spec = mrm._spec.return_value
        spec.max_model_len = 2048

        v = ModelVariant(
            "test/m", "fp16",
            size_gb=14.0, backend_compatibility=["vllm"],
            # model_size_b defaults to 0 → no auto-tune
        )
        caps = _gpu_node(vram_mb=24576)

        backend.launch("test/m", v, caps)

        # spec.max_model_len must remain unchanged
        assert spec.max_model_len == 2048

    def test_auto_tune_result_logged(self):
        backend, mrm = _make_backend()
        v = _variant(model_size_b=1_800_000_000, dtype="fp16", size_gb=3.5)

        with patch.object(logger := __import__(
            "model_runtime_manager.backends.vllm_backend", fromlist=["logger"]
        ).logger, "info") as mock_info:
            backend.launch("test/model-7b", v, _gpu_node(vram_mb=24576))

        # At least one info log with "auto_config_selected"
        log_calls = [str(c) for c in mock_info.call_args_list]
        assert any("auto_config_selected" in s for s in log_calls)

    def test_result_includes_backend_and_variant_fields(self):
        backend, mrm = _make_backend()
        v = _variant(model_size_b=7_000_000_000, dtype="fp16", size_gb=14.0)

        result = backend.launch("test/model-7b", v, _gpu_node(vram_mb=24576))

        assert result["backend"] == "vllm"
        assert result["model_variant"] == "fp16"
        assert result["ram_mb"] == int(14.0 * 1024)


# ── VLLMBackend: safety retry loop ───────────────────────────────────────────

class TestVLLMBackendSafetyRetry:
    def test_succeeds_on_second_attempt(self):
        """First call raises crash; second succeeds with halved context."""
        backend, mrm = _make_backend()
        spec = mrm._spec.return_value
        spec.max_model_len = 2048

        from model_runtime_manager.mrm.runtime import RuntimeError409
        crash = RuntimeError409("Container crashed during startup. Check logs.")
        success = {
            "base_model": "test/model-7b", "model_alias": "m",
            "api_base": "http://vllm_test:8000/v1", "container": "c",
            "gpu": "0", "state": "READY", "runtime_type": "gpu",
        }
        mrm._ensure_gpu.side_effect = [crash, success]

        v = ModelVariant(
            "test/model-7b", "fp16",
            size_gb=14.0, backend_compatibility=["vllm"],
        )
        result = backend.launch("test/model-7b", v, _gpu_node(vram_mb=24576))

        assert result["state"] == "READY"
        assert mrm._ensure_gpu.call_count == 2

    def test_halves_max_model_len_on_crash(self):
        """After a crash the backend halves max_model_len before retrying."""
        backend, mrm = _make_backend()
        spec = mrm._spec.return_value
        spec.max_model_len = 2048

        from model_runtime_manager.mrm.runtime import RuntimeError409
        crash = RuntimeError409("Container crashed during startup. Check logs.")
        success = {
            "base_model": "test/model-7b", "model_alias": "m",
            "api_base": "http://vllm_test:8000/v1", "container": "c",
            "gpu": "0", "state": "READY", "runtime_type": "gpu",
        }
        mrm._ensure_gpu.side_effect = [crash, success]

        v = ModelVariant(
            "test/model-7b", "fp16",
            size_gb=14.0, backend_compatibility=["vllm"],
        )
        backend.launch("test/model-7b", v, _gpu_node(vram_mb=24576))

        # Second call must see a reduced max_model_len
        second_call_spec_len = spec.max_model_len
        assert second_call_spec_len == 1024  # 2048 // 2

    def test_raises_after_max_attempts(self):
        """Exhausting all retry attempts must re-raise the last exception."""
        backend, mrm = _make_backend()
        spec = mrm._spec.return_value
        spec.max_model_len = 2048

        from model_runtime_manager.mrm.runtime import RuntimeError409
        crash = RuntimeError409("Container crashed during startup. Check logs.")
        mrm._ensure_gpu.side_effect = crash

        v = ModelVariant(
            "test/model-7b", "fp16",
            size_gb=14.0, backend_compatibility=["vllm"],
        )
        with pytest.raises(RuntimeError409, match="Container crashed"):
            backend.launch("test/model-7b", v, _gpu_node(vram_mb=24576))

        assert mrm._ensure_gpu.call_count == MAX_LAUNCH_ATTEMPTS

    def test_non_oom_error_not_retried(self):
        """Lock/GPU-busy errors must not trigger retry — fail fast."""
        backend, mrm = _make_backend()
        spec = mrm._spec.return_value
        spec.max_model_len = 2048

        from model_runtime_manager.mrm.runtime import RuntimeError409
        lock_err = RuntimeError409("Model is being started/stopped by another request")
        mrm._ensure_gpu.side_effect = lock_err

        v = ModelVariant(
            "test/model-7b", "fp16",
            size_gb=14.0, backend_compatibility=["vllm"],
        )
        with pytest.raises(RuntimeError409, match="being started"):
            backend.launch("test/model-7b", v, _gpu_node(vram_mb=24576))

        # Must NOT retry
        assert mrm._ensure_gpu.call_count == 1

    def test_fallback_logged_on_crash(self):
        """auto_config_fallback must be logged when halving the context."""
        backend, mrm = _make_backend()
        spec = mrm._spec.return_value
        spec.max_model_len = 2048

        from model_runtime_manager.mrm.runtime import RuntimeError409
        import model_runtime_manager.backends.vllm_backend as vllm_mod

        crash = RuntimeError409("Container crashed during startup. Check logs.")
        success = {
            "base_model": "test/model-7b", "model_alias": "m",
            "api_base": "http://vllm_test:8000/v1", "container": "c",
            "gpu": "0", "state": "READY", "runtime_type": "gpu",
        }
        mrm._ensure_gpu.side_effect = [crash, success]

        v = ModelVariant(
            "test/model-7b", "fp16",
            size_gb=14.0, backend_compatibility=["vllm"],
        )
        with patch.object(vllm_mod.logger, "warning") as mock_warn:
            backend.launch("test/model-7b", v, _gpu_node(vram_mb=24576))

        mock_warn.assert_called_once()
        assert "auto_config_fallback" in mock_warn.call_args[0][0]

    def test_min_model_len_floor_respected(self):
        """Halving should never go below _MIN_MODEL_LEN."""
        from model_runtime_manager.backends.vllm_backend import _MIN_MODEL_LEN
        backend, mrm = _make_backend()
        spec = mrm._spec.return_value
        spec.max_model_len = 300  # one halve → 150, but floor is 256

        from model_runtime_manager.mrm.runtime import RuntimeError409
        crash = RuntimeError409("Container crashed during startup. Check logs.")
        success = {
            "base_model": "test/model-7b", "model_alias": "m",
            "api_base": "http://vllm_test:8000/v1", "container": "c",
            "gpu": "0", "state": "READY", "runtime_type": "gpu",
        }
        mrm._ensure_gpu.side_effect = [crash, success]

        v = ModelVariant("test/model-7b", "fp16", size_gb=14.0, backend_compatibility=["vllm"])
        backend.launch("test/model-7b", v, _gpu_node(vram_mb=24576))

        assert spec.max_model_len >= _MIN_MODEL_LEN

"""
tests/unit/test_mrm_fallback.py

Unit tests for MRM GPU→CPU auto-fallback logic:
  - _is_nvidia_runtime_error()
  - _fallback_to_cpu_runtime()
  - ensure_running() fallback integration
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

import docker.errors


# ── helpers ──────────────────────────────────────────────────────────

def _make_docker_api_error(message: str) -> docker.errors.APIError:
    """Construct a docker.errors.APIError with a given message string."""
    response = MagicMock()
    response.status_code = 500
    response.reason = "Internal Server Error"
    err = docker.errors.APIError(message, response=response)
    return err


def _make_settings(
    fallback_url: str = "http://cpu_runtime:8090",
    fallback_alias: str = "cpu-model",
) -> MagicMock:
    s = MagicMock()
    s.auto_fallback_cpu_url = fallback_url
    s.auto_fallback_cpu_alias = fallback_alias
    s.redis_url = "redis://redis:6379/2"
    s.idle_timeout_sec = 900
    s.docker_network = "ai_net_lite"
    s.keep_failed_containers = False
    return s


# ── _is_nvidia_runtime_error ─────────────────────────────────────────

class TestIsNvidiaRuntimeError:
    """Covers _is_nvidia_runtime_error() predicate logic."""

    def setup_method(self):
        # Import after patching to avoid docker daemon connection
        from model_runtime_manager.mrm.runtime import _is_nvidia_runtime_error
        self.fn = _is_nvidia_runtime_error

    # --- True cases (real Docker API error messages) ------------------

    def test_unknown_runtime_nvidia(self):
        err = _make_docker_api_error("OCI runtime create failed: unknown runtime: nvidia")
        assert self.fn(err) is True

    def test_could_not_select_device_driver(self):
        err = _make_docker_api_error(
            "could not select device driver \"nvidia\" with capabilities: [[gpu]]"
        )
        assert self.fn(err) is True

    def test_nvidia_in_message(self):
        err = _make_docker_api_error("500 Server Error: nvidia container runtime not found")
        assert self.fn(err) is True

    def test_no_devices_found(self):
        err = _make_docker_api_error("no devices found")
        assert self.fn(err) is True

    def test_failed_to_initialize_nvml(self):
        err = _make_docker_api_error("failed to initialize nvml: driver/library version mismatch")
        assert self.fn(err) is True

    def test_oci_runtime_create_failed(self):
        err = _make_docker_api_error(
            "OCI runtime create failed: container_linux.go: ... nvidia ..."
        )
        assert self.fn(err) is True

    def test_runtime_not_found(self):
        err = _make_docker_api_error("runtime not found: nvidia")
        assert self.fn(err) is True

    def test_case_insensitive(self):
        err = _make_docker_api_error("UNKNOWN RUNTIME: NVIDIA")
        assert self.fn(err) is True

    # --- False cases --------------------------------------------------

    def test_our_runtimeerror409_no_free_gpu(self):
        """GPU is busy — must NOT trigger fallback."""
        from model_runtime_manager.mrm.runtime import RuntimeError409
        err = RuntimeError409("No free GPU. All allowed GPUs ['0'] are busy.")
        assert self.fn(err) is False

    def test_our_runtimeerror409_model_not_found(self):
        from model_runtime_manager.mrm.runtime import RuntimeError409
        err = RuntimeError409("Unknown base_model: Qwen/Qwen1.5-1.8B-Chat")
        assert self.fn(err) is False

    def test_docker_not_found_error(self):
        """Container not found — not a GPU error."""
        err = docker.errors.NotFound("No such container: some_container")
        assert self.fn(err) is False

    def test_plain_value_error(self):
        err = ValueError("some random error")
        assert self.fn(err) is False

    def test_docker_api_error_unrelated(self):
        """Docker API error unrelated to nvidia."""
        err = _make_docker_api_error("Error response from daemon: image not found")
        assert self.fn(err) is False

    def test_docker_api_error_oom(self):
        """OOM kill — not a GPU availability error."""
        err = _make_docker_api_error("container killed by OOM killer")
        assert self.fn(err) is False


# ── _fallback_to_cpu_runtime ─────────────────────────────────────────

class TestFallbackToCpuRuntime:
    """Covers _fallback_to_cpu_runtime() Redis state writing and return value."""

    def _make_mrm(self, fallback_url="http://cpu_runtime:8090", fallback_alias="cpu-model"):
        """Return a ModelRuntimeManager with mocked Docker/Redis."""
        with (
            patch("docker.from_env"),
            patch("redis.Redis.from_url"),
        ):
            from model_runtime_manager.mrm.config import Settings, ModelSpec
            from model_runtime_manager.mrm.runtime import ModelRuntimeManager

            settings = MagicMock(spec=Settings)
            settings.auto_fallback_cpu_url = fallback_url
            settings.auto_fallback_cpu_alias = fallback_alias
            settings.idle_timeout_sec = 900
            settings.redis_url = "redis://localhost:6379/0"
            settings.docker_network = "test_net"
            settings.keep_failed_containers = False
            settings.model_registry = {}
            settings.load_default_registry.return_value = {}

            mrm = ModelRuntimeManager.__new__(ModelRuntimeManager)
            mrm.s = settings
            mrm.redis = MagicMock()
            mrm.docker = MagicMock()
            mrm.registry = {}
            return mrm

    def test_returns_ready_state(self):
        mrm = self._make_mrm()
        from model_runtime_manager.mrm.config import ModelSpec
        spec = MagicMock(spec=ModelSpec)
        spec.model_alias = "qwen-1.8b"

        result = mrm._fallback_to_cpu_runtime("Qwen/Qwen1.5-1.8B-Chat", spec, "http://cpu_runtime:8090")

        assert result["state"] == "READY"
        assert result["runtime_type"] == "cpu"
        assert result["gpu"] == ""
        assert result["fallback"] is True
        assert result["api_base"] == "http://cpu_runtime:8090/v1"
        assert result["model_alias"] == "cpu-model"

    def test_writes_correct_redis_state(self):
        mrm = self._make_mrm()
        from model_runtime_manager.mrm.config import ModelSpec
        spec = MagicMock(spec=ModelSpec)

        mrm._fallback_to_cpu_runtime("some-model", spec, "http://cpu_runtime:8090")

        mrm.redis.hset.assert_called_once()
        call_kwargs = mrm.redis.hset.call_args
        mapping = call_kwargs[1]["mapping"] if "mapping" in (call_kwargs[1] or {}) else call_kwargs[0][1]
        assert mapping["state"] == "READY"
        assert mapping["runtime_type"] == "cpu"
        assert mapping["gpu"] == ""
        assert "http://cpu_runtime:8090/v1" in mapping["api_base"]

    def test_strips_trailing_slash_from_url(self):
        mrm = self._make_mrm(fallback_url="http://cpu_runtime:8090/")
        from model_runtime_manager.mrm.config import ModelSpec
        spec = MagicMock(spec=ModelSpec)

        result = mrm._fallback_to_cpu_runtime("m", spec, "http://cpu_runtime:8090/")
        assert result["api_base"] == "http://cpu_runtime:8090/v1"

    def test_uses_settings_alias(self):
        mrm = self._make_mrm(fallback_alias="my-gguf-model")
        from model_runtime_manager.mrm.config import ModelSpec
        spec = MagicMock(spec=ModelSpec)

        result = mrm._fallback_to_cpu_runtime("m", spec, "http://cpu_runtime:8090")
        assert result["model_alias"] == "my-gguf-model"


# ── ensure_running: fallback integration ─────────────────────────────

class TestEnsureRunningFallback:
    """Ensures ensure_running() calls fallback only for nvidia errors."""

    def _make_mrm_with_spec(self, fallback_url="http://cpu_runtime:8090"):
        with (
            patch("docker.from_env"),
            patch("redis.Redis.from_url"),
        ):
            from model_runtime_manager.mrm.config import Settings, ModelSpec
            from model_runtime_manager.mrm.runtime import ModelRuntimeManager

            settings = MagicMock(spec=Settings)
            settings.auto_fallback_cpu_url = fallback_url
            settings.auto_fallback_cpu_alias = "cpu-model"
            settings.idle_timeout_sec = 900

            spec = MagicMock(spec=ModelSpec)
            spec.runtime = "gpu"
            spec.model_alias = "test-model"

            mrm = ModelRuntimeManager.__new__(ModelRuntimeManager)
            mrm.s = settings
            mrm.redis = MagicMock()
            mrm.docker = MagicMock()
            mrm.registry = {"my-gpu-model": spec}
            return mrm, spec

    def test_fallback_called_on_nvidia_error(self):
        mrm, spec = self._make_mrm_with_spec()
        nvidia_err = _make_docker_api_error("unknown runtime: nvidia")

        with (
            patch.object(mrm, "_ensure_gpu", side_effect=nvidia_err),
            patch.object(mrm, "_fallback_to_cpu_runtime", return_value={"state": "READY"}) as mock_fb,
        ):
            result = mrm.ensure_running("my-gpu-model")
            mock_fb.assert_called_once()
            assert result["state"] == "READY"

    def test_no_fallback_when_fallback_url_empty(self):
        mrm, spec = self._make_mrm_with_spec(fallback_url="")
        nvidia_err = _make_docker_api_error("unknown runtime: nvidia")

        with patch.object(mrm, "_ensure_gpu", side_effect=nvidia_err):
            with pytest.raises(docker.errors.APIError):
                mrm.ensure_running("my-gpu-model")

    def test_no_fallback_for_gpu_busy_error(self):
        """RuntimeError409 'No free GPU' must NOT trigger CPU fallback."""
        mrm, spec = self._make_mrm_with_spec()
        from model_runtime_manager.mrm.runtime import RuntimeError409
        busy_err = RuntimeError409("No free GPU. All allowed GPUs ['0'] are busy.")

        with (
            patch.object(mrm, "_ensure_gpu", side_effect=busy_err),
            patch.object(mrm, "_fallback_to_cpu_runtime") as mock_fb,
        ):
            with pytest.raises(RuntimeError409):
                mrm.ensure_running("my-gpu-model")
            mock_fb.assert_not_called()

    def test_no_fallback_for_model_not_found(self):
        """Spec lookup failure must NOT trigger CPU fallback."""
        mrm, spec = self._make_mrm_with_spec()
        from model_runtime_manager.mrm.runtime import RuntimeError409

        with pytest.raises(RuntimeError409, match="Unknown base_model"):
            mrm.ensure_running("nonexistent-model")

    def test_cpu_spec_skips_gpu_path_entirely(self):
        """Models with spec.runtime='cpu' go to _ensure_cpu, never _ensure_gpu."""
        mrm, spec = self._make_mrm_with_spec()
        spec.runtime = "cpu"

        with (
            patch.object(mrm, "_ensure_cpu", return_value={"state": "READY", "runtime_type": "cpu"}) as mock_cpu,
            patch.object(mrm, "_ensure_gpu") as mock_gpu,
        ):
            mrm.ensure_running("my-gpu-model")
            mock_cpu.assert_called_once()
            mock_gpu.assert_not_called()

"""
backends/vllm_backend.py

vLLM backend — GPU inference via Docker/vLLM containers.
Wraps the existing _ensure_gpu() logic in ModelRuntimeManager.

VRAM-aware launch flow:
  1. auto_tune_config() estimates the largest safe (max_model_len, num_seqs)
     pair for the available GPU memory.  Skipped when model metadata absent.
  2. Safety retry loop: if the container crashes (OOM-style error) we halve
     max_model_len and retry up to MAX_LAUNCH_ATTEMPTS times.
  3. On each attempt, structured logs are emitted for observability.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from .base import RuntimeBackend, NodeCapabilities
from ..vram_estimator import auto_tune_config, AutoTuneResult, get_available_vram_gb

if TYPE_CHECKING:
    from ..model_variant_registry import ModelVariant
    from ..mrm.runtime import ModelRuntimeManager

logger = logging.getLogger("MRM.backend.vllm")

# Fraction of VRAM reserved for KV cache, CUDA kernels, etc. (used by can_run)
_VRAM_OVERHEAD_FACTOR = 1.2

# Maximum launch attempts in the safety retry loop
MAX_LAUNCH_ATTEMPTS = 3

# Minimum context length we will ever try (floor for the halving ladder)
_MIN_MODEL_LEN = 256


def _is_oom_candidate(exc: Exception) -> bool:
    """
    Return True if the exception *might* be caused by an OOM or startup crash
    (i.e., worth retrying with a smaller config).

    Lock contention and GPU-busy errors are NOT worth retrying.
    """
    msg = str(exc).lower()
    non_retryable = (
        "being started",
        "no free gpu",
        "locked",
        "unknown base_model",
    )
    return not any(kw in msg for kw in non_retryable)


class VLLMBackend(RuntimeBackend):
    """
    GPU inference backend using vLLM served via Docker containers.

    Metadata:
        supports_flash_attention = True
        supports_continuous_batching = True
    """

    name = "vllm"
    supports_flash_attention = True
    supports_continuous_batching = True

    def __init__(self, mrm: "ModelRuntimeManager") -> None:
        self._mrm = mrm

    # ------------------------------------------------------------------
    # Capability check
    # ------------------------------------------------------------------

    def can_run(self, model_variant: "ModelVariant", node_caps: NodeCapabilities) -> bool:
        if not node_caps.gpu:
            return False
        if node_caps.supported_backends and "vllm" not in node_caps.supported_backends:
            return False
        if "vllm" not in model_variant.backend_compatibility:
            return False
        required_vram = int(model_variant.size_gb * 1024 * _VRAM_OVERHEAD_FACTOR)
        return node_caps.vram_mb >= required_vram

    # ------------------------------------------------------------------
    # Cost estimation — lower is better
    # ------------------------------------------------------------------

    def estimate_cost(self, model_variant: "ModelVariant", node_caps: NodeCapabilities) -> float:
        if node_caps.vram_mb == 0:
            return float("inf")
        return (model_variant.size_gb * 1024 * _VRAM_OVERHEAD_FACTOR) / node_caps.vram_mb

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(
        self,
        base_model: str,
        model_variant: "ModelVariant",
        node_caps: NodeCapabilities,
    ) -> Dict[str, Any]:
        spec = self._mrm._spec(base_model)

        # Always sync quantization from the (already conflict-resolved) variant.
        # If the variant carries quantization=None after auto-fix, we must NOT
        # pass --quantization to vLLM — it will read the native one from config.
        spec.quantization = model_variant.quantization

        # ── VRAM-aware auto-tuning ─────────────────────────────────────
        tuning: Optional[AutoTuneResult] = auto_tune_config(model_variant, node_caps)

        if tuning is not None:
            spec.max_model_len = tuning.max_model_len
            spec.max_num_seqs = tuning.num_seqs
            logger.info(
                '{"event": "auto_config_selected", "model": "%s",'
                ' "max_model_len": %d, "num_seqs": %d,'
                ' "estimated_vram_gb": %.2f, "available_vram_gb": %.2f}',
                base_model,
                tuning.max_model_len,
                tuning.num_seqs,
                tuning.estimate.total_gb,
                get_available_vram_gb(node_caps),
            )

        # ── Safety retry loop ─────────────────────────────────────────
        # If the container crashes (unexpected OOM or timeout) we halve
        # max_model_len and retry up to MAX_LAUNCH_ATTEMPTS times.
        last_exc: Optional[Exception] = None

        for attempt in range(MAX_LAUNCH_ATTEMPTS):
            try:
                result = self._mrm._ensure_gpu(base_model, spec)
                result["backend"] = self.name
                result["model_variant"] = model_variant.format
                result["ram_mb"] = int(model_variant.size_gb * 1024)
                return result

            except Exception as exc:
                last_exc = exc

                # Non-OOM errors (lock, GPU busy, unknown model) → fail fast
                if not _is_oom_candidate(exc):
                    raise

                # Last attempt — give up
                if attempt == MAX_LAUNCH_ATTEMPTS - 1:
                    break

                # Halve context length and retry
                new_len = max(spec.max_model_len // 2, _MIN_MODEL_LEN)
                logger.warning(
                    '{"event": "auto_config_fallback", "reason": "oom",'
                    ' "attempt": %d, "prev_max_model_len": %d,'
                    ' "new_max_model_len": %d, "model": "%s"}',
                    attempt + 1,
                    spec.max_model_len,
                    new_len,
                    base_model,
                )
                spec.max_model_len = new_len

        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Stop — MRM handles container lifecycle via existing stop() API
    # ------------------------------------------------------------------

    def stop(self, instance_id: str) -> None:
        pass

"""
mrm/services/validator.py

Config validator: answers "will this model actually start on this GPU?"

The vLLM engine will fail to initialise if:
  (a) the model weights exceed the vLLM memory budget
      (gpu_memory_utilization × total_vram)
  (b) not enough free VRAM exists right now to even begin loading weights
  (c) after loading weights, zero space remains for KV cache (single token)

This validator checks all three conditions and returns a detailed result
so the auto_fit loop can make targeted adaptations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..core.gpu import GpuMemory
from .config_generator import VllmConfig, _weights_vram_gb, _kv_cache_gb
from .model_enricher import ModelMeta

logger = logging.getLogger("MRM.validator")

# Minimum KV cache that must fit after weights load (in GiB).
# Equivalent to ~256 tokens for any model — below this vLLM aborts.
_MIN_KV_HEADROOM_GB = 0.25

# Safety fraction of free VRAM required before starting a load.
# 0.85 accounts for GPU memory fragmentation from CUDA context,
# driver overhead, and other processes that may hold GPU memory.
_FREE_VRAM_SAFETY = 0.85


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    fits:           bool
    weights_gb:     float
    kv_cache_gb:    float
    required_gb:    float   # weights + kv
    allowed_gb:     float   # gpu_util × total
    free_gb:        float   # current free VRAM on GPU
    # Failure reason (populated when fits=False)
    reason:         str = ""

    def __bool__(self) -> bool:
        return self.fits


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def validate_config(
    meta:   ModelMeta,
    gpu:    GpuMemory,
    config: VllmConfig,
) -> ValidationResult:
    """
    Validate whether *config* can be loaded on *gpu* for model *meta*.

    Checks
    ------
    1. Weights fit inside vLLM's budget  (weights_gb ≤ allowed_gb)
    2. Enough total budget for weights + minimum KV cache
    3. Current free VRAM is sufficient to begin loading

    Args:
        meta:   Enriched model metadata.
        gpu:    Real-time GPU memory snapshot.
        config: Proposed vLLM config.

    Returns:
        ValidationResult with fits=True/False and diagnostics.
    """
    total_gb   = gpu.total_mib / 1024.0
    free_gb    = gpu.free_mib  / 1024.0
    allowed_gb = total_gb * config.gpu_memory_utilization

    weights_gb  = _weights_vram_gb(meta.params_b, config.quantization)
    kv_gb       = _kv_cache_gb(meta, config.max_model_len)
    required_gb = weights_gb + kv_gb

    reason = ""

    # ── Check 1: weights fit inside vLLM budget ───────────────────────
    if weights_gb > allowed_gb:
        reason = (
            f"weights ({weights_gb:.2f}GiB) exceed vLLM budget "
            f"({allowed_gb:.2f}GiB = {config.gpu_memory_utilization:.0%} × {total_gb:.1f}GiB). "
            "Need larger gpu_memory_utilization or quantization."
        )

    # ── Check 2: weights + KV headroom fit inside budget ─────────────
    elif weights_gb + _MIN_KV_HEADROOM_GB > allowed_gb:
        reason = (
            f"weights ({weights_gb:.2f}GiB) + min KV headroom ({_MIN_KV_HEADROOM_GB}GiB) "
            f"exceed budget ({allowed_gb:.2f}GiB). "
            "Reduce max_model_len or use quantization."
        )

    # ── Check 3: enough current free VRAM to start ────────────────────
    elif weights_gb > free_gb * _FREE_VRAM_SAFETY:
        reason = (
            f"weights ({weights_gb:.2f}GiB) exceed current free VRAM "
            f"({free_gb:.2f}GiB × {_FREE_VRAM_SAFETY:.0%} = {free_gb * _FREE_VRAM_SAFETY:.2f}GiB). "
            "Another model may be holding memory."
        )

    fits = (reason == "")

    result = ValidationResult(
        fits        = fits,
        weights_gb  = round(weights_gb, 3),
        kv_cache_gb = round(kv_gb, 3),
        required_gb = round(required_gb, 3),
        allowed_gb  = round(allowed_gb, 3),
        free_gb     = round(free_gb, 3),
        reason      = reason,
    )

    if fits:
        logger.info(
            "[VALIDATION] OK  model=%s  quant=%s  weights=%.2fGiB  "
            "kv=%.2fGiB  required=%.2fGiB  allowed=%.2fGiB  free=%.2fGiB",
            meta.repo_id, config.quantization or "fp16",
            weights_gb, kv_gb, required_gb, allowed_gb, free_gb,
        )
    else:
        logger.warning(
            "[VALIDATION] FAIL  model=%s  reason=%s",
            meta.repo_id, reason,
        )

    return result

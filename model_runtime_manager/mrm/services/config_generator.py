"""
mrm/services/config_generator.py

Dynamic vLLM config generator.

No presets.  No static rules.

Given:
    meta  — ModelMeta (params_b, quantization, architecture)
    gpu   — GpuMemory (total_mib, free_mib)

Computes:
    gpu_memory_utilization  — fraction of total GPU VRAM vLLM may use
    max_model_len           — context window in tokens
    dtype                   — torch dtype string
    quantization            — quantization scheme for vLLM flag

Design
------
vLLM reserves  ``gpu_memory_utilization × total_vram`` for its engine.
Of that budget:
  [weights_vram] — fixed cost, determined by model size + quantization
  [kv_cache]     — scales with max_model_len; vLLM fills leftover space

We therefore set:

    utilization = (weights_gb + kv_cache_gb) / total_gb × SAFETY

    where SAFETY = 1.10   (10 % overhead for framework state, CUDA context)

Then clamp: [MIN_UTIL, MAX_UTIL].

If params_b is unknown (0) we fall back to a conservative 0.85.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..core.gpu import GpuMemory
from .model_enricher import ModelMeta

logger = logging.getLogger("MRM.config_generator")

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────────────

_GiB             = 1024 ** 3
_WEIGHT_OVERHEAD = 1.15        # activation buffers, layer norm caches, etc.
_SAFETY_MARGIN   = 1.10        # headroom on top of estimated total
_MIN_UTIL        = 0.30
_MAX_UTIL        = 0.95

# Bytes per parameter per quantization scheme
_BYTES_PER_PARAM: dict[Optional[str], float] = {
    None:    2.0,   # fp16 / bf16
    "awq":   0.5,   # 4-bit
    "gptq":  0.5,
    "int8":  1.0,
    "bnb":   1.0,
    "int4":  0.5,
    "fp8":   1.0,
    "gguf":  0.5,
    "ggml":  0.5,
}

# Max context windows to try, in descending order
CONTEXT_LADDER = [4096, 2048, 1024, 512, 256]

# dtype mapping per quantization
_DTYPE_MAP: dict[Optional[str], str] = {
    None:    "auto",
    "awq":   "auto",
    "gptq":  "auto",
    "int8":  "auto",
    "bnb":   "auto",
    "int4":  "auto",
    "fp8":   "auto",
    "gguf":  "auto",
    "ggml":  "auto",
}


# ──────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VllmConfig:
    gpu_memory_utilization: float
    max_model_len:          int
    dtype:                  str
    quantization:           Optional[str]
    # Diagnostics (not passed to vLLM)
    weights_gb:             float = 0.0
    kv_cache_gb:            float = 0.0
    required_gb:            float = 0.0
    allowed_gb:             float = 0.0

    def as_spec_kwargs(self) -> dict:
        """Return only the keys ModelSpec / vLLM care about."""
        return {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len":          self.max_model_len,
            "dtype":                  self.dtype,
            "quantization":           self.quantization,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_config(
    meta:          ModelMeta,
    gpu:           GpuMemory,
    max_model_len: int = 2048,
) -> VllmConfig:
    """
    Compute a vLLM config from first principles.

    Steps
    -----
    1.  Estimate weight VRAM from params_b × bytes_per_param × overhead.
    2.  Estimate KV-cache VRAM from params_b × context_scaling.
    3.  Set gpu_memory_utilization = required / total × SAFETY.
    4.  Select dtype based on quantization.

    Args:
        meta:          Enriched model metadata.
        gpu:           Current GPU memory snapshot.
        max_model_len: Desired context window (tokens).

    Returns:
        VllmConfig with all fields filled in.
    """
    total_gb = gpu.total_mib / 1024.0

    weights_gb = _weights_vram_gb(meta.params_b, meta.quantization)
    kv_gb      = _kv_cache_gb(meta, max_model_len)
    required_gb = weights_gb + kv_gb

    if total_gb > 0 and meta.params_b > 0:
        raw_util = (required_gb / total_gb) * _SAFETY_MARGIN
        utilization = max(_MIN_UTIL, min(_MAX_UTIL, raw_util))
    else:
        # Unknown model size or no GPU info — use safe default
        utilization = 0.85

    allowed_gb = total_gb * utilization

    dtype = _DTYPE_MAP.get(meta.quantization, "auto")

    cfg = VllmConfig(
        gpu_memory_utilization = round(utilization, 3),
        max_model_len          = max_model_len,
        dtype                  = dtype,
        quantization           = meta.quantization,
        weights_gb             = round(weights_gb, 3),
        kv_cache_gb            = round(kv_gb, 3),
        required_gb            = round(required_gb, 3),
        allowed_gb             = round(allowed_gb, 3),
    )

    logger.info(
        "[CONFIG] generated  model=%s  quant=%s  params=%.1fB  "
        "weights=%.1fGiB  kv=%.1fGiB  required=%.1fGiB  "
        "allowed=%.1fGiB  util=%.2f  ctx=%d",
        meta.repo_id,
        meta.quantization or "fp16",
        meta.params_b,
        weights_gb,
        kv_gb,
        required_gb,
        allowed_gb,
        utilization,
        max_model_len,
    )

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Internal physics
# ──────────────────────────────────────────────────────────────────────────────

def _weights_vram_gb(params_b: float, quantization: Optional[str]) -> float:
    """Weight tensor VRAM in GiB including framework overhead."""
    if params_b <= 0:
        return 0.0
    bpp = _BYTES_PER_PARAM.get(quantization, 2.0)
    return params_b * 1e9 * bpp / _GiB * _WEIGHT_OVERHEAD


def _kv_cache_gb(meta: "ModelMeta", context_len: int) -> float:
    """
    KV-cache GiB estimate.

    Exact formula when architecture details are known (num_layers > 0, hidden_size > 0):

        kv_bytes = 4 × num_layers × hidden_size × context_len

    Derivation (standard transformer, fp16):
        - 2 tensors: K and V
        - 2 bytes per element (fp16)
        - total = 2 × 2 × num_layers × hidden_size × context_len
                = 4 × num_layers × hidden_size × context_len

    Falls back to empirical estimate when architecture is unknown:
        ~0.5 GiB per billion params at 2 048 tokens (fp16, 32-head MHA).
    """
    if meta.num_layers > 0 and meta.hidden_size > 0:
        kv_bytes = 4 * meta.num_layers * meta.hidden_size * context_len
        return kv_bytes / _GiB

    # Empirical fallback
    if meta.params_b <= 0:
        return 0.0
    return meta.params_b * 0.5 * (context_len / 2048.0)

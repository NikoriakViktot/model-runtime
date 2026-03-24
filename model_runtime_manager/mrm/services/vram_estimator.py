"""
mrm/services/vram_estimator.py

Estimates total GPU VRAM required to run a model for inference.

Formula
-------
  total = weights_gb + kv_cache_gb

  weights_gb  = params_b * bytes_per_param(quant) / 1 GiB * overhead_factor
  kv_cache_gb = num_layers * 2 * num_heads * head_dim * context_len * 2 bytes
                ─────────────────────────────────────────────────────────────
                                    1 GiB

When exact architecture details are unknown (most search results), we fall
back to empirical constants derived from commonly used models.
"""
from __future__ import annotations

from .model_enricher import ModelMeta

# ──────────────────────────────────────────────────────────────────────────────
# Bytes-per-parameter by quantization scheme
# ──────────────────────────────────────────────────────────────────────────────
_BYTES_PER_PARAM = {
    None:   2.0,   # fp16 / bf16
    "awq":  0.5,   # 4-bit
    "gptq": 0.5,
    "bnb":  1.0,   # 8-bit by default; 4-bit bnb would be 0.5
    "int8": 1.0,
    "int4": 0.5,
    "fp8":  1.0,
    "gguf": 0.5,
    "ggml": 0.5,
}

_GiB = 1024 ** 3
_WEIGHT_OVERHEAD = 1.15  # frameworks keep activations + grad buffers


# ──────────────────────────────────────────────────────────────────────────────
# Empirical KV-cache constants per parameter-scale
# (hidden_size^2 / num_heads * num_layers * 2 tensors * fp16)
# Tuned against Llama/Qwen/Mistral families.
# ──────────────────────────────────────────────────────────────────────────────
def _kv_cache_gb(params_b: float, context_len: int) -> float:
    """
    Rough KV-cache estimate in GiB.

    Empirically: ~0.5 GiB per 1 B params at 2 k context (fp16 KV, 32 heads).
    Scale linearly with context_len relative to 2048.
    """
    if params_b <= 0:
        return 0.0
    base_per_b_at_2k = 0.5          # GiB per billion params at 2 048 tokens
    scale = context_len / 2048.0
    return round(params_b * base_per_b_at_2k * scale, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def estimate_vram_requirement(meta: ModelMeta, context_len: int = 2048) -> float:
    """
    Return estimated total VRAM in GiB needed to load and run *meta* at
    *context_len* tokens.

    Returns 0.0 if parameter count is unknown.
    """
    if meta.params_b <= 0:
        return 0.0

    bpp = _BYTES_PER_PARAM.get(meta.quantization, 2.0)
    weights_gb = meta.params_b * 1e9 * bpp / _GiB * _WEIGHT_OVERHEAD
    kv_gb = _kv_cache_gb(meta.params_b, context_len)

    return round(weights_gb + kv_gb, 2)

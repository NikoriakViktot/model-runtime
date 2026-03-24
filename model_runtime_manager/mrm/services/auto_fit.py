"""
mrm/services/auto_fit.py

Auto-fit loop: finds a valid (model, vLLM config) pair that fits on the GPU.

Algorithm
---------

    auto_fit(meta, gpu)

    ┌─────────────────────────────────────────────────────┐
    │ 1. generate_config(meta, gpu, ctx=2048)             │
    │    validate_config  → fits?  ──► [SUCCESS]          │
    │                                                     │
    │ 2. Context ladder: 1024 → 512 → 256                │
    │    for each ctx:                                    │
    │      generate_config(meta, gpu, ctx)                │
    │      validate_config  → fits?  ──► [SUCCESS]        │
    │                                                     │
    │ 3. Quantized fallback (only if original is fp16)   │
    │    find_quantized_model(meta, hf_client)            │
    │    if found:                                        │
    │      auto_fit(quantized_meta, gpu,                  │
    │              _depth+1, _tried_quant=True)           │
    │                                                     │
    │ 4. raise AutoFitError                               │
    └─────────────────────────────────────────────────────┘

The recursive call for the quantized alternative is capped at depth 1
(we don't chase quantized models of quantized models).

Returns
-------
    (ModelMeta, VllmConfig)  — the model to actually load and its config.

The returned ModelMeta may differ from the input meta when a quantized
alternative was selected.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from ..core.gpu import GpuMemory
from .config_generator import (
    VllmConfig,
    generate_config,
    CONTEXT_LADDER,
)
from .model_enricher import ModelMeta
from .quant_finder import find_quantized_model
from .validator import ValidationResult, validate_config, _FREE_VRAM_SAFETY

logger = logging.getLogger("MRM.auto_fit")

_MAX_DEPTH = 1    # max recursion depth for quantized fallback


# ──────────────────────────────────────────────────────────────────────────────
# Exception
# ──────────────────────────────────────────────────────────────────────────────

class AutoFitError(Exception):
    """Raised when no valid config can be found for the model on the GPU."""


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def auto_fit(
    meta:            ModelMeta,
    gpu:             GpuMemory,
    hf_client=None,
    telemetry_store=None,
    _depth:          int = 0,
    _tried_quant:    bool = False,
) -> Tuple[ModelMeta, VllmConfig]:
    """
    Find the best (model, config) pair that fits on *gpu*.

    Args:
        meta:             Enriched model metadata.
        gpu:              Real-time GPU memory snapshot.
        hf_client:        HFClient for quantized model search (None = skip step 3).
        telemetry_store:  Optional TelemetryStore for reputation-aware quant ranking.
        _depth:           Internal recursion depth guard.
        _tried_quant:     Internal flag — True when we're already in quant branch.

    Returns:
        (ModelMeta, VllmConfig) ready for ModelSpec construction.

    Raises:
        AutoFitError if no config fits.
    """
    total_gb = gpu.total_mib / 1024.0
    logger.info(
        "[AUTO_FIT] start  model=%s  params=%.1fB  quant=%s  "
        "gpu_total=%.1fGiB  gpu_free=%.1fGiB  depth=%d",
        meta.repo_id, meta.params_b,
        meta.quantization or "fp16",
        total_gb, gpu.free_mib / 1024.0, _depth,
    )

    # ── Step 1: Try default context (2048) ───────────────────────────
    result = _try_context(meta, gpu, ctx=2048)
    if result:
        return meta, result.config

    # ── Step 2: Context ladder ────────────────────────────────────────
    for ctx in CONTEXT_LADDER:
        if ctx >= 2048:
            continue   # already tried 2048 above
        logger.info("[ADAPT] reducing context  model=%s  ctx=%d", meta.repo_id, ctx)
        result = _try_context(meta, gpu, ctx=ctx)
        if result:
            return meta, result.config

    # ── Step 3: Quantized fallback ────────────────────────────────────
    if (
        meta.quantization is None      # only try if fp16
        and not _tried_quant           # don't recurse on a quant alternative
        and _depth < _MAX_DEPTH
        and hf_client is not None
    ):
        logger.info(
            "[ADAPT] switching to quantized model  original=%s", meta.repo_id
        )
        quant_meta = find_quantized_model(meta, hf_client, telemetry_store=telemetry_store)
        if quant_meta is not None:
            logger.info(
                "[ADAPT] found quantized alternative  %s → %s  quant=%s",
                meta.repo_id, quant_meta.repo_id, quant_meta.quantization,
            )
            try:
                return auto_fit(
                    quant_meta, gpu,
                    hf_client=hf_client,
                    telemetry_store=telemetry_store,
                    _depth=_depth + 1,
                    _tried_quant=True,
                )
            except AutoFitError:
                logger.warning(
                    "[ADAPT] quantized alternative %s also doesn't fit — giving up",
                    quant_meta.repo_id,
                )

    # ── Step 4: Fail ─────────────────────────────────────────────────
    raise AutoFitError(
        f"[AUTO_FIT] FAIL  model={meta.repo_id}  "
        f"params={meta.params_b:.1f}B  quant={meta.quantization or 'fp16'}  "
        f"gpu_total={total_gb:.1f}GiB  gpu_free={gpu.free_mib/1024:.1f}GiB  "
        "No context window fits even with quantization. "
        "Use a smaller model or a larger GPU."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

class _TryResult:
    """Wraps a successful (config, validation) pair."""
    def __init__(self, config: VllmConfig, validation: ValidationResult) -> None:
        self.config     = config
        self.validation = validation

    def __bool__(self) -> bool:
        return True


def _try_context(
    meta: ModelMeta,
    gpu:  GpuMemory,
    ctx:  int,
) -> Optional[_TryResult]:
    """
    Generate config for *ctx* tokens and validate it.

    Returns _TryResult on success, None on failure.

    Fast-reject: if the model's estimated weight VRAM already exceeds the
    fragmentation-adjusted free VRAM, skip expensive config generation.
    """
    if meta.estimated_vram_gb > 0:
        effective_free_gb = (gpu.free_mib / 1024.0) * _FREE_VRAM_SAFETY
        if meta.estimated_vram_gb > effective_free_gb:
            return None

    cfg = generate_config(meta, gpu, max_model_len=ctx)
    v   = validate_config(meta, gpu, cfg)

    if v.fits:
        logger.info(
            "[SUCCESS] config found  model=%s  ctx=%d  "
            "util=%.2f  required=%.2fGiB  allowed=%.2fGiB",
            meta.repo_id, ctx,
            cfg.gpu_memory_utilization,
            v.required_gb, v.allowed_gb,
        )
        return _TryResult(cfg, v)

    return None

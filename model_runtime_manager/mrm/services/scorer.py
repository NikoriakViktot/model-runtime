"""
mrm/services/scorer.py

GPU-aware model filtering, preset auto-selection, and scoring/ranking.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .model_enricher import ModelMeta
from .vram_estimator import estimate_vram_requirement


# ──────────────────────────────────────────────────────────────────────────────
# GPU-aware filter
# ──────────────────────────────────────────────────────────────────────────────

def filter_models_by_gpu(
    models_with_meta: List[Tuple[Dict[str, Any], ModelMeta]],
    gpu_vram_available_gb: float,
    context_len: int = 2048,
    headroom_gb: float = 1.0,
) -> List[Tuple[Dict[str, Any], ModelMeta]]:
    """
    Return only (raw_model, meta) pairs whose estimated VRAM fits on the GPU.

    Models with unknown parameter counts (params_b == 0) are kept because we
    cannot rule them out — callers can choose to display them with a warning.

    Args:
        models_with_meta:     list of (raw HF dict, ModelMeta) tuples
        gpu_vram_available_gb: free VRAM on the target GPU in GiB
        context_len:           context window to use for KV-cache estimate
        headroom_gb:           safety margin to subtract from available VRAM
    """
    effective = gpu_vram_available_gb - headroom_gb
    result = []
    for raw, meta in models_with_meta:
        if meta.params_b <= 0:
            result.append((raw, meta))
            continue
        needed = estimate_vram_requirement(meta, context_len)
        if needed <= effective:
            result.append((raw, meta))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Preset auto-selection
# ──────────────────────────────────────────────────────────────────────────────

# Thresholds and overrides per preset
_PRESET_RULES = [
    # (max_params_b, quantization_hint,  preset_name, extra_overrides)
    (3.5,  None,    "small_chat", {}),
    (8.0,  "awq",   "7b_awq",    {}),
    (8.0,  "gptq",  "7b_awq",    {"quantization": "gptq"}),
    (8.0,  None,    "7b_awq",    {"quantization": "awq"}),
    # Larger models: try awq preset with reduced context
    (14.0, "awq",   "7b_awq",    {"max_model_len": 256}),
    (14.0, None,    "7b_awq",    {"quantization": "awq", "max_model_len": 256}),
]

def select_preset(
    meta: ModelMeta,
    gpu_vram_gb: float,
) -> Dict[str, Any]:
    """
    Pick the best preset name and any recommended overrides for *meta*.

    Returns::

        {
            "preset": "small_chat" | "7b_awq",
            "overrides": {"gpu_memory_utilization": 0.8, ...},
            "fits": True | False,   # whether the model likely fits
        }
    """
    # If we don't know params, play safe with small_chat
    if meta.params_b <= 0:
        return {"preset": "small_chat", "overrides": {}, "fits": None}

    needed = estimate_vram_requirement(meta)
    fits = needed <= gpu_vram_gb

    for max_b, quant_hint, preset, overrides in _PRESET_RULES:
        if meta.params_b > max_b:
            continue
        if quant_hint and meta.quantization != quant_hint:
            continue
        # Adjust gpu_memory_utilization to give some headroom
        util = min(0.95, needed / gpu_vram_gb + 0.05) if gpu_vram_gb > 0 else 0.9
        result_overrides = dict(overrides)
        result_overrides["gpu_memory_utilization"] = round(util, 2)
        return {"preset": preset, "overrides": result_overrides, "fits": fits}

    # Fallback
    return {"preset": "7b_awq", "overrides": {"quantization": "awq"}, "fits": fits}


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

# Bonus for architectures known to work well with vLLM
_ARCH_BONUS = {
    "LlamaForCausalLM": 1.0,
    "MistralForCausalLM": 1.0,
    "QWenLMHeadModel": 0.8,
    "Qwen2ForCausalLM": 0.8,
    "PhiForCausalLM": 0.5,
    "FalconForCausalLM": 0.3,
    "GPTNeoXForCausalLM": 0.3,
}

def score_model(
    raw: Dict[str, Any],
    meta: ModelMeta,
    gpu_vram_gb: float = 0.0,
    context_len: int = 2048,
) -> float:
    """
    Compute a scalar score for ranking.

    Higher = better.

    Formula:
        score = log1p(downloads) * 0.6
              + log1p(likes)     * 0.4
              + arch_bonus
              - vram_penalty
    """
    dl_score  = math.log1p(meta.downloads) * 0.6
    lk_score  = math.log1p(meta.likes)     * 0.4
    arch_bonus = _ARCH_BONUS.get(meta.architecture or "", 0.0)

    vram_penalty = 0.0
    if gpu_vram_gb > 0 and meta.params_b > 0:
        needed = estimate_vram_requirement(meta, context_len)
        # Penalty grows quadratically as model approaches / exceeds GPU capacity
        ratio = needed / gpu_vram_gb
        if ratio > 0.95:
            vram_penalty = (ratio - 0.95) * 10.0
        else:
            # Slight bonus for leaving more headroom
            vram_penalty = -0.2 * (1.0 - ratio)

    return dl_score + lk_score + arch_bonus - vram_penalty


def rank_models(
    models_with_meta: List[Tuple[Dict[str, Any], ModelMeta]],
    gpu_vram_gb: float = 0.0,
    context_len: int = 2048,
) -> List[Dict[str, Any]]:
    """
    Score and sort *models_with_meta* descending by score.

    Returns a list of dicts ready for API serialisation::

        [
            {
                "repo_id": "...",
                "score": 12.34,
                "params_b": 7.0,
                "estimated_vram_gb": 14.1,
                "quantization": "awq",
                "architecture": "LlamaForCausalLM",
                "downloads": 1000000,
                "likes": 5000,
                "pipeline_tag": "text-generation",
                "preset": "7b_awq",
                "preset_overrides": {...},
                "fits": True,
            },
            ...
        ]
    """
    scored = []
    for raw, meta in models_with_meta:
        s = score_model(raw, meta, gpu_vram_gb, context_len)
        preset_info = select_preset(meta, gpu_vram_gb) if gpu_vram_gb > 0 else {}
        scored.append({
            "repo_id": meta.repo_id,
            "score": round(s, 3),
            "params_b": meta.params_b,
            "estimated_vram_gb": estimate_vram_requirement(meta, context_len),
            "quantization": meta.quantization,
            "architecture": meta.architecture,
            "downloads": meta.downloads,
            "likes": meta.likes,
            "pipeline_tag": meta.pipeline_tag,
            "tags": meta.tags[:10],  # truncate for API response size
            "preset": preset_info.get("preset"),
            "preset_overrides": preset_info.get("overrides", {}),
            "fits": preset_info.get("fits"),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

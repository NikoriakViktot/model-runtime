"""
mrm/services/model_enricher.py

Enriches a raw HuggingFace API model response into a structured ModelMeta
dataclass that downstream services (VRAM estimator, scorer, preset selector)
can consume without re-parsing the raw dict.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelMeta:
    repo_id: str
    params_b: float = 0.0          # billions of parameters (0 = unknown)
    estimated_vram_gb: float = 0.0  # naive base-model VRAM at fp16 (no KV cache)
    quantization: Optional[str] = None  # awq | gptq | bnb | None
    architecture: Optional[str] = None  # LlamaForCausalLM | QWenLMHeadModel | …
    downloads: int = 0
    likes: int = 0
    tags: List[str] = field(default_factory=list)
    pipeline_tag: Optional[str] = None
    # Architecture details for exact KV-cache estimation
    num_layers:  int = 0  # num_hidden_layers from HF config (0 = unknown)
    hidden_size: int = 0  # hidden_size from HF config          (0 = unknown)
    # Raw fields kept for display
    private: bool = False
    disabled: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def enrich(raw: Dict[str, Any]) -> ModelMeta:
    """
    Convert a raw HF API model dict (``/api/models/{repo_id}?full=true``)
    into a :class:`ModelMeta`.
    """
    repo_id = raw.get("modelId") or raw.get("id") or ""
    tags: List[str] = raw.get("tags") or []
    pipeline_tag: Optional[str] = raw.get("pipeline_tag")

    params_b = _extract_params_b(raw)
    quantization = _detect_quantization(raw, tags)
    architecture = _extract_architecture(raw)
    estimated_vram_gb = _base_vram_gb(params_b, quantization)
    num_layers, hidden_size = _extract_arch_details(raw)

    return ModelMeta(
        repo_id=repo_id,
        params_b=params_b,
        estimated_vram_gb=estimated_vram_gb,
        quantization=quantization,
        architecture=architecture,
        downloads=raw.get("downloads") or 0,
        likes=raw.get("likes") or 0,
        tags=tags,
        pipeline_tag=pipeline_tag,
        num_layers=num_layers,
        hidden_size=hidden_size,
        private=raw.get("private", False),
        disabled=raw.get("disabled", False),
    )


def enrich_list(raw_list: List[Dict[str, Any]]) -> List[ModelMeta]:
    return [enrich(r) for r in raw_list]


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_params_b(raw: Dict[str, Any]) -> float:
    """
    Try several locations where HF stores parameter counts:
      1. safetensors.total (authoritative, direct byte count)
      2. cardData / model card metadata
      3. Heuristic from the repo name (e.g. "7B", "1.8b", "70B")
    """
    # 1. safetensors.total → number of parameters (not bytes)
    st = raw.get("safetensors") or {}
    total_params = st.get("total")
    if total_params and total_params > 0:
        return round(total_params / 1e9, 2)

    # 2. cardData.model-index or metadata
    card = raw.get("cardData") or {}
    for key in ("num_parameters", "model_parameters"):
        v = card.get(key)
        if v:
            try:
                return round(float(v) / 1e9, 2)
            except (ValueError, TypeError):
                pass

    # 3. Regex from repo name
    name = raw.get("modelId") or raw.get("id") or ""
    return _params_from_name(name)


_PARAM_RE = re.compile(r"[_\-./](\d+(?:\.\d+)?)[bB](?:[_\-./]|$)")

def _params_from_name(name: str) -> float:
    """Extract '7B', '1.8b', '70b' patterns from model name."""
    m = _PARAM_RE.search(name)
    if m:
        return float(m.group(1))
    return 0.0


_QUANT_TAGS = {
    "awq": "awq",
    "gptq": "gptq",
    "bnb": "bnb",
    "bitsandbytes": "bnb",
    "gguf": "gguf",
    "ggml": "ggml",
    "int8": "int8",
    "int4": "int4",
    "fp8": "fp8",
}

def _detect_quantization(raw: Dict[str, Any], tags: List[str]) -> Optional[str]:
    """Detect quantization type from tags, model name, or library."""
    name = (raw.get("modelId") or raw.get("id") or "").lower()
    tags_lower = [t.lower() for t in tags]
    library = (raw.get("library_name") or "").lower()

    combined = " ".join([name, library] + tags_lower)
    for keyword, quant in _QUANT_TAGS.items():
        if keyword in combined:
            return quant
    return None


def _extract_arch_details(raw: Dict[str, Any]) -> tuple:
    """
    Return (num_layers, hidden_size) from the HF model config dict.

    HuggingFace uses consistent field names across Llama, Mistral, Qwen, Falcon:
      - "num_hidden_layers"  → transformer depth
      - "hidden_size"        → embedding / attention dimension
    """
    config = raw.get("config") or {}
    num_layers  = int(config.get("num_hidden_layers") or 0)
    hidden_size = int(config.get("hidden_size") or 0)
    return num_layers, hidden_size


def _extract_architecture(raw: Dict[str, Any]) -> Optional[str]:
    """Pull the first architecture name from model config."""
    config = raw.get("config") or {}
    archs = config.get("architectures") or []
    if archs:
        return archs[0]
    # Fall back to pipeline_tag or library
    return raw.get("pipeline_tag")


# Bytes per parameter at fp16 = 2
_FP16_BYTES = 2.0
_QUANT_MULTIPLIERS = {
    "awq": 0.5,
    "gptq": 0.5,
    "bnb": 0.5,
    "int8": 0.5,
    "int4": 0.25,
    "gguf": 0.5,
    "ggml": 0.5,
    "fp8": 0.5,
}
# Add 15% overhead for activations / framework tensors
_OVERHEAD = 1.15

def _base_vram_gb(params_b: float, quantization: Optional[str]) -> float:
    """
    Naive minimum VRAM estimate for weights only (no KV cache, no activations
    beyond the 15% overhead).
    """
    if params_b <= 0:
        return 0.0
    mult = _QUANT_MULTIPLIERS.get(quantization or "", 1.0)
    raw_gb = params_b * 1e9 * _FP16_BYTES * mult / (1024 ** 3)
    return round(raw_gb * _OVERHEAD, 2)

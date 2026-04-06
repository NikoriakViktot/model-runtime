"""
model_variant_registry.py

ModelVariant — describes a specific build/quantization of a model.
VariantRegistry — maps base_model_id -> list[ModelVariant].

Each logical model may map to multiple variants:
    fp16   → vLLM
    awq    → vLLM
    gguf_q8 → llama.cpp
    gguf_q4 → llama.cpp / cpu

Usage example:

    from model_runtime_manager.model_variant_registry import ModelVariant, get_registry

    get_registry().register("Qwen/Qwen1.5-1.8B-Chat", [
        ModelVariant(
            base_model_id="Qwen/Qwen1.5-1.8B-Chat",
            format="fp16",
            size_gb=3.5,
            backend_compatibility=["vllm"],
        ),
        ModelVariant(
            base_model_id="Qwen/Qwen1.5-1.8B-Chat",
            format="gguf_q4",
            size_gb=1.2,
            gguf_path="/models/qwen-1.8b-q4.gguf",
            backend_compatibility=["llama.cpp", "cpu"],
        ),
    ])
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Fallback order: preferred first, cheapest/smallest last
VARIANT_FORMAT_PRIORITY: List[str] = ["fp16", "awq", "gguf_q8", "gguf_q4"]


@dataclass
class ModelVariant:
    """
    A concrete build of a model that can be served by one or more backends.

    Attributes:
        base_model_id:         HuggingFace or local model identifier.
        format:                One of "fp16", "awq", "gguf_q8", "gguf_q4".
        quantization:          Quantization tag used by vLLM (e.g. "awq"), or None.
        size_gb:               Approximate weight file size in GB.
        backend_compatibility: List of backend names that can serve this variant.
        hf_repo:               Override HF repo (e.g. quantised variant lives in a different repo).
        gguf_path:             Container-side path to the GGUF file (for llama.cpp / cpu backends).
        n_ctx:                 Context length for llama.cpp server.
    """
    base_model_id: str
    format: str                          # "fp16" | "awq" | "gguf_q8" | "gguf_q4"
    quantization: Optional[str] = None
    size_gb: float = 0.0
    backend_compatibility: List[str] = field(default_factory=list)
    hf_repo: Optional[str] = None
    gguf_path: Optional[str] = None
    n_ctx: int = 2048
    # VRAM estimation metadata (optional — auto-tuning is skipped when model_size_b == 0)
    model_size_b: int = 0               # e.g. 7_000_000_000 for a 7B model
    dtype: str = "fp16"                 # weight dtype for VRAM estimation ("fp16", "fp8", etc.)

    def copy_with(self, **changes: Any) -> "ModelVariant":
        """Return a new ModelVariant with the given fields overridden."""
        return dataclasses.replace(self, **changes)


class VariantRegistry:
    """Thread-safe-ish registry mapping base_model -> list[ModelVariant]."""

    def __init__(self) -> None:
        self._variants: Dict[str, List[ModelVariant]] = {}

    def register(self, base_model: str, variants: List[ModelVariant]) -> None:
        """Register variants for a model, replacing any previous entries."""
        self._variants[base_model] = list(variants)

    def get_variants(self, base_model: str) -> List[ModelVariant]:
        """Return registered variants for a model, or an empty list."""
        return list(self._variants.get(base_model, []))

    def has_variants(self, base_model: str) -> bool:
        """Return True if at least one variant is registered for this model."""
        return base_model in self._variants and bool(self._variants[base_model])

    def all_models(self) -> List[str]:
        return list(self._variants.keys())


# Module-level singleton — can be replaced in tests
_registry = VariantRegistry()


def get_registry() -> VariantRegistry:
    """Return the global variant registry singleton."""
    return _registry

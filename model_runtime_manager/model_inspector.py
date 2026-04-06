"""
model_inspector.py

Inspects a HuggingFace model's config to detect native quantization settings.

Results are cached per repo_id (lru_cache) so repeated calls within a process
pay zero network cost after the first lookup.

Fail-safe: any error (network, auth, malformed config) returns
ModelCapabilities(native_quantization=None), which means "unknown — let vLLM decide".
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass
class ModelCapabilities:
    """Capabilities detected from a model's HuggingFace config."""
    native_quantization: str | None


@lru_cache(maxsize=128)
def inspect_model(repo_id: str) -> ModelCapabilities:
    """
    Fetch and parse the model config to find any native quantization setting.

    Args:
        repo_id: HuggingFace model repo ID or local path.

    Returns:
        ModelCapabilities with native_quantization set to the quant_method
        string (e.g. "fp8", "awq", "gptq") or None if not present / unknown.
    """
    try:
        from transformers import AutoConfig  # type: ignore[import]

        cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        quant_cfg = getattr(cfg, "quantization_config", None)

        if quant_cfg:
            # quantization_config may be a dict or a QuantizationConfigMixin object
            if isinstance(quant_cfg, dict):
                quant_method = quant_cfg.get("quant_method")
            else:
                quant_method = getattr(quant_cfg, "quant_method", None)

            if quant_method:
                return ModelCapabilities(native_quantization=str(quant_method))

    except Exception:
        # Fail-safe: treat as unknown — caller will not apply any override
        pass

    return ModelCapabilities(native_quantization=None)

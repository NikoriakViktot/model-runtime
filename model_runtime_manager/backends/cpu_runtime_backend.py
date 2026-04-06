"""
backends/cpu_runtime_backend.py

CPU Runtime backend — wraps the existing cpu_runtime container service.

This backend delegates to _ensure_cpu() or _fallback_to_cpu_runtime() in
ModelRuntimeManager, depending on whether a fallback URL is configured.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

from .base import RuntimeBackend, NodeCapabilities

if TYPE_CHECKING:
    from ..model_variant_registry import ModelVariant
    from ..mrm.runtime import ModelRuntimeManager

logger = logging.getLogger("MRM.backend.cpu_runtime")

# CPU runtime has a higher overhead factor due to memory-mapped model loading
_RAM_OVERHEAD_FACTOR = 1.5

# Cost penalty: cpu runtime is the last resort after vllm and llama.cpp
_COST_PENALTY = 3.0


class CpuRuntimeBackend(RuntimeBackend):
    """
    Wraps the existing CPU runtime service as a backend.

    Supports any model format, limited only by available RAM.
    Preferred only when neither vLLM nor llama.cpp can run the model.
    """

    name = "cpu"

    def __init__(self, mrm: "ModelRuntimeManager") -> None:
        self._mrm = mrm

    # ------------------------------------------------------------------
    # Capability check
    # ------------------------------------------------------------------

    def can_run(self, model_variant: "ModelVariant", node_caps: NodeCapabilities) -> bool:
        if node_caps.supported_backends and "cpu" not in node_caps.supported_backends:
            return False
        if "cpu" not in model_variant.backend_compatibility:
            return False
        required_ram = self._estimate_ram_mb(model_variant)
        return node_caps.ram_mb >= required_ram

    # ------------------------------------------------------------------
    # Cost estimation — lower is better
    # ------------------------------------------------------------------

    def estimate_cost(self, model_variant: "ModelVariant", node_caps: NodeCapabilities) -> float:
        if node_caps.ram_mb == 0:
            return float("inf")
        return _COST_PENALTY + self._estimate_ram_mb(model_variant) / node_caps.ram_mb

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
        fallback_url = self._mrm.s.auto_fallback_cpu_url

        if fallback_url:
            result = self._mrm._fallback_to_cpu_runtime(base_model, spec, fallback_url)
        else:
            result = self._mrm._ensure_cpu(base_model, spec)

        result["backend"] = self.name
        result["model_variant"] = model_variant.format
        result["ram_mb"] = self._estimate_ram_mb(model_variant)
        return result

    # ------------------------------------------------------------------
    # Stop — MRM handles container lifecycle via existing stop() API
    # ------------------------------------------------------------------

    def stop(self, instance_id: str) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_ram_mb(self, model_variant: "ModelVariant") -> int:
        return int(model_variant.size_gb * 1024 * _RAM_OVERHEAD_FACTOR)

"""
backends/llama_cpp_backend.py

llama.cpp backend — CPU/GPU inference for GGUF models.

Runs llama.cpp server via the cpu_runtime Docker container which exposes an
OpenAI-compatible endpoint. Supports n_ctx, n_threads, and n_gpu_layers.

RAM estimation: model_size_gb * 1024 * RAM_OVERHEAD_FACTOR
"""
from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

from .base import RuntimeBackend, NodeCapabilities

if TYPE_CHECKING:
    from ..model_variant_registry import ModelVariant
    from ..mrm.runtime import ModelRuntimeManager

logger = logging.getLogger("MRM.backend.llama_cpp")

# llama.cpp RAM overhead: model weights + KV cache + runtime allocations
_RAM_OVERHEAD_FACTOR = 1.3

# Penalty added to cost so llama.cpp is preferred over vllm only when GPU unavailable
_COST_PENALTY = 2.0


class LlamaCppBackend(RuntimeBackend):
    """
    llama.cpp inference backend for GGUF models.

    Supports:
        - gguf_q4 and gguf_q8 model formats
        - n_ctx  — context length (mapped to ModelSpec.max_model_len)
        - n_threads — CPU threads (mapped to ModelSpec.cpu_cores)
        - n_gpu_layers — GPU offload layers (future; currently CPU-only)

    Estimates RAM usage as size_gb * 1024 * 1.3 MiB.
    """

    name = "llama.cpp"

    def __init__(self, mrm: "ModelRuntimeManager") -> None:
        self._mrm = mrm

    # ------------------------------------------------------------------
    # Capability check
    # ------------------------------------------------------------------

    def can_run(self, model_variant: "ModelVariant", node_caps: NodeCapabilities) -> bool:
        if model_variant.format not in ("gguf_q4", "gguf_q8"):
            return False
        if node_caps.supported_backends and "llama.cpp" not in node_caps.supported_backends:
            return False
        if "llama.cpp" not in model_variant.backend_compatibility:
            return False
        required_ram = self._estimate_ram_mb(model_variant)
        return node_caps.ram_mb >= required_ram

    # ------------------------------------------------------------------
    # Cost estimation — lower is better
    # ------------------------------------------------------------------

    def estimate_cost(self, model_variant: "ModelVariant", node_caps: NodeCapabilities) -> float:
        if node_caps.ram_mb == 0:
            return float("inf")
        ram_ratio = self._estimate_ram_mb(model_variant) / node_caps.ram_mb
        return _COST_PENALTY + ram_ratio

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(
        self,
        base_model: str,
        model_variant: "ModelVariant",
        node_caps: NodeCapabilities,
    ) -> Dict[str, Any]:
        from ..mrm.config import ModelSpec

        spec = self._mrm._spec(base_model)
        memory_mb = self._estimate_ram_mb(model_variant)

        # Build a CPU-flavoured spec override without mutating the original
        cpu_spec = ModelSpec(
            **{
                **spec.model_dump(),
                "runtime": "cpu",
                "gguf_path": model_variant.gguf_path or spec.gguf_path,
                "cpu_cores": node_caps.cpu_cores or spec.cpu_cores,
                "memory_mb": memory_mb,
                "max_model_len": model_variant.n_ctx,
            }
        )

        # Temporarily swap the registry entry so _ensure_cpu picks up the right spec
        original_spec = self._mrm.registry.get(base_model)
        self._mrm.registry[base_model] = cpu_spec
        try:
            result = self._mrm._ensure_cpu(base_model, cpu_spec)
        finally:
            # Restore the original spec so other callers are unaffected
            if original_spec is not None:
                self._mrm.registry[base_model] = original_spec

        result["backend"] = self.name
        result["model_variant"] = model_variant.format
        result["ram_mb"] = memory_mb
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

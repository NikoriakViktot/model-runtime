"""
backends/base.py

Abstract base for all runtime backends. Each backend knows:
- whether it *can* run a given model variant on specific hardware
- how to *estimate cost* (lower = better fit)
- how to *launch* the model and return a result dict
- how to *stop* a running instance
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..model_variant_registry import ModelVariant


@dataclass
class NodeCapabilities:
    """Hardware snapshot used by backends to decide whether they can run a variant."""
    gpu: bool = False
    vram_mb: int = 0
    ram_mb: int = 0
    cpu_cores: int = 4
    supported_backends: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NodeCapabilities":
        return cls(
            gpu=bool(d.get("gpu", False)),
            vram_mb=int(d.get("vram_mb", 0)),
            ram_mb=int(d.get("ram_mb", 0)),
            cpu_cores=int(d.get("cpu_cores", 4)),
            supported_backends=list(d.get("supported_backends", [])),
        )


@dataclass
class Instance:
    """Metadata about a running model instance."""
    api_base: str
    backend: str
    model_variant: str
    gpu: Optional[str] = None
    ram_mb: int = 0


class RuntimeBackend(ABC):
    """
    Base interface for all inference backends.

    Backends are stateless capability checkers + launch delegators.
    Actual Docker/process management is handled by ModelRuntimeManager;
    backends call into it via dependency injection.
    """

    name: str

    @abstractmethod
    def can_run(self, model_variant: "ModelVariant", node_caps: NodeCapabilities) -> bool:
        """Return True if this backend can serve the variant on the given node."""
        ...

    @abstractmethod
    def estimate_cost(self, model_variant: "ModelVariant", node_caps: NodeCapabilities) -> float:
        """
        Return a float cost score — lower means better fit.
        Used to rank candidates when multiple backends can run a variant.
        """
        ...

    @abstractmethod
    def launch(
        self,
        base_model: str,
        model_variant: "ModelVariant",
        node_caps: NodeCapabilities,
    ) -> Dict[str, Any]:
        """
        Launch the model and return a result dict compatible with the legacy
        ensure_running() return format, extended with 'backend', 'model_variant',
        and 'ram_mb' fields.
        """
        ...

    @abstractmethod
    def stop(self, instance_id: str) -> None:
        """Stop a running instance. May be a no-op if MRM handles lifecycle."""
        ...

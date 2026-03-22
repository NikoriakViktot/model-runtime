"""
core/specs.py

Domain types shared across engines, runtime manager, and infrastructure.
This module has no internal dependencies — it is the foundation layer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EngineType(str, Enum):
    """Supported inference engine backends."""

    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"


class ModelSource(str, Enum):
    """Where model weights are sourced from."""

    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    S3 = "s3"


class RuntimeState(str, Enum):
    """Lifecycle state of a single runtime instance."""

    ABSENT = "ABSENT"
    STARTING = "STARTING"
    READY = "READY"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Input specifications
# ---------------------------------------------------------------------------


@dataclass
class AdapterConfig:
    """
    A single adapter (e.g. LoRA) to be loaded alongside a model.

    ``adapter_id``  — stable identifier used as the name in the inference API.
    ``local_path``  — absolute path on the Docker host where weights live.
    """

    adapter_id: str
    local_path: str


@dataclass
class RuntimeSpec:
    """
    Complete specification for launching one model runtime instance.

    Generic fields live here. Engine-specific tuning (e.g. ``max_num_seqs``,
    ``enforce_eager``) goes in ``extra_args`` and is consumed by the
    EngineAdapter — never by the RuntimeManager directly.
    """

    # Identity
    model_id: str               # canonical lookup key, e.g. "Qwen/Qwen1.5-1.8B-Chat"
    hf_model: str               # HuggingFace path or absolute local path
    engine: EngineType
    source: ModelSource = ModelSource.HUGGINGFACE

    # GPU resources
    gpu_count: int = 1
    gpu_memory_utilization: float = 0.90
    allowed_gpus: list[str] = field(default_factory=lambda: ["0"])

    # Model configuration
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    quantization: Optional[str] = None

    # Networking — host_port is what callers connect to; container_port is
    # what the engine binds inside the container.
    host_port: int = 8000
    container_port: int = 8000

    # Container configuration
    image: Optional[str] = None         # override adapter's default_image()
    docker_network: Optional[str] = None
    shm_size: str = "1g"                # shared memory for PyTorch (increase for large models)
    ipc_mode: Optional[str] = "host"    # "host" recommended for multi-GPU

    # Auth and caching
    hf_token: Optional[str] = None
    hf_cache_host_path: str = "/root/.cache/huggingface"

    # Engine-specific pass-through (consumed exclusively by EngineAdapter)
    extra_args: dict[str, str] = field(default_factory=dict)

    # Adapters (LoRA, etc.)
    adapters: list[AdapterConfig] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Intermediate / output types produced by adapters
# ---------------------------------------------------------------------------


@dataclass
class AdapterMount:
    """
    Maps a resolved adapter to a concrete container mount point.

    The ``host_path`` → ``container_path`` binding is engine-specific and
    produced by ``EngineAdapter.build_adapter_mount()``.
    """

    adapter_id: str
    host_path: str        # absolute path on the Docker host
    container_path: str   # absolute path inside the container
    mode: str = "ro"


@dataclass
class FallbackSpec:
    """
    One entry in an OOM recovery ladder.

    ``overrides`` contains field names from ``RuntimeSpec`` that should be
    replaced before the next start attempt. The RuntimeManager applies these
    via ``dataclasses.replace()`` — it never inspects the contents.

    Example::

        FallbackSpec(label="max_model_len=4096", overrides={"max_model_len": 4096})
    """

    label: str
    overrides: dict[str, object]


# ---------------------------------------------------------------------------
# Live runtime reference
# ---------------------------------------------------------------------------


@dataclass
class RuntimeHandle:
    """
    Reference to a live (or recently started) runtime instance.

    Returned by ``ModelRuntimeManager.ensure_runtime()`` and stored in the
    runtime state store. Callers use ``api_base`` to route inference requests.
    """

    instance_id: str
    model_id: str
    container_id: str
    container_name: str
    engine: EngineType
    api_base: str       # e.g. "http://localhost:8000/v1"
    host_port: int
    gpu_index: str      # GPU index string, e.g. "0" or "1"
    state: RuntimeState

    @classmethod
    def new(
        cls,
        *,
        model_id: str,
        container_id: str,
        container_name: str,
        engine: EngineType,
        api_base: str,
        host_port: int,
        gpu_index: str,
    ) -> RuntimeHandle:
        """Create a new handle in STARTING state with a fresh instance_id."""
        return cls(
            instance_id=str(uuid.uuid4()),
            model_id=model_id,
            container_id=container_id,
            container_name=container_name,
            engine=engine,
            api_base=api_base,
            host_port=host_port,
            gpu_index=gpu_index,
            state=RuntimeState.STARTING,
        )

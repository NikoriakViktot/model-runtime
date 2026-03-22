"""
node_agent/models.py

Request/response types for the Node Agent API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GpuInfo(BaseModel):
    gpu_index: str
    memory_total_mb: int = 0
    memory_free_mb: int = 0
    memory_used_mb: int = 0


class LocalEnsureRequest(BaseModel):
    model: str


class LocalEnsureResponse(BaseModel):
    """Mirrors MRM's /models/ensure response, forwarded verbatim."""
    model_id: str
    model_alias: str
    api_base: str       # includes /v1
    gpu: str
    state: str
    container: str = ""


class LocalStopRequest(BaseModel):
    model: str


class LocalStatusResponse(BaseModel):
    model_id: str
    state: str
    api_base: str = ""
    gpu: str = ""
    running: bool = False
    active_loras: list[str] = Field(default_factory=list)

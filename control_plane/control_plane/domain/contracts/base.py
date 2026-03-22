# control_plane/domain/contracts/base.py
from pydantic import BaseModel, Field
from typing import Literal, Optional
from uuid import UUID
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class BaseContract(BaseModel):
    contract_version: str = Field(..., example="v1")
    job_type: str
    job_id: UUID

    class Config:
        frozen = True          # immutable
        extra = "forbid"



class ContractEnvelopeV1(BaseModel):
    """
    Strict envelope for all V1 contracts.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(..., description="Globally unique contract identifier")
    spec_version: Literal["v1"] = Field("v1", description="Pinned schema version")
    idempotency_key: str | None = Field(None, description="Optional idempotency key")
    payload: dict[str, Any] = Field(..., description="Validated canonical intent")
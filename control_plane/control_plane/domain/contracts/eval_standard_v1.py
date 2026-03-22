from typing import Literal, List
from pydantic import BaseModel, ConfigDict


class ModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["base", "lora"]
    name: str
    artifact_uri: str | None = None


class DatasetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    uri: str


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metrics_uri: str


class EvalStandardPayloadV1(BaseModel):
    """
    Payload for 'eval.standard.v1' contract.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ModelRef
    dataset: DatasetRef
    metrics: List[str]
    output: OutputConfig
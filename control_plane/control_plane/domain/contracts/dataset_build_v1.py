from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class DatasetOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generate_rejected: bool = True
    rejected_max_items: int = 500

    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = "latest"


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_uri: str


class DatasetBuildPayloadV1(BaseModel):
    """
    Payload for 'dataset.build.v1' contract.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    db_id: str
    target_name: str
    dataset_type: Literal["sft", "prefs", "graph", "linear"]
    base_model: str | None = None
    options: DatasetOptions = Field(default_factory=DatasetOptions)
    output: OutputConfig
from pydantic import BaseModel, ConfigDict, Field


class DatasetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    uri: str
    sha256: str | None = None


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(3, ge=1, le=20)
    learning_rate: float = Field(5e-6, ge=1e-7, le=1e-3)
    batch_size: int = Field(1, ge=1, le=8)
    gradient_accumulation: int = Field(4, ge=1, le=64)
    warmup_ratio: float = Field(0.03, ge=0.0, le=0.3)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lora_base_uri: str


class TrainQLoRAPayloadV1(BaseModel):
    """
    Payload for 'train.qlora.v1' contract.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    parent_run_id: str
    target_slug: str
    base_model: str
    dataset: DatasetRef
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig

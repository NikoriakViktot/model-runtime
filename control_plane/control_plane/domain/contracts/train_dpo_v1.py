from pydantic import BaseModel, ConfigDict, Field


class DatasetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    uri: str


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = 1
    learning_rate: float = 0.000005
    beta: float = 0.1
    batch_size: int = 1


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lora_base_uri: str


class TrainDPOPayloadV1(BaseModel):
    """
    Payload for 'train.dpo.v1' contract.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_slug: str
    base_model: str
    prefs_dataset: DatasetRef
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig
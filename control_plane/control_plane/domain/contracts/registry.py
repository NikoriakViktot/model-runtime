# domain/contracts/registry.py
from typing import Type
from pydantic import BaseModel

from domain.contracts.dataset_build_v1 import DatasetBuildPayloadV1
from .train_qlora_v1 import TrainQLoRAPayloadV1
from domain.contracts.train_dpo_v1 import TrainDPOPayloadV1
from .eval_standard_v1 import EvalStandardPayloadV1

CONTRACT_REGISTRY: dict[str, Type[BaseModel]] = {
    "dataset.build.v1": DatasetBuildPayloadV1,
    "train.qlora.v1": TrainQLoRAPayloadV1,
    "train.dpo.v1": TrainDPOPayloadV1,
    "eval.standard.v1": EvalStandardPayloadV1,
}
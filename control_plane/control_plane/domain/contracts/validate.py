from typing import Any, Tuple, Type
from pydantic import BaseModel, ValidationError

from .base import ContractEnvelopeV1
from .registry import CONTRACT_REGISTRY


def validate_contract(payload_dict: dict[str, Any]) -> Tuple[str, str, BaseModel]:
    """
    Validates a raw dictionary against the Contract Envelope and specific Payload schema.

    Returns:
        (contract_type, spec_version, payload_model_instance)

    Raises:
        ValueError: If envelope or payload is invalid.
    """
    # 1. Validate Envelope
    try:
        envelope = ContractEnvelopeV1.model_validate(payload_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid Contract Envelope: {e}")

    # 2. Lookup Payload Schema
    payload_cls: Type[BaseModel] | None = CONTRACT_REGISTRY.get(envelope.type)
    if not payload_cls:
        raise ValueError(f"Unknown contract type: {envelope.type}")

    # 3. Validate Payload
    try:
        payload_instance = payload_cls.model_validate(envelope.payload)
    except ValidationError as e:
        raise ValueError(f"Invalid Payload for {envelope.type}: {e}")

    return envelope.type, envelope.spec_version, payload_instance
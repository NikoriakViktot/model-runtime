# mrm/routers/factory.py ===
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal

router = APIRouter(tags=["factory"])


class ProvisionReq(BaseModel):
    repo_id: str
    preset: Literal["small_chat", "7b_awq"]
    gpu: str = "0"
    overrides: Dict[str, Any] = Field(default_factory=dict)


@router.post("/factory/provision")
def provision(req: ProvisionReq, request: Request):
    mrm = request.app.state.mrm

    reg = mrm.register_from_hf(req)              # 1) register
    ensured = mrm.ensure_running(req.repo_id)    # 2) ensure
    mat = mrm.materialize_litellm_config()       # 3) write config (+ optional restart)

    return {"registered": reg, "ensured": ensured, "litellm": mat}

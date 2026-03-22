import os
import httpx
from uuid import UUID
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Імпорти з вашої структури проекту
from control_plane.infrastructure.db.session import get_db
from control_plane.infrastructure.db.models import Run
from control_plane.infrastructure.dispatcher import dispatch
from control_plane.domain.states import RunState
from control_plane.domain.events import EventType
from control_plane.domain.rules import transition, next_action

router = APIRouter()



class EventEnvelope(BaseModel):
    run_id: UUID
    event: EventType
    artifacts: Dict[str, Any] = Field(default_factory=dict)


async def emit_contract(contract: dict):
    async with httpx.AsyncClient() as client:
        await client.post(
                "http://control_plane:8004/contracts",
                json=contract,
                timeout=10
            )

@router.post("/events")
async def handle_event(
    body: EventEnvelope,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # ==========================================
    # 1. LOAD RUN (STRICT)
    # ==========================================
    run = db.query(Run).get(body.run_id)

    if not run:
        # ❌ НІЯКИХ ghost.run
        raise HTTPException(
            status_code=409,
            detail=f"Run {body.run_id} does not exist. Event rejected."
        )

    current_state = RunState(run.state)

    # ==========================================
    # 2. UPDATE ARTIFACTS (RESULTS ONLY)
    # ==========================================
    if body.artifacts:
        run.artifacts = {
            **(run.artifacts or {}),
            **body.artifacts,
        }

    # ==========================================
    # 3. STATE TRANSITION (PURE)
    # ==========================================
    new_state = transition(current_state, body.event)
    run.state = new_state
    db.commit()

    # ==========================================
    # 4. NEXT ACTION (ORCHESTRATION)
    # ==========================================
    action = next_action(new_state, body.event)

    if action == "TRAIN":
        # 🔒 захист
        if "epitaph_id" not in run.artifacts:
            raise HTTPException(409, "TRAIN requires epitaph_id in run.artifacts")

        train_contract = {
            "type": "train.qlora.v1",
            "spec_version": "v1",
            "payload": {
                "parent_run_id": str(run.id),
                "epitaph_id": run.artifacts["epitaph_id"],  # ← ОСЬ ВІН
                "target_slug": run.artifacts["target_name"],
                "base_model": "Qwen/Qwen1.5-1.8B-Chat",
                "dataset": {
                    "uri": run.artifacts["dataset_uri"],
                    "sha256": None
                },
                "training": {
                    "epochs": 3,
                    "learning_rate": 2e-5
                },
                "output": {
                    "lora_base_uri": f"s3://epitaphs-loras/{run.artifacts['epitaph_id']}"
                }
            }
        }

        async with httpx.AsyncClient() as client:
            await client.post(
                "http://control_plane:8004/contracts",
                json=train_contract,
                timeout=10
            )

    # ==========================================
    # 5. NOTIFY (DELEGATED)
    # ==========================================
    if new_state in (RunState.DONE, RunState.FAILED):
        dispatch(
            action="NOTIFY",
            run_id=run.id,
            artifacts={
                "notify": {
                    "target": "EPITAPH_NODE",
                    "epitaph_id": run.artifacts["epitaph_id"],
                    "status": "done" if new_state == RunState.DONE else "failed",
                }
            }
        )

    return {
        "ok": True,
        "run_id": str(run.id),
        "previous_state": current_state,
        "new_state": run.state,
        "action_taken": action,
    }

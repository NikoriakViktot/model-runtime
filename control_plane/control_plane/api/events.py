from uuid import UUID
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

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


@router.post("/events")
async def handle_event(
    body: EventEnvelope,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # 1. LOAD RUN
    run = db.query(Run).get(body.run_id)
    if not run:
        raise HTTPException(
            status_code=409,
            detail=f"Run {body.run_id} does not exist. Event rejected."
        )

    current_state = RunState(run.state)

    # 2. MERGE ARTIFACTS
    if body.artifacts:
        run.artifacts = {**(run.artifacts or {}), **body.artifacts}

    # 3. STATE TRANSITION
    new_state = transition(current_state, body.event)
    run.state = new_state
    db.commit()

    # 4. NEXT ACTION
    action = next_action(new_state, body.event)

    if action is not None:
        contract_payload = (run.contract or {}).get("payload", {})
        await dispatch(action, run.id, contract_payload=contract_payload)

    return {
        "ok": True,
        "run_id": str(run.id),
        "previous_state": current_state,
        "new_state": run.state,
        "action_taken": action,
    }

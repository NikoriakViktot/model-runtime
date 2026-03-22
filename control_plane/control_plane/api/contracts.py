from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Any, Dict

from control_plane.infrastructure.db.session import get_db
from control_plane.infrastructure.db.models import Run
from control_plane.infrastructure.dispatcher import dispatch

from control_plane.domain.states import RunState
from control_plane.domain.events import EventType
from control_plane.domain.rules import transition, next_action
from control_plane.domain.contracts.validate import validate_contract

router = APIRouter()


@router.post("/contracts")
def submit_contract(payload: Dict[str, Any], db: Session = Depends(get_db)):
    # 1. Validate Contract
    try:
        c_type, c_ver, c_model = validate_contract(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Create Run (Initial State)
    # The run starts at CREATED.
    run = Run(
        state=RunState.CREATED,
        contract={
            "type": c_type,
            "spec_version": c_ver,
            "payload": c_model.model_dump(mode='json')
        },
        artifacts={}
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # 3. Apply Logic based on Contract Type (✅ ВИПРАВЛЕНО)

    # Визначаємо, що робити далі, в залежності від типу контракту
    if c_type.startswith("train."):
        # Якщо це тренування — пропускаємо етап датасету
        # Одразу ставимо стан TRAIN_RUNNING (або TRAIN_QUEUED, якщо є черга)
        run.state = RunState.TRAIN_RUNNING
        action = "TRAIN"

    elif c_type.startswith("eval."):
        # Якщо це евалюація
        run.state = RunState.EVAL_RUNNING
        action = "EVAL"

    else:
        # Стандартна поведінка (для датасетів)
        # Використовуємо машину станів: CREATED + CONTRACT_ACCEPTED -> DATASET_RUNNING
        next_state = transition(RunState(run.state), EventType.CONTRACT_ACCEPTED)
        run.state = next_state
        action = next_action(RunState(run.state), EventType.CONTRACT_ACCEPTED)

    db.commit()

    # 4. Dispatch Action
    if action:
        # Відправляємо задачу (TRAIN, DATASET_BUILD або EVAL)
        dispatch(
            action=action,
            run_id=run.id,
            contract_payload=run.contract["payload"],
            artifacts=run.artifacts,
        )
    return {
        "run_id": str(run.id),
        "state": run.state,
        "dispatched_action": action
    }
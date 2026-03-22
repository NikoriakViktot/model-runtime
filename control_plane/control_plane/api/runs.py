from sqlalchemy.orm import Session
from uuid import UUID
from typing import Optional, Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Query

from control_plane.infrastructure.db.session import get_db
from control_plane.infrastructure.db.models import Run

router = APIRouter()


def _run_to_dict(r: Run) -> Dict[str, Any]:
    return {
        "id": str(r.id),
        "state": r.state,
        "contract": r.contract,
        "artifacts": r.artifacts,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }


@router.get("/runs/{run_id}")
def get_run(run_id: UUID, db: Session = Depends(get_db)):
    run = db.query(Run).get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_dict(run)


@router.get("/runs")
def list_runs(
    skip: int = 0,
    limit: int = 100,
    run_type: Optional[str] = Query(None),
    base_model: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    exclude_status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(Run)

    if status:
        q = q.filter(Run.state == status)
    if exclude_status:
        q = q.filter(Run.state != exclude_status)
    if run_type:
        q = q.filter(Run.contract["type"].astext == run_type)
    if base_model:
        q = q.filter(Run.contract["payload"]["base_model"].astext == base_model)

    runs = q.order_by(Run.created_at.desc()).offset(skip).limit(limit).all()
    return [_run_to_dict(r) for r in runs]

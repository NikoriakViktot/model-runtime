# control_plane/api/prompts.py
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

from control_plane.infrastructure.db.session import get_db
from control_plane.infrastructure.db.models import Prompt, PromptVersion


router = APIRouter(prefix="/prompts", tags=["Prompts"])


class PromptDTO(BaseModel):
    id: str
    description: Optional[str] = None
    versions_count: int = 0


class VersionDTO(BaseModel):
    id: str
    version_tag: str
    template: str
    input_variables: List[str]
    commit_message: Optional[str]
    created_at: datetime


class CreateVersionReq(BaseModel):
    prompt_id: str
    version_tag: str
    template: str
    commit_message: str = "Update via UI"


@router.get("/", response_model=List[PromptDTO])
def list_prompts(db: Session = Depends(get_db)):
    prompts = db.query(Prompt).all()
    return [
        PromptDTO(id=p.id, description=p.description, versions_count=len(p.versions))
        for p in prompts
    ]


@router.post("/")
def create_prompt(id: str, description: str = None, db: Session = Depends(get_db)):
    if db.query(Prompt).filter(Prompt.id == id).first():
        raise HTTPException(400, "Prompt ID already exists")
    p = Prompt(id=id, description=description)
    db.add(p)
    db.commit()
    return {"status": "created", "id": id}


@router.get("/{prompt_id}/versions", response_model=List[VersionDTO])
def list_versions(prompt_id: str, db: Session = Depends(get_db)):
    versions = db.query(PromptVersion).filter(PromptVersion.prompt_id == prompt_id) \
        .order_by(PromptVersion.created_at.desc()).all()
    return [
        VersionDTO(
            id=v.id,
            version_tag=v.version_tag,
            template=v.template,
            input_variables=v.input_variables or [],
            commit_message=v.commit_message,
            created_at=v.created_at
        ) for v in versions
    ]


@router.post("/version")
def create_version(req: CreateVersionReq, db: Session = Depends(get_db)):
    # Simple var extractor (regex or jinja2 meta can be used here)
    import re
    # Знаходимо всі {{ variable }}
    vars_found = list(set(re.findall(r"\{\{\s*(\w+)\s*\}\}", req.template)))

    v = PromptVersion(
        id=uuid.uuid4().hex,
        prompt_id=req.prompt_id,
        version_tag=req.version_tag,
        template=req.template,
        input_variables=vars_found,
        commit_message=req.commit_message
    )
    db.add(v)
    db.commit()
    return {"status": "created", "version": req.version_tag}


@router.get("/{prompt_id}/{version_tag}")
def get_prompt_version(prompt_id: str, version_tag: str, db: Session = Depends(get_db)):
    """
    Повертає конкретну версію промпта для ETL воркера.
    Якщо version_tag='latest', повертає найсвіжішу.
    """
    query = db.query(PromptVersion).filter(PromptVersion.prompt_id == prompt_id)

    if version_tag == "latest":
        version = query.order_by(PromptVersion.created_at.desc()).first()
    else:
        version = query.filter(PromptVersion.version_tag == version_tag).first()

    if not version:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id}:{version_tag} not found")

    return {
        "template": version.template,
        "input_variables": version.input_variables
    }
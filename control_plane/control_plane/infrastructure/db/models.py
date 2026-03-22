import uuid
from sqlalchemy import Column, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from control_plane.infrastructure.db.base import Base


class Run(Base):
    __tablename__ = "runs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    state = Column(Text, nullable=False)
    contract = Column(JSONB, nullable=False)
    artifacts = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(Text, primary_key=True)
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

    versions = relationship("PromptVersion", back_populates="prompt")


class PromptVersion(Base):
    __tablename__ = "prompt_versions"
    id = Column(Text, primary_key=True)
    prompt_id = Column(Text, ForeignKey("prompts.id"))
    version_tag = Column(Text)
    template = Column(Text, nullable=False)

    prompt = relationship("Prompt", back_populates="versions")

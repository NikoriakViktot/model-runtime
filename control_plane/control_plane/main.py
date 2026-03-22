from fastapi import FastAPI
from contextlib import asynccontextmanager

from control_plane.infrastructure.db.session import engine
from control_plane.infrastructure.db.base import Base
from control_plane.api import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
app = FastAPI(
    title="AI Control Plane",
    root_path="/api",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok", "system": "control_plane"}

app.include_router(api_router.router)
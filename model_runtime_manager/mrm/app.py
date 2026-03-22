# mrm/app.py
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .config import Settings
from .runtime import (
    ModelRuntimeManager,
    RegisterReq,
    PlanFromHFReq,
    EnsureReq,
    StopReq,
    RemoveReq,
    RuntimeError409,
    idle_reaper_loop,
    LoraRegisterReq
)
from .gpu_metrics import get_gpu_metrics
from .hf_client import HFClient

from .routers.litellm import router as litellm_router  # ✅ NEW
from .routers.factory import router as factory_router

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("MRM")

settings = Settings()
mrm = ModelRuntimeManager(settings)
hf = HFClient(settings.hf_token)

app = FastAPI(title="Model Runtime Manager", version="v1")


app.state.mrm = mrm
app.state.settings = settings
app.state.hf = hf

app.include_router(litellm_router)
app.include_router(factory_router)


@app.get("/gpu/metrics")
def gpu_metrics():
    return get_gpu_metrics()

@app.get("/hf/search")
def hf_search(q: str, limit: int = 10):
    return {"items": hf.search_models(q, limit)}

@app.get("/hf/model_info")
def hf_model_info(repo_id: str):
    return hf.model_info(repo_id)

@app.post("/models/register_from_hf")
async def register_from_hf(req: PlanFromHFReq):
    return await asyncio.to_thread(mrm.register_from_hf, req)

@app.post("/models/register")
async def register(req: RegisterReq):
    return await asyncio.to_thread(mrm.register, req.spec)

@app.on_event("startup")
async def _startup():
    app.state.reaper_task = asyncio.create_task(
        idle_reaper_loop(mrm, settings.sweep_interval_sec)
    )

@app.on_event("shutdown")
async def _shutdown():
    task = getattr(app.state, "reaper_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info("Reaper task cancelled gracefully")

@app.exception_handler(RuntimeError409)
async def _handle_409(_, exc: RuntimeError409):
    return JSONResponse(status_code=409, content={"detail": str(exc)})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/models/ensure")
async def ensure(req: EnsureReq):
    try:
        logger.info(f"Ensuring model: {req.base_model}")
        return await asyncio.to_thread(mrm.ensure_running, req.base_model)
    except RuntimeError409 as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.post("/models/stop")
async def stop(req: StopReq):
    try:
        logger.info(f"Stopping model: {req.base_model}")
        return await asyncio.to_thread(mrm.stop, req.base_model)
    except RuntimeError409 as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.post("/models/remove")
async def remove(req: RemoveReq):
    try:
        logger.info(f"Removing model: {req.base_model}")
        return await asyncio.to_thread(mrm.remove, req.base_model)
    except RuntimeError409 as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.get("/models/status/{base_model:path}")
async def status(base_model: str):
    try:
        return await asyncio.to_thread(mrm.status, base_model)
    except RuntimeError409 as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.get("/models/status")
async def status_all():
    return await asyncio.to_thread(mrm.status_all)

@app.post("/models/touch")
async def touch(req: EnsureReq):
    await asyncio.to_thread(mrm.touch, req.base_model)
    return {"base_model": req.base_model, "touched": True}

@app.get("/models/status_one")
async def status_one(base_model: str):
    return await asyncio.to_thread(mrm.status, base_model)

@app.post("/models/lora/register")
def models_lora_register(req: LoraRegisterReq):
    return mrm.lora_register(req)
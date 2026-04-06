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
    LoraRegisterReq,
)
from .gpu_metrics import get_gpu_metrics
from .hf_client import HFClient

from .routers.litellm import router as litellm_router
from .routers.factory import router as factory_router
from .api.routes import router as production_router

from .services.model_enricher import enrich_list
from .services.scorer import filter_models_by_gpu, rank_models

# ── Production components ─────────────────────────────────────────────────────
from .core.state_machine import StateMachine
from .core.queue import LoadQueue
from .core.scheduler import LoadScheduler
from .core.health import HealthWatcher
from .services.loader import ModelLoader
from .services.fallback import FallbackService

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("MRM")

# ──────────────────────────────────────────────────────────────────────────────
# Singletons
# ──────────────────────────────────────────────────────────────────────────────

settings = Settings()
mrm      = ModelRuntimeManager(settings)
hf       = HFClient(settings.hf_token)

from .services.telemetry import TelemetryStore
from .services.feedback  import FeedbackLoop

# Production layer — built once, stored on app.state
state_machine = StateMachine(mrm.redis)
load_queue    = LoadQueue(maxsize=64)
telemetry     = TelemetryStore(mrm.redis)
loader        = ModelLoader(mrm, state_machine, telemetry=telemetry)
fallback      = FallbackService(mrm, state_machine, loader)
feedback      = FeedbackLoop(telemetry, mrm)
scheduler     = LoadScheduler(load_queue, loader, fallback)
health_watcher = HealthWatcher(mrm, state_machine)

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Model Runtime Manager", version="v2")

app.state.mrm           = mrm
app.state.settings      = settings
app.state.hf            = hf
app.state.hf_client     = hf   # alias used by register_auto
app.state.state_machine = state_machine
app.state.telemetry     = telemetry
app.state.feedback      = feedback
app.state.load_queue    = load_queue
app.state.scheduler     = scheduler
app.state.loader        = loader
app.state.fallback      = fallback

app.include_router(litellm_router)
app.include_router(factory_router)
app.include_router(production_router)


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    # Legacy idle reaper
    app.state.reaper_task = asyncio.create_task(
        idle_reaper_loop(mrm, settings.sweep_interval_sec)
    )
    # Production components
    scheduler.start()
    health_watcher.start()
    logger.info("MRM v2 started — scheduler + health watcher running")


@app.on_event("shutdown")
async def _shutdown():
    # Stop production components first
    await scheduler.stop()
    await health_watcher.stop()

    # Stop legacy reaper
    task = getattr(app.state, "reaper_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info("Reaper task cancelled gracefully")

    logger.info("MRM shutdown complete")


# ──────────────────────────────────────────────────────────────────────────────
# Exception handler
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(RuntimeError409)
async def _handle_409(_, exc: RuntimeError409):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "scheduler":    scheduler._running,
        "health_watcher": health_watcher._running,
        "queue_size":   load_queue.size,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Legacy routes (unchanged — backward compatible)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/gpu/metrics")
def gpu_metrics():
    return get_gpu_metrics()


@app.get("/hf/search")
def hf_search(q: str, limit: int = 10):
    return {"items": hf.search_models(q, limit)}


@app.get("/hf/model_info")
def hf_model_info(repo_id: str):
    return hf.model_info(repo_id)


@app.get("/hf/recommend")
def hf_recommend(q: str, gpu_id: str = "0", limit: int = 20, context_len: int = 2048):
    gpu_mem  = mrm._gpu_mem(gpu_id)
    free_gb  = gpu_mem.get("free_mib", 0) / 1024.0
    raw_list = hf.search_models(q, limit)
    metas    = enrich_list(raw_list)
    pairs    = list(zip(raw_list, metas))
    filtered = filter_models_by_gpu(pairs, free_gb, context_len=context_len)
    ranked   = rank_models(filtered, gpu_vram_gb=free_gb, context_len=context_len)
    return {
        "query": q, "gpu_id": gpu_id,
        "gpu_free_gb": round(free_gb, 2),
        "total_fetched": len(raw_list),
        "total_fit": len(filtered),
        "results": ranked,
    }


@app.post("/models/register_from_hf")
async def register_from_hf(req: PlanFromHFReq):
    return await asyncio.to_thread(mrm.register_from_hf, req)


@app.post("/models/register")
async def register(req: RegisterReq):
    return await asyncio.to_thread(mrm.register, req.spec)


@app.post("/models/ensure")
async def ensure(req: EnsureReq):
    try:
        logger.info("Ensuring model: %s profile=%s", req.base_model, req.profile)
        return await asyncio.to_thread(
            mrm.ensure_running,
            req.base_model,
            req.profile,
            req.node_capabilities,
        )
    except RuntimeError409 as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/models/stop")
async def stop(req: StopReq):
    try:
        logger.info("Stopping model: %s", req.base_model)
        return await asyncio.to_thread(mrm.stop, req.base_model)
    except RuntimeError409 as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/models/remove")
async def remove(req: RemoveReq):
    try:
        logger.info("Removing model: %s", req.base_model)
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

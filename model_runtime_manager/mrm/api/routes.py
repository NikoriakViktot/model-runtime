"""
mrm/api/routes.py

New production-grade API routes for the enhanced MRM.

Endpoints
---------
  POST /models/load
    Enqueue a model load request.  Returns immediately with a request_id;
    the client can poll GET /models/status/{base_model} to track progress,
    or set wait=true to block until the model is RUNNING (or times out).

  GET /models/status
    Return status of ALL registered models plus queue / GPU summary.

  GET /models/status/{base_model}
    Status of a single model (state, GPU, queue_position).

  GET /scheduler/info
    Scheduler internals: active model, queue depth.

  POST /models/register_auto
    Register a model with fully automatic config generation (no presets).
    Accepts {repo_id, gpu, search_quant}.  Runs auto_fit() internally.

  POST /models/stop_safe
    Stop a model via the Loader (state machine aware).
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..core.gpu import query_gpu_memory
from ..core.queue import LoadRequest, QueueFullError
from ..core.state_machine import ModelState
from ..services.auto_fit import AutoFitError, auto_fit
from ..services.config_generator import VllmConfig
from ..services.feedback import (
    ACTION_OK,
    ACTION_PREFER_QUANTIZED,
    ACTION_REDUCE_CONTEXT,
    ACTION_REDUCE_UTILIZATION,
)
from ..services.model_enricher import ModelMeta, enrich
from ..runtime import build_spec_from_config

logger = logging.getLogger("MRM.api")

router = APIRouter(tags=["production"])


# ──────────────────────────────────────────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────────────────────────────────────────

class TelemetryReportRequest(BaseModel):
    base_model:       str   = Field(..., description="HF repo ID / registry key")
    latency_ms:       float = Field(..., description="End-to-end latency in milliseconds")
    tokens_generated: int   = Field(0,   description="Output tokens generated")
    is_oom:           bool  = Field(False, description="Was this an OOM failure?")


class RegisterAutoRequest(BaseModel):
    repo_id:    str = Field(..., description="HuggingFace repo ID to register")
    gpu:        str = Field("0",  description="GPU index")
    search_quant: bool = Field(True, description="Allow quantized fallback search on HF")


class LoadModelRequest(BaseModel):
    base_model: str = Field(..., description="HuggingFace repo ID")
    preset:     str = Field("small_chat", description="small_chat | 7b_awq")
    gpu:        str = Field("0",          description="GPU index")
    overrides:  Dict[str, Any] = Field(default_factory=dict)
    wait:       bool = Field(False, description="Block until model is RUNNING")
    wait_timeout: float = Field(600.0, description="Max seconds to wait when wait=true")


class StopRequest(BaseModel):
    base_model: str


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — pull singletons from app.state
# ──────────────────────────────────────────────────────────────────────────────

def _sm(request: Request):
    return request.app.state.state_machine

def _queue(request: Request):
    return request.app.state.load_queue

def _scheduler(request: Request):
    return request.app.state.scheduler

def _loader(request: Request):
    return request.app.state.loader

def _mrm(request: Request):
    return request.app.state.mrm

def _hf(request: Request):
    return request.app.state.hf

def _settings(request: Request):
    return request.app.state.settings

def _telemetry(request: Request):
    return request.app.state.telemetry

def _feedback(request: Request):
    return request.app.state.feedback


# ──────────────────────────────────────────────────────────────────────────────
# POST /models/load
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/models/load")
async def load_model(body: LoadModelRequest, request: Request) -> Dict[str, Any]:
    """
    Enqueue a model load request.

    If ``wait=false`` (default) returns immediately with::

        {"queued": true, "request_id": "...", "queue_position": N}

    If ``wait=true`` blocks until the model is RUNNING or times out.
    """
    mrm       = _mrm(request)
    sm        = _sm(request)
    queue     = _queue(request)

    # ── Validate model is registered ─────────────────────────────────
    if body.base_model not in mrm.registry:
        raise HTTPException(
            status_code=404,
            detail=f"Model not registered: {body.base_model}. "
                   "Register it first via POST /models/register_from_hf",
        )

    # ── Check current state ───────────────────────────────────────────
    current = await sm.get(body.base_model)
    if current == ModelState.RUNNING:
        return {
            "queued":     False,
            "already_running": True,
            "base_model": body.base_model,
            "state":      current.value,
        }
    if current == ModelState.LOADING:
        return {
            "queued":     False,
            "already_loading": True,
            "base_model": body.base_model,
            "state":      current.value,
            "message":    "Model is already loading. Poll /models/status for progress.",
        }

    # ── Enqueue ───────────────────────────────────────────────────────
    req = LoadRequest(
        base_model=body.base_model,
        preset=body.preset,
        overrides=body.overrides,
        gpu=body.gpu,
    )
    try:
        future = await queue.enqueue(req)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc))

    queue_pos = queue.size
    logger.info(
        "[API] /models/load  model=%s  request_id=%s  wait=%s",
        body.base_model, req.request_id, body.wait,
    )

    if not body.wait:
        return {
            "queued":         True,
            "request_id":     req.request_id,
            "queue_position": queue_pos,
            "base_model":     body.base_model,
            "state":          "LOADING",
        }

    # ── Wait for result ───────────────────────────────────────────────
    try:
        result = await asyncio.wait_for(future, timeout=body.wait_timeout)
        return {
            "queued":     True,
            "request_id": req.request_id,
            "state":      "RUNNING",
            **result,
        }
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Model did not become RUNNING within {body.wait_timeout}s. "
                   "Check /models/status for current state.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# GET /models/status
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/models/status_v2")
async def status_all_v2(request: Request) -> Dict[str, Any]:
    """
    Return full status: all models + GPU snapshot + scheduler info.
    """
    mrm       = _mrm(request)
    sm        = _sm(request)
    scheduler = _scheduler(request)

    # GPU snapshot (GPU 0)
    gpu_mem = await query_gpu_memory("0")

    models = []
    for base_model, spec in mrm.registry.items():
        state = await sm.get(base_model)
        st    = mrm._get_state(base_model)
        models.append({
            "base_model":   base_model,
            "model_alias":  spec.model_alias,
            "state":        state.value,
            "container":    spec.container_name,
            "api_base":     mrm._api_base(spec),
            "gpu":          st.get("gpu", ""),
            "last_used":    st.get("last_used", ""),
            "active_loras": sorted(list(mrm.redis.smembers(f"mrm:loras:{base_model}") or [])),
        })

    return {
        "models": models,
        "gpu": {
            "gpu_id":    gpu_mem.gpu_id,
            "free_mib":  gpu_mem.free_mib,
            "used_mib":  gpu_mem.used_mib,
            "total_mib": gpu_mem.total_mib,
            "free_gb":   round(gpu_mem.free_gb, 2),
        },
        "scheduler": {
            "active_model": scheduler.active_model,
            "queue_size":   scheduler.queue_size,
        },
    }


@router.get("/models/status_v2/{base_model:path}")
async def status_one_v2(base_model: str, request: Request) -> Dict[str, Any]:
    """Single-model status with state machine state."""
    mrm  = _mrm(request)
    sm   = _sm(request)

    if base_model not in mrm.registry:
        raise HTTPException(status_code=404, detail=f"Unknown model: {base_model}")

    spec  = mrm.registry[base_model]
    state = await sm.get(base_model)
    st    = mrm._get_state(base_model)
    sched = _scheduler(request)

    gpu_mem = await query_gpu_memory(st.get("gpu", "0"))

    return {
        "base_model":   base_model,
        "model_alias":  spec.model_alias,
        "state":        state.value,
        "container":    spec.container_name,
        "api_base":     mrm._api_base(spec),
        "gpu":          st.get("gpu", ""),
        "last_used":    st.get("last_used", ""),
        "gpu_free_mib": gpu_mem.free_mib,
        "queue_size":   sched.queue_size,
        "active_model": sched.active_model,
    }


# ──────────────────────────────────────────────────────────────────────────────
# GET /scheduler/info
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/scheduler/info")
async def scheduler_info(request: Request) -> Dict[str, Any]:
    sched = _scheduler(request)
    return {
        "active_model": sched.active_model,
        "queue_size":   sched.queue_size,
        "running":      sched._running,
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /models/stop_safe
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Telemetry endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/models/telemetry/report")
async def report_telemetry(
    body: TelemetryReportRequest, request: Request
) -> Dict[str, Any]:
    """
    Record a completed inference call for *base_model*.

    Called by the API Dispatcher after each vLLM inference.
    Not required for system operation — telemetry is best-effort.
    """
    telem = _telemetry(request)
    mrm   = _mrm(request)

    if body.base_model not in mrm.registry:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.base_model}")

    tokens_sec = (
        body.tokens_generated / (body.latency_ms / 1000.0)
        if body.latency_ms > 0 else 0.0
    )

    if body.is_oom:
        await asyncio.to_thread(telem.record_oom, body.base_model)
    await asyncio.to_thread(
        telem.record_inference, body.base_model, body.latency_ms, tokens_sec
    )
    return {
        "recorded":   True,
        "base_model": body.base_model,
        "tokens_sec": round(tokens_sec, 2),
    }


@router.get("/models/telemetry")
async def get_all_telemetry(request: Request) -> Dict[str, Any]:
    """Return telemetry stats for all models that have recorded data."""
    telem  = _telemetry(request)
    models = await asyncio.to_thread(telem.all_models)
    result = {}
    for m in models:
        stats = await asyncio.to_thread(telem.get_stats, m)
        result[m] = dataclasses.asdict(stats)
    return {"models": result, "count": len(result)}


@router.get("/models/telemetry/{base_model:path}")
async def get_telemetry(base_model: str, request: Request) -> Dict[str, Any]:
    """Return telemetry stats for a single model."""
    telem = _telemetry(request)
    stats = await asyncio.to_thread(telem.get_stats, base_model)
    return dataclasses.asdict(stats)


# ──────────────────────────────────────────────────────────────────────────────
# Feedback loop endpoints
# ──────────────────────────────────────────────────────────────────────────────

def _config_and_meta_from_spec(spec) -> tuple:
    """Build a minimal VllmConfig + ModelMeta stub from a live registry ModelSpec."""
    config = VllmConfig(
        gpu_memory_utilization = spec.gpu_memory_utilization,
        max_model_len          = spec.max_model_len,
        dtype                  = getattr(spec, "dtype", "auto") or "auto",
        quantization           = getattr(spec, "quantization", None),
    )
    meta = ModelMeta(repo_id=spec.base_model)
    return config, meta


@router.post("/models/feedback/evaluate/{base_model:path}")
async def feedback_evaluate(base_model: str, request: Request) -> Dict[str, Any]:
    """
    Evaluate telemetry for *base_model* and return an advisory.

    Does NOT modify any registry state.
    """
    fb  = _feedback(request)
    mrm = _mrm(request)

    spec = mrm.registry.get(base_model)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {base_model}")

    gpu_id  = (spec.allowed_gpus[0] if getattr(spec, "allowed_gpus", None) else "0")
    gpu_mem = await query_gpu_memory(gpu_id)
    config, meta = _config_and_meta_from_spec(spec)

    advice = await asyncio.to_thread(fb.evaluate, base_model, config, meta, gpu_mem)

    return {
        "base_model":       base_model,
        "action":           advice.action,
        "reason":           advice.reason,
        "suggested_config": (
            dataclasses.asdict(advice.suggested_config)
            if advice.suggested_config else None
        ),
    }


@router.post("/models/feedback/apply/{base_model:path}")
async def feedback_apply(base_model: str, request: Request) -> Dict[str, Any]:
    """
    Evaluate telemetry and apply the suggested config change in-memory.

    When action is PREFER_QUANTIZED, no change is applied — the response
    will indicate the advisory with applied=false.
    """
    fb  = _feedback(request)
    mrm = _mrm(request)

    spec = mrm.registry.get(base_model)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {base_model}")

    gpu_id  = (spec.allowed_gpus[0] if getattr(spec, "allowed_gpus", None) else "0")
    gpu_mem = await query_gpu_memory(gpu_id)
    config, meta = _config_and_meta_from_spec(spec)

    advice = await asyncio.to_thread(fb.apply, base_model, config, meta, gpu_mem)

    return {
        "base_model":       base_model,
        "action":           advice.action,
        "applied":          advice.action != ACTION_OK and advice.suggested_config is not None,
        "reason":           advice.reason,
        "suggested_config": (
            dataclasses.asdict(advice.suggested_config)
            if advice.suggested_config else None
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /models/register_auto
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/models/register_auto")
async def register_auto(body: RegisterAutoRequest, request: Request) -> Dict[str, Any]:
    """
    Register a model using fully automatic config generation (no presets).

    Steps
    -----
    1. Fetch model metadata from HuggingFace.
    2. Snapshot current GPU memory.
    3. Run auto_fit() — tries context ladder then quantized alternatives.
    4. Build ModelSpec from the resulting VllmConfig.
    5. Register the spec in the MRM registry.

    Returns the registered spec dict plus the auto-fit diagnostics.
    """
    mrm      = _mrm(request)
    hf       = _hf(request)
    settings = _settings(request)

    # ── 1. Fetch + enrich model metadata ────────────────────────────────
    logger.info("[register_auto] fetching model info  repo_id=%s", body.repo_id)
    try:
        raw = await asyncio.to_thread(hf.model_info, body.repo_id)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"HuggingFace API error for {body.repo_id!r}: {exc}",
        )

    if raw is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found on HuggingFace: {body.repo_id!r}",
        )

    meta = enrich(raw)
    logger.info(
        "[register_auto] enriched  repo=%s  params=%.1fB  quant=%s",
        meta.repo_id, meta.params_b, meta.quantization or "fp16",
    )

    # ── 2. Snapshot GPU memory ───────────────────────────────────────────
    try:
        gpu_mem = await query_gpu_memory(body.gpu)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot read GPU {body.gpu!r} memory: {exc}",
        )

    # ── 3. Auto-fit ──────────────────────────────────────────────────────
    hf_client  = hf if body.search_quant else None
    telem_store = request.app.state.telemetry
    try:
        final_meta, config = await asyncio.to_thread(
            auto_fit, meta, gpu_mem, hf_client, telem_store
        )
    except AutoFitError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── 4. Build ModelSpec ───────────────────────────────────────────────
    spec = build_spec_from_config(
        repo_id       = body.repo_id,
        final_repo_id = final_meta.repo_id,
        config        = config,
        gpu           = body.gpu,
        settings      = settings,
    )

    # ── 5. Register ──────────────────────────────────────────────────────
    try:
        await asyncio.to_thread(mrm.register, spec)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    quant_fallback = (final_meta.repo_id != body.repo_id)
    logger.info(
        "[register_auto] registered  original=%s  final=%s  "
        "util=%.3f  ctx=%d  quant=%s  quant_fallback=%s",
        body.repo_id, final_meta.repo_id,
        config.gpu_memory_utilization, config.max_model_len,
        config.quantization or "fp16", quant_fallback,
    )

    return {
        "registered":      True,
        "original_repo_id": body.repo_id,
        "final_repo_id":   final_meta.repo_id,
        "quant_fallback":  quant_fallback,
        "config": {
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len":          config.max_model_len,
            "quantization":           config.quantization,
            "dtype":                  config.dtype,
            "weights_gb":             config.weights_gb,
            "kv_cache_gb":            config.kv_cache_gb,
            "required_gb":            config.required_gb,
        },
        "gpu": {
            "gpu_id":   gpu_mem.gpu_id,
            "free_gb":  round(gpu_mem.free_gb, 2),
            "total_gb": round(gpu_mem.total_gb, 2),
        },
        "model_alias":      spec.model_alias,
        "container_name":   spec.container_name,
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /models/stop_safe
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/models/stop_safe")
async def stop_model_safe(body: StopRequest, request: Request) -> Dict[str, Any]:
    """
    Stop a model via ModelLoader (state-machine aware).
    """
    loader = _loader(request)
    mrm    = _mrm(request)

    if body.base_model not in mrm.registry:
        raise HTTPException(status_code=404, detail=f"Unknown model: {body.base_model}")

    try:
        result = await loader.stop(body.base_model)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

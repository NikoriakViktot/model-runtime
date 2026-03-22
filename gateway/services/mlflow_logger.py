"""
gateway/services/mlflow_logger.py

Minimal, non-blocking MLflow integration.

Design rules:
  - Never block the request path.  All MLflow calls run in a thread pool
    via asyncio.to_thread() and are fire-and-forget.
  - Never raise to the caller.  MLflow failures are logged and swallowed.
  - One MLflow run per inference request.

Logged per request:
  - Params:  model, streaming (bool)
  - Metrics: latency_ms, response_bytes, status_code
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# MLflow experiment ID resolved at startup; None disables logging.
_experiment_id: str | None = None


def setup(tracking_uri: str, experiment_name: str) -> None:
    """
    Configure MLflow and resolve (or create) the experiment.

    Called once from the FastAPI lifespan before the server starts
    accepting requests.  Failures are non-fatal.
    """
    global _experiment_id
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            _experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(
                "MLflow: created experiment '%s' (id=%s)",
                experiment_name,
                _experiment_id,
            )
        else:
            _experiment_id = exp.experiment_id
            logger.info(
                "MLflow: using experiment '%s' (id=%s)",
                experiment_name,
                _experiment_id,
            )
    except Exception as exc:
        logger.warning("MLflow setup failed (logging disabled): %s", exc)
        _experiment_id = None


async def log_inference(
    *,
    model: str,
    latency_ms: float,
    response_bytes: int,
    status_code: int,
    streaming: bool,
) -> None:
    """
    Fire-and-forget: log one inference request to MLflow.

    This coroutine schedules a background task and returns immediately.
    The caller does not await the logging result.
    """
    if _experiment_id is None:
        return

    asyncio.create_task(
        _log_in_thread(
            model=model,
            latency_ms=latency_ms,
            response_bytes=response_bytes,
            status_code=status_code,
            streaming=streaming,
        )
    )


async def _log_in_thread(
    *,
    model: str,
    latency_ms: float,
    response_bytes: int,
    status_code: int,
    streaming: bool,
) -> None:
    """Run the synchronous MLflow call in the thread pool."""
    await asyncio.to_thread(
        _sync_log,
        model=model,
        latency_ms=latency_ms,
        response_bytes=response_bytes,
        status_code=status_code,
        streaming=streaming,
    )


def _sync_log(
    *,
    model: str,
    latency_ms: float,
    response_bytes: int,
    status_code: int,
    streaming: bool,
) -> None:
    """Synchronous MLflow logging — runs in a thread pool worker."""
    try:
        import mlflow

        with mlflow.start_run(experiment_id=_experiment_id):
            mlflow.log_params(
                {
                    "model": model,
                    "streaming": str(streaming),
                }
            )
            mlflow.log_metrics(
                {
                    "latency_ms": latency_ms,
                    "response_bytes": float(response_bytes),
                    "status_code": float(status_code),
                }
            )
    except Exception as exc:
        # Logging failures must never reach the caller.
        logger.debug("MLflow log_inference failed: %s", exc)

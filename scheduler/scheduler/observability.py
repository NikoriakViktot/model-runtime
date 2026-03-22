"""
scheduler/scheduler/observability.py

Prometheus metrics, OTel tracing, and structlog for the Scheduler service.
"""

from __future__ import annotations

import logging
import os

import structlog
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

PLACEMENTS_TOTAL = Counter(
    "scheduler_placements_total",
    "Model placement operations (new placements only, not cache hits)",
    ["model", "node", "strategy"],
)

FAILOVERS_TOTAL = Counter(
    "scheduler_failovers_total",
    "Dead-node detected during ensure — triggers re-placement",
    ["model"],
)

NODES_ALIVE = Gauge(
    "scheduler_nodes_alive",
    "Number of nodes currently alive in the registry",
)

ENSURE_LATENCY = Histogram(
    "scheduler_ensure_latency_seconds",
    "Duration of scheduler.ensure() call",
    ["model", "path"],            # path: cache_hit | new_placement
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0, 120.0],
)

HEARTBEATS_TOTAL = Counter(
    "scheduler_heartbeats_total",
    "Node heartbeats received",
    ["node_id"],
)

metrics_app = make_asgi_app()


# ---------------------------------------------------------------------------
# OpenTelemetry tracing
# ---------------------------------------------------------------------------


def setup_tracing(service_name: str, otlp_endpoint: str) -> None:
    if not otlp_endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {"service.name": service_name, "service.version": "1.0.0"}
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        HTTPXClientInstrumentor().instrument()

        logging.getLogger(__name__).info(
            "OTel tracing enabled  service=%s  endpoint=%s",
            service_name,
            otlp_endpoint,
        )
    except ImportError as exc:
        logging.getLogger(__name__).warning(
            "OTel packages not installed (%s) — tracing disabled", exc
        )


def instrument_fastapi(app) -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        pass


def get_tracer(name: str):
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NoopTracer()


class _NoopTracer:
    def start_as_current_span(self, *args, **kwargs):
        from contextlib import nullcontext

        return nullcontext()


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


def setup_logging(service_name: str) -> None:
    json_mode = os.environ.get("LOG_FORMAT", "").lower() == "json"

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_mode:
        processors = shared_processors + [
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )

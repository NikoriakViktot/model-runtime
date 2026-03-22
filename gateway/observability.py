"""
gateway/observability.py

Three-pillar observability for the Gateway service.

  Metrics  — Prometheus Counters / Histograms / Gauges
  Tracing  — OpenTelemetry SDK → OTLP/gRPC export (Jaeger)
  Logging  — structlog JSON (production) or console (dev)

All three are optional dependencies: if a package is missing the function
degrades gracefully so the gateway still starts without observability
packages installed.

Usage
-----
Call once at startup (before ``instrument_fastapi``):

    from gateway.observability import setup_tracing, setup_logging, instrument_fastapi

    setup_logging("gateway")
    setup_tracing("gateway", settings.otel_endpoint)
    instrument_fastapi(app)
"""

from __future__ import annotations

import logging
import os
import uuid

import structlog
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_TOTAL = Counter(
    "gateway_requests_total",
    "Total chat completion requests",
    ["model", "status"],          # status: 200 | 502 | 503 | …
)

REQUEST_LATENCY = Histogram(
    "gateway_request_latency_seconds",
    "End-to-end request latency in seconds",
    ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

IN_FLIGHT = Gauge(
    "gateway_in_flight_requests",
    "Number of chat requests currently being processed",
)

ERRORS_TOTAL = Counter(
    "gateway_errors_total",
    "Total request errors by type",
    ["model", "error_type"],      # error_type: upstream | ensure | timeout | …
)

ROUTING_DECISIONS_TOTAL = Counter(
    "gateway_routing_decisions_total",
    "Instance selections made by the router",
    ["instance", "strategy"],
)

# ASGI sub-app that serves Prometheus text format on /metrics
metrics_app = make_asgi_app()


# ---------------------------------------------------------------------------
# OpenTelemetry tracing
# ---------------------------------------------------------------------------


def setup_tracing(service_name: str, otlp_endpoint: str) -> None:
    """
    Initialise the OTel SDK and register a BatchSpanProcessor that exports to
    Jaeger (or any OTLP/gRPC receiver) at *otlp_endpoint*.

    Also instruments all httpx clients so W3C TraceContext headers are
    propagated automatically to Scheduler / MRM / vLLM.

    No-op if ``otlp_endpoint`` is empty or OTel packages are not installed.
    """
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

        # Propagate trace context in all outbound httpx calls automatically
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
    """
    Apply FastAPI auto-instrumentation (creates a server span for every
    HTTP request).  Must be called *after* ``setup_tracing()``.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        pass


def get_tracer(name: str):
    """Return the OTel Tracer for *name*, or a no-op tracer if OTel is absent."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NoopTracer()


class _NoopTracer:
    """Fallback tracer that produces no-op context managers."""

    def start_as_current_span(self, *args, **kwargs):
        from contextlib import nullcontext

        return nullcontext()


# ---------------------------------------------------------------------------
# Structured logging (structlog)
# ---------------------------------------------------------------------------


def setup_logging(service_name: str) -> None:
    """
    Configure structlog.

    JSON output when ``LOG_FORMAT=json`` (use in Docker / production).
    Pretty coloured console output otherwise (local development).

    Async-safe: uses contextvars so per-request fields (request_id, model,
    node_id) are automatically merged into every log event without explicit
    passing.
    """
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

    # Redirect stdlib loggers to structlog output as well
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )


def new_request_id() -> str:
    """Generate a short random request ID (8 hex chars)."""
    return uuid.uuid4().hex[:12]

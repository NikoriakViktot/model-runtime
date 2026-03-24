"""
cpu_runtime/observability.py

Prometheus metrics for the CPU inference service.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

CPU_INFERENCE_TOTAL = Counter(
    "cpu_inference_requests_total",
    "Total inference requests handled by the CPU runtime",
    ["model", "status"],
)

CPU_INFERENCE_LATENCY = Histogram(
    "cpu_inference_latency_seconds",
    "End-to-end inference latency (CPU runtime)",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
)

CPU_QUEUE_DEPTH = Gauge(
    "cpu_inference_queue_depth",
    "Number of requests currently queued or running in the CPU runtime",
)

CPU_ERRORS_TOTAL = Counter(
    "cpu_inference_errors_total",
    "Total inference errors in the CPU runtime",
    ["error_type"],
)

# ASGI sub-app for /metrics endpoint
metrics_app = make_asgi_app()

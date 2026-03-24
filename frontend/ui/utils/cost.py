"""
ui/utils/cost.py
Simple cost estimation for GPU vs CPU inference.

Rates are configurable via environment variables and intentionally
conservative estimates for comparison purposes — not billing.
"""
from __future__ import annotations
import os

# $/hour for a GPU A10G node (on-demand spot estimate)
_GPU_COST_PER_HOUR = float(os.getenv("COST_GPU_PER_HOUR", "0.50"))
# $/hour for a CPU-only node (8 cores, 16 GB RAM)
_CPU_COST_PER_HOUR = float(os.getenv("COST_CPU_PER_HOUR", "0.05"))

# Approximate GPU throughput: tokens/second
_GPU_TOKENS_PER_SEC = float(os.getenv("COST_GPU_TOK_PER_SEC", "400"))
# Approximate CPU throughput for a 7B GGUF Q4_K_M model
_CPU_TOKENS_PER_SEC = float(os.getenv("COST_CPU_TOK_PER_SEC", "12"))


def estimate_gpu(latency_ms: float, output_tokens: int) -> dict:
    secs = latency_ms / 1000
    cost = (_GPU_COST_PER_HOUR / 3600) * secs
    tok_per_sec = output_tokens / secs if secs > 0 else 0
    return {
        "runtime": "gpu",
        "latency_ms": latency_ms,
        "output_tokens": output_tokens,
        "tokens_per_sec": round(tok_per_sec, 1),
        "cost_usd": round(cost, 6),
        "cost_per_1k_tokens": round(cost / max(output_tokens, 1) * 1000, 4),
    }


def estimate_cpu(latency_ms: float, output_tokens: int) -> dict:
    secs = latency_ms / 1000
    cost = (_CPU_COST_PER_HOUR / 3600) * secs
    tok_per_sec = output_tokens / secs if secs > 0 else 0
    return {
        "runtime": "cpu",
        "latency_ms": latency_ms,
        "output_tokens": output_tokens,
        "tokens_per_sec": round(tok_per_sec, 1),
        "cost_usd": round(cost, 6),
        "cost_per_1k_tokens": round(cost / max(output_tokens, 1) * 1000, 4),
    }


def savings_pct(gpu_cost: float, cpu_cost: float) -> float:
    if gpu_cost <= 0:
        return 0.0
    return round((1 - cpu_cost / gpu_cost) * 100, 1)

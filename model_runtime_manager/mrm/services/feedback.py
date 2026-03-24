"""
mrm/services/feedback.py

Performance feedback loop for the self-optimizing LLM runtime.

Reads telemetry data and recommends (or applies) config adaptations:

  Rule 1 — REDUCE_CONTEXT
    Trigger:  p95 latency > 8 000 ms
    Action:   halve max_model_len (floor 256), regenerate config

  Rule 2 — REDUCE_UTILIZATION
    Trigger:  oom_count > 0 AND gpu_memory_utilization > 0.88
    Action:   multiply utilization by 0.85, floor at 0.30

  Rule 3 — PREFER_QUANTIZED
    Trigger:  reputation_score < 0.3
    Action:   advisory only — no suggested_config returned;
              caller should re-register via POST /models/register_auto
              (switching quantization requires fetching a different HF model)

FeedbackLoop.evaluate() — returns FeedbackAdvice without side effects
FeedbackLoop.apply()    — evaluates, then patches the live registry entry
                          (in-memory only; does not persist across restarts)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..core.gpu import GpuMemory
from .config_generator import VllmConfig, generate_config, CONTEXT_LADDER
from .model_enricher import ModelMeta
from .telemetry import TelemetryStore

logger = logging.getLogger("MRM.feedback")

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────────────

_SLOW_P95_MS        = 8_000    # ms — trigger context reduction
_OOM_UTIL_THRESHOLD = 0.88     # utilization above which OOMs prompt reduction
_REP_THRESHOLD      = 0.30     # reputation below which we recommend quantization
_UTIL_REDUCTION     = 0.85     # multiply utilization by this factor on reduction
_MIN_UTIL           = 0.30     # floor for gpu_memory_utilization
_MIN_CONTEXT        = 256      # floor for max_model_len


# ──────────────────────────────────────────────────────────────────────────────
# Action constants (exported for use in routes)
# ──────────────────────────────────────────────────────────────────────────────

ACTION_OK                 = "OK"
ACTION_REDUCE_CONTEXT     = "REDUCE_CONTEXT"
ACTION_REDUCE_UTILIZATION = "REDUCE_UTILIZATION"
ACTION_PREFER_QUANTIZED   = "PREFER_QUANTIZED"


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FeedbackAdvice:
    """
    Feedback recommendation returned by FeedbackLoop.

    Fields
    ------
    action:           One of ACTION_* constants.
    suggested_config: New VllmConfig when action is REDUCE_CONTEXT or
                      REDUCE_UTILIZATION; None otherwise.
    reason:           Human-readable explanation for logging / API response.
    """
    action:           str
    suggested_config: Optional[VllmConfig]
    reason:           str


# ──────────────────────────────────────────────────────────────────────────────
# FeedbackLoop
# ──────────────────────────────────────────────────────────────────────────────

class FeedbackLoop:
    """
    Evaluates telemetry and recommends adaptive config changes.

    The loop is stateless between calls — all state lives in TelemetryStore
    (Redis) and mrm.registry (in-memory).
    """

    def __init__(self, telemetry_store: TelemetryStore, mrm) -> None:
        self._telem = telemetry_store
        self._mrm   = mrm

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        base_model:     str,
        current_config: VllmConfig,
        meta:           ModelMeta,
        gpu:            GpuMemory,
    ) -> FeedbackAdvice:
        """
        Evaluate telemetry for *base_model* and return an advisory.

        Does NOT modify any registry state.

        Args:
            base_model:     HF repo ID / registry key.
            current_config: The VllmConfig currently in use for this model.
            meta:           ModelMeta for config regeneration (may be a stub).
            gpu:            Current GPU memory snapshot.

        Returns:
            FeedbackAdvice with action, optional suggested_config, and reason.
        """
        stats = self._telem.get_stats(base_model)

        logger.debug(
            "[FEEDBACK] evaluate  model=%s  p95=%.0fms  oom=%d  "
            "util=%.3f  rep=%.3f",
            base_model, stats.p95_latency_ms, stats.oom_count,
            current_config.gpu_memory_utilization, stats.reputation_score,
        )

        # Rule 1: high p95 latency → reduce context
        if stats.p95_latency_ms > _SLOW_P95_MS and stats.total_requests > 0:
            new_ctx = max(_MIN_CONTEXT, current_config.max_model_len // 2)
            if new_ctx < current_config.max_model_len:
                suggested = generate_config(meta, gpu, max_model_len=new_ctx)
                return FeedbackAdvice(
                    action           = ACTION_REDUCE_CONTEXT,
                    suggested_config = suggested,
                    reason           = (
                        f"p95_latency={stats.p95_latency_ms:.0f}ms exceeds "
                        f"{_SLOW_P95_MS}ms threshold. "
                        f"Reducing context: {current_config.max_model_len} → {new_ctx} tokens."
                    ),
                )

        # Rule 2: OOM event with high utilization → reduce utilization
        if (stats.oom_count > 0
                and current_config.gpu_memory_utilization > _OOM_UTIL_THRESHOLD):
            new_util = round(
                max(_MIN_UTIL, current_config.gpu_memory_utilization * _UTIL_REDUCTION),
                3,
            )
            total_gb  = gpu.total_mib / 1024.0
            suggested = VllmConfig(
                gpu_memory_utilization = new_util,
                max_model_len          = current_config.max_model_len,
                dtype                  = current_config.dtype,
                quantization           = current_config.quantization,
                weights_gb             = current_config.weights_gb,
                kv_cache_gb            = current_config.kv_cache_gb,
                required_gb            = current_config.required_gb,
                allowed_gb             = round(total_gb * new_util, 3),
            )
            return FeedbackAdvice(
                action           = ACTION_REDUCE_UTILIZATION,
                suggested_config = suggested,
                reason           = (
                    f"oom_count={stats.oom_count} with "
                    f"gpu_memory_utilization={current_config.gpu_memory_utilization:.3f} "
                    f"> {_OOM_UTIL_THRESHOLD} threshold. "
                    f"Reducing utilization: "
                    f"{current_config.gpu_memory_utilization:.3f} → {new_util:.3f}."
                ),
            )

        # Rule 3: low reputation → advisory to switch to quantized model
        if stats.reputation_score < _REP_THRESHOLD and stats.load_attempts > 0:
            return FeedbackAdvice(
                action           = ACTION_PREFER_QUANTIZED,
                suggested_config = None,
                reason           = (
                    f"reputation_score={stats.reputation_score:.3f} < "
                    f"{_REP_THRESHOLD} threshold "
                    f"(success_rate={stats.success_rate:.2f}, "
                    f"oom_count={stats.oom_count}, "
                    f"load_failures={stats.load_failures}). "
                    "Consider re-registering with POST /models/register_auto "
                    "(search_quant=true) to use a quantized alternative."
                ),
            )

        return FeedbackAdvice(
            action           = ACTION_OK,
            suggested_config = None,
            reason           = (
                f"All metrics within thresholds. "
                f"p95={stats.p95_latency_ms:.0f}ms  "
                f"oom={stats.oom_count}  "
                f"reputation={stats.reputation_score:.3f}"
            ),
        )

    def apply(
        self,
        base_model:     str,
        current_config: VllmConfig,
        meta:           ModelMeta,
        gpu:            GpuMemory,
    ) -> FeedbackAdvice:
        """
        Evaluate and patch the live registry entry if a config change is advised.

        Mutations are in-memory only.  The spec will be re-loaded from
        settings on the next restart.

        Args:
            Same as evaluate().

        Returns:
            FeedbackAdvice (same as evaluate).
        """
        advice = self.evaluate(base_model, current_config, meta, gpu)

        if advice.action == ACTION_OK or advice.suggested_config is None:
            return advice   # nothing to apply (OK or PREFER_QUANTIZED)

        spec = self._mrm.registry.get(base_model)
        if spec is None:
            logger.warning(
                "[FEEDBACK] apply: model %s not in registry — skipping patch",
                base_model,
            )
            return advice

        sc      = advice.suggested_config
        updates = {
            "gpu_memory_utilization": sc.gpu_memory_utilization,
            "max_model_len":          sc.max_model_len,
        }

        # Support both Pydantic v1 (.copy) and v2 (.model_copy)
        try:
            new_spec = spec.model_copy(update=updates)   # Pydantic v2
        except AttributeError:
            new_spec = spec.copy(update=updates)          # Pydantic v1

        self._mrm.registry[base_model] = new_spec

        logger.info(
            "[FEEDBACK] applied  model=%s  action=%s  "
            "util: %.3f → %.3f  ctx: %d → %d",
            base_model, advice.action,
            current_config.gpu_memory_utilization,
            sc.gpu_memory_utilization,
            current_config.max_model_len,
            sc.max_model_len,
        )
        return advice

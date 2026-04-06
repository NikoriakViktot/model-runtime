"""
vram_estimator.py

Estimates GPU VRAM requirements for a vLLM model configuration and selects
the largest context/batch config that safely fits in available memory.

Design:
  - estimate_vram() is a pure function — no I/O, fully testable
  - auto_tune_config() picks the best candidate from a fixed ladder
  - Returns None when model metadata is missing (caller falls back to spec defaults)
  - Safety margin: configs are accepted only when total_gb < available * 0.9

Estimation heuristic:
  weights_gb   = model_size_b * dtype_bytes / 1e9
  kv_cache_gb  = 2e-6 * max_model_len * num_seqs * (model_size_b / 1e9)
  total_gb     = weights_gb + kv_cache_gb

The KV-cache formula is a simplified approximation:
  KV per token ≈ 2 * num_layers * hidden_size bytes
  We fold num_layers and hidden_size into the 2e-6 * model_size_b heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .model_variant_registry import ModelVariant
    from .backends.base import NodeCapabilities

# ---------------------------------------------------------------------------
# dtype → bytes per element
# ---------------------------------------------------------------------------

DTYPE_BYTES: Dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8":  1.0,
    "int8": 1.0,
    "int4": 0.5,
}

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class VRAMEstimate:
    model_vram_gb: float
    kv_cache_gb: float
    total_gb: float


@dataclass
class AutoTuneResult:
    """Result of auto_tune_config — carries selected config + estimate for logging."""
    max_model_len: int
    num_seqs: int
    estimate: VRAMEstimate


# ---------------------------------------------------------------------------
# Tuning candidates — tried in order, first one that fits is selected
# ---------------------------------------------------------------------------

_TUNING_CANDIDATES: List[Dict[str, int]] = [
    {"max_model_len": 8192, "num_seqs": 4},
    {"max_model_len": 4096, "num_seqs": 2},
    {"max_model_len": 2048, "num_seqs": 2},
    {"max_model_len": 1024, "num_seqs": 1},
    {"max_model_len": 512,  "num_seqs": 1},
]

# Use at most this fraction of available VRAM
VRAM_SAFETY_MARGIN = 0.9

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def estimate_vram(
    model_size_b: int,
    dtype_bytes: float,
    max_model_len: int,
    num_seqs: int,
) -> VRAMEstimate:
    """
    Estimate VRAM needed to run a model with the given config.

    Args:
        model_size_b:  Total parameter count (e.g. 7_000_000_000 for 7B).
        dtype_bytes:   Bytes per weight element (use DTYPE_BYTES lookup).
        max_model_len: Maximum sequence length.
        num_seqs:      Maximum concurrent sequences.

    Returns:
        VRAMEstimate with per-component and total GB values.
    """
    weights_gb = model_size_b * dtype_bytes / 1e9
    kv_cache_gb = 0.000002 * max_model_len * num_seqs * model_size_b / 1e9
    total_gb = weights_gb + kv_cache_gb
    return VRAMEstimate(
        model_vram_gb=round(weights_gb, 3),
        kv_cache_gb=round(kv_cache_gb, 3),
        total_gb=round(total_gb, 3),
    )


def get_available_vram_gb(node_caps: "NodeCapabilities") -> float:
    """Convert node vram_mb to GB."""
    return node_caps.vram_mb / 1024.0


def auto_tune_config(
    model_variant: "ModelVariant",
    node_caps: "NodeCapabilities",
) -> Optional[AutoTuneResult]:
    """
    Select the largest (max_model_len, num_seqs) that fits in available VRAM.

    Returns:
        AutoTuneResult if model metadata is present and a fitting config exists.
        None if model_size_b == 0 (caller should use spec defaults unchanged).

    The minimal fallback config {"max_model_len": 256, "num_seqs": 1} is always
    returned as a last resort so the system never crashes due to this function.
    """
    model_size_b: int = getattr(model_variant, "model_size_b", 0)
    if not model_size_b:
        return None

    dtype_str: str = getattr(model_variant, "dtype", "fp16")
    dtype_bytes: float = DTYPE_BYTES.get(dtype_str, 2.0)
    vram_available = get_available_vram_gb(node_caps)
    threshold = vram_available * VRAM_SAFETY_MARGIN

    for candidate in _TUNING_CANDIDATES:
        est = estimate_vram(
            model_size_b=model_size_b,
            dtype_bytes=dtype_bytes,
            max_model_len=candidate["max_model_len"],
            num_seqs=candidate["num_seqs"],
        )
        if est.total_gb < threshold:
            return AutoTuneResult(
                max_model_len=candidate["max_model_len"],
                num_seqs=candidate["num_seqs"],
                estimate=est,
            )

    # Minimal fallback — always try 256 tokens / 1 seq
    fallback_est = estimate_vram(
        model_size_b=model_size_b,
        dtype_bytes=dtype_bytes,
        max_model_len=256,
        num_seqs=1,
    )
    return AutoTuneResult(max_model_len=256, num_seqs=1, estimate=fallback_est)

"""
tests/unit/test_runtime_explain.py

Unit tests for the pure-logic helpers in ui/components/runtime_explain.py.

These tests cover the VRAM threshold logic without importing Streamlit,
so they run in any Python environment without a display.
"""
import sys
import types
import pytest

# ---------------------------------------------------------------------------
# Stub out Streamlit so the import succeeds in headless CI
# ---------------------------------------------------------------------------

_st_stub = types.ModuleType("streamlit")
for _attr in ("markdown", "caption", "progress", "info", "expander", "json"):
    setattr(_st_stub, _attr, lambda *a, **kw: None)
sys.modules.setdefault("streamlit", _st_stub)

# Now we can import the component
from ui.components.runtime_explain import (  # noqa: E402
    vram_fit_status,
    runtime_summary_line,
)


# ---------------------------------------------------------------------------
# vram_fit_status — threshold logic
# ---------------------------------------------------------------------------

class TestVramFitStatus:
    """vram_fit_status(ratio) → (color, description)"""

    def test_green_well_below_threshold(self):
        color, desc = vram_fit_status(0.50)
        assert color == "green"
        assert "comfortably" in desc

    def test_green_at_80_pct(self):
        color, desc = vram_fit_status(0.80)
        assert color == "green"

    def test_orange_just_above_80_pct(self):
        color, desc = vram_fit_status(0.81)
        assert color == "orange"
        assert "close" in desc

    def test_orange_at_95_pct(self):
        color, desc = vram_fit_status(0.95)
        assert color == "orange"

    def test_red_just_above_95_pct(self):
        color, desc = vram_fit_status(0.951)
        assert color == "red"
        assert "too large" in desc

    def test_red_over_100_pct(self):
        color, desc = vram_fit_status(1.10)
        assert color == "red"

    def test_green_zero_ratio(self):
        color, _ = vram_fit_status(0.0)
        assert color == "green"


# ---------------------------------------------------------------------------
# runtime_summary_line
# ---------------------------------------------------------------------------

class TestRuntimeSummaryLine:
    def test_returns_none_when_no_backend(self):
        assert runtime_summary_line({}) is None
        assert runtime_summary_line({"state": "READY"}) is None

    def test_backend_only(self):
        result = runtime_summary_line({"backend": "vllm"})
        assert result is not None
        assert "vllm" in result

    def test_backend_and_variant(self):
        result = runtime_summary_line({"backend": "vllm", "model_variant": "awq"})
        assert "vllm" in result
        assert "awq" in result

    def test_full_debug_info(self):
        m = {
            "backend": "vllm",
            "model_variant": "fp16",
            "debug": {
                "max_model_len": 4096,
                "estimated_vram_gb": 14.2,
                "available_vram_gb": 24.0,
            },
        }
        result = runtime_summary_line(m)
        assert "vllm" in result
        assert "fp16" in result
        assert "4,096" in result
        assert "14.2" in result
        assert "24.0" in result

    def test_partial_debug_missing_vram(self):
        m = {
            "backend": "llama.cpp",
            "model_variant": "gguf_q4",
            "debug": {"max_model_len": 2048},
        }
        result = runtime_summary_line(m)
        assert "llama.cpp" in result
        assert "2,048" in result
        # No VRAM numbers should appear
        assert "GB" not in result

    def test_empty_backend_string_returns_none(self):
        assert runtime_summary_line({"backend": "", "model_variant": "fp16"}) is None

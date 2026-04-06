"""
ui/components/runtime_explain.py

Operator-facing components for backend selection explainability and VRAM previews.

Public API:
    render_vram_preview(estimated_gb, available_gb) — progress bar with color coding
    render_runtime_reasoning(reasoning)             — human-readable selection list
    render_runtime_debug(debug_info)                — structured debug panel
    runtime_summary_line(m)                         — compact one-line text summary
"""
from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Internal threshold helper (also used by unit tests)
# ---------------------------------------------------------------------------

def vram_fit_status(ratio: float) -> tuple[str, str]:
    """
    Return (css_color, description) for a VRAM utilisation ratio.

    Thresholds:
        ≤ 80 %  → green  — fits comfortably
        ≤ 95 %  → orange — close to limit
        > 95 %  → red    — likely too large
    """
    if ratio <= 0.80:
        return "green", "fits comfortably"
    if ratio <= 0.95:
        return "orange", "close to limit"
    return "red", "likely too large"


# ---------------------------------------------------------------------------
# VRAM preview bar
# ---------------------------------------------------------------------------

def render_vram_preview(
    estimated_gb: float | None,
    available_gb: float | None,
    *,
    label: str = "Est. VRAM",
) -> None:
    """
    Show a VRAM usage indicator with a colour-coded progress bar.

    Gracefully degrades when either value is missing.
    """
    if not available_gb:
        st.caption(f"{label}: unknown (GPU metrics unavailable)")
        return

    if estimated_gb is None:
        st.caption(
            f"{label}: unknown (insufficient model metadata)"
            f"  ·  {available_gb:.1f} GB available"
        )
        return

    ratio = estimated_gb / available_gb
    color, desc = vram_fit_status(ratio)
    pct = min(ratio, 1.0)

    st.markdown(
        f"**{label}:** `{estimated_gb:.1f}` / `{available_gb:.1f}` GB "
        f"<span style='color:{color}'>({ratio * 100:.0f}% — {desc})</span>",
        unsafe_allow_html=True,
    )
    st.progress(pct)


# ---------------------------------------------------------------------------
# Reasoning list
# ---------------------------------------------------------------------------

def render_runtime_reasoning(reasoning: list[str] | None) -> None:
    """Render a human-readable bullet list of backend-selection reasoning."""
    if not reasoning:
        return
    for line in reasoning:
        st.markdown(f"- {line}")


# ---------------------------------------------------------------------------
# Full debug panel
# ---------------------------------------------------------------------------

def render_runtime_debug(debug_info: dict | None) -> None:
    """
    Render a structured debug panel for a single runtime selection result.

    Layout:
      - compact summary line (backend · variant · context · auto-tuned)
      - reason tag
      - VRAM preview bar
      - reasoning bullet list
      - collapsible raw JSON
    """
    if not debug_info:
        st.info("Runtime selection debug metadata not available.")
        return

    backend  = debug_info.get("selected_backend") or "—"
    variant  = debug_info.get("selected_variant") or "—"
    quant    = debug_info.get("quantization") or None
    max_len  = debug_info.get("max_model_len")
    est_gb   = debug_info.get("estimated_vram_gb")
    avail_gb = debug_info.get("available_vram_gb")
    auto     = debug_info.get("auto_tuned", False)
    reason   = debug_info.get("reason") or "—"

    # ── compact summary ──────────────────────────────────────────────
    parts = [f"Backend: **{backend}**", f"Variant: **{variant}**"]
    if quant:
        parts.append(f"Quant: {quant}")
    if max_len:
        parts.append(f"Context: {max_len:,}")
    if auto:
        parts.append("Auto-tuned ✓")
    st.caption("  ·  ".join(parts))
    st.caption(f"Reason: `{reason}`")

    # ── VRAM bar ─────────────────────────────────────────────────────
    render_vram_preview(est_gb, avail_gb)

    # ── reasoning bullets ────────────────────────────────────────────
    reasoning = debug_info.get("reasoning")
    if reasoning:
        st.markdown("**Selection reasoning:**")
        render_runtime_reasoning(reasoning)

    # ── raw JSON (collapsed by default) ──────────────────────────────
    with st.expander("Raw debug JSON", expanded=False):
        st.json(debug_info)


# ---------------------------------------------------------------------------
# Compact one-line summary (for model registry cards)
# ---------------------------------------------------------------------------

def runtime_summary_line(m: dict) -> str | None:
    """
    Return a one-line runtime summary string for a model status dict.
    Returns None when no backend information is available.
    """
    backend  = m.get("backend") or ""
    variant  = m.get("model_variant") or ""
    debug    = m.get("debug") or {}
    max_len  = debug.get("max_model_len")
    est_gb   = debug.get("estimated_vram_gb")
    avail_gb = debug.get("available_vram_gb")

    if not backend:
        return None

    parts = [f"Backend: {backend}"]
    if variant:
        parts.append(f"Variant: {variant}")
    if max_len:
        parts.append(f"Context: {max_len:,}")
    if est_gb is not None and avail_gb is not None:
        parts.append(f"Est. VRAM: {est_gb:.1f} / {avail_gb:.1f} GB")
    return "  ·  ".join(parts)

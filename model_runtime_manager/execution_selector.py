"""
execution_selector.py

ExecutionSelector — ranks and picks the best (backend, variant) pair for a
given model + node capabilities + optional execution profile.

Selection flow:
    1. Fetch model variants from VariantRegistry
    2. Filter by profile-preferred formats (if profile given)
    3. Filter to (variant, backend) pairs where backend.can_run() == True
    4. If nothing fits, try quantization fallback order
    5. Rank by (backend_priority, estimated_cost)
    6. Return the best match as SelectionResult

Quantization fallback order:
    fp16 → awq → gguf_q8 → gguf_q4 → fail

Profile → preferred formats:
    quality  → fp16
    balanced → awq, fp16
    cheap    → gguf_q4, gguf_q8, awq
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .backends.base import RuntimeBackend, NodeCapabilities
from .model_inspector import inspect_model
from .model_variant_registry import ModelVariant, VariantRegistry, VARIANT_FORMAT_PRIORITY

logger = logging.getLogger("MRM.execution_selector")

# Maps profile names to ordered lists of preferred variant formats
PROFILE_FORMAT_PREFERENCE: Dict[str, List[str]] = {
    "quality": ["fp16"],
    "balanced": ["awq", "fp16"],
    "cheap": ["gguf_q4", "gguf_q8", "awq"],
}

# Lower number = higher priority (preferred when multiple backends can run)
BACKEND_PRIORITY: Dict[str, int] = {
    "vllm": 1,
    "llama.cpp": 2,
    "cpu": 3,
}


class SelectionResult:
    """Outcome of a successful backend selection."""

    def __init__(
        self,
        variant: ModelVariant,
        backend: RuntimeBackend,
        reason: str,
    ) -> None:
        self.variant = variant
        self.backend = backend
        self.reason = reason


class ExecutionSelector:
    """
    Stateless selector that chooses the best (variant, backend) pair.

    Backends and registry are injected at construction time, making the
    selector fully testable without Docker or Redis.
    """

    def __init__(
        self,
        backends: List[RuntimeBackend],
        registry: VariantRegistry,
    ) -> None:
        self._backends: Dict[str, RuntimeBackend] = {b.name: b for b in backends}
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        base_model: str,
        node_caps: NodeCapabilities,
        profile: Optional[str] = None,
    ) -> Optional[SelectionResult]:
        """
        Select the best backend + variant for base_model on node_caps.

        Returns None if:
        - No variants are registered for the model, OR
        - No registered variant can run on node_caps (even after fallback)
        """
        variants = self._registry.get_variants(base_model)
        if not variants:
            return None

        candidate_variants = self._apply_profile_filter(variants, profile)

        runnable = self._find_runnable(candidate_variants, node_caps)

        if not runnable:
            # Quantization fallback: try all variants in degradation order
            runnable = self._quantization_fallback(base_model, node_caps)
            if runnable:
                reason = "vram_insufficient"
            else:
                return None
        else:
            reason = f"profile={profile or 'default'}"

        best_variant, best_backend = self._rank(runnable, node_caps)[0]

        # Validate quantization against the model's native config and auto-fix
        best_variant = self._resolve_quantization(best_variant)

        self._log_selection(base_model, best_backend.name, best_variant.format, reason)
        return SelectionResult(best_variant, best_backend, reason)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_profile_filter(
        self,
        variants: List[ModelVariant],
        profile: Optional[str],
    ) -> List[ModelVariant]:
        if not profile or profile not in PROFILE_FORMAT_PREFERENCE:
            return variants
        preferred = PROFILE_FORMAT_PREFERENCE[profile]
        filtered = [v for v in variants if v.format in preferred]
        return filtered if filtered else variants  # fall through if no match

    def _find_runnable(
        self,
        variants: List[ModelVariant],
        node_caps: NodeCapabilities,
    ) -> List[Tuple[ModelVariant, RuntimeBackend]]:
        runnable = []
        for variant in variants:
            for backend_name in variant.backend_compatibility:
                backend = self._backends.get(backend_name)
                if backend and backend.can_run(variant, node_caps):
                    runnable.append((variant, backend))
        return runnable

    def _quantization_fallback(
        self,
        base_model: str,
        node_caps: NodeCapabilities,
    ) -> List[Tuple[ModelVariant, RuntimeBackend]]:
        """
        Try all variants in quantization fallback order (fp16 → awq → gguf_q8 → gguf_q4).
        Returns the first runnable match wrapped in a list, or empty list.
        """
        all_variants = self._registry.get_variants(base_model)
        format_order = {fmt: i for i, fmt in enumerate(VARIANT_FORMAT_PRIORITY)}
        sorted_variants = sorted(
            all_variants,
            key=lambda v: format_order.get(v.format, len(VARIANT_FORMAT_PRIORITY)),
        )
        for variant in sorted_variants:
            for backend_name in variant.backend_compatibility:
                backend = self._backends.get(backend_name)
                if backend and backend.can_run(variant, node_caps):
                    return [(variant, backend)]
        return []

    def _rank(
        self,
        candidates: List[Tuple[ModelVariant, RuntimeBackend]],
        node_caps: NodeCapabilities,
    ) -> List[Tuple[ModelVariant, RuntimeBackend]]:
        def sort_key(item: Tuple[ModelVariant, RuntimeBackend]) -> Tuple[int, float]:
            variant, backend = item
            priority = BACKEND_PRIORITY.get(backend.name, 99)
            cost = backend.estimate_cost(variant, node_caps)
            return (priority, cost)

        return sorted(candidates, key=sort_key)

    def _resolve_quantization(self, variant: ModelVariant) -> ModelVariant:
        """
        Check whether the selected variant's quantization conflicts with the
        model's native quantization (as declared in its HuggingFace config).

        Resolution rules:
          - native=None              → keep selected quantization as-is
          - native=X, selected=None  → no conflict; vLLM will use native
          - native=X, selected=X     → exact match; clear selected so we don't
                                       redundantly pass --quantization (vLLM
                                       reads it from config automatically)
          - native=X, selected=Y     → CONFLICT → auto-fix: clear selected
                                       and emit a structured warning

        Returns a (possibly new) ModelVariant with the resolved quantization.
        """
        repo_id = variant.hf_repo or variant.base_model_id
        capabilities = inspect_model(repo_id)
        native_quant = capabilities.native_quantization
        selected_quant = variant.quantization

        if not native_quant:
            # Model has no native quantization — honour whatever the variant says
            return variant

        if selected_quant and selected_quant != native_quant:
            # Conflict: variant requests a quant that differs from what's baked in
            logger.warning(
                '{"event": "quantization_conflict_auto_fixed", "model": "%s",'
                ' "selected": "%s", "native": "%s"}',
                variant.base_model_id,
                selected_quant,
                native_quant,
            )
            return variant.copy_with(quantization=None)

        if selected_quant == native_quant:
            # Exact match: clear the flag — vLLM reads it from config natively;
            # passing it explicitly would be redundant but harmless on most
            # versions, yet some builds reject the duplicate flag.
            return variant.copy_with(quantization=None)

        # selected_quant is None and native_quant is set → no conflict
        return variant

    def _log_selection(
        self,
        model: str,
        backend: str,
        variant: str,
        reason: str,
    ) -> None:
        logger.info(
            '{"event": "backend_selected", "model": "%s", "backend": "%s",'
            ' "variant": "%s", "reason": "%s"}',
            model,
            backend,
            variant,
            reason,
        )

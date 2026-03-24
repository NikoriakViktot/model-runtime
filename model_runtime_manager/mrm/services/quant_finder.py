"""
mrm/services/quant_finder.py

Finds a quantized (AWQ / GPTQ) alternative for a model on HuggingFace.

Strategy
--------
1. Extract the base model name from the repo_id.
2. Run several HF searches in priority order:
     "{base_name} AWQ"           — find org's own AWQ release
     "{org}/{base_name}-AWQ"
     "TheBloke {base_name} AWQ"  — most common community quantizer
     "TheBloke {base_name} GPTQ"
     "{base_name} GPTQ"
3. Score each candidate by:
     a. Name similarity to the original repo_id     (0–1)
     b. Quantization preference  AWQ > GPTQ         (+0.2 bonus for AWQ)
     c. Download count (log-normalized)             (0–1)
4. Return the ModelMeta for the highest-scoring candidate,
   or None if no quantized match is found.

Similarity score
----------------
Uses token-overlap (Jaccard) on lowercased word-tokens extracted from
both repo IDs.  This correctly matches e.g.:
    "mistralai/Mistral-7B-Instruct-v0.2"
  → "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
  → score ≈ 0.86
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import List, Optional

from .model_enricher import ModelMeta, enrich

logger = logging.getLogger("MRM.quant_finder")

_QUANT_PREFERENCE = {"awq": 1.0, "gptq": 0.8}


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantCandidate:
    meta:             ModelMeta
    similarity:       float    # 0–1 name overlap vs original
    quant_score:      float    # 0–1 quant preference
    download_score:   float    # 0–1 log-normalised downloads
    reputation_score: float    # 0–1 from TelemetryStore (0.5 = unknown)
    total_score:      float    # weighted sum


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def find_quantized_model(
    meta:             ModelMeta,
    hf_client,                   # HFClient instance
    limit:            int = 10,
    telemetry_store=None,        # Optional[TelemetryStore]
) -> Optional[ModelMeta]:
    """
    Search HuggingFace for an AWQ or GPTQ version of *meta*.

    Args:
        meta:       Original model metadata.
        hf_client:  HFClient with search_models / model_info methods.
        limit:      Max results per search query.

    Returns:
        ModelMeta of the best quantized alternative, or None.
    """
    queries = _build_queries(meta.repo_id)
    seen: set[str] = {meta.repo_id.lower()}
    candidates: List[QuantCandidate] = []

    for q in queries:
        logger.info("[QUANT_FINDER] searching HF  query=%r", q)
        try:
            raw_list = hf_client.search_models(q, limit)
        except Exception as exc:
            logger.warning("[QUANT_FINDER] search failed  query=%r  error=%s", q, exc)
            continue

        for raw in raw_list:
            candidate_meta = enrich(raw)
            rid = candidate_meta.repo_id.lower()

            # Only consider quantized models we haven't seen
            if rid in seen or candidate_meta.quantization not in _QUANT_PREFERENCE:
                continue
            seen.add(rid)

            reputation = 0.5   # neutral prior when no telemetry
            if telemetry_store is not None:
                try:
                    stats = telemetry_store.get_stats(candidate_meta.repo_id)
                    reputation = stats.reputation_score
                except Exception:
                    pass
            cand = _score(meta.repo_id, candidate_meta, reputation_score=reputation)
            candidates.append(cand)
            logger.debug(
                "[QUANT_FINDER] candidate  repo=%s  quant=%s  score=%.3f",
                candidate_meta.repo_id, candidate_meta.quantization, cand.total_score,
            )

    if not candidates:
        logger.warning(
            "[QUANT_FINDER] no quantized alternative found for %s", meta.repo_id
        )
        return None

    best = max(candidates, key=lambda c: c.total_score)

    # Reject if similarity is too low (different model family)
    if best.similarity < 0.30:
        logger.warning(
            "[QUANT_FINDER] best candidate %s has low similarity %.2f — skipping",
            best.meta.repo_id, best.similarity,
        )
        return None

    logger.info(
        "[QUANT_FINDER] best match  original=%s  candidate=%s  "
        "quant=%s  similarity=%.2f  score=%.3f",
        meta.repo_id,
        best.meta.repo_id,
        best.meta.quantization,
        best.similarity,
        best.total_score,
    )
    return best.meta


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_queries(repo_id: str) -> List[str]:
    """Generate HF search queries in descending priority."""
    base_name = repo_id.split("/")[-1]
    org       = repo_id.split("/")[0] if "/" in repo_id else ""

    queries = []
    # Same org's own quantized release (most accurate)
    if org:
        queries += [
            f"{org} {base_name} AWQ",
            f"{org} {base_name} GPTQ",
        ]
    # TheBloke is the dominant community quantizer
    queries += [
        f"TheBloke {base_name} AWQ",
        f"TheBloke {base_name} GPTQ",
    ]
    # Generic fallback
    queries += [
        f"{base_name} AWQ",
        f"{base_name} GPTQ",
    ]
    return queries


def _tokenise(repo_id: str) -> set[str]:
    """Split repo_id into lowercase word tokens, ignoring version strings."""
    text = re.sub(r"[_\-./]", " ", repo_id.lower())
    tokens = set(text.split())
    # Strip tokens that are noise (quantization keywords, version tags)
    noise = {"awq", "gptq", "bnb", "int4", "int8", "thebloke", "v0", "v1", "v2",
             "v0.1", "v0.2", "v0.3", "instruct", "chat"}
    return tokens - noise


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _score(
    original_repo_id: str,
    candidate_meta:   ModelMeta,
    reputation_score: float = 0.5,   # neutral prior: 0.5 when no telemetry data
) -> QuantCandidate:
    orig_tokens  = _tokenise(original_repo_id)
    cand_tokens  = _tokenise(candidate_meta.repo_id)
    similarity   = _jaccard(orig_tokens, cand_tokens)

    quant_score  = _QUANT_PREFERENCE.get(candidate_meta.quantization or "", 0.0)

    # Log-normalise downloads: log10(downloads+1) / 8  (caps at 10^8 downloads)
    dl_score = min(1.0, math.log10(max(1, candidate_meta.downloads)) / 8.0)

    # Reputation bonus: up to +0.10 for a proven, well-performing model
    rep_score = max(0.0, min(1.0, reputation_score))

    # Weights: similarity=0.45, quant=0.27, downloads=0.18, reputation=0.10
    # (sum = 1.00; reputation replaces 0.05 each from similarity and downloads)
    total = (
        0.45 * similarity
        + 0.27 * quant_score
        + 0.18 * dl_score
        + 0.10 * rep_score
    )

    return QuantCandidate(
        meta             = candidate_meta,
        similarity       = round(similarity, 3),
        quant_score      = quant_score,
        download_score   = round(dl_score, 3),
        reputation_score = round(rep_score, 3),
        total_score      = round(total, 4),
    )

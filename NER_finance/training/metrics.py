# -*- coding: utf-8 -*-
"""Post-training evaluation metrics for NER.

During training only the cross-entropy loss is tracked (memory-efficient).
This module provides F1 computation utilities intended for a *separate*
evaluation run after training (see ``evaluate.py``).

The functions here are called with *decoded text* (not raw logits) so
they work regardless of model vocabulary size.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Set, Tuple

from utils.span_utils import normalize_term

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def safe_parse_json(text: str) -> dict:
    """Attempt to parse *text* as JSON; return an empty dict on failure."""
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# NER metrics
# ---------------------------------------------------------------------------


def extract_ner_gold(parsed: dict) -> Set[Tuple[str, str]]:
    """Extract (label, normalised_term) pairs from a gold or predicted dict."""
    result: Set[Tuple[str, str]] = set()
    for ent in parsed.get("entities", []):
        label = ent.get("label", "")
        term = normalize_term(ent.get("term", ""))
        if label and term:
            result.add((label, term))
    return result


def compute_ner_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute entity-level precision, recall, and F1.

    Matching is done on ``(label, normalised_term)`` pairs (soft span match).

    Args:
        predictions: List of model-generated JSON strings.
        references:  List of gold-standard JSON strings.

    Returns:
        Dict with keys ``ner_precision``, ``ner_recall``, ``ner_f1``.
    """
    tp = fp = fn = 0
    for pred_str, ref_str in zip(predictions, references):
        pred_set = extract_ner_gold(safe_parse_json(pred_str))
        ref_set = extract_ner_gold(safe_parse_json(ref_str))
        tp += len(pred_set & ref_set)
        fp += len(pred_set - ref_set)
        fn += len(ref_set - pred_set)
    return _compute_prf(tp, fp, fn, prefix="ner")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _compute_prf(tp: int, fp: int, fn: int, prefix: str) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        f"{prefix}_precision": round(precision, 4),
        f"{prefix}_recall": round(recall, 4),
        f"{prefix}_f1": round(f1, 4),
    }

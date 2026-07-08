# -*- coding: utf-8 -*-
"""BIO label definitions and conversion utilities for FIRE NER dataset.

The 13 entity types from the FIRE financial NER dataset are mapped to
BIO tags, resulting in 27 labels total: ``O`` + 13 × (``B-`` + ``I-``).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Entity types (sorted alphabetically for reproducibility)
# ---------------------------------------------------------------------------

ENTITY_TYPES: List[str] = [
    "Action",
    "BusinessUnit",
    "Company",
    "Date",
    "Designation",
    "FinancialEntity",
    "GeopoliticalEntity",
    "Location",
    "Money",
    "Person",
    "Product",
    "Quantity",
    "Sector",
]


# ---------------------------------------------------------------------------
# BIO label list
# ---------------------------------------------------------------------------


def build_label_list() -> List[str]:
    """Build the full BIO label list: ``['O', 'B-Action', 'I-Action', ...]``.

    Returns:
        Ordered list of 27 string labels.
    """
    labels = ["O"]
    for ent in ENTITY_TYPES:
        labels.append(f"B-{ent}")
        labels.append(f"I-{ent}")
    return labels


# Pre-built for convenience
_LABEL_LIST: List[str] = build_label_list()


def label2id() -> Dict[str, int]:
    """Return mapping ``{label_str: int_id}``."""
    return {lbl: idx for idx, lbl in enumerate(_LABEL_LIST)}


def id2label() -> Dict[int, str]:
    """Return mapping ``{int_id: label_str}``."""
    return {idx: lbl for idx, lbl in enumerate(_LABEL_LIST)}


# ---------------------------------------------------------------------------
# Span → BIO conversion
# ---------------------------------------------------------------------------


def entities_to_bio_tags(
    tokens: List[str],
    entities: List[dict],
) -> List[str]:
    """Convert FIRE-format entity spans to a BIO tag sequence.

    Args:
        tokens: Whitespace-split word list (length *N*).
        entities: List of dicts with keys ``type``, ``start``, ``end``
            where ``start`` is inclusive and ``end`` is **exclusive**
            (FIRE convention).

    Returns:
        List of *N* BIO tag strings, e.g. ``['O', 'B-Company', 'I-Company', 'O', ...]``.

    Example::

        >>> entities_to_bio_tags(
        ...     ["Albertsons", "and", "Rite", "Aid"],
        ...     [{"type": "Company", "start": 0, "end": 1},
        ...      {"type": "Company", "start": 2, "end": 4}],
        ... )
        ['B-Company', 'O', 'B-Company', 'I-Company']
    """
    tags = ["O"] * len(tokens)

    for ent in entities:
        ent_type = ent.get("type", "UNKNOWN")
        start: int = ent.get("start", -1)
        end: int = ent.get("end", -1)  # exclusive

        if start < 0 or end <= start or start >= len(tokens):
            continue

        # Clamp end to token length
        end = min(end, len(tokens))

        tags[start] = f"B-{ent_type}"
        for i in range(start + 1, end):
            tags[i] = f"I-{ent_type}"

    return tags

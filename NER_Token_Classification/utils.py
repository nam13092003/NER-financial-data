# -*- coding: utf-8 -*-
"""Utility helpers for NER Token Classification with BIO tagging.

Provides:
- Entity type definitions from the FIRE dataset
- BIO label list construction
- label↔id mappings
- FIRE entity span → BIO tag sequence conversion
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ── 13 entity types from the FIRE financial NER dataset ──────────────
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

# ── BIO label list: O + B-<type> + I-<type> for each entity type ────
BIO_LABELS: List[str] = ["O"]
for _etype in ENTITY_TYPES:
    BIO_LABELS.append(f"B-{_etype}")
    BIO_LABELS.append(f"I-{_etype}")

NUM_LABELS: int = len(BIO_LABELS)  # 27

LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(BIO_LABELS)}


def build_bio_tags(
    tokens: List[str],
    entities: List[dict],
) -> List[str]:
    """Convert FIRE entity spans into a BIO tag sequence.

    Args:
        tokens: List of whitespace-split tokens from the FIRE record.
        entities: List of entity dicts, each with keys
            ``"type"`` (str), ``"start"`` (int, inclusive),
            ``"end"`` (int, exclusive — FIRE convention).

    Returns:
        A list of BIO tags, one per token.  Length == ``len(tokens)``.

    Example::

        >>> tokens = ["Albertsons", "and", "Rite", "Aid", "merged"]
        >>> entities = [
        ...     {"type": "Company", "start": 0, "end": 1},
        ...     {"type": "Company", "start": 2, "end": 4},
        ...     {"type": "Action",  "start": 4, "end": 5},
        ... ]
        >>> build_bio_tags(tokens, entities)
        ['B-Company', 'O', 'B-Company', 'I-Company', 'B-Action']
    """
    tags = ["O"] * len(tokens)

    for ent in entities:
        etype = ent.get("type", "UNKNOWN")
        start = ent.get("start", -1)
        end = ent.get("end", -1)  # exclusive

        # Skip malformed or unknown entity types
        if etype not in ENTITY_TYPES:
            continue
        if start < 0 or end <= start or end > len(tokens):
            continue

        tags[start] = f"B-{etype}"
        for i in range(start + 1, end):
            tags[i] = f"I-{etype}"

    return tags

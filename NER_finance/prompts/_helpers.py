# -*- coding: utf-8 -*-
"""Shared helpers for building prompt text content.

These functions are purely presentational and have no side effects,
making them easy to unit-test and reuse across Causal and Instruction
builders.
"""

from __future__ import annotations

import json
from typing import Dict, List

from data.schemas import EntitySpan


# ---------------------------------------------------------------------------
# Entity block formatters
# ---------------------------------------------------------------------------


def format_entity_definitions(entity_labels: Dict[str, str]) -> str:
    """Render entity label definitions as a bullet list."""
    lines = [
        f"- {label}: {desc}" for label, desc in entity_labels.items()
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output builders
# ---------------------------------------------------------------------------


def build_ner_output(entities: List[EntitySpan]) -> str:
    """Build the ground-truth JSON string for NER output."""
    return json.dumps(
        {
            "entities": [
                {"id": e.entity_id, "label": e.label, "term": e.term}
                for e in entities
            ]
        },
        ensure_ascii=False,
    )

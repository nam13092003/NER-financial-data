# -*- coding: utf-8 -*-
"""Shared data schemas for the Financial NER pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EntitySpan:
    """A named entity extracted from a document."""

    label: str
    """Entity type label, e.g. ``"Company"``, ``"Person"``, ``"Money"``."""

    term: str
    """Surface form of the entity as it appears in the text."""

    start_token_idx: int
    """0-based index of the first whitespace-split token (inclusive)."""

    end_token_idx: int
    """0-based index of the last whitespace-split token (inclusive)."""

    entity_id: str
    """Unique identifier within the document, e.g. ``"T0"``."""


@dataclass
class StandardizedDocument:
    """A normalised document ready for prompt construction."""

    text: str
    """Original sentence string (tokens joined by whitespace)."""

    words: List[str]
    """Whitespace-split tokens of ``text``."""

    entities: List[EntitySpan]
    """All labelled entities in the document."""

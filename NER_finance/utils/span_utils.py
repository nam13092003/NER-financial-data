# -*- coding: utf-8 -*-
"""Utility functions for resolving entity text spans to token indices."""

from __future__ import annotations

from typing import Tuple


def resolve_span_to_index(term: str, sentence: str) -> Tuple[int, int]:
    """Find the start and end **whitespace-token** indices of *term* in *sentence*.

    Performs case-insensitive matching on whitespace-split tokens.

    Args:
        term: The surface form to locate (e.g. ``"moderate as"``).
        sentence: The full sentence string (e.g. ``"...with moderate AS..."``).

    Returns:
        ``(start_idx, end_idx)`` — both indices are 0-based and inclusive.
        Returns ``(-1, -1)`` when the term cannot be matched.

    Example::

        >>> resolve_span_to_index("moderate as", "patient with moderate AS and...")
        (2, 3)
    """
    words = sentence.lower().split()
    term_words = term.lower().split()
    n = len(term_words)

    if n == 0:
        return -1, -1

    for i in range(len(words) - n + 1):
        if words[i : i + n] == term_words:
            return i, i + n - 1

    return -1, -1


def normalize_term(term: str) -> str:
    """Lowercase and collapse internal whitespace for stable comparison.

    Args:
        term: Raw entity surface form.

    Returns:
        Normalised string suitable for set-based equality checks.

    Example::

        >>> normalize_term("  Moderate AS  ")
        'moderate as'
    """
    return " ".join(term.lower().split())

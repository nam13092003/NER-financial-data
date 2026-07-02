# -*- coding: utf-8 -*-
"""Adapter that parses FIRE financial NER records into StandardizedDocuments.

Follows the Single Responsibility Principle: this class only handles
parsing, never prompt construction or training logic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .schemas import EntitySpan, StandardizedDocument

logger = logging.getLogger(__name__)


class FireFormatAdapter:
    """Converts a raw FIRE JSON record into a :class:`StandardizedDocument`.

    FIRE record schema::

        {
            "tokens": ["Albertsons", "and", ...],
            "entities": [
                {"id": "uuid", "text": "Albertsons", "type": "Company", "start": 0, "end": 1}
            ],
            ...
        }

    Note: ``end`` in FIRE is *exclusive* (``end = start + num_tokens``).
    We convert to inclusive ``end_token_idx = end - 1``.
    """

    def parse_record(
        self,
        record: dict,
        record_idx: int = 0,
    ) -> Optional[StandardizedDocument]:
        """Parse one FIRE JSON record.

        Args:
            record: A single element from the FIRE JSON array.
            record_idx: Index of the record in the source file (for logging).

        Returns:
            A ``StandardizedDocument``, or ``None`` if the record has no tokens.
        """
        tokens: List[str] = record.get("tokens", [])
        if not tokens:
            logger.debug("Empty tokens at record %d; skipping.", record_idx)
            return None

        text = " ".join(tokens)
        entities = self._parse_entities(record, record_idx)

        return StandardizedDocument(
            text=text,
            words=tokens,
            entities=entities,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_entities(
        record: dict,
        record_idx: int,
    ) -> List[EntitySpan]:
        """Parse the ``entities`` list from a FIRE record."""
        entities: List[EntitySpan] = []

        for idx, raw_ent in enumerate(record.get("entities", [])):
            start: int = raw_ent.get("start", -1)
            end: int = raw_ent.get("end", -1)

            # FIRE uses exclusive end; convert to inclusive
            end_inclusive = end - 1 if end > 0 else -1

            entities.append(
                EntitySpan(
                    label=raw_ent.get("type", "UNKNOWN"),
                    term=raw_ent.get("text", ""),
                    start_token_idx=start,
                    end_token_idx=end_inclusive,
                    entity_id=f"T{idx}",
                )
            )

        return entities

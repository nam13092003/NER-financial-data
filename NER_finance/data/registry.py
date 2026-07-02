# -*- coding: utf-8 -*-
"""Dataset registry that aggregates multiple FIRE JSON files.

Each file is expected to be a JSON array of FIRE-format records.
"""

from __future__ import annotations

import json
import logging
from typing import List

from config import DataFileEntry
from .adapter import FireFormatAdapter
from .schemas import StandardizedDocument

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Load and merge an arbitrary number of FIRE JSON files.

    Example::

        from config import DataFileEntry
        from data.registry import DatasetRegistry

        registry = DatasetRegistry([
            DataFileEntry(path="data/fire_train.json"),
        ])
        docs = registry.aggregate()
    """

    def __init__(self, entries: List[DataFileEntry]) -> None:
        self._entries = entries
        self._adapter = FireFormatAdapter()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def aggregate(self) -> List[StandardizedDocument]:
        """Parse all registered files and return a flat document list."""
        all_docs: List[StandardizedDocument] = []
        for entry in self._entries:
            docs = self._load_file(entry)
            logger.info(
                "Loaded %d documents from '%s'",
                len(docs),
                entry.path,
            )
            all_docs.extend(docs)
        logger.info("Total documents loaded: %d", len(all_docs))
        return all_docs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_file(self, entry: DataFileEntry) -> List[StandardizedDocument]:
        """Read a single FIRE JSON file (array of records)."""
        docs: List[StandardizedDocument] = []
        try:
            with open(entry.path, encoding="utf-8") as fh:
                records: list = json.load(fh)

            if not isinstance(records, list):
                logger.error(
                    "Expected a JSON array in '%s', got %s.",
                    entry.path,
                    type(records).__name__,
                )
                return docs

            for idx, record in enumerate(records):
                doc = self._adapter.parse_record(record, record_idx=idx)
                if doc is not None:
                    docs.append(doc)

        except FileNotFoundError:
            logger.error("File not found: '%s'", entry.path)
        except json.JSONDecodeError as exc:
            logger.error(
                "Malformed JSON in '%s': %s", entry.path, exc
            )
        return docs

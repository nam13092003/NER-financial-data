# -*- coding: utf-8 -*-
"""FIRE → BIO Dataset with subword-to-word alignment.

This module reads FIRE-format JSON files and produces HuggingFace-compatible
datasets ready for token classification training.

Key design decisions
--------------------
- **First-subword pooling for alignment**: Each word is tokenised into one or
  more subword tokens.  Only the *first* subword of each word receives the
  real BIO label; all subsequent subwords of the same word receive ``-100``
  so they are ignored by ``CrossEntropyLoss``.
- Special tokens (``[CLS]``, ``[SEP]``, etc.) also receive ``-100``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import LABEL2ID, build_bio_tags

logger = logging.getLogger(__name__)

# Label assigned to subword tokens that should be ignored in the loss
IGNORE_INDEX = -100


class FIREBIODataset(Dataset):
    """A ``torch.utils.data.Dataset`` that converts FIRE records to BIO.

    Each ``__getitem__`` returns a dict with:
        - ``input_ids``      : List[int]   — subword token ids
        - ``attention_mask``  : List[int]   — 1 for real tokens, 0 for padding
        - ``labels``          : List[int]   — BIO label ids (``-100`` for ignored)

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to FIRE JSON files (each is a JSON array of records).
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer (e.g. ``AutoTokenizer.from_pretrained(...)``).
    max_length : int
        Maximum subword sequence length (including special tokens).
    """

    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict[str, Any]] = []

        for fpath in file_paths:
            self._load_file(fpath)

        logger.info(
            "Loaded %d samples from %d file(s).", len(self.samples), len(file_paths)
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_file(self, file_path: Union[str, Path]) -> None:
        """Parse a single FIRE JSON file and append samples."""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning("File not found, skipping: %s", file_path)
            return

        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for idx, record in enumerate(records):
            tokens = record.get("tokens", [])
            if not tokens:
                continue

            entities = record.get("entities", [])
            bio_tags = build_bio_tags(tokens, entities)

            self.samples.append(
                {
                    "tokens": tokens,
                    "bio_tags": bio_tags,
                }
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        sample = self.samples[idx]
        tokens: List[str] = sample["tokens"]
        bio_tags: List[str] = sample["bio_tags"]

        return self._tokenize_and_align(tokens, bio_tags)

    # ------------------------------------------------------------------
    # Subword → Word alignment
    # ------------------------------------------------------------------

    def _tokenize_and_align(
        self,
        words: List[str],
        word_labels: List[str],
    ) -> Dict[str, List[int]]:
        """Tokenize words and align BIO labels to subword tokens.

        Strategy — **first subword pooling**:
        - For each word, only the first subword gets the real label.
        - Subsequent subwords of the same word get ``-100``.
        - Special tokens (``[CLS]``, ``[SEP]``) get ``-100``.

        Returns
        -------
        dict with ``input_ids``, ``attention_mask``, ``labels``
        """
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,           # Padding is handled by the collator
            return_offsets_mapping=False,
        )

        word_ids = tokenized.word_ids()  # None for special tokens, int for word idx

        labels: List[int] = []
        previous_word_id: Optional[int] = None

        for word_id in word_ids:
            if word_id is None:
                # Special token ([CLS], [SEP], etc.)
                labels.append(IGNORE_INDEX)
            elif word_id != previous_word_id:
                # First subword of a new word → assign real label
                labels.append(LABEL2ID[word_labels[word_id]])
            else:
                # Continuation subword of the same word → ignore
                labels.append(IGNORE_INDEX)

            previous_word_id = word_id

        tokenized["labels"] = labels
        return dict(tokenized)

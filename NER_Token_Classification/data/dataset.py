# -*- coding: utf-8 -*-
"""PyTorch Dataset for BIO-based NER token classification on FIRE data.

Implements **first-subword pooling**: for each word, only the first subword
produced by the tokenizer receives the true BIO label. All subsequent
subwords of the same word (plus special tokens [CLS], [SEP], padding) are
assigned ``-100`` so that ``CrossEntropyLoss`` ignores them.

This is equivalent to using the first subword's hidden state as the word
representation for classification.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from .label_utils import entities_to_bio_tags, label2id

logger = logging.getLogger(__name__)


class FireBIODataset(Dataset):
    """Load FIRE-format NER data and produce BIO-labelled token sequences.

    Each item returned is a dict with:
        - ``input_ids``:      (max_length,) int tensor
        - ``attention_mask``:  (max_length,) int tensor
        - ``labels``:          (max_length,) int tensor  (``-100`` for ignored positions)
        - ``word_ids``:        (max_length,) list[Optional[int]]  (for decoding)

    Args:
        file_paths: List of paths to FIRE JSON files.
        tokenizer: A HuggingFace *fast* tokenizer (required for ``word_ids()``).
        max_length: Maximum subword sequence length (truncation & padding).
        lab2id: Mapping from BIO label string to integer id.
    """

    def __init__(
        self,
        file_paths: List[str],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 256,
        lab2id: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lab2id = lab2id or label2id()

        # Load all records
        self.records: List[dict] = []
        for path in file_paths:
            self.records.extend(self._load_file(path))
        logger.info("Loaded %d records from %d file(s).", len(self.records), len(file_paths))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        tokens: List[str] = record["tokens"]
        entities: List[dict] = record.get("entities", [])

        # 1. Convert entities → word-level BIO tags
        bio_tags = entities_to_bio_tags(tokens, entities)

        # 2. Tokenize using the fast tokenizer with is_split_into_words=True
        #    This tells the tokenizer that `tokens` is already word-split,
        #    and it will further split each word into subwords.
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # collator handles padding
            return_tensors=None,
        )

        # 3. Align word-level BIO tags to subword tokens
        #    word_ids() returns None for special tokens ([CLS], [SEP], padding)
        #    and the word index for each subword token.
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special token → ignore in loss
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # First subword of a new word → assign the real BIO label
                if word_id < len(bio_tags):
                    tag = bio_tags[word_id]
                    aligned_labels.append(self.lab2id.get(tag, 0))
                else:
                    aligned_labels.append(-100)
            else:
                # Subsequent subword of the same word → ignore in loss
                aligned_labels.append(-100)
            previous_word_id = word_id

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": aligned_labels,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(path: str) -> List[dict]:
        """Read a FIRE JSON file (array of records)."""
        try:
            with open(path, encoding="utf-8") as fh:
                records = json.load(fh)
            if not isinstance(records, list):
                logger.error("Expected JSON array in '%s', got %s.", path, type(records).__name__)
                return []
            logger.info("Loaded %d records from '%s'.", len(records), path)
            return records
        except FileNotFoundError:
            logger.error("File not found: '%s'", path)
            return []
        except json.JSONDecodeError as exc:
            logger.error("Malformed JSON in '%s': %s", path, exc)
            return []

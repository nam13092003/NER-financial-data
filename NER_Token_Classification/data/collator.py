# -*- coding: utf-8 -*-
"""Data collator for BIO-based NER token classification.

Handles dynamic padding of ``input_ids``, ``attention_mask``, and ``labels``.
Labels are padded with ``-100`` so that padded positions do not contribute
to the cross-entropy loss.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class NERDataCollator:
    """Collate variable-length NER samples into a padded batch.

    Args:
        pad_token_id: Token id used for padding ``input_ids``.
        label_pad_token_id: Value used for padding ``labels`` (default: -100).
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Find maximum sequence length in the batch
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            input_ids_batch.append(
                f["input_ids"] + [self.pad_token_id] * pad_len
            )
            attention_mask_batch.append(
                f["attention_mask"] + [0] * pad_len
            )
            labels_batch.append(
                f["labels"] + [self.label_pad_token_id] * pad_len
            )

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }

# -*- coding: utf-8 -*-
"""Custom data collator with loss masking for instruction fine-tuning.

For the INSTRUCTION format, cross-entropy loss should only be computed
on the *assistant* (output JSON) tokens, not on the system or user prompt.
This collator masks all tokens before the assistant response with -100.

For the CAUSAL format, standard ``DataCollatorForLanguageModeling`` is
used; no masking beyond padding is needed.
"""

from __future__ import annotations

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase


class DataCollatorWithLossMask:
    """Pads sequences and masks prompt tokens from the loss computation.

    Expected dataset columns (produced by :func:`preprocess_instruction`):

    * ``input_ids``       — full tokenised sequence
    * ``attention_mask``  — 1/0 mask for padding
    * ``prompt_length``   — number of tokens belonging to the prompt
      (system + user), used to set ``labels[:prompt_length] = -100``

    The ``prompt_length`` field is removed before batching so it does not
    confuse the default collator.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._std_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

    def __call__(self, features: list) -> dict:
        # Extract and remove prompt_length before standard collation
        prompt_lengths = [int(f.pop("prompt_length", 0)) for f in features]

        batch = self._std_collator(features)

        # Apply additional masking: hide prompt tokens from the loss
        labels: torch.Tensor = batch["labels"].clone()
        for i, prompt_len in enumerate(prompt_lengths):
            if prompt_len > 0:
                labels[i, :prompt_len] = -100
        batch["labels"] = labels

        return batch

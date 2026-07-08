# -*- coding: utf-8 -*-
"""Model factory for BIO-based NER Token Classification.

Uses ``AutoModelForTokenClassification`` from HuggingFace Transformers
with a pre-trained encoder backbone. The classification head maps each
subword hidden state to one of the BIO labels.

**First-subword pooling** is handled implicitly: only the first subword
of each word has a real label; subsequent subwords are labelled ``-100``
and are therefore ignored by the loss function. This means the model
learns to classify based on the first subword's representation.
"""

from __future__ import annotations

import logging
from typing import Tuple

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from data.label_utils import build_label_list, label2id, id2label

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Load a pre-trained encoder with a token classification head.

    Args:
        model_name: HuggingFace model identifier
            (e.g., ``"xlm-roberta-base"``, ``"vinai/phobert-base-v2"``).

    Returns:
        ``(model, tokenizer)`` — both ready for fine-tuning.
    """
    label_list = build_label_list()
    num_labels = len(label_list)
    lab2id = label2id()
    id2lab = id2label()

    logger.info(
        "Loading encoder: %s (num_labels=%d)", model_name, num_labels
    )

    # Load tokenizer (must be a fast tokenizer for word_ids() support)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        add_prefix_space=True,  # Needed for RoBERTa-family models
    )

    # Load model config and override label mappings
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=lab2id,
        id2label=id2lab,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,  # classifier head size may differ
    )

    logger.info(
        "Model loaded: %s — %d parameters",
        model_name,
        sum(p.numel() for p in model.parameters()),
    )

    return model, tokenizer

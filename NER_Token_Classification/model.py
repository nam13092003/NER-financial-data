# -*- coding: utf-8 -*-
"""NER Token Classification model.

Provides a thin wrapper around ``AutoModelForTokenClassification`` from
HuggingFace Transformers, pre-configured with the BIO label set from the
FIRE financial NER dataset.

The model architecture is:

    Input IDs → Encoder (BERT / RoBERTa / DeBERTa) → last_hidden_state
        → Dropout → Linear(hidden_size, 27) → logits

Subword-to-word alignment is handled at the *data* level (see
``data/dataset.py``): only the first subword of each word receives a real
BIO label; all other subwords receive ``-100`` and are therefore ignored
by the cross-entropy loss computed internally by the HuggingFace model.
"""

from __future__ import annotations

from transformers import AutoModelForTokenClassification, AutoConfig

from utils import NUM_LABELS, LABEL2ID, ID2LABEL


def build_model(model_name: str) -> AutoModelForTokenClassification:
    """Instantiate a token classification model from a pre-trained encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"bert-base-uncased"``,
        ``"roberta-base"``, ``"microsoft/deberta-v3-base"``.

    Returns
    -------
    AutoModelForTokenClassification
        A model with a linear classification head on top of the encoder,
        configured with the BIO label set (27 labels).
    """
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,  # classifier head will be re-initialised
    )

    return model

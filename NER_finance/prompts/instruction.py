# -*- coding: utf-8 -*-
"""Instruction prompt builder for financial NER.

Produces samples with three columns — ``"system"``, ``"user"``, and
``"assistant"`` — that are later assembled into a ChatML sequence via
``tokenizer.apply_chat_template()``.

The system prompt contains the full entity label ontology so the model
learns the semantic meaning of every label, not just surface patterns.
"""

from __future__ import annotations

from typing import Dict, List

from datasets import Dataset as HFDataset

from config import LabelConfig
from data.schemas import StandardizedDocument
from ._helpers import (
    build_ner_output,
    format_entity_definitions,
)
from .builder import BasePromptBuilder


class InstructionPromptBuilder(BasePromptBuilder):
    """Builds ChatML-style training samples for instruction fine-tuning.

    The output ``HFDataset`` has three columns:

    * ``"system"``    — entity label ontology and output schema (constant per config)
    * ``"user"``      — the input sentence
    * ``"assistant"`` — the gold-standard JSON output

    Loss masking (training only on the ``assistant`` column) is handled by
    :class:`~training.collator.DataCollatorWithLossMask`.
    """

    def __init__(self, labels: LabelConfig) -> None:
        self._labels = labels
        self._system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------
    # BasePromptBuilder interface
    # ------------------------------------------------------------------

    def generate_training_data(
        self,
        documents: List[StandardizedDocument],
    ) -> HFDataset:
        records: List[Dict[str, str]] = []
        for doc in documents:
            records.append(self._ner_sample(doc))
        return HFDataset.from_list(records)

    # ------------------------------------------------------------------
    # System prompt (built once per config)
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        entity_defs = format_entity_definitions(self._labels.entities)
        return (
            "You are a financial Named Entity Recognition expert.\n"
            "Given a financial sentence, identify all named entities.\n\n"
            f"Entity types:\n{entity_defs}\n\n"
            "Output ONLY a valid JSON object. "
            'Schema: {"entities": [{"id": "T0", "label": "...", "term": "..."}]}'
        )

    # ------------------------------------------------------------------
    # NER sample
    # ------------------------------------------------------------------

    def _ner_sample(self, doc: StandardizedDocument) -> Dict[str, str]:
        user = f'Sentence: "{doc.text}"'
        return {
            "system": self._system_prompt,
            "user": user,
            "assistant": build_ner_output(doc.entities),
        }

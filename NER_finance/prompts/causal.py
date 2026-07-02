# -*- coding: utf-8 -*-
"""Causal prompt builder for financial NER.

Produces samples with a single ``"text"`` column containing the full
``input_prefix + ### Output:\\n{json}`` string.  No system/user/assistant
separation — the model learns the mapping as a causal language model.
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


class CausalPromptBuilder(BasePromptBuilder):
    """Builds plain-text training samples for causal LM fine-tuning.

    The output ``HFDataset`` has a single column ``"text"`` which is the
    concatenation of the instruction prefix and the gold-standard JSON output.

    Separating the instruction from the output with ``\\n\\n### Output:\\n``
    gives the model a clear boundary to learn where to start generating.
    """

    _OUTPUT_MARKER = "\n\n### Output:\n"

    def __init__(self, labels: LabelConfig) -> None:
        self._labels = labels
        self._entity_defs = format_entity_definitions(labels.entities)

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
    # NER
    # ------------------------------------------------------------------

    def _ner_sample(self, doc: StandardizedDocument) -> Dict[str, str]:
        prefix = (
            "Extract all named entities from the following financial sentence.\n\n"
            f"Entity types:\n{self._entity_defs}\n\n"
            "Output ONLY a valid JSON object. "
            'Schema: {"entities": [{"id": "T0", "label": "...", "term": "..."}]}\n\n'
            f'Sentence: "{doc.text}"'
        )
        return {"text": prefix + self._OUTPUT_MARKER + build_ner_output(doc.entities)}

# -*- coding: utf-8 -*-
"""Abstract base class for all prompt builders.

Follows the Open/Closed Principle: new formats are added by subclassing
``BasePromptBuilder`` without touching existing builders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from datasets import Dataset as HFDataset

from data.schemas import StandardizedDocument


class BasePromptBuilder(ABC):
    """Contract that every prompt builder must satisfy.

    Subclasses implement :meth:`generate_training_data` to convert a list
    of :class:`~ner_re_pipeline.data.schemas.StandardizedDocument` objects
    into a HuggingFace ``Dataset`` with columns ready for tokenisation.

    The concrete column schema depends on the format:

    * **Causal** – ``{"text": str}``
    * **Instruction** – ``{"system": str, "user": str, "assistant": str}``
    """

    @abstractmethod
    def generate_training_data(
        self,
        documents: List[StandardizedDocument],
    ) -> HFDataset:
        """Convert documents into a tokeniser-ready HuggingFace Dataset.

        Args:
            documents: Parsed documents from :class:`DatasetRegistry`.

        Returns:
            A ``datasets.Dataset`` instance with format-specific columns.
        """
        ...

# -*- coding: utf-8 -*-
"""Factory that creates the correct PromptBuilder from a PipelineConfig."""

from __future__ import annotations

from config import PipelineConfig, TrainingFormat
from .builder import BasePromptBuilder
from .causal import CausalPromptBuilder
from .instruction import InstructionPromptBuilder


class PromptBuilderFactory:
    """Creates a :class:`BasePromptBuilder` from a :class:`PipelineConfig`.

    Follows the Factory Method pattern (SOLID — Open/Closed): adding a new
    format only requires adding a new ``elif`` branch and a new subclass.
    """

    @staticmethod
    def create(config: PipelineConfig) -> BasePromptBuilder:
        """Instantiate and return the appropriate prompt builder.

        Args:
            config: Fully validated pipeline configuration.

        Returns:
            A concrete :class:`BasePromptBuilder` instance.

        Raises:
            ValueError: If ``config.training.format`` is not recognised.
        """
        labels = config.labels
        fmt = config.training.format

        if fmt == TrainingFormat.CAUSAL:
            return CausalPromptBuilder(labels)
        if fmt == TrainingFormat.INSTRUCTION:
            return InstructionPromptBuilder(labels)
        raise ValueError(
            f"Unknown training format: {fmt!r}. "
            f"Expected one of {[f.value for f in TrainingFormat]}."
        )

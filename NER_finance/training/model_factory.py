# -*- coding: utf-8 -*-
"""Unsloth model + LoRA adapter factory.

Centralises all Unsloth-specific calls so the rest of the codebase
remains framework-agnostic and testable without a GPU.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from config import PipelineConfig

logger = logging.getLogger(__name__)


class UnslothModelFactory:
    """Creates a LoRA-wrapped causal language model via Unsloth.

    The factory pattern (SOLID – Single Responsibility) ensures model
    loading logic is isolated and can be swapped for a pure HuggingFace
    implementation by replacing this class.
    """

    @staticmethod
    def load(config: PipelineConfig) -> Tuple[Any, Any]:
        """Load the backbone model and apply LoRA adapters.

        Args:
            config: Fully validated pipeline configuration.

        Returns:
            ``(model, tokenizer)`` — both ready for training.
        """
        # Import here so non-GPU environments can import other modules
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        model_cfg = config.model
        lora_cfg = config.lora

        logger.info("Loading backbone via Unsloth: %s", model_cfg.name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg.name,
            max_seq_length=model_cfg.max_seq_length,
            dtype=None,       # auto-detect
            load_in_4bit=True,
        )

        logger.info("Applying LoRA adapters (r=%d, alpha=%d)", lora_cfg.r, lora_cfg.alpha)
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.r,
            target_modules=lora_cfg.target_modules,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=model_cfg.seed,
            use_rslora=False,
            loftq_config=None,
        )

        # Ensure the tokenizer has a chat template
        if tokenizer.chat_template is None:
            logger.info("No chat template found — applying 'chatml'.")
            tokenizer = get_chat_template(tokenizer, chat_template="chatml")

        return model, tokenizer

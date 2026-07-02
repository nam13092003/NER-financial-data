# -*- coding: utf-8 -*-
"""Pipeline configuration loaded from a YAML file.

Provides strongly-typed dataclasses for every config section and enums
for every categorical option, enabling IDE auto-complete and early
validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrainingFormat(str, Enum):
    """Controls how training samples are formatted."""

    CAUSAL = "causal"
    """Plain text: ``sentence\\n\\n### Output:\\n{json}``."""

    INSTRUCTION = "instruction"
    """ChatML: System / User / Assistant roles."""


# ---------------------------------------------------------------------------
# Config sub-sections
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Backbone model settings."""

    name: str = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
    max_seq_length: int = 1024
    seed: int = 42


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation hyper-parameters."""

    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainingConfig:
    """Training loop hyper-parameters."""

    format: TrainingFormat = TrainingFormat.INSTRUCTION
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    output_dir: str = "./outputs"
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50

    # ── Dev subset (for faster evaluation during training) ───────
    # If float: fraction of the eval dataset (e.g., 0.2 for 20%)
    # If int: absolute number of samples (e.g., 500)
    # If None or 1.0: use the full dataset
    eval_subset_size: Union[int, float] = 0.2

    # ── Early Stopping ───────────────────────────────────────────
    # Number of eval calls with no improvement before stopping
    # Set to 0 to disable early stopping
    early_stopping_patience: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.format, str):
            self.format = TrainingFormat(self.format)


@dataclass
class DataFileEntry:
    """A single JSON data file path."""

    path: str


@dataclass
class DataConfig:
    """Dataset file lists for training and evaluation."""

    train_files: List[DataFileEntry] = field(default_factory=list)
    eval_files: List[DataFileEntry] = field(default_factory=list)
    test_files: List[DataFileEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.train_files = [
            DataFileEntry(**f) if isinstance(f, dict) else f
            for f in self.train_files
        ]
        self.eval_files = [
            DataFileEntry(**f) if isinstance(f, dict) else f
            for f in self.eval_files
        ]
        self.test_files = [
            DataFileEntry(**f) if isinstance(f, dict) else f
            for f in self.test_files
        ]


@dataclass
class LabelConfig:
    """Entity label definitions embedded into prompts."""

    entities: Dict[str, str] = field(default_factory=dict)
    """``{label: description}`` for each entity type."""


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Root configuration object assembled from YAML."""

    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    data: DataConfig
    labels: LabelConfig

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load and validate configuration from a YAML file."""
        with open(path, encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh)

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            lora=LoRAConfig(**raw.get("lora", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            labels=LabelConfig(**raw.get("labels", {})),
        )

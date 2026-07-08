# -*- coding: utf-8 -*-
"""Configuration for BIO-based NER Token Classification pipeline.

Provides strongly-typed dataclasses loaded from a YAML file, similar to
the ``NER_finance`` pipeline but tailored for encoder-based token
classification rather than LLM-based generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    """Encoder backbone settings."""

    name: str = "xlm-roberta-base"
    """HuggingFace model identifier for the pre-trained encoder."""

    max_length: int = 256
    """Maximum subword sequence length (truncation & padding)."""

    seed: int = 42


@dataclass
class DataFileEntry:
    """A single JSON data file path."""

    path: str


@dataclass
class DataConfig:
    """Dataset file lists for training, evaluation, and testing."""

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

    @property
    def train_paths(self) -> List[str]:
        return [f.path for f in self.train_files]

    @property
    def eval_paths(self) -> List[str]:
        return [f.path for f in self.eval_files]

    @property
    def test_paths(self) -> List[str]:
        return [f.path for f in self.test_files]


@dataclass
class TrainingConfig:
    """Training loop hyper-parameters."""

    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    output_dir: str = "./outputs_token_cls"
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    fp16: bool = True

    # Early stopping
    early_stopping_patience: int = 5
    metric_for_best_model: str = "eval_f1"


@dataclass
class TokenClassificationConfig:
    """Root configuration object for the Token Classification pipeline."""

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str) -> "TokenClassificationConfig":
        """Load and validate configuration from a YAML file."""
        with open(path, encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh)

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )

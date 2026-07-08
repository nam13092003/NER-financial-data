# -*- coding: utf-8 -*-
"""Training entry point for NER Token Classification (BIO tagging).

Usage::

    python train.py --config configs/default.yaml

Or with Accelerate for multi-GPU::

    accelerate launch train.py --config configs/default.yaml

The script uses the HuggingFace ``Trainer`` with ``seqeval`` metrics and
supports early stopping.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

import numpy as np
import torch
import yaml
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ── Ensure package imports work when run as a script ─────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data.dataset import FIREBIODataset
from model import build_model
from utils import BIO_LABELS, ID2LABEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric computation (seqeval)
# ---------------------------------------------------------------------------

def build_compute_metrics_fn():
    """Return a ``compute_metrics`` callable for the HuggingFace Trainer.

    Uses the ``seqeval`` library to compute entity-level P / R / F1.
    """
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Convert ids back to BIO string labels, skipping -100
        true_labels: list[list[str]] = []
        pred_labels: list[list[str]] = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_seq: list[str] = []
            pred_seq_str: list[str] = []

            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                true_seq.append(ID2LABEL[l])
                pred_seq_str.append(ID2LABEL[p])

            true_labels.append(true_seq)
            pred_labels.append(pred_seq_str)

        f1 = f1_score(true_labels, pred_labels, average="micro")
        precision = precision_score(true_labels, pred_labels, average="micro")
        recall = recall_score(true_labels, pred_labels, average="micro")

        # Log full report every evaluation
        report = classification_report(true_labels, pred_labels, digits=4)
        logger.info("\n%s", report)

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config and return as a plain dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Training arguments builder
# ---------------------------------------------------------------------------

def build_training_args(cfg: dict, num_train_samples: int) -> TrainingArguments:
    """Build HuggingFace ``TrainingArguments`` from the config dict."""
    tr = cfg["training"]
    model_cfg = cfg["model"]

    total_steps = math.ceil(
        num_train_samples
        / (tr["batch_size"] * tr.get("gradient_accumulation_steps", 1))
    ) * tr["epochs"]
    warmup_steps = max(1, int(total_steps * tr.get("warmup_ratio", 0.1)))

    patience = tr.get("early_stopping_patience", 0)

    return TrainingArguments(
        output_dir=tr["output_dir"],
        per_device_train_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr.get("gradient_accumulation_steps", 1),
        learning_rate=tr["learning_rate"],
        num_train_epochs=tr["epochs"],
        weight_decay=tr.get("weight_decay", 0.01),
        warmup_steps=warmup_steps,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=tr.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=tr.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=tr.get("save_steps", 100),
        seed=model_cfg.get("seed", 42),
        report_to="none",
        load_best_model_at_end=(patience > 0),
        metric_for_best_model="f1" if (patience > 0) else None,
        greater_is_better=True if (patience > 0) else None,
        save_total_limit=1,
        # DDP / gradient-checkpointing safety flags
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an encoder model for NER token classification (BIO)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="latest",
        default=None,
        help=(
            "Resume from a checkpoint. "
            "Use --resume to auto-detect the latest, or --resume path/to/ckpt."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load config ───────────────────────────────────────────────
    logger.info("Loading config from: %s", args.config)
    cfg = load_config(args.config)
    seed = cfg["model"].get("seed", 42)
    set_seed(seed)

    model_name = cfg["model"]["name"]
    max_seq_length = cfg["model"].get("max_seq_length", 256)

    # ── 2. Load tokenizer ────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── 3. Build datasets ────────────────────────────────────────────
    train_paths = [f["path"] for f in cfg["data"]["train_files"]]
    eval_paths = [f["path"] for f in cfg["data"]["eval_files"]]

    logger.info("Building training dataset …")
    train_dataset = FIREBIODataset(train_paths, tokenizer, max_length=max_seq_length)
    logger.info("Building evaluation dataset …")
    eval_dataset = FIREBIODataset(eval_paths, tokenizer, max_length=max_seq_length)

    logger.info(
        "Train samples: %d  |  Eval samples: %d",
        len(train_dataset),
        len(eval_dataset),
    )

    # ── 4. Build model ───────────────────────────────────────────────
    logger.info("Loading model: %s", model_name)
    model = build_model(model_name)

    # ── 5. Data collator (dynamic padding) ───────────────────────────
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_seq_length,
    )

    # ── 6. Training arguments ────────────────────────────────────────
    training_args = build_training_args(cfg, len(train_dataset))

    # ── 7. Build Trainer ─────────────────────────────────────────────
    patience = cfg["training"].get("early_stopping_patience", 0)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": build_compute_metrics_fn(),
    }

    if patience > 0:
        trainer_kwargs["callbacks"] = [
            EarlyStoppingCallback(early_stopping_patience=patience)
        ]
        logger.info("Early stopping enabled (patience=%d)", patience)

    trainer = Trainer(**trainer_kwargs)

    # ── 8. Train ─────────────────────────────────────────────────────
    logger.info("Starting training …")

    resume_ckpt = None
    if args.resume:
        if args.resume == "latest":
            resume_ckpt = True
            logger.info(
                "Resuming from latest checkpoint in %s …",
                cfg["training"]["output_dir"],
            )
        else:
            resume_ckpt = args.resume
            logger.info("Resuming from checkpoint: %s …", resume_ckpt)

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── 9. Save ──────────────────────────────────────────────────────
    output_dir = cfg["training"]["output_dir"]
    final_dir = os.path.join(output_dir, "final_model")

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Model saved to: %s", final_dir)

    logger.info(
        "\n[DONE] Outputs in %s\n"
        "  └── final_model/   ← Best model (encoder + classification head)",
        output_dir,
    )


if __name__ == "__main__":
    main()

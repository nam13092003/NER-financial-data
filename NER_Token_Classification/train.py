# -*- coding: utf-8 -*-
"""Training entry point for BIO-based NER Token Classification.

Usage::

    # Single GPU
    python train.py --config configs/default.yaml

    # Multi-GPU via accelerate
    accelerate launch --multi_gpu --num_processes=2 train.py \\
        --config configs/default.yaml

The script uses HuggingFace ``Trainer`` with ``seqeval``-based metrics
computed at each evaluation step for entity-level F1 monitoring.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import TokenClassificationConfig
from data.dataset import FireBIODataset
from data.collator import NERDataCollator
from data.label_utils import build_label_list, label2id, id2label
from model import load_model_and_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_ID2LABEL = id2label()


def compute_metrics(eval_preds):
    """Compute entity-level Precision, Recall, F1 using seqeval.

    ``eval_preds`` is a ``EvalPrediction`` namedtuple with:
        - ``predictions``: (batch, seq_len, num_labels) logits
        - ``label_ids``:   (batch, seq_len) integer labels
    """
    predictions, labels = eval_preds
    # argmax to get predicted label ids
    preds = np.argmax(predictions, axis=-1)

    # Convert integer predictions/labels back to BIO tag strings,
    # ignoring positions where label == -100
    true_labels = []
    true_preds = []

    for pred_seq, label_seq in zip(preds, labels):
        seq_preds = []
        seq_labels = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            seq_labels.append(_ID2LABEL.get(l, "O"))
            seq_preds.append(_ID2LABEL.get(p, "O"))
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an encoder for BIO-based financial NER token classification."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume", type=str, nargs="?", const="latest", default=None,
        help=(
            "Resume from a checkpoint. Use --resume to auto-detect the latest "
            "in output_dir, or --resume path/to/checkpoint."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading config from: %s", args.config)
    config = TokenClassificationConfig.from_yaml(args.config)
    set_seed(config.model.seed)

    # ------------------------------------------------------------------
    # 1. Load model & tokenizer
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(config.model.name)

    # ------------------------------------------------------------------
    # 2. Build datasets
    # ------------------------------------------------------------------
    lab2id = label2id()

    logger.info("Loading training data...")
    train_dataset = FireBIODataset(
        file_paths=config.data.train_paths,
        tokenizer=tokenizer,
        max_length=config.model.max_length,
        lab2id=lab2id,
    )

    logger.info("Loading evaluation data...")
    eval_dataset = FireBIODataset(
        file_paths=config.data.eval_paths,
        tokenizer=tokenizer,
        max_length=config.model.max_length,
        lab2id=lab2id,
    )

    logger.info("Train samples: %d  |  Eval samples: %d", len(train_dataset), len(eval_dataset))

    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty. Check file paths in config."
        )

    # ------------------------------------------------------------------
    # 3. Data collator
    # ------------------------------------------------------------------
    data_collator = NERDataCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    # ------------------------------------------------------------------
    # 4. Training arguments
    # ------------------------------------------------------------------
    tr = config.training
    training_args = TrainingArguments(
        output_dir=tr.output_dir,
        per_device_train_batch_size=tr.batch_size,
        per_device_eval_batch_size=tr.batch_size * 2,
        gradient_accumulation_steps=tr.gradient_accumulation_steps,
        learning_rate=tr.learning_rate,
        num_train_epochs=tr.epochs,
        weight_decay=tr.weight_decay,
        warmup_ratio=tr.warmup_ratio,
        fp16=tr.fp16 and torch.cuda.is_available(),
        bf16=not tr.fp16 and torch.cuda.is_bf16_supported(),
        logging_steps=tr.logging_steps,
        eval_strategy="steps",
        eval_steps=tr.eval_steps,
        save_strategy="steps",
        save_steps=tr.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=tr.metric_for_best_model,
        greater_is_better=True,
        report_to="none",
        seed=config.model.seed,
        ddp_find_unused_parameters=False,
    )

    # ------------------------------------------------------------------
    # 5. Build trainer
    # ------------------------------------------------------------------
    callbacks = []
    if tr.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=tr.early_stopping_patience)
        )
        logger.info("Early stopping enabled (patience=%d).", tr.early_stopping_patience)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    logger.info("Starting training...")
    resume_ckpt = None
    if args.resume:
        if args.resume == "latest":
            resume_ckpt = True
            logger.info("Resuming from latest checkpoint in %s...", tr.output_dir)
        else:
            resume_ckpt = args.resume
            logger.info("Resuming from: %s...", resume_ckpt)

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ------------------------------------------------------------------
    # 7. Final evaluation
    # ------------------------------------------------------------------
    logger.info("Running final evaluation on eval set...")
    eval_results = trainer.evaluate()
    logger.info("Final eval results: %s", eval_results)

    # ------------------------------------------------------------------
    # 8. Save best model
    # ------------------------------------------------------------------
    if trainer.is_world_process_zero():
        save_path = os.path.join(tr.output_dir, "best_model")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info("Best model saved to: %s", save_path)

        # Also print full seqeval classification report
        logger.info("Generating detailed classification report...")
        preds_output = trainer.predict(eval_dataset)
        preds = np.argmax(preds_output.predictions, axis=-1)

        id2lab = id2label()
        true_labels = []
        true_preds = []
        for pred_seq, label_seq in zip(preds, preds_output.label_ids):
            seq_preds = []
            seq_labels = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                seq_labels.append(id2lab.get(l, "O"))
                seq_preds.append(id2lab.get(p, "O"))
            true_labels.append(seq_labels)
            true_preds.append(seq_preds)

        report = classification_report(true_labels, true_preds, digits=4)
        logger.info("\n%s", report)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

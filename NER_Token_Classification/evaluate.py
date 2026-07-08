# -*- coding: utf-8 -*-
"""Evaluation script for NER Token Classification.

Usage::

    python evaluate.py \\
        --config configs/default.yaml \\
        --checkpoint /path/to/final_model \\
        --output /path/to/predictions.json

Loads a fine-tuned ``AutoModelForTokenClassification``, runs inference on
the test set, and prints entity-level seqeval metrics (P / R / F1 per
entity type + micro / macro averages).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List

import numpy as np
import torch
import yaml
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

# ── Ensure package imports work when run as a script ─────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data.dataset import FIREBIODataset
from utils import ID2LABEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict(
    model: AutoModelForTokenClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[List[List[str]], List[List[str]]]:
    """Run inference and return (true_labels, pred_labels) as BIO strings.

    Only positions with label != -100 are included (i.e. only the first
    subword of each word, excluding special tokens and padding).
    """
    model.eval()
    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, seq_len, num_labels)
            preds = torch.argmax(logits, dim=-1)  # (batch, seq_len)

            # Move to CPU for processing
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for pred_seq, label_seq in zip(preds_np, labels_np):
                true_seq: List[str] = []
                pred_seq_str: List[str] = []

                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    true_seq.append(ID2LABEL[l])
                    pred_seq_str.append(ID2LABEL[p])

                all_true.append(true_seq)
                all_pred.append(pred_seq_str)

    return all_true, all_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned NER token classification model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved model directory (e.g. final_model/).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the evaluation results JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["eval", "test"],
        help="Which split to evaluate: 'eval' (dev) or 'test'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load config ───────────────────────────────────────────────
    logger.info("Loading config from: %s", args.config)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    max_seq_length = cfg["model"].get("max_seq_length", 256)

    # ── 2. Load model & tokenizer from checkpoint ────────────────────
    logger.info("Loading model from checkpoint: %s", args.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Using device: %s", device)

    # ── 3. Build test dataset ────────────────────────────────────────
    if args.split == "test":
        file_paths = [f["path"] for f in cfg["data"]["test_files"]]
    else:
        file_paths = [f["path"] for f in cfg["data"]["eval_files"]]

    logger.info("Loading %s dataset …", args.split)
    dataset = FIREBIODataset(file_paths, tokenizer, max_length=max_seq_length)
    logger.info("Loaded %d samples.", len(dataset))

    # ── 4. Build DataLoader ──────────────────────────────────────────
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_seq_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # ── 5. Run prediction ────────────────────────────────────────────
    logger.info("Running inference …")
    true_labels, pred_labels = predict(model, dataloader, device)

    # ── 6. Compute metrics ───────────────────────────────────────────
    report = classification_report(true_labels, pred_labels, digits=4)
    micro_f1 = f1_score(true_labels, pred_labels, average="micro")

    logger.info("\n===== Evaluation Results (%s set) =====\n%s", args.split, report)
    logger.info("Micro F1: %.4f", micro_f1)

    # ── 7. Save results ──────────────────────────────────────────────
    if args.output:
        results = {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "num_samples": len(dataset),
            "micro_f1": float(micro_f1),
            "report": report,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to: %s", args.output)


if __name__ == "__main__":
    main()

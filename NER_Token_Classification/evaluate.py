# -*- coding: utf-8 -*-
"""Evaluation entry point for BIO-based NER Token Classification.

Usage::

    # Single GPU
    python evaluate.py \\
        --config configs/default.yaml \\
        --checkpoint /path/to/best_model \\
        --output predictions.jsonl

    # Multi-GPU via accelerate
    accelerate launch --multi_gpu --num_processes=2 evaluate.py \\
        --config configs/default.yaml \\
        --checkpoint /path/to/best_model \\
        --batch_size 32 \\
        --output predictions.jsonl

Loads a trained model checkpoint, runs inference on the test set,
converts BIO predictions back to entity spans, computes entity-level
metrics, and saves results.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    set_seed,
)

from config import TokenClassificationConfig
from data.dataset import FireBIODataset
from data.collator import NERDataCollator
from data.label_utils import build_label_list, id2label, label2id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BIO → Entity span conversion
# ---------------------------------------------------------------------------


def bio_tags_to_entities(
    tokens: List[str],
    tags: List[str],
) -> List[dict]:
    """Convert a BIO tag sequence back to FIRE-format entity spans.

    Args:
        tokens: Word-level tokens.
        tags: BIO tag sequence (same length as tokens).

    Returns:
        List of entity dicts: ``{"text", "type", "start", "end"}``.
    """
    entities = []
    current_entity = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            # Save previous entity if any
            if current_entity is not None:
                entities.append(current_entity)
            ent_type = tag[2:]
            current_entity = {
                "type": ent_type,
                "start": i,
                "end": i + 1,
                "text": tokens[i] if i < len(tokens) else "",
            }
        elif tag.startswith("I-") and current_entity is not None:
            ent_type = tag[2:]
            if ent_type == current_entity["type"]:
                # Continue the current entity
                current_entity["end"] = i + 1
                if i < len(tokens):
                    current_entity["text"] += " " + tokens[i]
            else:
                # Type mismatch → close current, start new as B-
                entities.append(current_entity)
                current_entity = {
                    "type": ent_type,
                    "start": i,
                    "end": i + 1,
                    "text": tokens[i] if i < len(tokens) else "",
                }
        else:
            # O tag or I- without a preceding B-
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    # Don't forget the last entity
    if current_entity is not None:
        entities.append(current_entity)

    return entities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained token classification model for financial NER."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config.yaml"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the saved model directory (best_model).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Inference batch size."
    )
    parser.add_argument(
        "--output", type=str, default="predictions.jsonl",
        help="Path to save prediction results in JSONL format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    config = TokenClassificationConfig.from_yaml(args.config)
    set_seed(config.model.seed)

    # ------------------------------------------------------------------
    # 1. Load model & tokenizer from checkpoint
    # ------------------------------------------------------------------
    logger.info("Loading model from checkpoint: %s", args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    id2lab = id2label()
    lab2id_map = label2id()

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    test_files = config.data.test_paths
    if not test_files:
        logger.warning("No test_files in config; falling back to eval_files.")
        test_files = config.data.eval_paths

    logger.info("Loading test data from: %s", test_files)
    test_dataset = FireBIODataset(
        file_paths=test_files,
        tokenizer=tokenizer,
        max_length=config.model.max_length,
        lab2id=lab2id_map,
    )
    logger.info("Test samples: %d", len(test_dataset))

    collator = NERDataCollator(pad_token_id=tokenizer.pad_token_id or 0)
    dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    # ------------------------------------------------------------------
    # 3. Inference
    # ------------------------------------------------------------------
    logger.info("Running inference...")
    all_preds: List[List[str]] = []
    all_labels: List[List[str]] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            preds = np.argmax(logits, axis=-1)

            for pred_seq, label_seq in zip(preds, labels):
                seq_preds = []
                seq_labels = []
                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    seq_labels.append(id2lab.get(l, "O"))
                    seq_preds.append(id2lab.get(p, "O"))
                all_preds.append(seq_preds)
                all_labels.append(seq_labels)

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    logger.info("Computing entity-level metrics...")

    p = precision_score(all_labels, all_preds)
    r = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    logger.info("Entity-level Precision: %.4f", p)
    logger.info("Entity-level Recall:    %.4f", r)
    logger.info("Entity-level F1:        %.4f", f1)

    report = classification_report(all_labels, all_preds, digits=4)
    logger.info("\nDetailed Classification Report:\n%s", report)

    # ------------------------------------------------------------------
    # 5. Save predictions as JSONL
    # ------------------------------------------------------------------
    logger.info("Converting BIO predictions to entity spans...")

    # Reload raw records to get original tokens
    raw_records = []
    for path in test_files:
        with open(path, encoding="utf-8") as fh:
            raw_records.extend(json.load(fh))

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f_out:
        for idx, (pred_tags, raw_record) in enumerate(zip(all_preds, raw_records)):
            tokens = raw_record.get("tokens", [])
            gold_entities = raw_record.get("entities", [])
            pred_entities = bio_tags_to_entities(tokens, pred_tags)

            result = {
                "id": idx,
                "tokens": tokens,
                "gold_entities": gold_entities,
                "predicted_entities": pred_entities,
                "predicted_bio_tags": pred_tags,
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info("Predictions saved to: %s", args.output)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    summary = {
        "model": args.checkpoint,
        "test_samples": len(test_dataset),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }
    summary_path = args.output.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved to: %s", summary_path)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()

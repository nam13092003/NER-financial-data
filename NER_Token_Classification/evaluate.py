# -*- coding: utf-8 -*-
"""Evaluation script for NER Token Classification with Multi-GPU support.

Usage::

    accelerate launch --multi_gpu --num_processes=2 evaluate.py \
        --config configs/default.yaml \
        --checkpoint /path/to/final_model \
        --output /path/to/predictions.json
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
from accelerate import Accelerator

# ── Ensure package imports work when run as a script ─────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data.dataset import FIREBIODataset
from utils import ID2LABEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict(
    model: AutoModelForTokenClassification,
    dataloader: DataLoader,
    accelerator: Accelerator,
) -> tuple[List[List[str]], List[List[str]]]:
    """Run DDP inference and return (true_labels, pred_labels) as BIO strings.

    Only positions with label != -100 are included.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, max_len, num_labels)

            # Gather across all processes. Requires equal tensor sizes.
            # Due to padding="max_length" in collator, all processes have the same sequence length.
            gathered_logits = accelerator.gather_for_metrics(logits)
            gathered_labels = accelerator.gather_for_metrics(labels)

            all_logits.append(gathered_logits.cpu().numpy())
            all_labels.append(gathered_labels.cpu().numpy())

    # Concatenate all steps
    if len(all_logits) > 0:
        all_logits_np = np.concatenate(all_logits, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)
        predictions = np.argmax(all_logits_np, axis=-1)
    else:
        predictions = np.empty((0, 0))
        all_labels_np = np.empty((0, 0))

    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []

    for pred_seq, label_seq in zip(predictions, all_labels_np):
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


def bio_tags_to_entities(tokens: List[str], tags: List[str]) -> List[dict]:
    """Convert a sequence of word tokens and corresponding BIO tags to entity dicts."""
    entities = []
    current_entity = None
    entity_idx = 0

    for idx, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "id": f"T{entity_idx}",
                "label": tag[2:],
                "tokens": [token],
            }
            entity_idx += 1
        elif tag.startswith("I-"):
            etype = tag[2:]
            if current_entity and current_entity["label"] == etype:
                current_entity["tokens"].append(token)
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "id": f"T{entity_idx}",
                    "label": etype,
                    "tokens": [token],
                }
                entity_idx += 1
        else:  # "O"
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    formatted_entities = []
    for ent in entities:
        term = " ".join(ent["tokens"])
        formatted_entities.append({
            "id": ent["id"],
            "label": ent["label"],
            "term": term,
        })
    return formatted_entities


def reorder_gathered_predictions(gathered_list: list, num_samples: int, num_processes: int) -> list:
    """Reorder a list gathered by accelerator.gather_for_metrics back to original dataset order."""
    if num_processes <= 1:
        return gathered_list

    import math
    k = math.ceil(num_samples / num_processes)

    # Pre-allocate reconstructed list
    reconstructed = [None] * num_samples

    for r in range(num_processes):
        for j in range(k):
            concat_idx = r * k + j
            orig_idx = j * num_processes + r
            if orig_idx < num_samples and concat_idx < len(gathered_list):
                reconstructed[orig_idx] = gathered_list[concat_idx]

    return reconstructed


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
        "--predictions",
        type=str,
        default=None,
        help="Path to save the model prediction outputs in JSONL format (similar to NER_finance).",
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

    # Initialize Accelerator
    accelerator = Accelerator()

    # Only print logs from the main process
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # ── 1. Load config ───────────────────────────────────────────────
    logger.info("Loading config from: %s", args.config)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    max_seq_length = cfg["model"].get("max_seq_length", 256)

    # ── 2. Load model & tokenizer from checkpoint ────────────────────
    logger.info("Loading model from checkpoint: %s", args.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # ── 3. Build test dataset ────────────────────────────────────────
    if args.split == "test":
        file_paths = [f["path"] for f in cfg["data"]["test_files"]]
    else:
        file_paths = [f["path"] for f in cfg["data"]["eval_files"]]

    logger.info("Loading %s dataset …", args.split)
    dataset = FIREBIODataset(file_paths, tokenizer, max_length=max_seq_length)
    logger.info("Loaded %d samples.", len(dataset))

    # ── 4. Build DataLoader ──────────────────────────────────────────
    # CRITICAL: use padding="max_length" to ensure identical shapes across GPUs
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_seq_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # Prepare model and dataloader with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    # ── 5. Run prediction ────────────────────────────────────────────
    logger.info("Running inference …")
    true_labels, pred_labels = predict(model, dataloader, accelerator)

    # ── 6. Compute metrics and save (only on main process) ───────────
    if accelerator.is_main_process:
        # Reorder gathered predictions back to the original dataset order
        true_labels = reorder_gathered_predictions(
            true_labels, len(dataset), accelerator.num_processes
        )
        pred_labels = reorder_gathered_predictions(
            pred_labels, len(dataset), accelerator.num_processes
        )
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

        # ── 8. Save raw predictions (JSONL format, similar to NER_finance) ──
        pred_out_path = args.predictions
        if not pred_out_path and args.output:
            # Auto-derive predictions path if only --output is provided
            base, ext = os.path.splitext(args.output)
            pred_out_path = f"{base}_predictions.jsonl"

        if pred_out_path:
            os.makedirs(os.path.dirname(pred_out_path) or ".", exist_ok=True)
            with open(pred_out_path, "w", encoding="utf-8") as f:
                for idx in range(len(dataset)):
                    tokens = dataset.samples[idx]["tokens"]
                    tags = pred_labels[idx]
                    entities = bio_tags_to_entities(tokens, tags)
                    f.write(json.dumps({"entities": entities}, ensure_ascii=False) + "\n")
            logger.info("Predictions saved to: %s", pred_out_path)


if __name__ == "__main__":
    main()

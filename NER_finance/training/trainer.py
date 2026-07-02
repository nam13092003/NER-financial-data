# -*- coding: utf-8 -*-
"""Custom Trainer that handles Unsloth-specific quirks in multi-GPU training.

``NativeSafeTrainer`` overrides ``training_step`` and ``prediction_step``
to isolate the training loop from known Unsloth compatibility issues while
preserving all standard HuggingFace ``Trainer`` functionality.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from transformers import Trainer


class NativeSafeTrainer(Trainer):
    """HuggingFace ``Trainer`` subclass safe for Unsloth + multi-GPU use.

    Overrides:

    * :meth:`training_step` — adds DDP-safe loss averaging and
      explicit ``accelerator.backward()``.
    * :meth:`prediction_step` — handles ``EmptyLogits`` from Unsloth's
      generation path and enforces type-safe loss tensor handling.

    All other ``Trainer`` behaviour (LR scheduling, checkpointing,
    logging, gradient accumulation) is inherited unchanged.
    """

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self,
        model: Any,
        inputs: Dict[str, Any],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # Average across GPUs in DDP
        if self.args.n_gpu > 1 and hasattr(loss, "mean"):
            loss = loss.mean()

        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    # ------------------------------------------------------------------
    # Prediction step
    # ------------------------------------------------------------------

    def prediction_step(
        self,
        model: Any,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        # Type-safe loss normalisation
        loss = self._safe_detach_loss(loss)

        if prediction_loss_only:
            return (loss, None, None)

        # Handle Unsloth's EmptyLogits sentinel
        logits = getattr(outputs, "logits", None)
        if logits is None or type(logits).__name__ == "EmptyLogits":
            clean_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            with torch.no_grad():
                logits = model(**clean_inputs).logits

        if logits is None:
            return (loss, None, inputs.get("labels"))

        return (loss, logits.detach(), inputs.get("labels"))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_detach_loss(loss: Any) -> Optional[torch.Tensor]:
        """Ensure loss is a detached scalar ``torch.Tensor`` or ``None``."""
        if loss is None:
            return None
        if isinstance(loss, torch.Tensor):
            return loss.mean().detach() if loss.dim() > 0 else loss.detach()
        # Scalar Python float/int — wrap in tensor
        return torch.tensor(float(loss), dtype=torch.float32)

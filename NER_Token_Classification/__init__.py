# -*- coding: utf-8 -*-
"""NER Token Classification — BIO-based Encoder Fine-tuning.

This package implements financial NER as a token classification task using
the BIO tagging scheme. A pre-trained encoder (e.g., XLM-RoBERTa) is used
as the backbone, with a linear classification head on top.

Subword-to-word alignment uses **first-subword pooling**: only the first
subword token of each word receives the BIO label; remaining subwords are
masked with -100 so they do not contribute to the loss.
"""

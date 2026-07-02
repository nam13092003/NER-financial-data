"""Data sub-package: schemas, adapter, and registry."""

from .adapter import FireFormatAdapter
from .registry import DatasetRegistry
from .schemas import EntitySpan, StandardizedDocument

__all__ = [
    "FireFormatAdapter",
    "DatasetRegistry",
    "EntitySpan",
    "StandardizedDocument",
]

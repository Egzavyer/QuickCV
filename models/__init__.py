"""Detection models and decision logic for the hybrid pipeline.

The torch/ultralytics-backed ``Model`` is imported lazily so that the pure
decision and geometry helpers can be used (and unit-tested) without pulling in
heavy inference dependencies.
"""

from __future__ import annotations

from typing import Any

from .decision import SIMILAR_CLASS_PAIRS, choose_final_detection
from .geometry import get_area, pad_box

__all__ = [
    "Model",
    "PredictionResult",
    "SIMILAR_CLASS_PAIRS",
    "choose_final_detection",
    "get_area",
    "pad_box",
]


def __getattr__(name: str) -> Any:
    if name in ("Model", "PredictionResult"):
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

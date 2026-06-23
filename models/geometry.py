"""Pure 2D geometry helpers shared across the detection pipeline."""

from __future__ import annotations

from collections.abc import Sequence


def get_area(box: Sequence[float]) -> float:
    """Return the area of an ``[x1, y1, x2, y2]`` box, clamped at zero."""
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(0.0, float(box[3]) - float(box[1]))
    return width * height


def pad_box(
    box: Sequence[float],
    image_width: int,
    image_height: int,
    min_pad: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    """Expand a box by ``max(min_pad, pad_ratio * longest_side)`` and clip to bounds.

    Returns the padded ``(x1, y1, x2, y2)`` as integers clamped to the image.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad = max(min_pad, int(round(max(box_w, box_h) * pad_ratio)))

    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(image_width, x2 + pad),
        min(image_height, y2 + pad),
    )

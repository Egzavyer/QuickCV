"""Conservative decision logic for reconciling stage-1 and stage-2 detections.

This module is intentionally free of heavy dependencies (no torch / OpenCV) so
the core relabeling rules can be unit-tested in isolation.
"""

from __future__ import annotations

from typing import Any

# Visually similar class pairs that require stricter evidence before relabeling.
SIMILAR_CLASS_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("Cyclist", "Pedestrian"),
        ("Pedestrian", "Cyclist"),
    }
)


def choose_final_detection(
    stage1_class: str,
    stage1_conf: float,
    best: dict[str, Any] | None,
    similar_min_conf: float,
    similar_min_delta: float,
    general_min_conf: float,
    general_min_delta: float,
) -> tuple[str, float, str, str | None]:
    """Decide the final label/confidence for an escalated detection.

    Returns ``(final_class, final_conf, reason, stage2_class)``. The stage-1
    prediction is only overridden when the stage-2 model is both sufficiently
    confident and sufficiently more confident than stage 1, with a stricter bar
    for visually similar classes.
    """
    if best is None:
        return stage1_class, stage1_conf, "stage2_no_detection", None

    stage2_class = str(best["class_name"])
    stage2_conf = float(best["confidence"])
    delta = stage2_conf - stage1_conf

    if stage2_class == stage1_class:
        return stage2_class, stage2_conf, "agree", stage2_class

    if (stage1_class, stage2_class) in SIMILAR_CLASS_PAIRS:
        if stage2_conf >= similar_min_conf and delta >= similar_min_delta:
            return stage2_class, stage2_conf, "similar_class_relabel", stage2_class
        return stage1_class, stage1_conf, "kept_stage1_similar", stage2_class

    if stage2_conf >= general_min_conf and delta >= general_min_delta:
        return stage2_class, stage2_conf, "relabel", stage2_class

    return stage1_class, stage1_conf, "kept_stage1", stage2_class

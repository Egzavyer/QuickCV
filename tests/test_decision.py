"""Unit tests for the conservative stage-1/stage-2 relabeling logic."""

from __future__ import annotations

import pytest

from models.decision import choose_final_detection

GATES = {
    "similar_min_conf": 0.85,
    "similar_min_delta": 0.20,
    "general_min_conf": 0.75,
    "general_min_delta": 0.15,
}


def detection(class_name: str, confidence: float) -> dict[str, object]:
    return {"class_name": class_name, "confidence": confidence}


def test_no_stage2_detection_keeps_stage1():
    final_class, final_conf, reason, stage2_class = choose_final_detection(
        "Car", 0.42, None, **GATES
    )
    assert (final_class, reason, stage2_class) == ("Car", "stage2_no_detection", None)
    assert final_conf == 0.42


def test_agreement_takes_stage2_confidence():
    final_class, final_conf, reason, _ = choose_final_detection(
        "Car", 0.40, detection("Car", 0.91), **GATES
    )
    assert (final_class, reason) == ("Car", "agree")
    assert final_conf == pytest.approx(0.91)


def test_similar_class_relabel_accepted_when_confident():
    final_class, _, reason, _ = choose_final_detection(
        "Cyclist", 0.30, detection("Pedestrian", 0.90), **GATES
    )
    assert (final_class, reason) == ("Pedestrian", "similar_class_relabel")


def test_similar_class_relabel_rejected_below_confidence():
    # Confident delta but stage-2 confidence below the stricter similar bar.
    final_class, final_conf, reason, stage2_class = choose_final_detection(
        "Cyclist", 0.30, detection("Pedestrian", 0.80), **GATES
    )
    assert (final_class, reason, stage2_class) == (
        "Cyclist",
        "kept_stage1_similar",
        "Pedestrian",
    )
    assert final_conf == 0.30


def test_similar_class_relabel_rejected_below_delta():
    final_class, _, reason, _ = choose_final_detection(
        "Pedestrian", 0.70, detection("Cyclist", 0.86), **GATES
    )
    assert (final_class, reason) == ("Pedestrian", "kept_stage1_similar")


def test_general_relabel_accepted():
    final_class, _, reason, _ = choose_final_detection(
        "Car", 0.40, detection("Truck", 0.80), **GATES
    )
    assert (final_class, reason) == ("Truck", "relabel")


def test_general_relabel_rejected_keeps_stage1():
    final_class, _, reason, _ = choose_final_detection(
        "Car", 0.70, detection("Truck", 0.74), **GATES
    )
    assert (final_class, reason) == ("Car", "kept_stage1")

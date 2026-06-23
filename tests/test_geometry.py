"""Unit tests for the shared 2D geometry helpers."""

from __future__ import annotations

import pytest

from models.geometry import get_area, pad_box


def test_get_area_basic():
    assert get_area([0, 0, 10, 5]) == pytest.approx(50.0)


def test_get_area_clamped_to_zero_for_inverted_box():
    assert get_area([10, 10, 0, 0]) == 0.0


def test_pad_box_uses_min_pad_for_small_box():
    # Longest side is 10, ratio padding = 2, so min_pad (8) wins.
    assert pad_box([20, 20, 30, 30], 200, 200, min_pad=8, pad_ratio=0.2) == (
        12,
        12,
        38,
        38,
    )


def test_pad_box_uses_ratio_for_large_box():
    # Longest side is 100, ratio padding = 25, beating min_pad (8).
    assert pad_box([100, 100, 200, 200], 1000, 1000, min_pad=8, pad_ratio=0.25) == (
        75,
        75,
        225,
        225,
    )


def test_pad_box_clips_to_image_bounds():
    assert pad_box([5, 5, 95, 95], 100, 100, min_pad=32, pad_ratio=0.0) == (
        0,
        0,
        100,
        100,
    )

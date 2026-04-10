from __future__ import annotations

import pytest
import torch

from flapping_bot.physics.qsm_wang2016 import WingGeometry
from flapping_bot.physics.wing_equivalent_ac import compute_area_weighted_quarter_chord_link_points


def test_area_weighted_quarter_chord_returns_expected_link_frame_point() -> None:
    geom = WingGeometry.from_input(
        {
            "R": 2.0,
            "N": 2,
            "c": torch.tensor([1.0, 1.0]),
            "dhat": 0.0,
            "aspect_ratio": 2.0,
        },
        device="cpu",
        dtype=torch.float32,
    )

    points = compute_area_weighted_quarter_chord_link_points(geom)

    expected_left = torch.tensor([-0.25, 1.0, 0.0], dtype=torch.float32)
    expected_right = torch.tensor([-0.25, -1.0, 0.0], dtype=torch.float32)
    expected = torch.stack((expected_left, expected_right), dim=0)
    torch.testing.assert_close(points, expected)


def test_area_weighted_quarter_chord_uses_strip_area_weights() -> None:
    geom = WingGeometry.from_input(
        {
            "R": 2.0,
            "N": 2,
            "c": torch.tensor([1.0, 3.0]),
            "dhat": 0.0,
            "aspect_ratio": 1.0,
        },
        device="cpu",
        dtype=torch.float32,
    )

    points = compute_area_weighted_quarter_chord_link_points(geom)

    expected_span = (0.5 * 1.0 + 1.5 * 3.0) / (1.0 + 3.0)
    expected_chord = -0.25 * (1.0 * 1.0 + 3.0 * 3.0) / (1.0 + 3.0)
    assert float(points[0, 0]) == expected_chord
    assert float(points[0, 1]) == expected_span
    assert float(points[1, 0]) == expected_chord
    assert float(points[1, 1]) == -expected_span


def test_area_weighted_quarter_chord_respects_pitch_axis_offset() -> None:
    geom = WingGeometry.from_input(
        {
            "R": 2.0,
            "N": 2,
            "c": torch.tensor([2.0, 2.0]),
            "dhat": torch.tensor([0.10, 0.10]),
            "aspect_ratio": 1.0,
        },
        device="cpu",
        dtype=torch.float32,
    )

    points = compute_area_weighted_quarter_chord_link_points(geom)

    expected_chord = (0.10 - 0.25) * 2.0
    assert float(points[0, 0]) == pytest.approx(expected_chord)
    assert float(points[1, 0]) == pytest.approx(expected_chord)

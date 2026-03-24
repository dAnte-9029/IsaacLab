from __future__ import annotations

import math

import pytest
import torch

from flapping_bot.direct.flapping_bot.path_tracking_env import _compute_alignment_error


def test_alignment_error_uses_ground_track_against_path_tangent() -> None:
    tangent_xy = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    ground_vel_xy = torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float32)

    align_error = _compute_alignment_error(ground_vel_xy=ground_vel_xy, tangent_xy=tangent_xy)

    assert align_error[0].item() == pytest.approx(0.0)
    assert align_error[1].item() == pytest.approx(0.5 * math.pi, rel=1.0e-5)

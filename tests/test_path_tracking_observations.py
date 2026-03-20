from __future__ import annotations

import torch

from flapping_bot.direct.flapping_bot.path_tracking_env import _build_preview_observation, _compute_alignment_error


def test_preview_observation_has_five_xyz_points():
    query = {
        "preview_points_body_xyz": torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
        "lateral_error_m": torch.tensor([0.2]),
        "height_error_m": torch.tensor([0.5]),
        "curvature_m_inv": torch.tensor([0.01]),
        "progress_s": torch.tensor([3.0]),
    }
    obs = _build_preview_observation(query)
    assert obs.shape[-1] >= 15


def test_alignment_error_is_zero_for_aligned_track():
    tangent_xy = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    ground_vel_xy = torch.tensor([[5.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    align_error = _compute_alignment_error(tangent_xy=tangent_xy, ground_vel_xy=ground_vel_xy)
    assert torch.allclose(align_error, torch.zeros_like(align_error))

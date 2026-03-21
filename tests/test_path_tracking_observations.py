from __future__ import annotations

import torch

from flapping_bot.direct.flapping_bot.path_tracking_env import _build_preview_observation, _compute_alignment_error


def test_preview_observation_has_fixed_path_tracking_contract():
    query = {
        "closest_point_body_xyz": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        "tangent_xy": torch.tensor([[0.6, 0.8]], dtype=torch.float32),
        "preview_points_body_xyz": torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
        "lateral_error_m": torch.tensor([0.2]),
        "height_error_m": torch.tensor([0.5]),
        "align_error_rad": torch.tensor([0.1]),
        "curvature_m_inv": torch.tensor([0.01]),
        "progress_s": torch.tensor([3.0]),
        "previous_action": torch.tensor([[0.4, -0.3, 0.2, -0.1]], dtype=torch.float32),
    }
    obs = _build_preview_observation(query)
    expected = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                0.6,
                0.8,
                0.01,
                0.2,
                0.5,
                0.1,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                0.4,
                -0.3,
                0.2,
                -0.1,
            ]
        ],
        dtype=torch.float32,
    )
    assert obs.shape == (1, 28)
    assert torch.allclose(obs, expected)


def test_alignment_error_is_zero_for_aligned_track():
    tangent_xy = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    ground_vel_xy = torch.tensor([[5.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    align_error = _compute_alignment_error(tangent_xy=tangent_xy, ground_vel_xy=ground_vel_xy)
    assert torch.allclose(align_error, torch.zeros_like(align_error))

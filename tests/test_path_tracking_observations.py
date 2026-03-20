from __future__ import annotations

import torch

from flapping_bot.direct.flapping_bot.path_tracking_env import _build_preview_observation


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

"""Generic path-tracking environment skeleton."""

from __future__ import annotations

import torch


class FlappingBotPathTrackingEnvCfg:
    """Configuration placeholder for the path-tracking environment."""


class FlappingBotPathTrackingEnv:
    """Minimal path-tracking environment placeholder."""


def _build_preview_observation(query: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build a flat preview-based observation vector."""
    preview = query["preview_points_body_xyz"].reshape(query["preview_points_body_xyz"].shape[0], -1)
    extras = torch.stack(
        [
            query["lateral_error_m"],
            query["height_error_m"],
            query["curvature_m_inv"],
            query["progress_s"],
        ],
        dim=-1,
    )
    return torch.cat((preview, extras), dim=-1)

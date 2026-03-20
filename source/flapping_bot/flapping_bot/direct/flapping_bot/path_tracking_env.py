"""Generic path-tracking environment skeleton."""

from __future__ import annotations

import torch

PATH_TRACKING_RUNTIME_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from isaaclab.utils import configclass

    from .straight_flight_env import FlappingBotStraightFlightDeLaurierPureRLEnvCfg, FlappingBotStraightFlightEnv
except ModuleNotFoundError as exc:
    PATH_TRACKING_RUNTIME_IMPORT_ERROR = exc
    PATH_TRACKING_RUNTIME_AVAILABLE = False

    class FlappingBotPathTrackingEnvCfg:
        """Headless fallback config placeholder for path-tracking tests."""


    class FlappingBotPathTrackingEnv:
        """Headless fallback env placeholder for path-tracking tests."""

else:
    PATH_TRACKING_RUNTIME_AVAILABLE = True

    @configclass
    class FlappingBotPathTrackingEnvCfg(FlappingBotStraightFlightDeLaurierPureRLEnvCfg):
        """Configuration skeleton for the generic path-tracking environment."""


    class FlappingBotPathTrackingEnv(FlappingBotStraightFlightEnv):
        """Generic path-tracking environment skeleton."""


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


def _compute_tracking_reward(
    *,
    lateral_error: torch.Tensor,
    height_error: torch.Tensor,
    align_error: torch.Tensor,
    delta_s: torch.Tensor,
    airspeed: torch.Tensor,
    action: torch.Tensor,
    action_delta: torch.Tensor,
    tilt: torch.Tensor,
    ang_rate: torch.Tensor,
) -> torch.Tensor:
    """Compute a minimal tracking-plus-progress reward."""
    del action, action_delta, tilt, ang_rate
    low_speed_penalty = 0.2 * torch.clamp(4.0 - airspeed, min=0.0)
    return delta_s - 0.2 * lateral_error - 0.2 * height_error - 0.1 * align_error - low_speed_penalty


def _teacher_recovery_mask(**kwargs) -> torch.Tensor:
    """Thin wrapper placeholder for recovery-teacher masking."""
    from ...px4_like.rl_training_utils import compute_recovery_teacher_mask

    return compute_recovery_teacher_mask(**kwargs)

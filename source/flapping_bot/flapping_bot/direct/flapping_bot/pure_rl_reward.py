"""Pure Tensor reward and termination contracts for PureRL curriculum 1."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

Tensor = torch.Tensor


@dataclass
class PureRLRewardConfig:
    """Hydra-restorable scales and weights for straight-flight curriculum 1."""

    cross_track_scale_m: float = 1.5
    height_scale_m: float = 1.0
    progress_speed_scale_mps: float = 3.0
    cross_track_speed_scale_mps: float = 1.0
    vertical_speed_scale_mps: float = 1.0
    roll_scale_rad: float = math.radians(20.0)
    angular_rate_scale_rad_s: float = 3.0
    pitch_envelope_rad: float = math.radians(30.0)
    pitch_excess_scale_rad: float = math.radians(15.0)
    maximum_flap_frequency_hz: float = 5.0
    tail_action_limit_threshold: float = 0.8

    path_reward_weight: float = 0.40
    progress_reward_weight: float = 0.20
    velocity_reward_weight: float = 0.15
    roll_reward_weight: float = 0.15
    angular_rate_reward_weight: float = 0.10
    pitch_envelope_penalty_weight: float = 0.03
    flap_penalty_weight: float = 0.04
    frequency_slew_scale_hz_per_s: float = 2.0
    frequency_slew_penalty_weight: float = 0.01
    requested_frequency_action_delta_penalty_weight: float = 0.0
    requested_frequency_action_delta_penalty_mode: str = "squared"
    requested_applied_frequency_action_gap_penalty_weight: float = 0.0
    tail_action_delta_penalty_weight: float = 0.01
    tail_action_limit_penalty_weight: float = 0.02


@dataclass(frozen=True)
class PureRLRewardTerms:
    """Per-environment unweighted terms and their weighted total."""

    path_reward: Tensor
    progress_reward: Tensor
    velocity_reward: Tensor
    roll_reward: Tensor
    angular_rate_reward: Tensor
    pitch_envelope_penalty: Tensor
    flap_penalty: Tensor
    frequency_slew_penalty: Tensor
    requested_frequency_action_delta_penalty: Tensor
    requested_applied_frequency_action_gap_penalty: Tensor
    tail_action_delta_penalty: Tensor
    tail_action_limit_penalty: Tensor
    total_reward: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        """Return stable telemetry names without copying tensors."""

        return {
            "path": self.path_reward,
            "progress": self.progress_reward,
            "velocity": self.velocity_reward,
            "roll": self.roll_reward,
            "angular_rate": self.angular_rate_reward,
            "pitch_envelope_penalty": self.pitch_envelope_penalty,
            "flap_penalty": self.flap_penalty,
            "frequency_slew_penalty": self.frequency_slew_penalty,
            "requested_frequency_action_delta_penalty": self.requested_frequency_action_delta_penalty,
            "requested_applied_frequency_action_gap_penalty": (
                self.requested_applied_frequency_action_gap_penalty
            ),
            "tail_action_delta_penalty": self.tail_action_delta_penalty,
            "tail_action_limit_penalty": self.tail_action_limit_penalty,
            "total": self.total_reward,
        }


@dataclass(frozen=True)
class PureRLTerminationConfig:
    """Termination thresholds for straight-flight curriculum 1."""

    ground_height_m: float = 0.05
    maximum_tilt_rad: float = math.radians(75.0)
    maximum_cross_track_error_m: float = 3.0
    maximum_height_error_m: float = 3.0


@dataclass(frozen=True)
class PureRLTerminationTerms:
    """Per-environment termination causes and their union."""

    ground: Tensor
    tilt: Tensor
    cross_track: Tensor
    height_error: Tensor
    terminated: Tensor
    tilt_rad: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        """Return stable telemetry names without copying tensors."""

        return {
            "ground": self.ground,
            "tilt": self.tilt,
            "cross_track": self.cross_track,
            "height_error": self.height_error,
            "terminated": self.terminated,
        }


@dataclass(frozen=True)
class PureRLSpatialTerminationTerms:
    """Per-environment spatial termination causes and their union."""

    ground: Tensor
    tilt: Tensor
    cross_track: Tensor
    height_error: Tensor
    roll_limit: Tensor
    terminated: Tensor
    tilt_rad: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        """Return stable telemetry names without copying tensors."""

        return {
            "ground": self.ground,
            "tilt": self.tilt,
            "cross_track": self.cross_track,
            "height_error": self.height_error,
            "roll_limit": self.roll_limit,
            "terminated": self.terminated,
        }


PURE_RL_CURRICULUM1_REWARD_CONFIG = PureRLRewardConfig()
PURE_RL_CURRICULUM1_TERMINATION_CONFIG = PureRLTerminationConfig()


def compute_pure_rl_reward_terms(
    *,
    cross_track_error_m: Tensor,
    height_error_m: Tensor,
    along_track_velocity_mps: Tensor,
    cross_track_velocity_mps: Tensor,
    vertical_velocity_mps: Tensor,
    roll_rad: Tensor,
    pitch_rad: Tensor,
    angular_velocity_body_rad_s: Tensor,
    actual_flap_frequency_hz: Tensor,
    frequency_slew_hz_per_s: Tensor,
    applied_action: Tensor,
    previous_applied_action: Tensor,
    requested_frequency_action: Tensor | None = None,
    previous_requested_frequency_action: Tensor | None = None,
    config: PureRLRewardConfig = PURE_RL_CURRICULUM1_REWARD_CONFIG,
) -> PureRLRewardTerms:
    """Compute C1 terms through the three-dimensional path reward API."""

    return compute_pure_rl_path_reward_terms(
        cross_track_error_m=cross_track_error_m,
        height_error_m=height_error_m,
        tangent_velocity_mps=along_track_velocity_mps,
        lateral_normal_velocity_mps=cross_track_velocity_mps,
        vertical_normal_velocity_mps=vertical_velocity_mps,
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        angular_velocity_body_rad_s=angular_velocity_body_rad_s,
        actual_flap_frequency_hz=actual_flap_frequency_hz,
        frequency_slew_hz_per_s=frequency_slew_hz_per_s,
        applied_action=applied_action,
        previous_applied_action=previous_applied_action,
        requested_frequency_action=requested_frequency_action,
        previous_requested_frequency_action=previous_requested_frequency_action,
        config=config,
    )


def compute_pure_rl_path_reward_terms(
    *,
    cross_track_error_m: Tensor,
    height_error_m: Tensor,
    tangent_velocity_mps: Tensor,
    lateral_normal_velocity_mps: Tensor,
    vertical_normal_velocity_mps: Tensor,
    roll_rad: Tensor,
    pitch_rad: Tensor,
    angular_velocity_body_rad_s: Tensor,
    actual_flap_frequency_hz: Tensor,
    frequency_slew_hz_per_s: Tensor,
    applied_action: Tensor,
    previous_applied_action: Tensor,
    requested_frequency_action: Tensor | None = None,
    previous_requested_frequency_action: Tensor | None = None,
    config: PureRLRewardConfig = PURE_RL_CURRICULUM1_REWARD_CONFIG,
) -> PureRLRewardTerms:
    """Compute dense reward terms in an orthonormal three-dimensional path basis."""

    vectors = {
        "cross_track_error_m": cross_track_error_m,
        "height_error_m": height_error_m,
        "tangent_velocity_mps": tangent_velocity_mps,
        "lateral_normal_velocity_mps": lateral_normal_velocity_mps,
        "vertical_normal_velocity_mps": vertical_normal_velocity_mps,
        "roll_rad": roll_rad,
        "pitch_rad": pitch_rad,
        "actual_flap_frequency_hz": actual_flap_frequency_hz,
        "frequency_slew_hz_per_s": frequency_slew_hz_per_s,
    }
    reference = cross_track_error_m
    for name, value in vectors.items():
        _validate_vector(name, value)
        _validate_aligned(reference, name, value)
    _validate_matrix("angular_velocity_body_rad_s", angular_velocity_body_rad_s, columns=3)
    _validate_matrix("applied_action", applied_action, columns=4)
    _validate_matrix("previous_applied_action", previous_applied_action, columns=4)
    for name, value in (
        ("angular_velocity_body_rad_s", angular_velocity_body_rad_s),
        ("applied_action", applied_action),
        ("previous_applied_action", previous_applied_action),
    ):
        _validate_aligned(reference, name, value)
    if bool(torch.any(actual_flap_frequency_hz < 0.0)):
        raise ValueError("actual_flap_frequency_hz must be non-negative.")
    _validate_reward_config(config)
    if (requested_frequency_action is None) != (previous_requested_frequency_action is None):
        raise ValueError(
            "requested_frequency_action and previous_requested_frequency_action must be provided together."
        )
    if requested_frequency_action is None:
        if (
            config.requested_frequency_action_delta_penalty_weight > 0.0
            or config.requested_applied_frequency_action_gap_penalty_weight > 0.0
        ):
            raise ValueError(
                "Positive requested-frequency penalty weight requires requested frequency actions."
            )
        requested_frequency_action_delta_penalty = torch.zeros_like(reference)
        requested_applied_frequency_action_gap_penalty = torch.zeros_like(reference)
    else:
        assert previous_requested_frequency_action is not None
        _validate_vector("requested_frequency_action", requested_frequency_action)
        _validate_vector(
            "previous_requested_frequency_action",
            previous_requested_frequency_action,
        )
        _validate_aligned(reference, "requested_frequency_action", requested_frequency_action)
        _validate_aligned(
            reference,
            "previous_requested_frequency_action",
            previous_requested_frequency_action,
        )
        for name, value in (
            ("requested_frequency_action", requested_frequency_action),
            ("previous_requested_frequency_action", previous_requested_frequency_action),
        ):
            if bool(torch.any(torch.abs(value) > 1.0)):
                raise ValueError(f"{name} must lie in [-1, 1].")
        normalized_requested_frequency_delta = 0.5 * (
            requested_frequency_action - previous_requested_frequency_action
        )
        if config.requested_frequency_action_delta_penalty_mode == "absolute":
            requested_frequency_action_delta_penalty = torch.abs(
                normalized_requested_frequency_delta
            )
        else:
            requested_frequency_action_delta_penalty = torch.square(
                normalized_requested_frequency_delta
            )
        requested_applied_frequency_action_gap_penalty = torch.abs(
            requested_frequency_action - applied_action[:, 0]
        )

    cross_track_reward = torch.exp(-torch.square(cross_track_error_m / config.cross_track_scale_m))
    height_reward = torch.exp(-torch.square(height_error_m / config.height_scale_m))
    path_reward = 0.5 * (cross_track_reward + height_reward)
    progress_reward = torch.tanh(tangent_velocity_mps / config.progress_speed_scale_mps)
    lateral_normal_velocity_reward = torch.exp(
        -torch.square(lateral_normal_velocity_mps / config.cross_track_speed_scale_mps)
    )
    vertical_normal_velocity_reward = torch.exp(
        -torch.square(vertical_normal_velocity_mps / config.vertical_speed_scale_mps)
    )
    velocity_reward = 0.5 * (lateral_normal_velocity_reward + vertical_normal_velocity_reward)
    roll_reward = torch.exp(-torch.square(roll_rad / config.roll_scale_rad))
    normalized_angular_rate = angular_velocity_body_rad_s / config.angular_rate_scale_rad_s
    angular_rate_reward = torch.exp(-torch.sum(torch.square(normalized_angular_rate), dim=1))

    pitch_excess_rad = torch.relu(torch.abs(pitch_rad) - config.pitch_envelope_rad)
    pitch_envelope_penalty = torch.tanh(torch.square(pitch_excess_rad / config.pitch_excess_scale_rad))
    flap_penalty = torch.pow(actual_flap_frequency_hz / config.maximum_flap_frequency_hz, 3.0)
    normalized_action_delta = 0.5 * (applied_action - previous_applied_action)
    frequency_slew_penalty = torch.square(
        frequency_slew_hz_per_s / config.frequency_slew_scale_hz_per_s
    )
    tail_action_delta_penalty = torch.mean(torch.square(normalized_action_delta[:, 1:4]), dim=1)
    tail_limit_margin = 1.0 - config.tail_action_limit_threshold
    tail_limit_excess = torch.relu(torch.abs(applied_action[:, 1:4]) - config.tail_action_limit_threshold)
    tail_action_limit_penalty = torch.mean(torch.square(tail_limit_excess / tail_limit_margin), dim=1)

    total_reward = (
        config.path_reward_weight * path_reward
        + config.progress_reward_weight * progress_reward
        + config.velocity_reward_weight * velocity_reward
        + config.roll_reward_weight * roll_reward
        + config.angular_rate_reward_weight * angular_rate_reward
        - config.pitch_envelope_penalty_weight * pitch_envelope_penalty
        - config.flap_penalty_weight * flap_penalty
        - config.frequency_slew_penalty_weight * frequency_slew_penalty
        - config.requested_frequency_action_delta_penalty_weight
        * requested_frequency_action_delta_penalty
        - config.requested_applied_frequency_action_gap_penalty_weight
        * requested_applied_frequency_action_gap_penalty
        - config.tail_action_delta_penalty_weight * tail_action_delta_penalty
        - config.tail_action_limit_penalty_weight * tail_action_limit_penalty
    )
    return PureRLRewardTerms(
        path_reward=path_reward,
        progress_reward=progress_reward,
        velocity_reward=velocity_reward,
        roll_reward=roll_reward,
        angular_rate_reward=angular_rate_reward,
        pitch_envelope_penalty=pitch_envelope_penalty,
        flap_penalty=flap_penalty,
        frequency_slew_penalty=frequency_slew_penalty,
        requested_frequency_action_delta_penalty=requested_frequency_action_delta_penalty,
        requested_applied_frequency_action_gap_penalty=(
            requested_applied_frequency_action_gap_penalty
        ),
        tail_action_delta_penalty=tail_action_delta_penalty,
        tail_action_limit_penalty=tail_action_limit_penalty,
        total_reward=total_reward,
    )


def compute_pure_rl_spatial_path_reward_terms(
    *,
    cross_track_error_m: Tensor,
    height_error_m: Tensor,
    tangent_velocity_mps: Tensor,
    lateral_normal_velocity_mps: Tensor,
    vertical_normal_velocity_mps: Tensor,
    roll_rad: Tensor,
    pitch_rad: Tensor,
    angular_velocity_body_rad_s: Tensor,
    actual_flap_frequency_hz: Tensor,
    frequency_slew_hz_per_s: Tensor,
    applied_action: Tensor,
    previous_applied_action: Tensor,
    turn_activity: Tensor,
    requested_frequency_action: Tensor | None = None,
    previous_requested_frequency_action: Tensor | None = None,
    config: PureRLRewardConfig = PURE_RL_CURRICULUM1_REWARD_CONFIG,
) -> PureRLRewardTerms:
    """Compute path reward terms with relaxed roll reward during spatial turns."""

    base = compute_pure_rl_path_reward_terms(
        cross_track_error_m=cross_track_error_m,
        height_error_m=height_error_m,
        tangent_velocity_mps=tangent_velocity_mps,
        lateral_normal_velocity_mps=lateral_normal_velocity_mps,
        vertical_normal_velocity_mps=vertical_normal_velocity_mps,
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        angular_velocity_body_rad_s=angular_velocity_body_rad_s,
        actual_flap_frequency_hz=actual_flap_frequency_hz,
        frequency_slew_hz_per_s=frequency_slew_hz_per_s,
        applied_action=applied_action,
        previous_applied_action=previous_applied_action,
        requested_frequency_action=requested_frequency_action,
        previous_requested_frequency_action=previous_requested_frequency_action,
        config=config,
    )
    _validate_vector("turn_activity", turn_activity)
    _validate_aligned(roll_rad, "turn_activity", turn_activity)
    if bool(torch.any((turn_activity < 0.0) | (turn_activity > 1.0))):
        raise ValueError("turn_activity must lie in [0, 1].")

    normalized_excess = torch.clamp(
        (torch.abs(roll_rad) - math.radians(25.0)) / math.radians(10.0),
        min=0.0,
        max=1.0,
    )
    active_turn_roll_reward = 1.0 - torch.square(normalized_excess)
    spatial_roll_reward = torch.lerp(base.roll_reward, active_turn_roll_reward, turn_activity)
    spatial_total_reward = base.total_reward + config.roll_reward_weight * (
        spatial_roll_reward - base.roll_reward
    )
    return replace(
        base,
        roll_reward=spatial_roll_reward,
        total_reward=spatial_total_reward,
    )


def compute_pure_rl_termination_terms(
    *,
    height_m: Tensor,
    cross_track_error_m: Tensor,
    height_error_m: Tensor,
    projected_gravity_body: Tensor,
    config: PureRLTerminationConfig = PURE_RL_CURRICULUM1_TERMINATION_CONFIG,
) -> PureRLTerminationTerms:
    """Compute route-relative termination causes with a robust tilt angle."""

    _validate_vector("height_m", height_m)
    _validate_vector("cross_track_error_m", cross_track_error_m)
    _validate_vector("height_error_m", height_error_m)
    _validate_matrix("projected_gravity_body", projected_gravity_body, columns=3)
    for name, value in (
        ("cross_track_error_m", cross_track_error_m),
        ("height_error_m", height_error_m),
        ("projected_gravity_body", projected_gravity_body),
    ):
        _validate_aligned(height_m, name, value)
    _validate_termination_config(config)

    gravity_norm = torch.linalg.vector_norm(projected_gravity_body, dim=1).clamp_min(
        torch.finfo(projected_gravity_body.dtype).eps
    )
    upright_cosine = torch.clamp(-projected_gravity_body[:, 2] / gravity_norm, min=-1.0, max=1.0)
    tilt_rad = torch.acos(upright_cosine)
    ground = height_m <= config.ground_height_m
    tilt = tilt_rad > config.maximum_tilt_rad
    cross_track = torch.abs(cross_track_error_m) > config.maximum_cross_track_error_m
    height_error = torch.abs(height_error_m) > config.maximum_height_error_m
    terminated = ground | tilt | cross_track | height_error
    return PureRLTerminationTerms(
        ground=ground,
        tilt=tilt,
        cross_track=cross_track,
        height_error=height_error,
        terminated=terminated,
        tilt_rad=tilt_rad,
    )


def compute_pure_rl_spatial_termination_terms(
    *,
    height_m: Tensor,
    cross_track_error_m: Tensor,
    height_error_m: Tensor,
    projected_gravity_body: Tensor,
    roll_rad: Tensor,
    maximum_abs_roll_rad: float = math.radians(35.0),
    config: PureRLTerminationConfig = PURE_RL_CURRICULUM1_TERMINATION_CONFIG,
) -> PureRLSpatialTerminationTerms:
    """Compute spatial termination causes with an absolute-roll limit."""

    base = compute_pure_rl_termination_terms(
        height_m=height_m,
        cross_track_error_m=cross_track_error_m,
        height_error_m=height_error_m,
        projected_gravity_body=projected_gravity_body,
        config=config,
    )
    _validate_vector("roll_rad", roll_rad)
    _validate_aligned(height_m, "roll_rad", roll_rad)
    if (
        not math.isfinite(float(maximum_abs_roll_rad))
        or maximum_abs_roll_rad <= 0.0
        or maximum_abs_roll_rad > math.pi
    ):
        raise ValueError("maximum_abs_roll_rad must be finite and lie in (0, pi].")

    roll_limit = torch.abs(roll_rad) >= maximum_abs_roll_rad
    return PureRLSpatialTerminationTerms(
        ground=base.ground,
        tilt=base.tilt,
        cross_track=base.cross_track,
        height_error=base.height_error,
        roll_limit=roll_limit,
        terminated=base.terminated | roll_limit,
        tilt_rad=base.tilt_rad,
    )


def _validate_reward_config(config: PureRLRewardConfig) -> None:
    if config.requested_frequency_action_delta_penalty_mode not in ("squared", "absolute"):
        raise ValueError(
            "requested_frequency_action_delta_penalty_mode must be 'squared' or 'absolute'."
        )
    positive = (
        config.cross_track_scale_m,
        config.height_scale_m,
        config.progress_speed_scale_mps,
        config.cross_track_speed_scale_mps,
        config.vertical_speed_scale_mps,
        config.roll_scale_rad,
        config.angular_rate_scale_rad_s,
        config.pitch_excess_scale_rad,
        config.maximum_flap_frequency_hz,
        config.frequency_slew_scale_hz_per_s,
    )
    if any((not math.isfinite(float(value))) or float(value) <= 0.0 for value in positive):
        raise ValueError("Reward scales and maximum flap frequency must be finite and positive.")
    if not math.isfinite(float(config.pitch_envelope_rad)) or config.pitch_envelope_rad < 0.0:
        raise ValueError("pitch_envelope_rad must be finite and non-negative.")
    if not 0.0 <= config.tail_action_limit_threshold < 1.0:
        raise ValueError("tail_action_limit_threshold must lie in [0, 1).")
    weights = (
        config.path_reward_weight,
        config.progress_reward_weight,
        config.velocity_reward_weight,
        config.roll_reward_weight,
        config.angular_rate_reward_weight,
        config.pitch_envelope_penalty_weight,
        config.flap_penalty_weight,
        config.frequency_slew_penalty_weight,
        config.requested_frequency_action_delta_penalty_weight,
        config.requested_applied_frequency_action_gap_penalty_weight,
        config.tail_action_delta_penalty_weight,
        config.tail_action_limit_penalty_weight,
    )
    if any((not math.isfinite(float(value))) or float(value) < 0.0 for value in weights):
        raise ValueError("Reward and penalty weights must be finite and non-negative.")


def _validate_termination_config(config: PureRLTerminationConfig) -> None:
    finite = (
        config.ground_height_m,
        config.maximum_tilt_rad,
        config.maximum_cross_track_error_m,
        config.maximum_height_error_m,
    )
    if any(not math.isfinite(float(value)) for value in finite):
        raise ValueError("Termination thresholds must be finite.")
    if not 0.0 < config.maximum_tilt_rad < math.pi:
        raise ValueError("maximum_tilt_rad must lie in (0, pi).")
    if config.maximum_cross_track_error_m <= 0.0 or config.maximum_height_error_m <= 0.0:
        raise ValueError("Path-error termination thresholds must be positive.")


def _validate_vector(name: str, value: Tensor) -> None:
    _validate_tensor(name, value)
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape (N,).")


def _validate_matrix(name: str, value: Tensor, *, columns: int) -> None:
    _validate_tensor(name, value)
    if value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}).")


def _validate_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor.")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating dtype.")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must be finite.")


def _validate_aligned(reference: Tensor, name: str, value: Tensor) -> None:
    if reference.shape[0] != value.shape[0]:
        raise ValueError(f"{name} must share the reference batch dimension.")
    if reference.device != value.device or reference.dtype != value.dtype:
        raise ValueError(f"{name} must share the reference device and dtype.")


__all__ = [
    "PURE_RL_CURRICULUM1_REWARD_CONFIG",
    "PURE_RL_CURRICULUM1_TERMINATION_CONFIG",
    "PureRLRewardConfig",
    "PureRLRewardTerms",
    "PureRLTerminationConfig",
    "PureRLTerminationTerms",
    "PureRLSpatialTerminationTerms",
    "compute_pure_rl_path_reward_terms",
    "compute_pure_rl_reward_terms",
    "compute_pure_rl_spatial_path_reward_terms",
    "compute_pure_rl_spatial_termination_terms",
    "compute_pure_rl_termination_terms",
]

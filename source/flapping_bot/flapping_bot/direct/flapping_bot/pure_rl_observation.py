"""Raw actor-observation contract for the measured PureRL task.

This module defines both the raw physical-unit contract and the fixed-scale
actor contract. Sensor imperfections and environment integration remain
separate stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

Tensor = torch.Tensor

PURE_RL_PREVIEW_TIMES_S: tuple[float, ...] = (0.12, 0.24, 0.36, 0.48, 0.60)


@dataclass(frozen=True)
class PureRLRawObservationLayout:
    """Fixed raw actor-observation dimensions and temporal coverage."""

    sensor_history_steps: int = 30
    sensor_frame_dim: int = 14
    action_history_steps: int = 30
    action_dim: int = 4
    preview_point_count: int = 5
    preview_point_dim: int = 3

    @property
    def sensor_history_dim(self) -> int:
        """Flattened sensor-history dimension."""

        return self.sensor_history_steps * self.sensor_frame_dim

    @property
    def action_history_dim(self) -> int:
        """Flattened action-history dimension."""

        return self.action_history_steps * self.action_dim

    @property
    def preview_dim(self) -> int:
        """Flattened path-preview dimension."""

        return self.preview_point_count * self.preview_point_dim

    @property
    def observation_dim(self) -> int:
        """Total raw actor-observation dimension."""

        return self.sensor_history_dim + self.action_history_dim + self.preview_dim


PURE_RL_RAW_OBSERVATION_LAYOUT = PureRLRawObservationLayout()


@dataclass(frozen=True)
class PureRLObservationNormalization:
    """Fixed physical scales for the first PureRL curriculum."""

    forward_ground_velocity_scale_mps: float = 12.0
    lateral_vertical_ground_velocity_scale_mps: float = 5.0
    angular_velocity_scale_rad_s: float = 5.0
    forward_air_velocity_scale_mps: float = 12.0
    minimum_frequency_hz: float = 0.0
    maximum_frequency_hz: float = 5.0
    preview_position_scale_m: float = 5.0
    safety_clip_abs: float = 5.0


PURE_RL_OBSERVATION_NORMALIZATION = PureRLObservationNormalization()


def normalize_quaternion_wxyz(quaternion_wxyz: Tensor) -> Tensor:
    """Return normalized batched ``wxyz`` quaternions without choosing a sign."""

    _validate_matrix("quaternion_wxyz", quaternion_wxyz, columns=4)
    norm = torch.linalg.vector_norm(quaternion_wxyz, dim=1, keepdim=True)
    epsilon = torch.finfo(quaternion_wxyz.dtype).eps
    if bool(torch.any(norm <= epsilon)):
        raise ValueError("quaternion_wxyz must have non-zero norm.")
    return quaternion_wxyz / norm


def align_quaternion_sign_to_previous(
    quaternion_wxyz: Tensor,
    previous_quaternion_wxyz: Tensor,
) -> Tensor:
    """Choose the current quaternion sign closest to the preceding sample.

    The two inputs represent body orientation relative to the fixed world
    frame. Both are normalized before the dot-product test. The returned
    quaternion describes the same rotation as ``quaternion_wxyz``.
    """

    current = normalize_quaternion_wxyz(quaternion_wxyz)
    previous = normalize_quaternion_wxyz(previous_quaternion_wxyz)
    _validate_same_shape_and_metadata(
        "quaternion_wxyz",
        current,
        "previous_quaternion_wxyz",
        previous,
    )
    choose_negative = torch.sum(current * previous, dim=1, keepdim=True) < 0.0
    return torch.where(choose_negative, -current, current)


def canonicalize_orientation_to_route_heading(
    orientation_world_wxyz: Tensor,
    route_heading_rad: Tensor,
) -> Tensor:
    """Express world orientation relative to a fixed route heading.

    The returned ``wxyz`` quaternion is ``q_z(-route_heading) * q_world``.
    Its sign is chosen with a non-negative scalar component so equivalent
    world headings and the quaternion double cover share one actor input.
    """

    _validate_matrix("orientation_world_wxyz", orientation_world_wxyz, columns=4)
    _validate_vector("route_heading_rad", route_heading_rad)
    _validate_same_batch_and_metadata(
        "orientation_world_wxyz",
        orientation_world_wxyz,
        "route_heading_rad",
        route_heading_rad,
    )

    orientation = normalize_quaternion_wxyz(orientation_world_wxyz)
    half_heading = -0.5 * route_heading_rad
    cosine = torch.cos(half_heading)
    sine = torch.sin(half_heading)
    w, x, y, z = orientation.unbind(dim=1)
    canonical = torch.stack(
        (
            cosine * w - sine * z,
            cosine * x - sine * y,
            cosine * y + sine * x,
            cosine * z + sine * w,
        ),
        dim=1,
    )
    canonical = normalize_quaternion_wxyz(canonical)
    return torch.where(canonical[:, 0:1] < 0.0, -canonical, canonical)


def build_raw_sensor_frame(
    *,
    orientation_world_wxyz: Tensor,
    ground_velocity_body_mps: Tensor,
    angular_velocity_body_rad_s: Tensor,
    forward_air_velocity_body_mps: Tensor,
    actual_flap_frequency_hz: Tensor,
    flap_phase_rad: Tensor,
    previous_orientation_world_wxyz: Tensor | None = None,
) -> Tensor:
    """Build one unscaled 14-value sensor frame for every environment.

    Component order is ``[q_wxyz(4), v_ground_b(3), omega_b(3),
    u_air_x(1), actual_frequency(1), sin(phase), cos(phase)]``. Positions,
    velocities, rates and frequency retain SI units. If a previous quaternion
    is supplied, only the equivalent quaternion sign is made continuous.
    """

    _validate_matrix("orientation_world_wxyz", orientation_world_wxyz, columns=4)
    _validate_matrix("ground_velocity_body_mps", ground_velocity_body_mps, columns=3)
    _validate_matrix("angular_velocity_body_rad_s", angular_velocity_body_rad_s, columns=3)
    _validate_vector("forward_air_velocity_body_mps", forward_air_velocity_body_mps)
    _validate_vector("actual_flap_frequency_hz", actual_flap_frequency_hz)
    _validate_vector("flap_phase_rad", flap_phase_rad)

    reference = orientation_world_wxyz
    named_inputs = (
        ("ground_velocity_body_mps", ground_velocity_body_mps),
        ("angular_velocity_body_rad_s", angular_velocity_body_rad_s),
        ("forward_air_velocity_body_mps", forward_air_velocity_body_mps),
        ("actual_flap_frequency_hz", actual_flap_frequency_hz),
        ("flap_phase_rad", flap_phase_rad),
    )
    for name, value in named_inputs:
        _validate_same_batch_and_metadata("orientation_world_wxyz", reference, name, value)

    if bool(torch.any(actual_flap_frequency_hz < 0.0)):
        raise ValueError("actual_flap_frequency_hz must be non-negative.")

    if previous_orientation_world_wxyz is None:
        orientation = normalize_quaternion_wxyz(orientation_world_wxyz)
    else:
        orientation = align_quaternion_sign_to_previous(
            orientation_world_wxyz,
            previous_orientation_world_wxyz,
        )

    phase_features = torch.stack((torch.sin(flap_phase_rad), torch.cos(flap_phase_rad)), dim=1)
    sensor_frame = torch.cat(
        (
            orientation,
            ground_velocity_body_mps,
            angular_velocity_body_rad_s,
            forward_air_velocity_body_mps.unsqueeze(1),
            actual_flap_frequency_hz.unsqueeze(1),
            phase_features,
        ),
        dim=1,
    )
    if sensor_frame.shape[1] != PURE_RL_RAW_OBSERVATION_LAYOUT.sensor_frame_dim:
        raise RuntimeError("Raw sensor-frame implementation disagrees with its declared layout.")
    return sensor_frame


def initialize_raw_history(sample: Tensor, *, history_steps: int) -> Tensor:
    """Fill a reset history with one real sample instead of zero padding."""

    _validate_matrix("sample", sample)
    steps = int(history_steps)
    if steps <= 0:
        raise ValueError("history_steps must be positive.")
    return sample.unsqueeze(1).expand(-1, steps, -1).clone()


def append_raw_history(history: Tensor, sample: Tensor) -> Tensor:
    """Append one newest sample while preserving oldest-to-newest ordering."""

    _validate_history("history", history)
    _validate_matrix("sample", sample, columns=history.shape[2])
    _validate_same_batch_and_metadata("history", history, "sample", sample)
    return torch.cat((history[:, 1:, :], sample.unsqueeze(1)), dim=1)


def compute_preview_query_progress_m(
    *,
    closest_path_progress_m: Tensor,
    ground_velocity_world_mps: Tensor,
    path_tangent_world: Tensor,
    minimum_preview_speed_mps: float,
    maximum_preview_speed_mps: float,
    preview_times_s: tuple[float, ...] = PURE_RL_PREVIEW_TIMES_S,
) -> tuple[Tensor, Tensor]:
    """Return geometry-preview path queries driven by current along-track speed.

    This function has no target-speed input. Negative along-track motion is
    replaced by the configured minimum preview speed, then all preview speeds
    are bounded by the supplied interval. It returns query progress with shape
    ``(N, K)`` and the resolved current preview speed with shape ``(N,)``.
    """

    _validate_vector("closest_path_progress_m", closest_path_progress_m)
    _validate_matrix("ground_velocity_world_mps", ground_velocity_world_mps, columns=3)
    _validate_matrix("path_tangent_world", path_tangent_world, columns=3)
    _validate_same_batch_and_metadata(
        "closest_path_progress_m",
        closest_path_progress_m,
        "ground_velocity_world_mps",
        ground_velocity_world_mps,
    )
    _validate_same_shape_and_metadata(
        "ground_velocity_world_mps",
        ground_velocity_world_mps,
        "path_tangent_world",
        path_tangent_world,
    )

    minimum_speed = float(minimum_preview_speed_mps)
    maximum_speed = float(maximum_preview_speed_mps)
    if not math.isfinite(minimum_speed) or not math.isfinite(maximum_speed):
        raise ValueError("Preview-speed bounds must be finite.")
    if minimum_speed < 0.0 or maximum_speed < minimum_speed:
        raise ValueError("Preview-speed bounds must satisfy 0 <= minimum <= maximum.")
    if len(preview_times_s) != PURE_RL_RAW_OBSERVATION_LAYOUT.preview_point_count:
        raise ValueError("preview_times_s must contain exactly five entries.")
    if any((not math.isfinite(float(value))) or float(value) <= 0.0 for value in preview_times_s):
        raise ValueError("preview_times_s entries must be finite and positive.")
    if any(float(later) <= float(earlier) for earlier, later in zip(preview_times_s, preview_times_s[1:])):
        raise ValueError("preview_times_s must be strictly increasing.")

    tangent_norm = torch.linalg.vector_norm(path_tangent_world, dim=1, keepdim=True)
    epsilon = torch.finfo(path_tangent_world.dtype).eps
    if bool(torch.any(tangent_norm <= epsilon)):
        raise ValueError("path_tangent_world must have non-zero norm.")
    unit_tangent = path_tangent_world / tangent_norm
    along_track_speed = torch.sum(ground_velocity_world_mps * unit_tangent, dim=1)
    preview_speed = torch.clamp(along_track_speed, min=minimum_speed, max=maximum_speed)
    times = closest_path_progress_m.new_tensor(preview_times_s)
    query_progress = closest_path_progress_m.unsqueeze(1) + preview_speed.unsqueeze(1) * times.unsqueeze(0)
    return query_progress, preview_speed


def transform_world_preview_points_to_body(
    *,
    preview_points_world_m: Tensor,
    vehicle_position_world_m: Tensor,
    orientation_world_wxyz: Tensor,
) -> Tensor:
    """Express world-frame preview points as body-FLU relative coordinates."""

    _validate_preview_points("preview_points_world_m", preview_points_world_m)
    _validate_matrix("vehicle_position_world_m", vehicle_position_world_m, columns=3)
    _validate_matrix("orientation_world_wxyz", orientation_world_wxyz, columns=4)
    _validate_same_batch_and_metadata(
        "preview_points_world_m",
        preview_points_world_m,
        "vehicle_position_world_m",
        vehicle_position_world_m,
    )
    _validate_same_batch_and_metadata(
        "preview_points_world_m",
        preview_points_world_m,
        "orientation_world_wxyz",
        orientation_world_wxyz,
    )

    orientation = normalize_quaternion_wxyz(orientation_world_wxyz)
    relative_world = preview_points_world_m - vehicle_position_world_m.unsqueeze(1)
    vector_part = orientation[:, None, 1:4].expand_as(relative_world)
    scalar_part = orientation[:, None, 0:1]
    first_cross = torch.cross(vector_part, relative_world, dim=2)
    second_cross = torch.cross(vector_part, first_cross, dim=2)
    return relative_world - 2.0 * scalar_part * first_cross + 2.0 * second_cross


def build_raw_actor_observation(
    *,
    sensor_history: Tensor,
    action_history: Tensor,
    preview_points_body_m: Tensor,
    layout: PureRLRawObservationLayout = PURE_RL_RAW_OBSERVATION_LAYOUT,
) -> Tensor:
    """Flatten and concatenate the fixed raw actor observation in contract order."""

    _validate_history("sensor_history", sensor_history)
    _validate_history("action_history", action_history)
    _validate_preview_points("preview_points_body_m", preview_points_body_m)
    expected_sensor_shape = (layout.sensor_history_steps, layout.sensor_frame_dim)
    expected_action_shape = (layout.action_history_steps, layout.action_dim)
    expected_preview_shape = (layout.preview_point_count, layout.preview_point_dim)
    if sensor_history.shape[1:] != expected_sensor_shape:
        raise ValueError(f"sensor_history must have trailing shape {expected_sensor_shape}.")
    if action_history.shape[1:] != expected_action_shape:
        raise ValueError(f"action_history must have trailing shape {expected_action_shape}.")
    if preview_points_body_m.shape[1:] != expected_preview_shape:
        raise ValueError(f"preview_points_body_m must have trailing shape {expected_preview_shape}.")
    _validate_same_batch_and_metadata("sensor_history", sensor_history, "action_history", action_history)
    _validate_same_batch_and_metadata(
        "sensor_history",
        sensor_history,
        "preview_points_body_m",
        preview_points_body_m,
    )

    observation = torch.cat(
        (
            sensor_history.reshape(sensor_history.shape[0], -1),
            action_history.reshape(action_history.shape[0], -1),
            preview_points_body_m.reshape(preview_points_body_m.shape[0], -1),
        ),
        dim=1,
    )
    if observation.shape[1] != layout.observation_dim:
        raise RuntimeError("Raw actor observation disagrees with its declared layout.")
    return observation


def normalize_actor_observation(
    *,
    sensor_history: Tensor,
    action_history: Tensor,
    preview_points_body_m: Tensor,
    normalization: PureRLObservationNormalization = PURE_RL_OBSERVATION_NORMALIZATION,
    apply_safety_clip: bool = True,
    layout: PureRLRawObservationLayout = PURE_RL_RAW_OBSERVATION_LAYOUT,
) -> Tensor:
    """Build the fixed-scale 555-value actor observation.

    Quaternion and phase features are already dimensionless and are retained.
    Applied-action history is already normalized to ``[-1, 1]``. Frequency is
    affinely mapped from the configured physical interval to ``[-1, 1]``;
    other physical quantities are divided by fixed, documented scales.
    """

    _validate_normalization(normalization)
    # Reuse the raw builder as the authoritative shape, batch and metadata gate.
    build_raw_actor_observation(
        sensor_history=sensor_history,
        action_history=action_history,
        preview_points_body_m=preview_points_body_m,
        layout=layout,
    )

    normalized_sensor = sensor_history.clone()
    normalized_sensor[..., 4] /= normalization.forward_ground_velocity_scale_mps
    normalized_sensor[..., 5:7] /= normalization.lateral_vertical_ground_velocity_scale_mps
    normalized_sensor[..., 7:10] /= normalization.angular_velocity_scale_rad_s
    normalized_sensor[..., 10] /= normalization.forward_air_velocity_scale_mps
    frequency_span_hz = normalization.maximum_frequency_hz - normalization.minimum_frequency_hz
    normalized_sensor[..., 11] = (
        2.0 * (sensor_history[..., 11] - normalization.minimum_frequency_hz) / frequency_span_hz - 1.0
    )
    normalized_preview = preview_points_body_m / normalization.preview_position_scale_m
    observation = build_raw_actor_observation(
        sensor_history=normalized_sensor,
        action_history=action_history,
        preview_points_body_m=normalized_preview,
        layout=layout,
    )
    if apply_safety_clip:
        observation = torch.clamp(
            observation,
            min=-normalization.safety_clip_abs,
            max=normalization.safety_clip_abs,
        )
    return observation


def _validate_normalization(normalization: PureRLObservationNormalization) -> None:
    positive_values = (
        normalization.forward_ground_velocity_scale_mps,
        normalization.lateral_vertical_ground_velocity_scale_mps,
        normalization.angular_velocity_scale_rad_s,
        normalization.forward_air_velocity_scale_mps,
        normalization.preview_position_scale_m,
        normalization.safety_clip_abs,
    )
    if any((not math.isfinite(float(value))) or float(value) <= 0.0 for value in positive_values):
        raise ValueError("Observation scales and safety clip must be finite and positive.")
    if not math.isfinite(float(normalization.minimum_frequency_hz)) or not math.isfinite(
        float(normalization.maximum_frequency_hz)
    ):
        raise ValueError("Frequency normalization bounds must be finite.")
    if normalization.maximum_frequency_hz <= normalization.minimum_frequency_hz:
        raise ValueError("Frequency normalization requires maximum_frequency_hz > minimum_frequency_hz.")


def _validate_vector(name: str, value: Tensor) -> None:
    _validate_tensor(name, value)
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape (N,).")


def _validate_matrix(name: str, value: Tensor, *, columns: int | None = None) -> None:
    _validate_tensor(name, value)
    if value.ndim != 2 or (columns is not None and value.shape[1] != columns):
        suffix = "D" if columns is None else str(columns)
        raise ValueError(f"{name} must have shape (N, {suffix}).")


def _validate_history(name: str, value: Tensor) -> None:
    _validate_tensor(name, value)
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape (N, K, D).")


def _validate_preview_points(name: str, value: Tensor) -> None:
    _validate_tensor(name, value)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{name} must have shape (N, K, 3).")


def _validate_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor.")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating dtype.")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must be finite.")


def _validate_same_batch_and_metadata(
    first_name: str,
    first: Tensor,
    second_name: str,
    second: Tensor,
) -> None:
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"{first_name} and {second_name} must have the same batch dimension.")
    if first.device != second.device or first.dtype != second.dtype:
        raise ValueError(f"{first_name} and {second_name} must have the same device and dtype.")


def _validate_same_shape_and_metadata(
    first_name: str,
    first: Tensor,
    second_name: str,
    second: Tensor,
) -> None:
    if first.shape != second.shape:
        raise ValueError(f"{first_name} and {second_name} must have the same shape.")
    _validate_same_batch_and_metadata(first_name, first, second_name, second)


__all__ = [
    "PURE_RL_PREVIEW_TIMES_S",
    "PURE_RL_OBSERVATION_NORMALIZATION",
    "PURE_RL_RAW_OBSERVATION_LAYOUT",
    "PureRLObservationNormalization",
    "PureRLRawObservationLayout",
    "align_quaternion_sign_to_previous",
    "append_raw_history",
    "build_raw_actor_observation",
    "build_raw_sensor_frame",
    "canonicalize_orientation_to_route_heading",
    "compute_preview_query_progress_m",
    "initialize_raw_history",
    "normalize_actor_observation",
    "normalize_quaternion_wxyz",
    "transform_world_preview_points_to_body",
]

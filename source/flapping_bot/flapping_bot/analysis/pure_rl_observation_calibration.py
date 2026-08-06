"""Calibration gates for the measured PureRL actor observation.

The gates retain physical-unit samples, then independently evaluate the frozen
normalization contract before its safety clip. Runtime jobs are designed to be
called after Isaac Sim starts in a fresh process.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from flapping_bot.direct.flapping_bot.action_contract import frequency_hz_to_normalized_action
from flapping_bot.direct.flapping_bot.pure_rl_observation import (
    PURE_RL_OBSERVATION_NORMALIZATION,
    PURE_RL_RAW_OBSERVATION_LAYOUT,
    append_raw_history,
    build_raw_actor_observation,
    build_raw_sensor_frame,
    compute_preview_query_progress_m,
    initialize_raw_history,
    normalize_actor_observation,
    transform_world_preview_points_to_body,
)


Tensor = torch.Tensor
GateMode = Literal["synthetic", "reset", "scripted", "boundary"]

SENSOR_CHANNEL_NAMES: tuple[str, ...] = (
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "ground_velocity_body_x_mps",
    "ground_velocity_body_y_mps",
    "ground_velocity_body_z_mps",
    "angular_velocity_body_x_rad_s",
    "angular_velocity_body_y_rad_s",
    "angular_velocity_body_z_rad_s",
    "forward_air_velocity_body_mps",
    "actual_flap_frequency_hz",
    "flap_phase_sin",
    "flap_phase_cos",
)
ACTION_CHANNEL_NAMES: tuple[str, ...] = (
    "action_frequency_normalized",
    "action_rudder_normalized",
    "action_left_elevon_normalized",
    "action_right_elevon_normalized",
)
PARTITION_CALIBRATION = np.uint8(0)
PARTITION_VERIFICATION = np.uint8(1)
_PREVIEW_MIN_SPEED_MPS = 1.0
_PREVIEW_MAX_SPEED_MPS = 12.0


@dataclass(frozen=True)
class PureRLObservationCalibrationResult:
    """JSON summary and compact raw samples from one calibration gate."""

    summary: dict[str, object]
    samples: dict[str, np.ndarray]


def quaternion_wxyz_from_roll_pitch_yaw(roll_rad: Tensor, pitch_rad: Tensor, yaw_rad: Tensor) -> Tensor:
    """Return batched world-from-body ``wxyz`` quaternions."""

    if roll_rad.ndim != 1 or pitch_rad.shape != roll_rad.shape or yaw_rad.shape != roll_rad.shape:
        raise ValueError("roll_rad, pitch_rad and yaw_rad must be aligned 1D tensors.")
    half_roll = 0.5 * roll_rad
    half_pitch = 0.5 * pitch_rad
    half_yaw = 0.5 * yaw_rad
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=1,
    )


def rotate_body_vectors_to_world(quaternion_world_wxyz: Tensor, vectors_body: Tensor) -> Tensor:
    """Rotate batched body-frame vectors into the world frame."""

    if quaternion_world_wxyz.ndim != 2 or quaternion_world_wxyz.shape[1] != 4:
        raise ValueError("quaternion_world_wxyz must have shape (N, 4).")
    if vectors_body.ndim != 2 or vectors_body.shape != (quaternion_world_wxyz.shape[0], 3):
        raise ValueError("vectors_body must have shape (N, 3).")
    quaternion = quaternion_world_wxyz / torch.linalg.vector_norm(
        quaternion_world_wxyz, dim=1, keepdim=True
    ).clamp_min(torch.finfo(quaternion_world_wxyz.dtype).eps)
    vector_part = quaternion[:, 1:4]
    twice_cross = 2.0 * torch.cross(vector_part, vectors_body, dim=1)
    return vectors_body + quaternion[:, 0:1] * twice_cross + torch.cross(
        vector_part, twice_cross, dim=1
    )


def rotate_world_vectors_to_body(quaternion_world_wxyz: Tensor, vectors_world: Tensor) -> Tensor:
    """Rotate batched world-frame vectors into the body frame."""

    conjugate = quaternion_world_wxyz.clone()
    conjugate[:, 1:4] *= -1.0
    return rotate_body_vectors_to_world(conjugate, vectors_world)


def build_level_straight_preview(
    *,
    vehicle_position_world_m: Tensor,
    orientation_world_wxyz: Tensor,
    ground_velocity_world_mps: Tensor,
    route_origin_world_m: Tensor,
    route_tangent_world: Tensor,
    minimum_preview_speed_mps: float = _PREVIEW_MIN_SPEED_MPS,
    maximum_preview_speed_mps: float = _PREVIEW_MAX_SPEED_MPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build five body-frame preview points on an unbounded level line."""

    tangent_norm = torch.linalg.vector_norm(route_tangent_world, dim=1, keepdim=True)
    unit_tangent = route_tangent_world / tangent_norm.clamp_min(
        torch.finfo(route_tangent_world.dtype).eps
    )
    relative_position = vehicle_position_world_m - route_origin_world_m
    closest_progress = torch.sum(relative_position * unit_tangent, dim=1)
    query_progress, preview_speed = compute_preview_query_progress_m(
        closest_path_progress_m=closest_progress,
        ground_velocity_world_mps=ground_velocity_world_mps,
        path_tangent_world=unit_tangent,
        minimum_preview_speed_mps=minimum_preview_speed_mps,
        maximum_preview_speed_mps=maximum_preview_speed_mps,
    )
    preview_world = route_origin_world_m.unsqueeze(1) + query_progress.unsqueeze(2) * unit_tangent.unsqueeze(1)
    preview_body = transform_world_preview_points_to_body(
        preview_points_world_m=preview_world,
        vehicle_position_world_m=vehicle_position_world_m,
        orientation_world_wxyz=orientation_world_wxyz,
    )
    along_track_speed = torch.sum(ground_velocity_world_mps * unit_tangent, dim=1)
    return preview_body, preview_speed, along_track_speed


class RawObservationCollector:
    """Stateful oldest-to-newest raw-history collector for one environment batch."""

    def __init__(self) -> None:
        self._previous_orientation: Tensor | None = None
        self._sensor_history: Tensor | None = None
        self._action_history: Tensor | None = None

    def collect(
        self,
        *,
        orientation_world_wxyz: Tensor,
        ground_velocity_body_mps: Tensor,
        angular_velocity_body_rad_s: Tensor,
        forward_air_velocity_body_mps: Tensor,
        actual_flap_frequency_hz: Tensor,
        flap_phase_rad: Tensor,
        applied_action: Tensor,
        preview_points_body_m: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Append one policy-rate sample and return its frame and full observation."""

        frame = build_raw_sensor_frame(
            orientation_world_wxyz=orientation_world_wxyz,
            ground_velocity_body_mps=ground_velocity_body_mps,
            angular_velocity_body_rad_s=angular_velocity_body_rad_s,
            forward_air_velocity_body_mps=forward_air_velocity_body_mps,
            actual_flap_frequency_hz=actual_flap_frequency_hz,
            flap_phase_rad=flap_phase_rad,
            previous_orientation_world_wxyz=self._previous_orientation,
        )
        if self._sensor_history is None:
            self._sensor_history = initialize_raw_history(
                frame,
                history_steps=PURE_RL_RAW_OBSERVATION_LAYOUT.sensor_history_steps,
            )
            self._action_history = initialize_raw_history(
                applied_action,
                history_steps=PURE_RL_RAW_OBSERVATION_LAYOUT.action_history_steps,
            )
        else:
            self._sensor_history = append_raw_history(self._sensor_history, frame)
            assert self._action_history is not None
            self._action_history = append_raw_history(self._action_history, applied_action)
        self._previous_orientation = frame[:, 0:4].detach().clone()
        assert self._action_history is not None
        observation = build_raw_actor_observation(
            sensor_history=self._sensor_history,
            action_history=self._action_history,
            preview_points_body_m=preview_points_body_m,
        )
        return frame, observation


def summarize_scalar_samples(values: np.ndarray) -> dict[str, float | int]:
    """Return robust and tail statistics for one finite scalar channel."""

    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(sample)
    if sample.size == 0 or not np.all(finite):
        raise ValueError("Scalar calibration samples must be non-empty and finite.")
    quantiles = np.quantile(sample, (0.01, 0.05, 0.50, 0.95, 0.99, 0.999))
    median = float(quantiles[2])
    return {
        "count": int(sample.size),
        "minimum": float(np.min(sample)),
        "maximum": float(np.max(sample)),
        "mean": float(np.mean(sample)),
        "standard_deviation": float(np.std(sample)),
        "median": median,
        "median_absolute_deviation": float(np.median(np.abs(sample - median))),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "p99_9": float(quantiles[5]),
        "p99_absolute": float(np.quantile(np.abs(sample), 0.99)),
    }


def build_scripted_actions(
    *,
    base_action: Tensor,
    policy_step: int,
    policy_dt_s: float,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
) -> Tensor:
    """Return the deterministic eight-family action calibration matrix."""

    if base_action.ndim != 2 or base_action.shape[1] != 4:
        raise ValueError("base_action must have shape (N, 4).")
    if policy_step < 0 or policy_dt_s <= 0.0:
        raise ValueError("policy_step must be nonnegative and policy_dt_s must be positive.")
    action = base_action.clone()
    env_ids = torch.arange(base_action.shape[0], device=base_action.device)
    family = torch.remainder(env_ids, 8)
    repetition = torch.div(env_ids, 8, rounding_mode="floor")
    time_s = float(policy_step) * float(policy_dt_s)
    for family_index, frequency_hz in ((1, 0.0), (2, 2.5), (3, 5.0)):
        mask = family == family_index
        target = torch.full(
            (int(mask.sum().item()),),
            frequency_hz,
            device=base_action.device,
            dtype=base_action.dtype,
        )
        action[mask, 0] = frequency_hz_to_normalized_action(
            target,
            minimum_frequency_hz=minimum_frequency_hz,
            maximum_frequency_hz=maximum_frequency_hz,
        )
    phase_offset_s = 0.011 * repetition.to(dtype=base_action.dtype)
    sine = 0.60 * torch.sin(2.0 * math.pi * (time_s + phase_offset_s))
    action[family == 4, 1] = torch.clamp(
        base_action[family == 4, 1] + sine[family == 4], -1.0, 1.0
    )
    action[family == 5, 2] = torch.clamp(
        base_action[family == 5, 2] + sine[family == 5], -1.0, 1.0
    )
    action[family == 6, 3] = torch.clamp(
        base_action[family == 6, 3] + sine[family == 6], -1.0, 1.0
    )
    prbs_mask = family == 7
    if bool(torch.any(prbs_mask)):
        block = policy_step // max(1, int(round(0.10 / policy_dt_s)))
        selected_ids = env_ids[prbs_mask]
        signs = torch.where(
            torch.remainder(
                selected_ids * 17
                + torch.div(selected_ids, 8, rounding_mode="floor") * 7
                + block * 13,
                2,
            )
            == 0,
            torch.ones_like(selected_ids, dtype=base_action.dtype),
            -torch.ones_like(selected_ids, dtype=base_action.dtype),
        )
        action[prbs_mask, 0] = signs
        action[prbs_mask, 1] = 0.45 * signs
        action[prbs_mask, 2] = -0.35 * signs
        action[prbs_mask, 3] = 0.35 * signs
    return torch.clamp(action, -1.0, 1.0)


def _uniform(
    count: int,
    low: float,
    high: float,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    return low + (high - low) * torch.rand((count,), generator=generator, dtype=dtype)


def _partition_from_index(count: int) -> np.ndarray:
    partition = np.full((count,), PARTITION_CALIBRATION, dtype=np.uint8)
    partition[(3 * count) // 4 :] = PARTITION_VERIFICATION
    return partition


def _build_gate_summary(
    *,
    mode: GateMode,
    sensor_frames: np.ndarray,
    preview_points_body_m: np.ndarray,
    actions: np.ndarray,
    partition: np.ndarray,
    observation_shape_ok: bool,
    env_ids: np.ndarray | None = None,
    sample_steps: np.ndarray | None = None,
    extras: dict[str, object] | None = None,
) -> dict[str, object]:
    sensor = np.asarray(sensor_frames, dtype=np.float64)
    preview = np.asarray(preview_points_body_m, dtype=np.float64)
    action = np.asarray(actions, dtype=np.float64)
    split = np.asarray(partition, dtype=np.uint8)
    expected_shapes = (
        sensor.ndim == 2 and sensor.shape[1] == len(SENSOR_CHANNEL_NAMES),
        preview.ndim == 3 and preview.shape[1:] == (5, 3),
        action.ndim == 2 and action.shape[1] == len(ACTION_CHANNEL_NAMES),
        split.shape == (sensor.shape[0],),
    )
    shape_ok = bool(all(expected_shapes) and observation_shape_ok)
    finite = bool(np.all(np.isfinite(sensor)) and np.all(np.isfinite(preview)) and np.all(np.isfinite(action)))
    normalized_preclip = normalize_actor_observation(
        sensor_history=torch.from_numpy(sensor).unsqueeze(1).expand(
            -1,
            PURE_RL_RAW_OBSERVATION_LAYOUT.sensor_history_steps,
            -1,
        ),
        action_history=torch.from_numpy(action).unsqueeze(1).expand(
            -1,
            PURE_RL_RAW_OBSERVATION_LAYOUT.action_history_steps,
            -1,
        ),
        preview_points_body_m=torch.from_numpy(preview),
        apply_safety_clip=False,
    ).numpy()
    normalized_finite = bool(np.all(np.isfinite(normalized_preclip)))
    safety_clip_abs = float(PURE_RL_OBSERVATION_NORMALIZATION.safety_clip_abs)
    would_clip = np.abs(normalized_preclip) > safety_clip_abs
    would_clip_value_count = int(np.count_nonzero(would_clip))
    would_clip_sample_count = int(np.count_nonzero(np.any(would_clip, axis=1)))
    partition_summary: dict[str, object] = {}
    for partition_value, partition_name in (
        (PARTITION_CALIBRATION, "calibration"),
        (PARTITION_VERIFICATION, "verification"),
    ):
        mask = split == partition_value
        if not bool(np.any(mask)):
            raise ValueError(f"Missing {partition_name} samples.")
        channels = {
            name: summarize_scalar_samples(sensor[mask, index])
            for index, name in enumerate(SENSOR_CHANNEL_NAMES)
        }
        channels.update(
            {
                name: summarize_scalar_samples(action[mask, index])
                for index, name in enumerate(ACTION_CHANNEL_NAMES)
            }
        )
        for coordinate, coordinate_name in enumerate(("x", "y", "z")):
            channels[f"preview_body_{coordinate_name}_m"] = summarize_scalar_samples(
                preview[mask, :, coordinate]
            )
        distances = np.linalg.norm(preview[mask], axis=2)
        partition_summary[partition_name] = {
            "sample_count": int(np.count_nonzero(mask)),
            "channels": channels,
            "preview_distance_by_point_m": {
                f"point_{point_index + 1}": summarize_scalar_samples(distances[:, point_index])
                for point_index in range(distances.shape[1])
            },
        }
    quaternion_norm_error = np.abs(np.linalg.norm(sensor[:, 0:4], axis=1) - 1.0)
    temporal_delta: dict[str, object] = {}
    minimum_adjacent_quaternion_dot = 1.0
    if env_ids is not None and sample_steps is not None:
        env_index = np.asarray(env_ids, dtype=np.int64)
        step_index = np.asarray(sample_steps, dtype=np.int64)
        deltas: list[np.ndarray] = []
        dots: list[float] = []
        for current in range(1, sensor.shape[0]):
            if env_index[current] != env_index[current - 1]:
                continue
            if step_index[current] != step_index[current - 1] + 1:
                continue
            deltas.append(sensor[current] - sensor[current - 1])
            dots.append(float(np.dot(sensor[current, 0:4], sensor[current - 1, 0:4])))
        if deltas:
            delta_array = np.asarray(deltas)
            temporal_delta = {
                name: summarize_scalar_samples(delta_array[:, index])
                for index, name in enumerate(SENSOR_CHANNEL_NAMES)
            }
        if dots:
            minimum_adjacent_quaternion_dot = float(np.min(dots))
    accepted = bool(
        shape_ok
        and finite
        and float(np.max(quaternion_norm_error)) <= 2.0e-6
        and minimum_adjacent_quaternion_dot >= -1.0e-6
        and normalized_finite
        and would_clip_value_count == 0
    )
    summary: dict[str, object] = {
        "schema_version": "pure_rl_observation_calibration_v2",
        "gate_mode": mode,
        "raw_samples_retained": True,
        "normalization_contract_evaluated": True,
        "safety_clipping_applied_to_saved_samples": False,
        "raw_observation_dimension": PURE_RL_RAW_OBSERVATION_LAYOUT.observation_dim,
        "sample_count": int(sensor.shape[0]),
        "shape_contract_ok": shape_ok,
        "all_samples_finite": finite,
        "maximum_quaternion_norm_error": float(np.max(quaternion_norm_error)),
        "minimum_adjacent_quaternion_dot": minimum_adjacent_quaternion_dot,
        "normalized_preclip_all_finite": normalized_finite,
        "normalized_preclip_maximum_absolute_value": float(np.max(np.abs(normalized_preclip))),
        "safety_clip_absolute_value": safety_clip_abs,
        "would_clip_value_count": would_clip_value_count,
        "would_clip_value_fraction": float(np.mean(would_clip)),
        "would_clip_sample_count": would_clip_sample_count,
        "would_clip_sample_fraction": float(np.mean(np.any(would_clip, axis=1))),
        "partitions": partition_summary,
        "temporal_sensor_delta": temporal_delta,
        "all_cases_accepted": accepted,
    }
    if extras:
        summary.update(extras)
    return summary


def run_synthetic_envelope_gate(
    *, sample_count: int = 8192, seed: int = 20260806
) -> PureRLObservationCalibrationResult:
    """Exercise the raw builder over a broad deterministic task envelope."""

    if sample_count < 64:
        raise ValueError("sample_count must be at least 64.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    count = int(sample_count)
    yaw = _uniform(count, -math.pi, math.pi, generator=generator)
    roll = _uniform(count, -math.radians(72.0), math.radians(72.0), generator=generator)
    pitch = _uniform(count, -math.radians(72.0), math.radians(72.0), generator=generator)
    quaternion = quaternion_wxyz_from_roll_pitch_yaw(roll, pitch, yaw)
    velocity_body = torch.stack(
        (
            _uniform(count, -2.0, 20.0, generator=generator),
            _uniform(count, -8.0, 8.0, generator=generator),
            _uniform(count, -8.0, 8.0, generator=generator),
        ),
        dim=1,
    )
    velocity_world = rotate_body_vectors_to_world(quaternion, velocity_body)
    angular_velocity = torch.stack(
        tuple(_uniform(count, -20.0, 20.0, generator=generator) for _ in range(3)),
        dim=1,
    )
    frequency = _uniform(count, 0.0, 5.0, generator=generator)
    phase = _uniform(count, 0.0, 2.0 * math.pi, generator=generator)
    wind_world = torch.stack(
        (
            _uniform(count, -3.0, 3.0, generator=generator),
            _uniform(count, -5.0, 5.0, generator=generator),
            torch.zeros(count, dtype=torch.float64),
        ),
        dim=1,
    )
    wind_body = rotate_world_vectors_to_body(quaternion, wind_world)
    route_tangent = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), dim=1)
    route_normal = torch.stack((-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)), dim=1)
    route_origin = torch.zeros((count, 3), dtype=torch.float64)
    route_origin[:, 2] = 10.0
    progress = _uniform(count, -20.0, 120.0, generator=generator)
    cross_track = _uniform(count, -19.0, 19.0, generator=generator)
    height_error = _uniform(count, -9.9, 10.0, generator=generator)
    position = (
        route_origin
        + progress.unsqueeze(1) * route_tangent
        + cross_track.unsqueeze(1) * route_normal
    )
    position[:, 2] += height_error
    preview, preview_speed, along_track_speed = build_level_straight_preview(
        vehicle_position_world_m=position,
        orientation_world_wxyz=quaternion,
        ground_velocity_world_mps=velocity_world,
        route_origin_world_m=route_origin,
        route_tangent_world=route_tangent,
    )
    action = _uniform(count * 4, -1.0, 1.0, generator=generator).reshape(count, 4)
    collector = RawObservationCollector()
    frame, observation = collector.collect(
        orientation_world_wxyz=quaternion,
        ground_velocity_body_mps=velocity_body,
        angular_velocity_body_rad_s=angular_velocity,
        forward_air_velocity_body_mps=velocity_body[:, 0] - wind_body[:, 0],
        actual_flap_frequency_hz=frequency,
        flap_phase_rad=phase,
        applied_action=action,
        preview_points_body_m=preview,
    )
    partition = _partition_from_index(count)
    sensor_np = frame.numpy()
    preview_np = preview.numpy()
    action_np = action.numpy()
    summary = _build_gate_summary(
        mode="synthetic",
        sensor_frames=sensor_np,
        preview_points_body_m=preview_np,
        actions=action_np,
        partition=partition,
        observation_shape_ok=observation.shape == (count, PURE_RL_RAW_OBSERVATION_LAYOUT.observation_dim),
        extras={
            "seed": int(seed),
            "envelope": {
                "roll_pitch_limit_deg": 72.0,
                "ground_velocity_body_x_mps": [-2.0, 20.0],
                "ground_velocity_body_yz_mps": [-8.0, 8.0],
                "angular_velocity_body_rad_s": [-20.0, 20.0],
                "actual_frequency_hz": [0.0, 5.0],
                "cross_track_m": [-19.0, 19.0],
                "height_error_m": [-9.9, 10.0],
            },
            "preview_speed_bounds_are_provisional": False,
            "preview_speed_bounds_are_frozen": True,
            "preview_minimum_speed_mps": _PREVIEW_MIN_SPEED_MPS,
            "preview_maximum_speed_mps": _PREVIEW_MAX_SPEED_MPS,
            "along_track_speed_mps": summarize_scalar_samples(along_track_speed.numpy()),
            "resolved_preview_speed_mps": summarize_scalar_samples(preview_speed.numpy()),
            "preview_minimum_clamp_fraction": float(
                torch.mean((along_track_speed < _PREVIEW_MIN_SPEED_MPS).to(dtype=torch.float64)).item()
            ),
            "preview_maximum_clamp_fraction": float(
                torch.mean((along_track_speed > _PREVIEW_MAX_SPEED_MPS).to(dtype=torch.float64)).item()
            ),
        },
    )
    return PureRLObservationCalibrationResult(
        summary=summary,
        samples={
            "sensor_frame": sensor_np,
            "preview_points_body_m": preview_np,
            "applied_action": action_np,
            "partition": partition,
            "along_track_speed_mps": along_track_speed.numpy(),
            "resolved_preview_speed_mps": preview_speed.numpy(),
        },
    )


def _configure_robot_asset(cfg: object, *, asset_path: Path | None, usd_dir: Path | None) -> None:
    if asset_path is None and usd_dir is None:
        return
    replacements: dict[str, str] = {}
    if asset_path is not None:
        replacements["asset_path"] = str(Path(asset_path).resolve())
    if usd_dir is not None:
        replacements["usd_dir"] = str(Path(usd_dir).resolve())
    cfg.robot = cfg.robot.replace(spawn=cfg.robot.spawn.replace(**replacements))


def make_runtime_cfg(
    *,
    num_envs: int,
    seed: int,
    asset_path: Path | None,
    usd_dir: Path | None,
):
    """Build the unchanged measured PureRL plant for observation diagnostics."""

    from flapping_bot.direct.flapping_bot.straight_flight_env import (
        FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
    )

    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg()
    cfg.seed = int(seed)
    cfg.scene.num_envs = int(num_envs)
    cfg.scene.env_spacing = 5.0
    cfg.sim.device = "cpu"
    cfg.freeze_steps_after_reset = 0
    cfg.episode_length_s = 100.0
    cfg.teacher_guidance_enabled = False
    cfg.wind_enabled = False
    cfg.randomize_wind = False
    cfg.wind_ou_enabled = False
    cfg.wind_curriculum_enabled = False
    _configure_robot_asset(cfg, asset_path=asset_path, usd_dir=usd_dir)
    return cfg


def _runtime_sample(
    env: object,
    collector: RawObservationCollector,
    *,
    route_origin_world_m: Tensor,
    route_tangent_world: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    orientation = env._robot.data.root_quat_w
    ground_body = env._robot.data.root_lin_vel_b
    angular_body = env._robot.data.root_ang_vel_b
    wind_body = rotate_world_vectors_to_body(orientation, env._wind_w)
    local_position = env._robot.data.root_pos_w - env.scene.env_origins
    local_route_origin = route_origin_world_m - env.scene.env_origins
    preview, preview_speed, along_track_speed = build_level_straight_preview(
        vehicle_position_world_m=local_position,
        orientation_world_wxyz=orientation,
        ground_velocity_world_mps=env._robot.data.root_lin_vel_w,
        route_origin_world_m=local_route_origin,
        route_tangent_world=route_tangent_world,
    )
    frame, observation = collector.collect(
        orientation_world_wxyz=orientation,
        ground_velocity_body_mps=ground_body,
        angular_velocity_body_rad_s=angular_body,
        forward_air_velocity_body_mps=ground_body[:, 0] - wind_body[:, 0],
        actual_flap_frequency_hz=env._freq,
        flap_phase_rad=env._phase,
        applied_action=env._act_cmd,
        preview_points_body_m=preview,
    )
    return frame, preview, observation, preview_speed, along_track_speed


def _to_numpy(value: Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _route_batch(env: object, yaw_rad: Tensor | None = None) -> tuple[Tensor, Tensor]:
    if yaw_rad is None:
        yaw_rad = env._straight_line_heading_rad
    tangent = torch.stack((torch.cos(yaw_rad), torch.sin(yaw_rad), torch.zeros_like(yaw_rad)), dim=1)
    origin = env.scene.env_origins.clone()
    origin[:, 2] += env._height_cmd
    return origin, tangent


def _runtime_result(
    *,
    mode: GateMode,
    sensor_frames: list[np.ndarray],
    previews: list[np.ndarray],
    actions: list[np.ndarray],
    partitions: list[np.ndarray],
    env_ids: list[np.ndarray],
    steps: list[np.ndarray],
    observation_shape_ok: bool,
    extras: dict[str, object],
) -> PureRLObservationCalibrationResult:
    sensor = np.concatenate(sensor_frames, axis=0)
    preview = np.concatenate(previews, axis=0)
    action = np.concatenate(actions, axis=0)
    partition = np.concatenate(partitions, axis=0)
    environment = np.concatenate(env_ids, axis=0)
    step = np.concatenate(steps, axis=0)
    order = np.lexsort((step, environment))
    sensor = sensor[order]
    preview = preview[order]
    action = action[order]
    partition = partition[order]
    environment = environment[order]
    step = step[order]
    summary = _build_gate_summary(
        mode=mode,
        sensor_frames=sensor,
        preview_points_body_m=preview,
        actions=action,
        partition=partition,
        observation_shape_ok=observation_shape_ok,
        env_ids=environment,
        sample_steps=step,
        extras=extras,
    )
    return PureRLObservationCalibrationResult(
        summary=summary,
        samples={
            "sensor_frame": sensor,
            "preview_points_body_m": preview,
            "applied_action": action,
            "partition": partition,
            "environment_id": environment,
            "sample_step": step,
        },
    )


def run_reset_distribution_job(env: object, *, reset_batches: int = 32) -> PureRLObservationCalibrationResult:
    """Collect the actual randomized reset distribution from the current task."""

    if reset_batches < 4:
        raise ValueError("reset_batches must be at least four.")
    env.reset()
    all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    sensor_samples: list[np.ndarray] = []
    preview_samples: list[np.ndarray] = []
    action_samples: list[np.ndarray] = []
    partition_samples: list[np.ndarray] = []
    env_id_samples: list[np.ndarray] = []
    step_samples: list[np.ndarray] = []
    observation_shape_ok = True
    for batch_index in range(int(reset_batches)):
        env._reset_idx(all_env_ids)
        env.sim.forward()
        env._robot.update(env.physics_dt)
        route_origin, route_tangent = _route_batch(env)
        collector = RawObservationCollector()
        frame, preview, observation, _preview_speed, _along_track = _runtime_sample(
            env,
            collector,
            route_origin_world_m=route_origin,
            route_tangent_world=route_tangent,
        )
        observation_shape_ok &= observation.shape == (
            env.num_envs,
            PURE_RL_RAW_OBSERVATION_LAYOUT.observation_dim,
        )
        sensor_samples.append(_to_numpy(frame))
        preview_samples.append(_to_numpy(preview))
        action_samples.append(_to_numpy(env._act_cmd))
        partition_value = (
            PARTITION_CALIBRATION
            if batch_index < (3 * reset_batches) // 4
            else PARTITION_VERIFICATION
        )
        partition_samples.append(np.full((env.num_envs,), partition_value, dtype=np.uint8))
        env_id_samples.append(np.arange(env.num_envs, dtype=np.int64) + batch_index * env.num_envs)
        step_samples.append(np.zeros((env.num_envs,), dtype=np.int64))
    unique_frames = np.unique(np.round(np.concatenate(sensor_samples, axis=0), decimals=6), axis=0)
    has_randomized_coverage = bool(unique_frames.shape[0] > 1)
    return _runtime_result(
        mode="reset",
        sensor_frames=sensor_samples,
        previews=preview_samples,
        actions=action_samples,
        partitions=partition_samples,
        env_ids=env_id_samples,
        steps=step_samples,
        observation_shape_ok=observation_shape_ok,
        extras={
            "reset_batches": int(reset_batches),
            "route_heading_randomized_by_current_task": bool(env.cfg.randomize_straight_line_heading),
            "reset_phase_randomized_by_current_task": bool(env.cfg.randomize_flap_phase_at_reset),
            "observed_unique_sensor_frame_count_at_1e_6": int(unique_frames.shape[0]),
            "coverage_ready_for_scale_selection_by_itself": has_randomized_coverage,
            "coverage_note": (
                "Reset heading and flap phase are randomized; dynamic and near-boundary gates remain necessary "
                "to audit the frozen scales away from initial conditions."
            ),
        },
    )


def run_scripted_dynamics_job(env: object, *, duration_s: float = 1.5) -> PureRLObservationCalibrationResult:
    """Collect bounded free-root observations under eight scripted action families."""

    if duration_s <= 0.0 or duration_s > 3.0:
        raise ValueError("duration_s must be in (0, 3].")
    env.reset()
    collector = RawObservationCollector()
    route_origin, route_tangent = _route_batch(env)
    base_action = env._act_cmd.detach().clone()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    sensor_samples: list[np.ndarray] = []
    preview_samples: list[np.ndarray] = []
    action_samples: list[np.ndarray] = []
    partition_samples: list[np.ndarray] = []
    env_id_samples: list[np.ndarray] = []
    step_samples: list[np.ndarray] = []
    termination_step = np.full((env.num_envs,), -1, dtype=np.int64)
    observation_shape_ok = True
    total_policy_steps = int(math.ceil(float(duration_s) / float(env.step_dt)))
    verification_env = torch.arange(env.num_envs, device=env.device) >= (3 * env.num_envs) // 4
    for policy_step in range(total_policy_steps):
        scripted_action = build_scripted_actions(
            base_action=base_action,
            policy_step=policy_step,
            policy_dt_s=float(env.step_dt),
            minimum_frequency_hz=float(env.cfg.min_flap_hz),
            maximum_frequency_hz=float(env.cfg.max_flap_hz),
        )
        _legacy_observation, _reward, terminated, truncated, _extras = env.step(scripted_action)
        newly_done = active & (terminated | truncated)
        if bool(torch.any(newly_done)):
            termination_step[_to_numpy(torch.nonzero(newly_done, as_tuple=False).squeeze(1)).astype(int)] = policy_step
        active &= ~(terminated | truncated)
        frame, preview, observation, _preview_speed, _along_track = _runtime_sample(
            env,
            collector,
            route_origin_world_m=route_origin,
            route_tangent_world=route_tangent,
        )
        observation_shape_ok &= observation.shape == (
            env.num_envs,
            PURE_RL_RAW_OBSERVATION_LAYOUT.observation_dim,
        )
        ids = torch.nonzero(active, as_tuple=False).squeeze(1)
        if ids.numel() == 0:
            break
        sensor_samples.append(_to_numpy(frame[ids]))
        preview_samples.append(_to_numpy(preview[ids]))
        action_samples.append(_to_numpy(env._act_cmd[ids]))
        partition_samples.append(_to_numpy(verification_env[ids]).astype(np.uint8))
        env_id_samples.append(_to_numpy(ids).astype(np.int64))
        step_samples.append(np.full((ids.numel(),), policy_step, dtype=np.int64))
    if not sensor_samples:
        raise RuntimeError("All scripted environments terminated before one raw sample was collected.")
    result = _runtime_result(
        mode="scripted",
        sensor_frames=sensor_samples,
        previews=preview_samples,
        actions=action_samples,
        partitions=partition_samples,
        env_ids=env_id_samples,
        steps=step_samples,
        observation_shape_ok=observation_shape_ok,
        extras={
            "duration_requested_s": float(duration_s),
            "policy_dt_s": float(env.step_dt),
            "physics_dt_s": float(env.physics_dt),
            "action_family_by_environment_modulo_8": {
                "0": "hold_reset_action",
                "1": "frequency_0_hz",
                "2": "frequency_2_5_hz",
                "3": "frequency_5_hz",
                "4": "rudder_sine",
                "5": "left_elevon_sine",
                "6": "right_elevon_sine",
                "7": "bounded_prbs_all_channels",
            },
            "termination_step_by_environment": termination_step.tolist(),
            "terminated_environment_count": int(np.count_nonzero(termination_step >= 0)),
            "terminal_reset_samples_included": False,
        },
    )
    return result


def run_near_boundary_job(env: object) -> PureRLObservationCalibrationResult:
    """Inject finite states just inside current ground, tilt and lateral limits."""

    env.reset()
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    family = torch.remainder(env_ids, 8)
    verification = env_ids >= (3 * env.num_envs) // 4
    severity = torch.where(
        verification,
        torch.full((env.num_envs,), 0.95, device=env.device),
        torch.full((env.num_envs,), 0.90, device=env.device),
    )
    yaw = -math.pi + 2.0 * math.pi * (env_ids.to(dtype=torch.float32) + 0.5) / float(env.num_envs)
    route_tangent = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), dim=1)
    route_normal = torch.stack((-torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)), dim=1)
    env._straight_line_heading_rad.copy_(yaw)
    env._straight_line_tangent_w.copy_(route_tangent)
    env._straight_line_normal_w.copy_(route_normal)
    tilt = severity * math.radians(float(env.cfg.terminate_tilt_deg))
    roll = torch.zeros(env.num_envs, device=env.device)
    pitch = torch.zeros_like(roll)
    roll[family == 0] = tilt[family == 0]
    roll[family == 1] = -tilt[family == 1]
    pitch[family == 2] = tilt[family == 2]
    pitch[family == 3] = -tilt[family == 3]
    quaternion = quaternion_wxyz_from_roll_pitch_yaw(roll, pitch, yaw)

    position = env.scene.env_origins.clone()
    position[:, 2] += env._height_cmd
    position[family == 4] += (
        severity[family == 4].unsqueeze(1)
        * float(env.cfg.terminate_abs_y)
        * route_normal[family == 4]
    )
    position[family == 5] -= (
        severity[family == 5].unsqueeze(1)
        * float(env.cfg.terminate_abs_y)
        * route_normal[family == 5]
    )
    position[family == 6, 2] = (
        env.scene.env_origins[family == 6, 2]
        + float(env.cfg.terminate_ground_height)
        + 0.05
    )

    velocity_body = torch.zeros((env.num_envs, 3), device=env.device)
    velocity_body[:, 0] = 7.0
    velocity_body[family == 0, 1] = 8.0
    velocity_body[family == 1, 1] = -8.0
    velocity_body[family == 2, 2] = 8.0
    velocity_body[family == 3, 2] = -8.0
    velocity_body[family == 6, 0] = -2.0
    velocity_body[family == 7, 0] = 20.0
    angular_body = torch.zeros_like(velocity_body)
    angular_body[family == 0, 0] = 20.0
    angular_body[family == 1, 0] = -20.0
    angular_body[family == 2, 1] = 20.0
    angular_body[family == 3, 1] = -20.0
    angular_body[family == 4, 2] = 20.0
    angular_body[family == 5, 2] = -20.0
    velocity_world = rotate_body_vectors_to_world(quaternion, velocity_body)
    angular_world = rotate_body_vectors_to_world(quaternion, angular_body)
    root_state = torch.cat((position, quaternion, velocity_world, angular_world), dim=1)
    env._robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    env.sim.forward()
    env._robot.update(env.physics_dt)

    route_origin, route_tangent = _route_batch(env, yaw)
    collector = RawObservationCollector()
    frame, preview, observation, _preview_speed, _along_track = _runtime_sample(
        env,
        collector,
        route_origin_world_m=route_origin,
        route_tangent_world=route_tangent,
    )
    terminated, truncated = env._get_dones()
    inside_current_boundaries = ~(terminated | truncated)
    result = _runtime_result(
        mode="boundary",
        sensor_frames=[_to_numpy(frame)],
        previews=[_to_numpy(preview)],
        actions=[_to_numpy(env._act_cmd)],
        partitions=[_to_numpy(verification).astype(np.uint8)],
        env_ids=[np.arange(env.num_envs, dtype=np.int64)],
        steps=[np.zeros((env.num_envs,), dtype=np.int64)],
        observation_shape_ok=observation.shape
        == (env.num_envs, PURE_RL_RAW_OBSERVATION_LAYOUT.observation_dim),
        extras={
            "calibration_boundary_fraction": 0.90,
            "verification_boundary_fraction": 0.95,
            "inside_current_termination_boundaries_count": int(inside_current_boundaries.sum().item()),
            "injected_state_count": int(env.num_envs),
            "termination_flags_before_physics_step": _to_numpy(~inside_current_boundaries).astype(bool).tolist(),
            "boundary_families": {
                "0": "positive_roll_and_rate",
                "1": "negative_roll_and_rate",
                "2": "positive_pitch_and_rate",
                "3": "negative_pitch_and_rate",
                "4": "positive_lateral_position_and_yaw_rate",
                "5": "negative_lateral_position_and_yaw_rate",
                "6": "near_ground_and_reverse_speed",
                "7": "high_forward_speed",
            },
        },
    )
    result.summary["all_cases_accepted"] = bool(
        result.summary["all_cases_accepted"] and bool(torch.all(inside_current_boundaries))
    )
    return result


def run_runtime_gate(
    *,
    mode: Literal["reset", "scripted", "boundary"],
    num_envs: int = 64,
    seed: int = 20260806,
    reset_batches: int = 32,
    duration_s: float = 1.5,
    asset_path: Path | None = None,
    usd_dir: Path | None = None,
) -> PureRLObservationCalibrationResult:
    """Construct one environment and dispatch one fresh-process runtime gate."""

    from flapping_bot.direct.flapping_bot.straight_flight_env import FlappingBotStraightFlightEnv

    if num_envs < 16 or num_envs % 8 != 0:
        raise ValueError("num_envs must be a multiple of eight and at least 16.")
    cfg = make_runtime_cfg(
        num_envs=num_envs,
        seed=seed,
        asset_path=asset_path,
        usd_dir=usd_dir,
    )
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        if mode == "reset":
            return run_reset_distribution_job(env, reset_batches=reset_batches)
        if mode == "scripted":
            return run_scripted_dynamics_job(env, duration_s=duration_s)
        if mode == "boundary":
            return run_near_boundary_job(env)
        raise ValueError(f"Unsupported runtime gate: {mode}")
    finally:
        import omni.physx

        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()


__all__ = [
    "ACTION_CHANNEL_NAMES",
    "PARTITION_CALIBRATION",
    "PARTITION_VERIFICATION",
    "PureRLObservationCalibrationResult",
    "RawObservationCollector",
    "SENSOR_CHANNEL_NAMES",
    "build_level_straight_preview",
    "build_scripted_actions",
    "make_runtime_cfg",
    "quaternion_wxyz_from_roll_pitch_yaw",
    "rotate_body_vectors_to_world",
    "rotate_world_vectors_to_body",
    "run_near_boundary_job",
    "run_reset_distribution_job",
    "run_runtime_gate",
    "run_scripted_dynamics_job",
    "run_synthetic_envelope_gate",
    "summarize_scalar_samples",
]

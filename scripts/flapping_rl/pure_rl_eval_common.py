"""Shared PureRL curriculum-1 evaluation and checkpoint-scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch


MEASURED_PURE_RL_TASK_ID = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0"
GPU_IMPLICIT_PURE_RL_TASK_ID = (
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuImplicit-Direct-v0"
)
GPU_IMPLICIT_PURE_RL_C2A_TASK_ID = (
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuImplicit-Direct-v0"
)
GPU_PHASE_MATCHED_PURE_RL_TASK_ID = (
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuPhaseMatched-Direct-v0"
)
GPU_PHASE_MATCHED_PURE_RL_C2A_TASK_ID = (
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuPhaseMatched-Direct-v0"
)
MEASURED_PURE_RL_LONGITUDINAL_TASK_STAGES: dict[str, str] = {
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-Direct-v0": "c2a",
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2b-Direct-v0": "c2b",
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2c-Direct-v0": "c2c",
    GPU_IMPLICIT_PURE_RL_C2A_TASK_ID: "c2a",
    GPU_PHASE_MATCHED_PURE_RL_C2A_TASK_ID: "c2a",
}
MEASURED_PURE_RL_SPATIAL_TASK_STAGES: dict[str, str] = {
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0": "c3a",
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3b-Direct-v0": "c3b",
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3c-Direct-v0": "c3c",
}
PURE_RL_TASK_BACKENDS: dict[str, str] = {
    MEASURED_PURE_RL_TASK_ID: "cpu_native_authority",
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-Direct-v0": "cpu_native_authority",
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2b-Direct-v0": "cpu_native_authority",
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2c-Direct-v0": "cpu_native_authority",
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0": "cpu_native_authority",
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3b-Direct-v0": "cpu_native_authority",
    "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3c-Direct-v0": "cpu_native_authority",
    GPU_IMPLICIT_PURE_RL_TASK_ID: "gpu_implicit_candidate",
    GPU_IMPLICIT_PURE_RL_C2A_TASK_ID: "gpu_implicit_candidate",
    GPU_PHASE_MATCHED_PURE_RL_TASK_ID: "gpu_implicit_candidate",
    GPU_PHASE_MATCHED_PURE_RL_C2A_TASK_ID: "gpu_implicit_candidate",
}
PURE_RL_CURRICULUM1_EVAL_SUITE = "pure_rl_curriculum1_nowind_v2"
PURE_RL_CURRICULUM1_EVAL_CONTRACT = "pure_rl_curriculum1_v2"


@dataclass(frozen=True)
class PureRLEvaluationGate:
    """Pre-registered curriculum-1 success thresholds."""

    minimum_timeout_rate: float = 0.80
    maximum_termination_rate: float = 0.20
    maximum_mean_abs_cross_track_error_m: float = 0.50
    maximum_mean_abs_height_error_m: float = 0.50
    minimum_mean_along_track_progress_m: float = 0.0
    maximum_frequency_limit_fraction: float = 0.25
    maximum_tail_limit_fraction: float = 0.10


PURE_RL_CURRICULUM1_EVALUATION_GATE = PureRLEvaluationGate()


def is_measured_pure_rl_task(task: str) -> bool:
    """Return whether ``task`` belongs to the measured PureRL family."""

    return str(task) in PURE_RL_TASK_BACKENDS


def longitudinal_stage_for_task(task: str) -> str | None:
    """Return the C2 stage selected by a measured task, or ``None`` for C1."""

    return MEASURED_PURE_RL_LONGITUDINAL_TASK_STAGES.get(str(task))


def spatial_stage_for_task(task: str) -> str | None:
    """Return the C3 stage selected by a measured task, or ``None`` otherwise."""

    return MEASURED_PURE_RL_SPATIAL_TASK_STAGES.get(str(task))


def curriculum_stage_for_task(task: str) -> str | None:
    """Return the active C2/C3 curriculum stage, or ``None`` for C1."""

    return longitudinal_stage_for_task(task) or spatial_stage_for_task(task)


def pure_rl_backend_for_task(task: str) -> str:
    """Return the explicit backend contract for a measured PureRL task."""

    task_id = str(task)
    try:
        return PURE_RL_TASK_BACKENDS[task_id]
    except KeyError as error:
        raise ValueError(f"Unknown measured PureRL task: {task_id!r}.") from error


def allocate_episode_quotas(total_episodes: int, num_envs: int) -> tuple[int, ...]:
    """Allocate exact per-environment episode quotas without fast-failure bias."""

    if total_episodes <= 0:
        raise ValueError("total_episodes must be positive.")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    base, remainder = divmod(int(total_episodes), int(num_envs))
    return tuple(base + int(env_id < remainder) for env_id in range(num_envs))


def assert_pure_rl_reset_schedule(
    *,
    actual_heading_rad: torch.Tensor,
    actual_flap_phase_rad: torch.Tensor,
    expected_heading_schedule_rad: Sequence[float],
    expected_flap_phase_schedule_rad: Sequence[float],
    atol: float = 1.0e-5,
) -> None:
    """Fail closed unless reset state matches the registered heading/phase grid."""

    if len(expected_heading_schedule_rad) == 0 or len(expected_heading_schedule_rad) != len(
        expected_flap_phase_schedule_rad
    ):
        raise ValueError("PureRL evaluation schedules must be non-empty and aligned.")
    if actual_heading_rad.ndim != 1 or actual_flap_phase_rad.shape != actual_heading_rad.shape:
        raise ValueError("PureRL reset heading and flap phase must be aligned one-dimensional tensors.")

    device = actual_heading_rad.device
    env_ids = torch.arange(actual_heading_rad.numel(), device=device)
    heading_schedule = torch.as_tensor(expected_heading_schedule_rad, device=device, dtype=actual_heading_rad.dtype)
    phase_schedule = torch.as_tensor(
        expected_flap_phase_schedule_rad,
        device=device,
        dtype=actual_flap_phase_rad.dtype,
    )
    expected_heading = heading_schedule[env_ids % heading_schedule.numel()]
    expected_phase = torch.remainder(phase_schedule[env_ids % phase_schedule.numel()], 2.0 * math.pi)
    heading_error = torch.abs(torch.atan2(
        torch.sin(actual_heading_rad - expected_heading),
        torch.cos(actual_heading_rad - expected_heading),
    ))
    phase_error = torch.abs(torch.atan2(
        torch.sin(actual_flap_phase_rad - expected_phase),
        torch.cos(actual_flap_phase_rad - expected_phase),
    ))
    if bool(torch.any(heading_error > atol)) or bool(torch.any(phase_error > atol)):
        raise RuntimeError(
            "PureRL reset did not reproduce the registered heading/phase schedule: "
            f"max_heading_error={float(torch.max(heading_error).item()):.3e}, "
            f"max_phase_error={float(torch.max(phase_error).item()):.3e}."
        )


def assert_pure_rl_longitudinal_reset_schedule(
    *,
    actual_heading_rad: torch.Tensor,
    actual_flap_phase_rad: torch.Tensor,
    actual_task_id: torch.Tensor,
    actual_signed_slope_rad: torch.Tensor,
    expected_heading_schedule_rad: Sequence[float],
    expected_flap_phase_schedule_rad: Sequence[float],
    expected_task_schedule: Sequence[int],
    expected_signed_slope_deg_schedule: Sequence[float],
    atol: float = 1.0e-5,
) -> None:
    """Fail closed unless a C2 reset reproduces the full registered schedule."""

    assert_pure_rl_reset_schedule(
        actual_heading_rad=actual_heading_rad,
        actual_flap_phase_rad=actual_flap_phase_rad,
        expected_heading_schedule_rad=expected_heading_schedule_rad,
        expected_flap_phase_schedule_rad=expected_flap_phase_schedule_rad,
        atol=atol,
    )
    expected_length = len(expected_heading_schedule_rad)
    if len(expected_task_schedule) != expected_length or len(expected_signed_slope_deg_schedule) != expected_length:
        raise ValueError("PureRL longitudinal reset schedules must be aligned.")
    if actual_task_id.shape != actual_heading_rad.shape or actual_signed_slope_rad.shape != actual_heading_rad.shape:
        raise ValueError("PureRL longitudinal reset tensors must be aligned one-dimensional tensors.")
    env_ids = torch.arange(actual_heading_rad.numel(), device=actual_heading_rad.device)
    task_schedule = torch.as_tensor(expected_task_schedule, device=actual_task_id.device, dtype=actual_task_id.dtype)
    slope_schedule = torch.deg2rad(
        torch.as_tensor(
            expected_signed_slope_deg_schedule,
            device=actual_signed_slope_rad.device,
            dtype=actual_signed_slope_rad.dtype,
        )
    )
    expected_task = task_schedule[env_ids % expected_length]
    expected_slope = slope_schedule[env_ids % expected_length]
    if bool(torch.any(actual_task_id != expected_task)) or bool(
        torch.any(torch.abs(actual_signed_slope_rad - expected_slope) > atol)
    ):
        raise RuntimeError("PureRL reset did not reproduce the registered longitudinal task/slope schedule.")


def assert_pure_rl_spatial_reset_schedule(
    *,
    actual_heading_rad: torch.Tensor,
    actual_flap_phase_rad: torch.Tensor,
    actual_template_id: torch.Tensor,
    actual_geometry_roll_rad: torch.Tensor,
    actual_slope_rad: torch.Tensor,
    actual_turn_sign: torch.Tensor,
    expected_heading_schedule_rad: Sequence[float],
    expected_flap_phase_schedule_rad: Sequence[float],
    expected_template_schedule: Sequence[int],
    expected_geometry_roll_deg_schedule: Sequence[float],
    expected_slope_deg_schedule: Sequence[float],
    expected_turn_sign_schedule: Sequence[int],
    atol: float = 1.0e-5,
) -> None:
    """Fail closed unless a C3 reset reproduces the full registered schedule."""

    assert_pure_rl_reset_schedule(
        actual_heading_rad=actual_heading_rad,
        actual_flap_phase_rad=actual_flap_phase_rad,
        expected_heading_schedule_rad=expected_heading_schedule_rad,
        expected_flap_phase_schedule_rad=expected_flap_phase_schedule_rad,
        atol=atol,
    )
    expected_length = len(expected_heading_schedule_rad)
    schedules = (
        expected_template_schedule,
        expected_geometry_roll_deg_schedule,
        expected_slope_deg_schedule,
        expected_turn_sign_schedule,
    )
    if any(len(schedule) != expected_length for schedule in schedules):
        raise ValueError("PureRL spatial reset schedules must be aligned.")
    actual_tensors = (
        actual_template_id,
        actual_geometry_roll_rad,
        actual_slope_rad,
        actual_turn_sign,
    )
    if any(tensor.shape != actual_heading_rad.shape for tensor in actual_tensors):
        raise ValueError("PureRL spatial reset tensors must be aligned one-dimensional tensors.")
    env_ids = torch.arange(actual_heading_rad.numel(), device=actual_heading_rad.device)
    expected_template = torch.as_tensor(
        expected_template_schedule,
        device=actual_template_id.device,
        dtype=actual_template_id.dtype,
    )[env_ids % expected_length]
    expected_roll = torch.deg2rad(torch.as_tensor(
        expected_geometry_roll_deg_schedule,
        device=actual_geometry_roll_rad.device,
        dtype=actual_geometry_roll_rad.dtype,
    ))[env_ids % expected_length]
    expected_slope = torch.deg2rad(torch.as_tensor(
        expected_slope_deg_schedule,
        device=actual_slope_rad.device,
        dtype=actual_slope_rad.dtype,
    ))[env_ids % expected_length]
    expected_turn = torch.as_tensor(
        expected_turn_sign_schedule,
        device=actual_turn_sign.device,
        dtype=actual_turn_sign.dtype,
    )[env_ids % expected_length]
    mismatch = (
        torch.any(actual_template_id != expected_template)
        or torch.any(torch.abs(actual_geometry_roll_rad - expected_roll) > atol)
        or torch.any(torch.abs(actual_slope_rad - expected_slope) > atol)
        or torch.any(actual_turn_sign != expected_turn)
    )
    if bool(mismatch):
        raise RuntimeError("PureRL reset did not reproduce the registered spatial path schedule.")


def read_pure_rl_step_metrics(env) -> dict[str, torch.Tensor]:
    """Read the latest pre-reset route-relative evaluation buffers."""

    required = {
        "cross_track_error_m": "_eval_pure_rl_cross_track_error_m",
        "height_error_m": "_eval_pure_rl_height_error_m",
        "along_track_progress_m": "_eval_pure_rl_along_track_progress_m",
        "along_track_velocity_mps": "_eval_pure_rl_along_track_velocity_mps",
        "tilt_rad": "_eval_pure_rl_tilt_rad",
        "angular_rate_rad_s": "_eval_pure_rl_angular_rate_rad_s",
        "actual_flap_frequency_hz": "_eval_pure_rl_actual_flap_frequency_hz",
        "frequency_limit_active": "_eval_pure_rl_frequency_limit_active",
        "tail_limit_active": "_eval_pure_rl_tail_limit_active",
        "normalized_action_delta": "_eval_pure_rl_normalized_action_delta",
        "frequency_slew_hz_per_s": "_eval_pure_rl_frequency_slew_hz_per_s",
        "frequency_governor_limited": "_eval_pure_rl_frequency_governor_limited",
        "ground_termination": "_eval_pure_rl_ground_termination",
        "tilt_termination": "_eval_pure_rl_tilt_termination",
        "cross_track_termination": "_eval_pure_rl_cross_track_termination",
        "height_termination": "_eval_pure_rl_height_termination",
    }
    missing = [attribute for attribute in required.values() if not hasattr(env, attribute)]
    if missing:
        raise AttributeError(f"PureRL evaluation buffers are missing: {missing}")
    result = {
        name: getattr(env, attribute).detach().cpu().clone()
        for name, attribute in required.items()
    }
    if longitudinal_stage_for_task(getattr(getattr(env, "spec", None), "id", "")) is not None or getattr(
        env,
        "_pure_rl_longitudinal_path",
        None,
    ) is not None:
        longitudinal_required = {
            "lateral_normal_velocity_mps": "_eval_pure_rl_lateral_normal_velocity_mps",
            "vertical_normal_velocity_mps": "_eval_pure_rl_vertical_normal_velocity_mps",
            "active_slope_rad": "_eval_pure_rl_active_slope_rad",
            "reached_recovery": "_eval_pure_rl_reached_recovery",
        }
        missing_longitudinal = [
            attribute
            for attribute in longitudinal_required.values()
            if getattr(env, attribute, None) is None
        ]
        path = getattr(env, "_pure_rl_longitudinal_path", None)
        if missing_longitudinal or path is None:
            raise AttributeError(
                "PureRL longitudinal evaluation buffers are missing: "
                f"{missing_longitudinal}"
            )
        result.update(
            {
                name: getattr(env, attribute).detach().cpu().clone()
                for name, attribute in longitudinal_required.items()
            }
        )
        result["sampled_task_id"] = path.task_id.detach().cpu().clone()
        result["sampled_signed_slope_rad"] = path.signed_slope_rad.detach().cpu().clone()
    if spatial_stage_for_task(getattr(getattr(env, "spec", None), "id", "")) is not None or getattr(
        env,
        "_pure_rl_spatial_path",
        None,
    ) is not None:
        spatial_required = {
            "lateral_normal_velocity_mps": "_eval_pure_rl_lateral_normal_velocity_mps",
            "vertical_normal_velocity_mps": "_eval_pure_rl_vertical_normal_velocity_mps",
            "active_slope_rad": "_eval_pure_rl_active_slope_rad",
            "active_curvature_rad_per_m": "_eval_pure_rl_active_curvature_rad_per_m",
            "turn_activity": "_eval_pure_rl_turn_activity",
            "reached_all_events": "_eval_pure_rl_reached_all_events",
            "roll_limit_termination": "_eval_pure_rl_roll_limit_termination",
            "abs_roll_rad": "_eval_pure_rl_abs_roll_rad",
        }
        missing_spatial = [
            attribute for attribute in spatial_required.values() if getattr(env, attribute, None) is None
        ]
        path = getattr(env, "_pure_rl_spatial_path", None)
        if missing_spatial or path is None:
            raise AttributeError(f"PureRL spatial evaluation buffers are missing: {missing_spatial}")
        result.update(
            {
                name: getattr(env, attribute).detach().cpu().clone()
                for name, attribute in spatial_required.items()
            }
        )
        result["sampled_template_id"] = path.template_id.detach().cpu().clone()
        result["sampled_turn_sign"] = path.turn_sign.detach().cpu().clone()
        result["sampled_vertical_sign"] = path.vertical_sign.detach().cpu().clone()
        result["sampled_geometry_roll_rad"] = path.peak_geometry_roll_rad.detach().cpu().clone()
        result["sampled_slope_rad"] = path.peak_slope_rad.detach().cpu().clone()
    return result


def summarize_pure_rl_episode(
    *,
    step_metrics: Mapping[str, Sequence[float]],
    step_dt_s: float,
    terminated: bool,
    time_out: bool,
    termination_causes: Mapping[str, bool],
) -> dict[str, float | int]:
    """Summarize one fixed-condition PureRL episode."""

    if step_dt_s <= 0.0:
        raise ValueError("step_dt_s must be positive.")
    required = (
        "cross_track_error_m",
        "height_error_m",
        "along_track_progress_m",
        "along_track_velocity_mps",
        "tilt_rad",
        "angular_rate_rad_s",
        "actual_flap_frequency_hz",
        "frequency_limit_active",
        "tail_limit_active",
        "normalized_action_delta",
    )
    lengths = {name: len(step_metrics[name]) for name in required}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values()), 0) <= 0:
        raise ValueError(f"PureRL step metrics must be non-empty and aligned, got {lengths}.")

    count = next(iter(lengths.values()))
    cross_track = [abs(float(value)) for value in step_metrics["cross_track_error_m"]]
    height = [abs(float(value)) for value in step_metrics["height_error_m"]]
    tilt_deg = [math.degrees(float(value)) for value in step_metrics["tilt_rad"]]
    along_velocity = [float(value) for value in step_metrics["along_track_velocity_mps"]]
    frequency_slew = step_metrics.get("frequency_slew_hz_per_s", [0.0] * count)
    governor_limited = step_metrics.get("frequency_governor_limited", [0.0] * count)
    if len(frequency_slew) != count or len(governor_limited) != count:
        raise ValueError("PureRL frequency-governor metrics must align with the episode trace.")
    return {
        "episode_duration_s": float(count) * float(step_dt_s),
        "along_track_progress_m": float(step_metrics["along_track_progress_m"][-1]),
        "mean_along_track_velocity_mps": _mean(along_velocity),
        "reverse_motion_fraction": _mean([float(value < 0.0) for value in along_velocity]),
        "mean_abs_cross_track_error_m": _mean(cross_track),
        "max_abs_cross_track_error_m": max(cross_track),
        "mean_abs_height_error_m": _mean(height),
        "max_abs_height_error_m": max(height),
        "mean_tilt_deg": _mean(tilt_deg),
        "max_tilt_deg": max(tilt_deg),
        "mean_angular_rate_rad_s": _mean(step_metrics["angular_rate_rad_s"]),
        "mean_actual_flap_frequency_hz": _mean(step_metrics["actual_flap_frequency_hz"]),
        "frequency_limit_fraction": _mean(step_metrics["frequency_limit_active"]),
        "tail_limit_fraction": _mean(step_metrics["tail_limit_active"]),
        "mean_normalized_action_delta": _mean(step_metrics["normalized_action_delta"]),
        "mean_abs_frequency_slew_hz_per_s": _mean(
            [abs(float(value)) for value in frequency_slew]
        ),
        "frequency_governor_limited_fraction": _mean(governor_limited),
        "terminated": int(bool(terminated)),
        "time_out": int(bool(time_out)),
        "ground_termination": int(bool(termination_causes["ground"])),
        "tilt_termination": int(bool(termination_causes["tilt"])),
        "cross_track_termination": int(bool(termination_causes["cross_track"])),
        "height_termination": int(bool(termination_causes["height"])),
    }


def aggregate_pure_rl_case_row(
    *,
    checkpoint: Path,
    ckpt_index: int,
    case: Mapping[str, object],
    episode_rows: Sequence[Mapping[str, float | int]],
) -> dict[str, object]:
    """Aggregate equally weighted PureRL episodes into one evaluation row."""

    if not episode_rows:
        raise ValueError("episode_rows must not be empty.")
    row: dict[str, object] = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "evaluation_contract": PURE_RL_CURRICULUM1_EVAL_CONTRACT,
        "case": str(case["name"]),
        "ckpt_index": int(ckpt_index),
        "episodes": len(episode_rows),
    }
    mean_fields = (
        "episode_duration_s",
        "along_track_progress_m",
        "mean_along_track_velocity_mps",
        "reverse_motion_fraction",
        "mean_abs_cross_track_error_m",
        "max_abs_cross_track_error_m",
        "mean_abs_height_error_m",
        "max_abs_height_error_m",
        "mean_tilt_deg",
        "max_tilt_deg",
        "mean_angular_rate_rad_s",
        "mean_actual_flap_frequency_hz",
        "frequency_limit_fraction",
        "tail_limit_fraction",
        "mean_normalized_action_delta",
        "mean_abs_frequency_slew_hz_per_s",
        "frequency_governor_limited_fraction",
    )
    output_names = {
        "episode_duration_s": "mean_episode_duration_s",
        "along_track_progress_m": "mean_along_track_progress_m",
        "max_abs_cross_track_error_m": "mean_max_abs_cross_track_error_m",
        "max_abs_height_error_m": "mean_max_abs_height_error_m",
        "max_tilt_deg": "mean_max_tilt_deg",
    }
    for field_name in mean_fields:
        row[output_names.get(field_name, field_name)] = _mean(
            [float(episode[field_name]) for episode in episode_rows]
        )

    rate_fields = (
        "terminated",
        "time_out",
        "ground_termination",
        "tilt_termination",
        "cross_track_termination",
        "height_termination",
    )
    output_rates = {
        "terminated": "termination_rate",
        "time_out": "timeout_rate",
        "ground_termination": "ground_termination_rate",
        "tilt_termination": "tilt_termination_rate",
        "cross_track_termination": "cross_track_termination_rate",
        "height_termination": "height_termination_rate",
    }
    for field_name in rate_fields:
        row[output_rates[field_name]] = _mean([float(episode[field_name]) for episode in episode_rows])

    row.update(
        {
            "heading_phase_pairs": len(tuple(case.get("straight_line_heading_schedule_rad", ()))),
            "wind_enabled": int(bool(case["wind_enabled"])),
            "wind_x_mps": float(case["wind_xy_mps"][0]),
            "wind_y_mps": float(case["wind_xy_mps"][1]),
            "wind_ou_enabled": int(bool(case["wind_ou_enabled"])),
        }
    )
    row["success_gate_passed"] = int(row_meets_pure_rl_success_gate(row))
    row["score"] = compute_pure_rl_score(row)
    return row


def aggregate_pure_rl_suite_row(case_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate PureRL case rows while preserving the versioned schema."""

    if not case_rows:
        raise ValueError("case_rows must not be empty.")
    if len(case_rows) == 1:
        row = dict(case_rows[0])
        row["case"] = "suite"
        return row

    total_episodes = sum(int(row["episodes"]) for row in case_rows)
    if total_episodes <= 0:
        raise ValueError("PureRL case rows must contain positive episode counts.")
    row = dict(case_rows[0])
    row["case"] = "suite"
    row["episodes"] = total_episodes
    excluded = {"checkpoint", "evaluation_contract", "case", "ckpt_index", "episodes", "score", "success_gate_passed"}
    for key in tuple(row):
        if key in excluded or not isinstance(row[key], (int, float)):
            continue
        row[key] = sum(float(case[key]) * int(case["episodes"]) for case in case_rows) / total_episodes
    row["success_gate_passed"] = int(row_meets_pure_rl_success_gate(row))
    row["score"] = compute_pure_rl_score(row)
    return row


def row_meets_pure_rl_success_gate(
    row: Mapping[str, object],
    gate: PureRLEvaluationGate = PURE_RL_CURRICULUM1_EVALUATION_GATE,
) -> bool:
    """Return the pre-registered curriculum-1 success decision."""

    return bool(
        float(row["timeout_rate"]) >= gate.minimum_timeout_rate
        and float(row["termination_rate"]) <= gate.maximum_termination_rate
        and float(row["mean_abs_cross_track_error_m"]) <= gate.maximum_mean_abs_cross_track_error_m
        and float(row["mean_abs_height_error_m"]) <= gate.maximum_mean_abs_height_error_m
        and float(row["mean_along_track_progress_m"]) > gate.minimum_mean_along_track_progress_m
        and float(row["frequency_limit_fraction"]) <= gate.maximum_frequency_limit_fraction
        and float(row["tail_limit_fraction"]) <= gate.maximum_tail_limit_fraction
    )


def compute_pure_rl_score(row: Mapping[str, object]) -> float:
    """Compute a diagnostic score without a target-speed error."""

    timeout_score = _clip01(float(row["timeout_rate"]))
    progress_score = math.tanh(max(float(row["mean_along_track_progress_m"]), 0.0) / 12.0)
    cross_track_score = math.exp(-math.pow(float(row["mean_abs_cross_track_error_m"]) / 0.50, 2.0))
    height_score = math.exp(-math.pow(float(row["mean_abs_height_error_m"]) / 0.50, 2.0))
    tilt_score = math.exp(-math.pow(float(row["mean_max_tilt_deg"]) / 45.0, 2.0))
    authority_score = 1.0 - 0.5 * (
        _clip01(float(row["frequency_limit_fraction"]))
        + _clip01(float(row["tail_limit_fraction"]))
    )
    return 100.0 * (
        0.40 * timeout_score
        + 0.20 * progress_score
        + 0.15 * cross_track_score
        + 0.15 * height_score
        + 0.05 * tilt_score
        + 0.05 * authority_score
    )


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

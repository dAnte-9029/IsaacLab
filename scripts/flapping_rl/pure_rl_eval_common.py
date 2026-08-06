"""Shared PureRL curriculum-1 evaluation and checkpoint-scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch


MEASURED_PURE_RL_TASK_ID = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0"
PURE_RL_CURRICULUM1_EVAL_SUITE = "pure_rl_curriculum1_nowind_v1"
PURE_RL_CURRICULUM1_EVAL_CONTRACT = "pure_rl_curriculum1_v1"


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
    """Return whether ``task`` is the canonical measured PureRL task."""

    return str(task) == MEASURED_PURE_RL_TASK_ID


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
        "ground_termination": "_eval_pure_rl_ground_termination",
        "tilt_termination": "_eval_pure_rl_tilt_termination",
        "cross_track_termination": "_eval_pure_rl_cross_track_termination",
        "height_termination": "_eval_pure_rl_height_termination",
    }
    missing = [attribute for attribute in required.values() if not hasattr(env, attribute)]
    if missing:
        raise AttributeError(f"PureRL evaluation buffers are missing: {missing}")
    return {
        name: getattr(env, attribute).detach().cpu().clone()
        for name, attribute in required.items()
    }


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

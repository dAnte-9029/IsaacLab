"""Shared helpers for path-tracking evaluation and checkpoint scoring."""

from __future__ import annotations

from pathlib import Path


def is_path_tracking_task(task: str) -> bool:
    """Return whether the gym task id is a path-tracking task."""
    return "PathTracking" in str(task)


def resolve_eval_suite(task: str, eval_suite: str) -> str:
    """Resolve the effective evaluation suite for the requested task."""
    if eval_suite == "straight_standard" and is_path_tracking_task(task):
        return "path_tracking_truth_nowind_v1"
    return str(eval_suite)


def apply_eval_case_to_cfg(case: dict, cfg, *, vx_cmd: float | None, height_cmd: float | None) -> None:
    """Apply one evaluation-case override set onto an environment config."""
    cfg.randomize_commands = False
    if vx_cmd is not None:
        cfg.vx_cmd = float(vx_cmd)
    if height_cmd is not None:
        cfg.height_cmd = float(height_cmd)

    if hasattr(cfg, "teacher_guidance_enabled"):
        cfg.teacher_guidance_enabled = False
    if hasattr(cfg, "wind_curriculum_enabled"):
        cfg.wind_curriculum_enabled = False

    cfg.wind_enabled = bool(case["wind_enabled"])
    cfg.randomize_wind = False
    cfg.wind_xy_mps = tuple(float(v) for v in case["wind_xy_mps"])
    cfg.wind_x_range_mps = (float(case["wind_xy_mps"][0]), float(case["wind_xy_mps"][0]))
    cfg.wind_y_range_mps = (float(case["wind_xy_mps"][1]), float(case["wind_xy_mps"][1]))
    cfg.wind_ou_enabled = bool(case["wind_ou_enabled"])
    cfg.wind_ou_tau_s = float(case["wind_ou_tau_s"])
    cfg.wind_ou_sigma_xy_mps = tuple(float(v) for v in case["wind_ou_sigma_xy_mps"])
    cfg.wind_ou_clip_to_range = False

    mission_fields = (
        "mission_seed",
        "mission_increment_seed_per_reset",
        "mission_num_segments_min",
        "mission_num_segments_max",
        "mission_allow_straight",
        "mission_allow_turn",
        "mission_allow_loiter",
        "mission_allow_climb_on_straight",
    )
    for field_name in mission_fields:
        if field_name in case and hasattr(cfg, field_name):
            setattr(cfg, field_name, case[field_name])


def compute_path_tracking_score(
    *,
    completion_rate: float,
    mean_abs_lateral_error_m: float,
    mean_abs_height_error_m: float,
    mean_abs_align_error_deg: float,
    termination_rate: float,
) -> float:
    """Compute a scalar score for one aggregated path-tracking evaluation row."""
    completion_deficit = max(0.0, 1.0 - float(completion_rate))
    cost = (
        4.0 * completion_deficit
        + float(mean_abs_lateral_error_m)
        + float(mean_abs_height_error_m)
        + 0.02 * float(mean_abs_align_error_deg)
        + 2.0 * float(termination_rate)
    )
    return 100.0 / (1.0 + cost)


def aggregate_case_row(
    *,
    checkpoint: Path,
    ckpt_index: int,
    case: dict,
    episode_rows: list[dict[str, float | int]],
) -> dict[str, float | int | str]:
    """Aggregate per-episode path-tracking metrics into one summary row."""
    if len(episode_rows) == 0:
        raise ValueError("episode_rows must not be empty.")

    num_episodes = len(episode_rows)

    def _mean(key: str) -> float:
        return float(sum(float(row[key]) for row in episode_rows) / num_episodes)

    row = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "case": str(case["name"]),
        "ckpt_index": int(ckpt_index),
        "episodes": int(num_episodes),
        "mean_abs_lateral_error_m": _mean("mean_abs_lateral_error_m"),
        "mean_abs_height_error_m": _mean("mean_abs_height_error_m"),
        "mean_abs_align_error_deg": _mean("mean_abs_align_error_deg"),
        "mean_p95_abs_lateral_error_m": _mean("p95_abs_lateral_error_m"),
        "mean_p95_abs_height_error_m": _mean("p95_abs_height_error_m"),
        "mean_p95_abs_align_error_deg": _mean("p95_abs_align_error_deg"),
        "mean_final_progress_ratio": _mean("final_progress_ratio"),
        "completion_rate": _mean("completed"),
        "termination_rate": _mean("terminated"),
        "timeout_rate": _mean("time_out"),
        "wind_enabled": int(bool(case["wind_enabled"])),
        "wind_x_mps": float(case["wind_xy_mps"][0]),
        "wind_y_mps": float(case["wind_xy_mps"][1]),
        "wind_ou_enabled": int(bool(case["wind_ou_enabled"])),
        "wind_ou_sigma_x_mps": float(case["wind_ou_sigma_xy_mps"][0]),
        "wind_ou_sigma_y_mps": float(case["wind_ou_sigma_xy_mps"][1]),
    }
    row["score"] = compute_path_tracking_score(
        completion_rate=float(row["completion_rate"]),
        mean_abs_lateral_error_m=float(row["mean_abs_lateral_error_m"]),
        mean_abs_height_error_m=float(row["mean_abs_height_error_m"]),
        mean_abs_align_error_deg=float(row["mean_abs_align_error_deg"]),
        termination_rate=float(row["termination_rate"]),
    )
    return row


def aggregate_suite_row(case_rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    """Aggregate multiple case rows into one suite row."""
    if len(case_rows) == 0:
        raise ValueError("case_rows must not be empty.")

    total_episodes = sum(int(row["episodes"]) for row in case_rows)
    if total_episodes <= 0:
        raise ValueError("total episodes must be positive.")

    def _weighted_mean(key: str) -> float:
        return float(
            sum(float(row[key]) * int(row["episodes"]) for row in case_rows) / total_episodes
        )

    suite_row = {
        "checkpoint": str(case_rows[0]["checkpoint"]),
        "case": "suite",
        "ckpt_index": int(case_rows[0]["ckpt_index"]),
        "episodes": int(total_episodes),
        "mean_abs_lateral_error_m": _weighted_mean("mean_abs_lateral_error_m"),
        "mean_abs_height_error_m": _weighted_mean("mean_abs_height_error_m"),
        "mean_abs_align_error_deg": _weighted_mean("mean_abs_align_error_deg"),
        "mean_p95_abs_lateral_error_m": _weighted_mean("mean_p95_abs_lateral_error_m"),
        "mean_p95_abs_height_error_m": _weighted_mean("mean_p95_abs_height_error_m"),
        "mean_p95_abs_align_error_deg": _weighted_mean("mean_p95_abs_align_error_deg"),
        "mean_final_progress_ratio": _weighted_mean("mean_final_progress_ratio"),
        "completion_rate": _weighted_mean("completion_rate"),
        "termination_rate": _weighted_mean("termination_rate"),
        "timeout_rate": _weighted_mean("timeout_rate"),
        "wind_enabled": 0,
        "wind_x_mps": float("nan"),
        "wind_y_mps": float("nan"),
        "wind_ou_enabled": 0,
        "wind_ou_sigma_x_mps": float("nan"),
        "wind_ou_sigma_y_mps": float("nan"),
    }
    suite_row["score"] = compute_path_tracking_score(
        completion_rate=float(suite_row["completion_rate"]),
        mean_abs_lateral_error_m=float(suite_row["mean_abs_lateral_error_m"]),
        mean_abs_height_error_m=float(suite_row["mean_abs_height_error_m"]),
        mean_abs_align_error_deg=float(suite_row["mean_abs_align_error_deg"]),
        termination_rate=float(suite_row["termination_rate"]),
    )
    return suite_row

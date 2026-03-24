"""Shared helpers for path-tracking evaluation and checkpoint scoring."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from path_tracking_success_gate import row_meets_path_tracking_success_gate


_DEFAULT_STRAIGHT_LENGTH_M = 60.0
_DEFAULT_TURN_RADIUS_M = 20.0
_DEFAULT_LOITER_RADIUS_M = 20.0
_DEFAULT_TURN_SWEEP_DEG = 90.0
_DEFAULT_LOITER_TURNS = 1.0
_DEFAULT_COMPLETION_MARGIN_S = 2.0


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values = sorted(float(v) for v in values)
    idx = (len(values) - 1) * max(0.0, min(1.0, q))
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return values[lower]
    alpha = idx - lower
    return values[lower] + (values[upper] - values[lower]) * alpha


def _single_segment_kind(case: dict) -> str | None:
    enabled = [
        kind
        for kind, is_enabled in (
            ("straight", bool(case.get("mission_allow_straight", False))),
            ("turn", bool(case.get("mission_allow_turn", False))),
            ("loiter", bool(case.get("mission_allow_loiter", False))),
        )
        if is_enabled
    ]
    if len(enabled) != 1:
        return None
    return enabled[0]


def compute_path_tracking_eval_episode_length_s(case: dict, cfg, *, vx_cmd: float | None) -> float:
    """Return a fair evaluation horizon for simple single-segment path cases."""
    current_episode_length_s = float(getattr(cfg, "episode_length_s", 0.0))
    segment_kind = _single_segment_kind(case)
    if segment_kind is None:
        return current_episode_length_s

    speed_mps = float(vx_cmd) if vx_cmd is not None else float(getattr(cfg, "vx_cmd", 7.0))
    speed_mps = max(abs(speed_mps), 1.0)

    sim_cfg = getattr(cfg, "sim", None)
    sim_dt = float(getattr(sim_cfg, "dt", 1.0 / 240.0))
    freeze_steps = max(int(getattr(cfg, "freeze_steps_after_reset", 0)), 0)
    freeze_time_s = freeze_steps * sim_dt

    if segment_kind == "straight":
        path_length_m = float(getattr(cfg, "straight_length_m", _DEFAULT_STRAIGHT_LENGTH_M))
    elif segment_kind == "turn":
        turn_radius_m = float(getattr(cfg, "turn_radius_m", _DEFAULT_TURN_RADIUS_M))
        turn_sweep_deg = float(getattr(cfg, "turn_sweep_deg", _DEFAULT_TURN_SWEEP_DEG))
        path_length_m = abs(turn_radius_m * math.radians(turn_sweep_deg))
    else:
        loiter_radius_m = float(getattr(cfg, "loiter_radius_m", _DEFAULT_LOITER_RADIUS_M))
        loiter_turns = float(getattr(cfg, "loiter_turns", _DEFAULT_LOITER_TURNS))
        path_length_m = abs(loiter_radius_m * 2.0 * math.pi * loiter_turns)

    required_episode_length_s = freeze_time_s + path_length_m / speed_mps + _DEFAULT_COMPLETION_MARGIN_S
    return max(current_episode_length_s, required_episode_length_s)


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
    if hasattr(cfg, "mission_curriculum_enabled"):
        cfg.mission_curriculum_enabled = False

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

    if hasattr(cfg, "episode_length_s"):
        cfg.episode_length_s = compute_path_tracking_eval_episode_length_s(case, cfg, vx_cmd=vx_cmd)


def read_path_tracking_step_metrics(env) -> dict[str, torch.Tensor]:
    """Read per-env path-tracking metrics from the latest pre-reset step."""
    num_envs = int(env.num_envs)
    default_nan = torch.full((num_envs,), float("nan"), dtype=torch.float32)
    default_false = torch.zeros((num_envs,), dtype=torch.bool)
    if all(
        hasattr(env, attr)
        for attr in (
            "_eval_abs_lateral_error_m",
            "_eval_abs_height_error_m",
            "_eval_abs_align_error_deg",
            "_eval_progress_ratio",
            "_eval_stalled",
        )
    ):
        return {
            "abs_lateral_error_m": env._eval_abs_lateral_error_m.detach().cpu().clone(),
            "abs_height_error_m": env._eval_abs_height_error_m.detach().cpu().clone(),
            "abs_align_error_deg": env._eval_abs_align_error_deg.detach().cpu().clone(),
            "progress_ratio": env._eval_progress_ratio.detach().cpu().clone(),
            "loiter_radial_error_m": getattr(env, "_eval_loiter_radial_error_m", default_nan).detach().cpu().clone(),
            "loiter_progress_ratio": getattr(env, "_eval_loiter_progress_ratio", default_nan).detach().cpu().clone(),
            "loiter_quarter_turn_complete": getattr(env, "_eval_loiter_quarter_turn_complete", default_false)
            .detach()
            .cpu()
            .clone(),
            "stalled": env._eval_stalled.detach().cpu().clone(),
        }

    lateral_error = torch.abs(env._path_lateral_error_m).detach().cpu()
    height_error = torch.abs(env._path_height_error_m).detach().cpu()
    align_error_deg = torch.rad2deg(torch.abs(env._path_align_error_rad)).detach().cpu()
    progress_s = env._path_progress_s.detach().cpu()
    progress_ratio = torch.zeros(num_envs, dtype=progress_s.dtype)

    for env_id in range(num_envs):
        manager = env._path_managers[env_id]
        total_length = float(manager.total_length_m) if manager is not None else 1.0
        progress_ratio[env_id] = min(max(float(progress_s[env_id].item()) / max(total_length, 1.0e-6), 0.0), 1.0)

    return {
        "abs_lateral_error_m": lateral_error,
        "abs_height_error_m": height_error,
        "abs_align_error_deg": align_error_deg,
        "progress_ratio": progress_ratio,
        "loiter_radial_error_m": getattr(env, "_path_loiter_radial_error_m", default_nan).detach().cpu().clone(),
        "loiter_progress_ratio": getattr(env, "_path_loiter_progress_ratio", default_nan).detach().cpu().clone(),
        "loiter_quarter_turn_complete": getattr(env, "_path_loiter_quarter_turn_complete", default_false)
        .detach()
        .cpu()
        .clone(),
        "stalled": getattr(env, "_stalled", default_false).detach().cpu().clone(),
    }


def compute_path_tracking_finish_masks(
    *,
    progress_ratio: torch.Tensor,
    dones: torch.Tensor,
    completion_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return finish masks for path-tracking evaluation episodes.

    The returned masks are:
    - finish_mask: episodes that should be finalized now
    - early_success_mask: completed episodes that need an eval-side reset
    - completed_mask: episodes that reached the mission completion threshold
    """
    done_mask = dones.to(dtype=torch.bool)
    completed_mask = progress_ratio.to(device=dones.device) >= float(completion_ratio)
    finish_mask = done_mask | completed_mask
    early_success_mask = completed_mask & ~done_mask
    return finish_mask, early_success_mask, completed_mask


def summarize_path_tracking_episode(
    *,
    step_abs_lateral: list[float],
    step_abs_height: list[float],
    step_abs_align_deg: list[float],
    step_loiter_radial_error: list[float] | None = None,
    step_loiter_progress_ratio: list[float] | None = None,
    loiter_quarter_turn_complete: bool = False,
    final_progress_ratio: float,
    completion_ratio: float,
    terminated: bool,
    time_out: bool,
    stalled: bool,
) -> dict[str, float | int]:
    """Summarize one path-tracking episode from accumulated step metrics."""
    completed = float(final_progress_ratio) >= float(completion_ratio)
    mean_loiter_radial_error_m = float("nan")
    if step_loiter_radial_error:
        mean_loiter_radial_error_m = float(sum(step_loiter_radial_error) / len(step_loiter_radial_error))
    mean_loiter_progress_ratio = float("nan")
    if step_loiter_progress_ratio:
        mean_loiter_progress_ratio = float(sum(step_loiter_progress_ratio) / len(step_loiter_progress_ratio))
    return {
        "mean_abs_lateral_error_m": float(sum(step_abs_lateral) / max(len(step_abs_lateral), 1)),
        "mean_abs_height_error_m": float(sum(step_abs_height) / max(len(step_abs_height), 1)),
        "mean_abs_align_error_deg": float(sum(step_abs_align_deg) / max(len(step_abs_align_deg), 1)),
        "p95_abs_lateral_error_m": _percentile(step_abs_lateral, 0.95),
        "p95_abs_height_error_m": _percentile(step_abs_height, 0.95),
        "p95_abs_align_error_deg": _percentile(step_abs_align_deg, 0.95),
        "mean_loiter_radial_error_m": mean_loiter_radial_error_m,
        "mean_loiter_progress_ratio": mean_loiter_progress_ratio,
        "quarter_turn_completed": int(bool(loiter_quarter_turn_complete)),
        "final_progress_ratio": float(final_progress_ratio),
        "completed": int(completed),
        "terminated": int(False if completed else terminated),
        "time_out": int(False if completed else time_out),
        "stalled": int(False if completed else stalled),
    }


def reset_path_tracking_eval_envs(vec_env, env_ids: torch.Tensor):
    """Reset selected vectorized envs during evaluation and return refreshed observations."""
    if env_ids.numel() == 0:
        return None

    from tensordict import TensorDict

    base_env = vec_env.unwrapped
    base_env._reset_idx(env_ids)
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()

    if base_env.sim.has_rtx_sensors() and getattr(base_env.cfg, "num_rerenders_on_reset", 0) > 0:
        for _ in range(int(base_env.cfg.num_rerenders_on_reset)):
            base_env.sim.render()

    obs_dict = base_env._get_observations()
    if getattr(base_env.cfg, "observation_noise_model", None):
        obs_dict["policy"] = base_env._observation_noise_model(obs_dict["policy"])
    base_env.obs_buf = obs_dict
    return TensorDict(obs_dict, batch_size=[vec_env.num_envs])


def compute_path_tracking_score(
    *,
    completion_rate: float,
    mean_final_progress_ratio: float | None = None,
    mean_abs_lateral_error_m: float,
    mean_abs_height_error_m: float,
    mean_abs_align_error_deg: float,
    termination_rate: float,
) -> float:
    """Compute a scalar score for one aggregated path-tracking evaluation row."""
    completion_ratio = min(max(float(completion_rate), 0.0), 1.0)
    progress_ratio = completion_ratio if mean_final_progress_ratio is None else min(max(float(mean_final_progress_ratio), 0.0), 1.0)
    mission_signal = 0.75 * completion_ratio + 0.25 * progress_ratio
    tracking_cost = (
        + float(mean_abs_lateral_error_m)
        + float(mean_abs_height_error_m)
        + 0.02 * float(mean_abs_align_error_deg)
        + 2.0 * float(termination_rate)
    )
    return 100.0 * mission_signal / (1.0 + tracking_cost)


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

    def _mean(key: str, *, default: float = 0.0, skip_nan: bool = False) -> float:
        values = [float(row.get(key, default)) for row in episode_rows]
        if skip_nan:
            values = [value for value in values if not math.isnan(value)]
            if len(values) == 0:
                return float("nan")
        return float(sum(values) / max(len(values), 1))

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
        "mean_loiter_radial_error_m": _mean("mean_loiter_radial_error_m", default=float("nan"), skip_nan=True),
        "mean_loiter_progress_ratio": _mean("mean_loiter_progress_ratio", default=float("nan"), skip_nan=True),
        "quarter_turn_rate": _mean("quarter_turn_completed", default=float("nan"), skip_nan=True),
        "mean_final_progress_ratio": _mean("final_progress_ratio"),
        "completion_rate": _mean("completed"),
        "termination_rate": _mean("terminated"),
        "timeout_rate": _mean("time_out"),
        "stall_rate": _mean("stalled"),
        "wind_enabled": int(bool(case["wind_enabled"])),
        "wind_x_mps": float(case["wind_xy_mps"][0]),
        "wind_y_mps": float(case["wind_xy_mps"][1]),
        "wind_ou_enabled": int(bool(case["wind_ou_enabled"])),
        "wind_ou_sigma_x_mps": float(case["wind_ou_sigma_xy_mps"][0]),
        "wind_ou_sigma_y_mps": float(case["wind_ou_sigma_xy_mps"][1]),
    }
    row["score"] = compute_path_tracking_score(
        completion_rate=float(row["completion_rate"]),
        mean_final_progress_ratio=float(row["mean_final_progress_ratio"]),
        mean_abs_lateral_error_m=float(row["mean_abs_lateral_error_m"]),
        mean_abs_height_error_m=float(row["mean_abs_height_error_m"]),
        mean_abs_align_error_deg=float(row["mean_abs_align_error_deg"]),
        termination_rate=float(row["termination_rate"]),
    )
    row["success_gate_passed"] = int(row_meets_path_tracking_success_gate(row))
    return row


def aggregate_suite_row(case_rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    """Aggregate multiple case rows into one suite row."""
    if len(case_rows) == 0:
        raise ValueError("case_rows must not be empty.")

    total_episodes = sum(int(row["episodes"]) for row in case_rows)
    if total_episodes <= 0:
        raise ValueError("total episodes must be positive.")

    def _weighted_mean(key: str, *, default: float = 0.0, skip_nan: bool = False) -> float:
        weighted_sum = 0.0
        weight_total = 0
        for row in case_rows:
            value = float(row.get(key, default))
            if skip_nan and math.isnan(value):
                continue
            episodes = int(row["episodes"])
            weighted_sum += value * episodes
            weight_total += episodes
        if skip_nan and weight_total == 0:
            return float("nan")
        return float(weighted_sum / max(weight_total, 1))

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
        "mean_loiter_radial_error_m": _weighted_mean("mean_loiter_radial_error_m", default=float("nan"), skip_nan=True),
        "mean_loiter_progress_ratio": _weighted_mean("mean_loiter_progress_ratio", default=float("nan"), skip_nan=True),
        "quarter_turn_rate": _weighted_mean("quarter_turn_rate", default=float("nan"), skip_nan=True),
        "mean_final_progress_ratio": _weighted_mean("mean_final_progress_ratio"),
        "completion_rate": _weighted_mean("completion_rate"),
        "termination_rate": _weighted_mean("termination_rate"),
        "timeout_rate": _weighted_mean("timeout_rate"),
        "stall_rate": _weighted_mean("stall_rate"),
        "wind_enabled": 0,
        "wind_x_mps": float("nan"),
        "wind_y_mps": float("nan"),
        "wind_ou_enabled": 0,
        "wind_ou_sigma_x_mps": float("nan"),
        "wind_ou_sigma_y_mps": float("nan"),
    }
    suite_row["score"] = compute_path_tracking_score(
        completion_rate=float(suite_row["completion_rate"]),
        mean_final_progress_ratio=float(suite_row["mean_final_progress_ratio"]),
        mean_abs_lateral_error_m=float(suite_row["mean_abs_lateral_error_m"]),
        mean_abs_height_error_m=float(suite_row["mean_abs_height_error_m"]),
        mean_abs_align_error_deg=float(suite_row["mean_abs_align_error_deg"]),
        termination_rate=float(suite_row["termination_rate"]),
    )
    suite_row["success_gate_passed"] = int(row_meets_path_tracking_success_gate(suite_row))
    return suite_row

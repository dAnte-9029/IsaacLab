"""Run pure teacher PX4-like path-mission rollouts on the generic path-tracking env."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FLAPPING_BOT_SOURCE = REPO_ROOT / "source" / "flapping_bot"
if str(FLAPPING_BOT_SOURCE) not in sys.path:
    sys.path.append(str(FLAPPING_BOT_SOURCE))

try:
    from flapping_bot.path_tracking.mission_primitives import Mission, MissionGeneratorCfg, MissionSegment, sample_mission
except ModuleNotFoundError:
    from flapping_bot.flapping_bot.path_tracking.mission_primitives import (
        Mission,
        MissionGeneratorCfg,
        MissionSegment,
        sample_mission,
    )


def _compute_injected_path_episode_length_s(
    *,
    current_episode_length_s: float,
    path_total_length_m: float,
    speed_ref_mps: float,
    freeze_steps: int,
    sim_dt: float,
    completion_margin_s: float,
    max_episode_length_s: float,
) -> float:
    try:
        from flapping_bot.direct.flapping_bot.path_tracking_env import _compute_path_episode_length_s
    except ModuleNotFoundError:
        from flapping_bot.flapping_bot.direct.flapping_bot.path_tracking_env import _compute_path_episode_length_s

    return _compute_path_episode_length_s(
        current_episode_length_s=current_episode_length_s,
        path_total_length_m=path_total_length_m,
        speed_ref_mps=speed_ref_mps,
        freeze_steps=freeze_steps,
        sim_dt=sim_dt,
        completion_margin_s=completion_margin_s,
        max_episode_length_s=max_episode_length_s,
    )


PHASE_CHOICES: tuple[str, ...] = (
    "level_straight",
    "climb_straight",
    "descent_straight",
    "level_turn",
    "level_loiter",
    "multi_segment",
)


def build_phase_mission(phase: str) -> Mission:
    """Build a deterministic canonical mission for the requested phase."""
    if phase == "level_straight":
        return Mission(segments=[MissionSegment(kind="straight", altitude_changes=False)])
    if phase == "climb_straight":
        return Mission(segments=[MissionSegment(kind="straight", altitude_changes=True, altitude_direction=1)])
    if phase == "descent_straight":
        return Mission(segments=[MissionSegment(kind="straight", altitude_changes=True, altitude_direction=-1)])
    if phase == "level_turn":
        return Mission(segments=[MissionSegment(kind="turn", altitude_changes=False)])
    if phase == "level_loiter":
        return Mission(segments=[MissionSegment(kind="loiter", altitude_changes=False)])
    if phase == "multi_segment":
        return Mission(
            segments=[
                MissionSegment(kind="straight", altitude_changes=False),
                MissionSegment(kind="turn", altitude_changes=False),
                MissionSegment(kind="straight", altitude_changes=True, altitude_direction=1),
                MissionSegment(kind="loiter", altitude_changes=False),
                MissionSegment(kind="straight", altitude_changes=True, altitude_direction=-1),
            ]
        )
    raise ValueError(f"Unsupported phase: {phase}")


def build_random_mission(
    *,
    seed: int,
    num_segments_min: int,
    num_segments_max: int,
    allow_straight: bool,
    allow_turn: bool,
    allow_loiter: bool,
    allow_climb_on_straight: bool,
) -> Mission:
    """Build a deterministic random mission from the requested seed and mission family."""
    return sample_mission(
        MissionGeneratorCfg(
            seed=int(seed),
            num_segments_min=int(num_segments_min),
            num_segments_max=int(num_segments_max),
            allow_straight=bool(allow_straight),
            allow_turn=bool(allow_turn),
            allow_loiter=bool(allow_loiter),
            allow_climb_on_straight=bool(allow_climb_on_straight),
        )
    )


def build_path_mission_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for path-mission baseline rollouts."""
    parser = argparse.ArgumentParser(description="Run a PX4-like teacher baseline on a canonical path mission.")
    parser.add_argument("--task", type=str, default="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0")
    parser.add_argument("--mission_mode", type=str, default="fixed", choices=("fixed", "random"))
    parser.add_argument("--phase", type=str, default=None, choices=PHASE_CHOICES)
    parser.add_argument("--mission_seed", type=int, default=0)
    parser.add_argument("--mission_label", type=str, default=None)
    parser.add_argument("--mission_num_segments_min", type=int, default=2)
    parser.add_argument("--mission_num_segments_max", type=int, default=4)
    parser.add_argument("--mission_allow_straight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mission_allow_turn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mission_allow_loiter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mission_allow_climb_on_straight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2600)
    parser.add_argument(
        "--auto_extend_steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-extend rollout length to cover the mission at a nominal cruise speed.",
    )
    parser.add_argument("--nominal_speed_mps", type=float, default=7.0)
    parser.add_argument("--completion_margin_s", type=float, default=2.0)
    parser.add_argument("--height_sp", type=float, default=10.0)
    parser.add_argument("--metrics_warmup_s", type=float, default=3.0)
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help="Override environment episode length. Default auto-expands to cover --steps.",
    )
    parser.add_argument("--straight_length_m", type=float, default=60.0)
    parser.add_argument("--turn_radius_m", type=float, default=20.0)
    parser.add_argument("--loiter_radius_m", type=float, default=20.0)
    parser.add_argument("--turn_sweep_deg", type=float, default=90.0)
    parser.add_argument("--loiter_turns", type=float, default=1.0)
    parser.add_argument("--climb_delta_m", type=float, default=3.0)
    parser.add_argument("--path_manager_max_roll_deg", type=float, default=35.0)
    parser.add_argument("--path_manager_max_flight_path_angle_deg", type=float, default=10.0)
    parser.add_argument("--wind_x_mps", type=float, default=0.0)
    parser.add_argument("--wind_y_mps", type=float, default=0.0)
    parser.add_argument(
        "--wind_ou",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable time-varying gusts with an OU / Gauss-Markov model.",
    )
    parser.add_argument("--wind_ou_tau_s", type=float, default=2.0)
    parser.add_argument("--wind_ou_sigma_x_mps", type=float, default=0.0)
    parser.add_argument("--wind_ou_sigma_y_mps", type=float, default=0.0)
    parser.add_argument("--wind_ou_clip_to_range", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out_dir", type=Path, default=Path("logs/flapping_px4/path_tracking"))
    parser.add_argument("--print_every", type=int, default=250)
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        AppLauncher = None
    if AppLauncher is not None:
        AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_args() -> argparse.Namespace:
    parser = build_path_mission_parser()
    args, _ = parser.parse_known_args()
    if args.mission_mode == "fixed" and args.phase is None:
        parser.error("--phase is required when --mission_mode=fixed")
    return args


def _quantile(values: list[float], q: float) -> float:
    import torch

    if not values:
        return float("nan")
    return float(torch.quantile(torch.tensor(values), q).item())


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _mean_abs(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(abs(v) for v in values) / len(values))


def finalize_rollout_metrics(
    *,
    progress_ratio_hist: list[float],
    last_observed_progress_ratio: float | None,
    completed_path: bool,
    failure_step: int | None,
    completion_threshold: float,
) -> tuple[float, bool]:
    """Finalize rollout progress and completion from the last evaluated step."""
    if last_observed_progress_ratio is not None:
        final_progress_ratio = float(last_observed_progress_ratio)
    elif progress_ratio_hist:
        final_progress_ratio = float(progress_ratio_hist[-1])
    else:
        final_progress_ratio = 0.0
    completed_path_final = bool(completed_path or (final_progress_ratio >= completion_threshold and failure_step is None))
    return final_progress_ratio, completed_path_final


def resolve_rollout_status(
    *,
    progress_ratio: float,
    terminated: bool,
    truncated: bool,
    completion_threshold: float,
) -> tuple[bool, bool, str | None]:
    """Resolve whether a rollout should stop, and whether it ended in success or failure."""
    if progress_ratio >= completion_threshold:
        return True, True, None
    if terminated:
        return False, True, "terminated"
    if truncated:
        return False, True, "truncated"
    return False, False, None


def ensure_episode_horizon(env, *, effective_steps: int, env_step_dt: float) -> None:
    """Extend the environment timeout to cover the effective rollout horizon."""
    required_steps = max(int(effective_steps) + 1, 1)
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "cfg") and hasattr(unwrapped.cfg, "episode_length_s"):
        required_length_s = float(required_steps) * float(env_step_dt)
        unwrapped.cfg.episode_length_s = max(float(unwrapped.cfg.episode_length_s), required_length_s)


def _configure_env(args: argparse.Namespace):
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.randomize_commands = False
    env_cfg.height_cmd = float(args.height_sp)
    env_cfg.action_space = 4
    if hasattr(env_cfg, "act_lpf_tau_s"):
        env_cfg.act_lpf_tau_s = 0.0
    if hasattr(env_cfg, "act_rate_limit_per_s"):
        env_cfg.act_rate_limit_per_s = 0.0

    env_step_dt = float(env_cfg.sim.dt) * float(env_cfg.decimation)
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = float(args.episode_length_s)
    else:
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(args.steps) * env_step_dt + 1.0)

    env_cfg.teacher_guidance_enabled = True
    env_cfg.teacher_guidance_delta_init = 0.0
    env_cfg.teacher_guidance_delta_final = 0.0
    env_cfg.teacher_guidance_schedule_steps = ()
    env_cfg.teacher_guidance_schedule_deltas = ()
    env_cfg.teacher_guidance_disable_after_steps = -1

    wind_enabled = (
        abs(float(args.wind_x_mps)) > 1.0e-6
        or abs(float(args.wind_y_mps)) > 1.0e-6
        or bool(args.wind_ou)
        or abs(float(args.wind_ou_sigma_x_mps)) > 1.0e-6
        or abs(float(args.wind_ou_sigma_y_mps)) > 1.0e-6
    )
    if hasattr(env_cfg, "wind_enabled"):
        env_cfg.wind_enabled = bool(wind_enabled)
    if hasattr(env_cfg, "randomize_wind"):
        env_cfg.randomize_wind = False
    if hasattr(env_cfg, "wind_xy_mps"):
        env_cfg.wind_xy_mps = (float(args.wind_x_mps), float(args.wind_y_mps))
    if hasattr(env_cfg, "wind_x_range_mps"):
        env_cfg.wind_x_range_mps = (float(args.wind_x_mps), float(args.wind_x_mps))
    if hasattr(env_cfg, "wind_y_range_mps"):
        env_cfg.wind_y_range_mps = (float(args.wind_y_mps), float(args.wind_y_mps))
    if hasattr(env_cfg, "wind_ou_enabled"):
        env_cfg.wind_ou_enabled = bool(args.wind_ou)
    if hasattr(env_cfg, "wind_ou_tau_s"):
        env_cfg.wind_ou_tau_s = float(args.wind_ou_tau_s)
    if hasattr(env_cfg, "wind_ou_sigma_xy_mps"):
        env_cfg.wind_ou_sigma_xy_mps = (float(args.wind_ou_sigma_x_mps), float(args.wind_ou_sigma_y_mps))
    if hasattr(env_cfg, "wind_ou_clip_to_range"):
        env_cfg.wind_ou_clip_to_range = bool(args.wind_ou_clip_to_range)

    if hasattr(env_cfg, "path_manager_max_roll_deg"):
        env_cfg.path_manager_max_roll_deg = float(args.path_manager_max_roll_deg)
    if hasattr(env_cfg, "path_manager_max_flight_path_angle_deg"):
        env_cfg.path_manager_max_flight_path_angle_deg = float(args.path_manager_max_flight_path_angle_deg)

    env = gym.make(args.task, cfg=env_cfg)
    return env, env_cfg, env_step_dt


def _build_selected_mission(args: argparse.Namespace) -> tuple[Mission, str]:
    if args.mission_mode == "fixed":
        assert args.phase is not None
        return build_phase_mission(args.phase), args.phase
    mission = build_random_mission(
        seed=int(args.mission_seed),
        num_segments_min=int(args.mission_num_segments_min),
        num_segments_max=int(args.mission_num_segments_max),
        allow_straight=bool(args.mission_allow_straight),
        allow_turn=bool(args.mission_allow_turn),
        allow_loiter=bool(args.mission_allow_loiter),
        allow_climb_on_straight=bool(args.mission_allow_climb_on_straight),
    )
    mission_label = args.mission_label or f"random_seed_{int(args.mission_seed):06d}"
    return mission, mission_label


def _inject_mission(env, args: argparse.Namespace, mission: Mission) -> Any:
    import torch

    try:
        from flapping_bot.path_tracking.path_manager import PathManager, PathManagerCfg
    except ModuleNotFoundError:
        from flapping_bot.flapping_bot.path_tracking.path_manager import PathManager, PathManagerCfg

    unwrapped = env.unwrapped
    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device, dtype=torch.long)
    unwrapped._ensure_path_buffers()
    for env_id in env_ids.tolist():
        phase_mission = Mission(segments=list(mission.segments))
        manager = PathManager(
            PathManagerCfg(
                max_roll_deg=float(args.path_manager_max_roll_deg),
                max_flight_path_angle_deg=float(args.path_manager_max_flight_path_angle_deg),
                straight_length_m=float(args.straight_length_m),
                turn_radius_m=float(args.turn_radius_m),
                loiter_radius_m=float(args.loiter_radius_m),
                turn_sweep_deg=float(args.turn_sweep_deg),
                loiter_turns=float(args.loiter_turns),
                climb_delta_m=float(args.climb_delta_m),
                initial_altitude_m=float(unwrapped._height_cmd[env_id].item()),
            ),
            phase_mission,
        )
        unwrapped._missions[env_id] = phase_mission
        unwrapped._path_managers[env_id] = manager
        if hasattr(unwrapped, "_path_episode_horizon_steps"):
            episode_length_s = _compute_injected_path_episode_length_s(
                current_episode_length_s=float(getattr(unwrapped.cfg, "episode_length_s", 0.0)),
                path_total_length_m=float(manager.total_length_m),
                speed_ref_mps=float(getattr(unwrapped.cfg, "path_episode_speed_ref_mps", 6.0)),
                freeze_steps=int(getattr(unwrapped.cfg, "freeze_steps_after_reset", 0)),
                sim_dt=float(getattr(getattr(unwrapped.cfg, "sim", None), "dt", 1.0 / 240.0)),
                completion_margin_s=float(getattr(unwrapped.cfg, "path_episode_completion_margin_s", 0.0)),
                max_episode_length_s=float(getattr(unwrapped.cfg, "path_episode_max_s", 0.0)),
            )
            horizon_steps = max(int(math.ceil(episode_length_s / max(float(unwrapped.step_dt), 1.0e-6))), 1)
            unwrapped._path_episode_horizon_steps[env_id] = horizon_steps
    unwrapped._path_progress_prev_s[env_ids] = 0.0
    unwrapped._path_delta_s[env_ids] = 0.0
    unwrapped._path_progress_s[env_ids] = 0.0
    unwrapped._path_lateral_error_m[env_ids] = 0.0
    unwrapped._path_height_error_m[env_ids] = 0.0
    unwrapped._path_align_error_rad[env_ids] = 0.0
    unwrapped._path_curvature_m_inv[env_ids] = 0.0
    unwrapped._path_height_sp_m[env_ids] = float(args.height_sp)
    unwrapped._path_closest_point_xyz[env_ids] = 0.0
    unwrapped._path_tangent_xy[env_ids, 0] = 1.0
    unwrapped._path_tangent_xy[env_ids, 1] = 0.0
    unwrapped._path_preview_points_xyz[env_ids] = 0.0
    unwrapped._path_preview_points_body_xyz[env_ids] = 0.0
    unwrapped._path_action_delta[env_ids] = 0.0
    unwrapped._path_query_dirty = True
    if getattr(unwrapped, "_teacher_controller", None) is not None:
        unwrapped._teacher_controller.reset()
    unwrapped._refresh_path_state()
    manager0 = unwrapped._path_managers[0]
    return manager0


def _sample_reference_path(manager: Any) -> list[dict[str, float]]:
    total_length_m = float(manager.total_length_m)
    num_samples = max(int(math.ceil(total_length_m / 1.0)) + 1, 128)
    rows: list[dict[str, float]] = []
    for idx in range(num_samples):
        progress_s = total_length_m * float(idx) / float(max(num_samples - 1, 1))
        x, y, z = manager.sample(progress_s)
        rows.append({"progress_s": progress_s, "x_ref": float(x), "y_ref": float(y), "z_ref": float(z)})
    return rows


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    from isaaclab.utils.math import euler_xyz_from_quat

    env, env_cfg, env_step_dt = _configure_env(args)
    env.reset()
    mission, mission_label = _build_selected_mission(args)
    manager0 = _inject_mission(env, args, mission)

    reference_rows = _sample_reference_path(manager0)
    total_path_length_m = float(manager0.total_length_m)
    freeze_time_s = float(getattr(env_cfg, "freeze_steps_after_reset", 0)) * float(env_cfg.sim.dt)
    effective_steps = int(args.steps)
    if bool(args.auto_extend_steps):
        required_time_s = total_path_length_m / max(float(args.nominal_speed_mps), 1.0e-3)
        required_time_s += freeze_time_s + max(float(args.completion_margin_s), 0.0)
        required_steps = int(math.ceil(required_time_s / env_step_dt))
        effective_steps = max(int(args.steps), required_steps)
    ensure_episode_horizon(env, effective_steps=effective_steps, env_step_dt=env_step_dt)

    abs_lateral_hist: list[float] = []
    abs_lateral_post_hist: list[float] = []
    abs_height_hist: list[float] = []
    abs_height_post_hist: list[float] = []
    abs_align_deg_hist: list[float] = []
    abs_align_deg_post_hist: list[float] = []
    progress_ratio_hist: list[float] = []
    progress_ratio_post_hist: list[float] = []
    speed_hist: list[float] = []
    speed_post_hist: list[float] = []

    traj_rows: list[dict[str, float]] = []
    failure_step: int | None = None
    failure_kind: str | None = None
    completed_path = False
    last_observed_progress_ratio: float | None = None
    warmup_s = max(float(args.metrics_warmup_s), 0.0)
    completion_threshold = 0.995

    for step in range(int(effective_steps)):
        unwrapped = env.unwrapped
        robot = unwrapped._robot
        unwrapped._refresh_path_state()

        pos_local = (robot.data.root_pos_w - unwrapped.scene.env_origins).clone()
        vel_w = robot.data.root_lin_vel_w.clone()
        quat_w = robot.data.root_quat_w.clone()
        roll, pitch, yaw = euler_xyz_from_quat(quat_w)
        ang_vel_b = robot.data.root_ang_vel_b.clone()
        wind_w = unwrapped._wind_w.clone() if getattr(unwrapped, "_wind_w", None) is not None else torch.zeros_like(vel_w)
        airspeed = torch.linalg.norm(vel_w - wind_w, dim=1)

        closest_point_xyz = unwrapped._path_closest_point_xyz.clone()
        lateral_error_m = unwrapped._path_lateral_error_m.clone()
        height_error_m = unwrapped._path_height_error_m.clone()
        align_error_rad = unwrapped._path_align_error_rad.clone()
        curvature_m_inv = unwrapped._path_curvature_m_inv.clone()
        progress_s = unwrapped._path_progress_s.clone()
        height_sp_m = unwrapped._path_height_sp_m.clone()

        actions, diag = unwrapped._compute_teacher_actions()
        _, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated

        idx = 0
        t_now = float(step) * env_step_dt
        speed = float(torch.linalg.norm(vel_w[idx]).item())
        lateral_abs = float(torch.abs(lateral_error_m[idx]).item())
        height_abs = float(torch.abs(height_error_m[idx]).item())
        align_abs_deg = float(torch.rad2deg(torch.abs(align_error_rad[idx])).item())
        progress_ratio = float(progress_s[idx].item()) / max(total_path_length_m, 1.0e-6)
        last_observed_progress_ratio = progress_ratio

        completed_path, stop_now, _ = resolve_rollout_status(
            progress_ratio=progress_ratio,
            terminated=False,
            truncated=False,
            completion_threshold=completion_threshold,
        )
        if stop_now:
            break

        abs_lateral_hist.append(lateral_abs)
        abs_height_hist.append(height_abs)
        abs_align_deg_hist.append(align_abs_deg)
        progress_ratio_hist.append(progress_ratio)
        speed_hist.append(speed)
        if t_now >= warmup_s:
            abs_lateral_post_hist.append(lateral_abs)
            abs_height_post_hist.append(height_abs)
            abs_align_deg_post_hist.append(align_abs_deg)
            progress_ratio_post_hist.append(progress_ratio)
            speed_post_hist.append(speed)

        traj_rows.append(
            {
                "episode_id": 0.0,
                "step": float(step),
                "t": t_now,
                "x": float(pos_local[idx, 0].item()),
                "y": float(pos_local[idx, 1].item()),
                "z": float(pos_local[idx, 2].item()),
                "vx": float(vel_w[idx, 0].item()),
                "vy": float(vel_w[idx, 1].item()),
                "vz": float(vel_w[idx, 2].item()),
                "speed": speed,
                "airspeed": float(airspeed[idx].item()),
                "wind_x_mps": float(wind_w[idx, 0].item()),
                "wind_y_mps": float(wind_w[idx, 1].item()),
                "wind_z_mps": float(wind_w[idx, 2].item()),
                "roll_deg": float(torch.rad2deg(roll[idx]).item()),
                "pitch_deg": float(torch.rad2deg(pitch[idx]).item()),
                "yaw_deg": float(torch.rad2deg(yaw[idx]).item()),
                "reference_x": float(closest_point_xyz[idx, 0].item()),
                "reference_y": float(closest_point_xyz[idx, 1].item()),
                "reference_z": float(closest_point_xyz[idx, 2].item()),
                "height_sp_m": float(height_sp_m[idx].item()),
                "lateral_error_m": float(lateral_error_m[idx].item()),
                "height_error_m": float(height_error_m[idx].item()),
                "align_error_deg": align_abs_deg,
                "curvature_m_inv": float(curvature_m_inv[idx].item()),
                "progress_s": float(progress_s[idx].item()),
                "progress_ratio": progress_ratio,
                "course_sp_deg": float(torch.rad2deg(diag["course_sp"][idx]).item()),
                "heading_sp_deg": float(torch.rad2deg(diag["heading_sp"][idx]).item()),
                "roll_sp_deg": float(torch.rad2deg(diag["roll_sp"][idx]).item()),
                "pitch_sp_deg": float(torch.rad2deg(diag["pitch_sp"][idx]).item()),
                "freq_hz": float(diag["freq_hz"][idx].item()),
                "tecs_tas_sp": float(diag["tecs_tas_sp"][idx].item()) if "tecs_tas_sp" in diag else float("nan"),
                "tecs_tas": float(diag["tecs_tas"][idx].item()) if "tecs_tas" in diag else float("nan"),
                "tecs_throttle_sp": float(diag["tecs_throttle_sp"][idx].item())
                if "tecs_throttle_sp" in diag
                else float("nan"),
                "tecs_pitch_sp_deg": float(torch.rad2deg(diag["tecs_pitch_sp"][idx]).item())
                if "tecs_pitch_sp" in diag
                else float("nan"),
                "action_freq": float(actions[idx, 0].item()),
                "action_rudder": float(actions[idx, 1].item()),
                "action_elevon_pitch": float(actions[idx, 2].item()),
                "action_elevon_roll": float(actions[idx, 3].item()),
                "reward": float(rewards[idx].item()),
                "done": float(done[idx].item()),
            }
        )

        if args.print_every > 0 and step % int(args.print_every) == 0:
            print(
                f"[step {step:05d}] mission={mission_label} "
                f"|lat|={lateral_abs:.3f} m |h|={height_abs:.3f} m "
                f"progress={progress_ratio:.3f} speed={speed:.3f} m/s"
            )

        completed_path, stop_now, failure_kind_now = resolve_rollout_status(
            progress_ratio=progress_ratio,
            terminated=bool(terminated.any().item()),
            truncated=bool(truncated.any().item()),
            completion_threshold=completion_threshold,
        )
        if stop_now and not completed_path:
            failure_step = int(step)
            failure_kind = failure_kind_now
            break

    run_dir = args.out_dir / mission_label / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_path = run_dir / "trajectory_env0.csv"
    summary_path = run_dir / "summary.json"
    ref_path = run_dir / "reference_path.csv"

    with traj_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        writer.writeheader()
        writer.writerows(traj_rows)

    with ref_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(reference_rows[0].keys()))
        writer.writeheader()
        writer.writerows(reference_rows)

    steps_completed = len(traj_rows)
    final_progress_ratio, completed_path_final = finalize_rollout_metrics(
        progress_ratio_hist=progress_ratio_hist,
        last_observed_progress_ratio=last_observed_progress_ratio,
        completed_path=completed_path,
        failure_step=failure_step,
        completion_threshold=completion_threshold,
    )
    summary = {
        "task": args.task,
        "phase": args.phase,
        "mission_mode": args.mission_mode,
        "mission_label": mission_label,
        "mission_seed": int(args.mission_seed) if args.mission_mode == "random" else None,
        "mission_segments": [segment.kind for segment in mission.segments],
        "mission_altitude_changes": [bool(segment.altitude_changes) for segment in mission.segments],
        "mission_altitude_directions": [int(segment.altitude_direction) for segment in mission.segments],
        "teacher_mode": "path_tracking_teacher",
        "num_envs": int(args.num_envs),
        "steps_requested": int(args.steps),
        "steps_effective": int(effective_steps),
        "steps_completed": int(steps_completed),
        "auto_extend_steps": bool(args.auto_extend_steps),
        "nominal_speed_mps": float(args.nominal_speed_mps),
        "completion_margin_s": float(args.completion_margin_s),
        "freeze_time_s": float(freeze_time_s),
        "env_dt_s": float(env_step_dt),
        "metrics_warmup_s": float(warmup_s),
        "height_sp_m": float(args.height_sp),
        "straight_length_m": float(args.straight_length_m),
        "turn_radius_m": float(args.turn_radius_m),
        "loiter_radius_m": float(args.loiter_radius_m),
        "turn_sweep_deg": float(args.turn_sweep_deg),
        "loiter_turns": float(args.loiter_turns),
        "climb_delta_m": float(args.climb_delta_m),
        "path_manager_max_roll_deg": float(args.path_manager_max_roll_deg),
        "path_manager_max_flight_path_angle_deg": float(args.path_manager_max_flight_path_angle_deg),
        "total_path_length_m": float(total_path_length_m),
        "mission_num_segments_min": int(args.mission_num_segments_min),
        "mission_num_segments_max": int(args.mission_num_segments_max),
        "mission_allow_straight": bool(args.mission_allow_straight),
        "mission_allow_turn": bool(args.mission_allow_turn),
        "mission_allow_loiter": bool(args.mission_allow_loiter),
        "mission_allow_climb_on_straight": bool(args.mission_allow_climb_on_straight),
        "wind_x_mps": float(args.wind_x_mps),
        "wind_y_mps": float(args.wind_y_mps),
        "wind_ou_enabled": bool(args.wind_ou),
        "wind_ou_tau_s": float(args.wind_ou_tau_s),
        "wind_ou_sigma_x_mps": float(args.wind_ou_sigma_x_mps),
        "wind_ou_sigma_y_mps": float(args.wind_ou_sigma_y_mps),
        "wind_ou_clip_to_range": bool(args.wind_ou_clip_to_range),
        "rudder_max_deg": float(getattr(env_cfg, "rudder_max_deg", 25.0)),
        "elevon_max_deg": float(getattr(env_cfg, "elevon_max_deg", 25.0)),
        "elevon_trim_deg": float(getattr(env_cfg, "elevon_trim_deg", 0.0)),
        "elevon_pitch_mix": float(getattr(env_cfg, "elevon_pitch_mix", 1.0)),
        "elevon_roll_mix": float(getattr(env_cfg, "elevon_roll_mix", 1.0)),
        "mean_abs_lateral_error_m": _mean(abs_lateral_hist),
        "p95_abs_lateral_error_m": _quantile(abs_lateral_hist, 0.95),
        "max_abs_lateral_error_m": max(abs_lateral_hist) if abs_lateral_hist else float("nan"),
        "mean_abs_height_error_m": _mean(abs_height_hist),
        "p95_abs_height_error_m": _quantile(abs_height_hist, 0.95),
        "max_abs_height_error_m": max(abs_height_hist) if abs_height_hist else float("nan"),
        "mean_abs_align_error_deg": _mean(abs_align_deg_hist),
        "p95_abs_align_error_deg": _quantile(abs_align_deg_hist, 0.95),
        "mean_speed_mps": _mean(speed_hist),
        "final_progress_ratio": float(final_progress_ratio),
        "completed_path": bool(completed_path_final),
        "failure_step": failure_step,
        "failure_kind": failure_kind,
        "output_dir": str(run_dir),
    }
    if abs_lateral_post_hist:
        summary["mean_abs_lateral_error_post_warmup_m"] = _mean(abs_lateral_post_hist)
        summary["p95_abs_lateral_error_post_warmup_m"] = _quantile(abs_lateral_post_hist, 0.95)
    if abs_height_post_hist:
        summary["mean_abs_height_error_post_warmup_m"] = _mean(abs_height_post_hist)
        summary["p95_abs_height_error_post_warmup_m"] = _quantile(abs_height_post_hist, 0.95)
    if abs_align_deg_post_hist:
        summary["mean_abs_align_error_post_warmup_deg"] = _mean(abs_align_deg_post_hist)
        summary["p95_abs_align_error_post_warmup_deg"] = _quantile(abs_align_deg_post_hist, 0.95)
    if progress_ratio_post_hist:
        summary["mean_progress_ratio_post_warmup"] = _mean(progress_ratio_post_hist)
    if speed_post_hist:
        summary["mean_speed_post_warmup_mps"] = _mean(speed_post_hist)

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    simulation_app.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

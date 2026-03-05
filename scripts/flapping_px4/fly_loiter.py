"""Run PX4-like loiter-circle control on the flapping DeLaurier environment.

Example:
  ./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0 \
    --num_envs 1 --steps 3000 --headless
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PX4-like loiter-circle control for FlappingBot.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0",
        help="Gym task id.",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--height_sp", type=float, default=10.0)
    parser.add_argument("--loiter_center_x", type=float, default=40.0)
    parser.add_argument("--loiter_center_y", type=float, default=0.0)
    parser.add_argument("--loiter_radius_m", type=float, default=20.0)
    parser.add_argument(
        "--loiter_clockwise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, fly clockwise. Otherwise counter-clockwise.",
    )
    parser.add_argument("--wind_x_mps", type=float, default=0.0)
    parser.add_argument("--wind_y_mps", type=float, default=0.0)
    parser.add_argument(
        "--random_wind",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, sample per-episode constant wind from the configured ranges.",
    )
    parser.add_argument("--wind_x_min_mps", type=float, default=0.0)
    parser.add_argument("--wind_x_max_mps", type=float, default=0.0)
    parser.add_argument("--wind_y_min_mps", type=float, default=0.0)
    parser.add_argument("--wind_y_max_mps", type=float, default=0.0)
    parser.add_argument(
        "--wind_ou",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable time-varying gusts with an OU (first-order Gauss-Markov) wind model.",
    )
    parser.add_argument("--wind_ou_tau_s", type=float, default=2.0, help="OU wind time constant in seconds.")
    parser.add_argument("--wind_ou_sigma_x_mps", type=float, default=0.0, help="OU stationary std-dev for wind_x (m/s).")
    parser.add_argument("--wind_ou_sigma_y_mps", type=float, default=0.0, help="OU stationary std-dev for wind_y (m/s).")
    parser.add_argument(
        "--wind_ou_clip_to_range",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip OU wind to [wind_x_range, wind_y_range] each step.",
    )
    parser.add_argument("--pitch_trim_deg", type=float, default=10.0)
    parser.add_argument("--height_kp", type=float, default=0.06)
    parser.add_argument("--height_rate_kd", type=float, default=0.02)
    parser.add_argument("--max_pitch_up_deg", type=float, default=25.0)
    parser.add_argument("--max_pitch_down_deg", type=float, default=20.0)
    parser.add_argument("--max_roll_deg", type=float, default=45.0)
    parser.add_argument("--roll_kp", type=float, default=2.5)
    parser.add_argument("--roll_kd", type=float, default=0.35)
    parser.add_argument("--pitch_kp", type=float, default=2.5)
    parser.add_argument("--pitch_kd", type=float, default=0.16)
    parser.add_argument("--yaw_kp", type=float, default=0.1)
    parser.add_argument("--yaw_kd", type=float, default=0.2)
    parser.add_argument("--freq_trim_hz", type=float, default=2.5)
    parser.add_argument(
        "--enable_tecs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable PX4-like TECS for longitudinal control (pitch + flapping frequency).",
    )
    parser.add_argument("--tecs_max_climb_rate_mps", type=float, default=3.0)
    parser.add_argument("--tecs_min_sink_rate_mps", type=float, default=2.0)
    parser.add_argument("--tecs_altitude_error_gain", type=float, default=0.55)
    parser.add_argument("--tecs_airspeed_error_gain", type=float, default=0.8)
    parser.add_argument("--tecs_pitch_speed_weight", type=float, default=0.8)
    parser.add_argument("--tecs_pitch_damping_gain", type=float, default=0.08)
    parser.add_argument("--tecs_integrator_gain_pitch", type=float, default=0.12)
    parser.add_argument("--tecs_throttle_damping_gain", type=float, default=0.35)
    parser.add_argument("--tecs_integrator_gain_throttle", type=float, default=0.22)
    parser.add_argument("--tecs_ste_rate_time_const_s", type=float, default=0.4)
    parser.add_argument("--tecs_tas_min_mps", type=float, default=5.0)
    parser.add_argument("--tecs_tas_error_percentage", type=float, default=0.15)
    parser.add_argument("--tecs_altitude_filter_tau_s", type=float, default=0.3)
    parser.add_argument("--tecs_altitude_rate_filter_tau_s", type=float, default=0.2)
    parser.add_argument("--tecs_pitch_sp_filter_tau_s", type=float, default=0.35)
    parser.add_argument("--tecs_pitch_sp_rate_limit_deg_s", type=float, default=20.0)
    parser.add_argument("--tecs_throttle_sp_filter_tau_s", type=float, default=0.25)
    parser.add_argument("--tecs_altitude_hold_error_band_m", type=float, default=0.25)
    parser.add_argument("--tecs_altitude_capture_error_m", type=float, default=0.8)
    parser.add_argument("--tecs_altitude_capture_time_const_s", type=float, default=1.0)
    parser.add_argument("--tecs_airspeed_error_gain_capture_scale", type=float, default=0.35)
    parser.add_argument("--tecs_pitch_speed_weight_capture", type=float, default=0.35)
    parser.add_argument("--tecs_capture_extra_climb_rate_mps", type=float, default=0.7)
    parser.add_argument("--tecs_capture_extra_sink_rate_mps", type=float, default=0.2)
    parser.add_argument("--inner_pitch_lpf_tau_s", type=float, default=0.12)
    parser.add_argument("--inner_pitch_rate_lpf_tau_s", type=float, default=0.1)
    parser.add_argument("--inner_elevon_pitch_rate_limit_per_s", type=float, default=2.0)
    parser.add_argument("--inner_elevon_roll_rate_limit_per_s", type=float, default=6.0)
    parser.add_argument("--inner_pitch_ki", type=float, default=0.8)
    parser.add_argument("--inner_pitch_integrator_limit", type=float, default=0.6)
    parser.add_argument("--inner_pitch_integrator_leak_per_s", type=float, default=0.04)
    parser.add_argument(
        "--tecs_detect_underspeed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TECS underspeed detection/ramp to max throttle.",
    )
    parser.add_argument(
        "--freq_height_kp_hz_per_m",
        type=float,
        default=0.0,
        help="Throttle assist: freq_hz += Kp * height_err (m).",
    )
    parser.add_argument(
        "--freq_height_rate_kd_hz_per_mps",
        type=float,
        default=0.0,
        help="Throttle assist: freq_hz += Kd * (-vz) (m/s).",
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help="Override environment episode length. Default auto-expands to avoid timeout before --steps.",
    )
    parser.add_argument("--enable_speed_hold", action="store_true")
    parser.add_argument("--speed_sp", type=float, default=7.0)
    parser.add_argument(
        "--speed_kp_hz_per_mps",
        type=float,
        default=0.10,
        help="Speed-hold proportional gain: freq_hz += Kp * (speed_sp - ground_speed).",
    )
    parser.add_argument(
        "--aero_mode",
        type=str,
        choices=("full", "wings_only", "tail_only", "none"),
        default="full",
        help="Which aerodynamic components to apply (for debugging).",
    )
    parser.add_argument(
        "--delaurier-enable-separation",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="delaurier_enable_separation",
        help="Override DeLaurier separated-flow model toggle in the environment (if supported).",
    )
    parser.add_argument(
        "--delaurier-induced-drag-efficiency",
        type=float,
        default=None,
        help="Override env_cfg.delaurier_induced_drag_efficiency (if supported).",
    )
    parser.add_argument(
        "--fuselage-drag-cda",
        type=float,
        default=None,
        help="Override env_cfg.fuselage_drag_cda (if supported).",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("logs/flapping_px4/loiter"))
    parser.add_argument("--print_every", type=int, default=250)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def _quantile(values: list[float], q: float) -> float:
    import torch

    if not values:
        return float("nan")
    return float(torch.quantile(torch.tensor(values), q).item())


def main():
    args = _parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.math import euler_xyz_from_quat
    from isaaclab.utils.math import quat_apply_inverse
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    try:
        from flapping_bot.px4_like import PX4LikeLoiterController, PX4LikeLoiterControllerCfg
    except ModuleNotFoundError:
        from flapping_bot.flapping_bot.px4_like import PX4LikeLoiterController, PX4LikeLoiterControllerCfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.randomize_commands = False
    env_cfg.height_cmd = float(args.height_sp)
    env_cfg.action_space = 4
    env_step_dt = float(env_cfg.sim.dt) * float(env_cfg.decimation)
    if hasattr(env_cfg, "act_lpf_tau_s"):
        env_cfg.act_lpf_tau_s = 0.0
    if hasattr(env_cfg, "act_rate_limit_per_s"):
        env_cfg.act_rate_limit_per_s = 0.0
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = float(args.episode_length_s)
    else:
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(args.steps) * env_step_dt + 1.0)
    if args.enable_speed_hold:
        env_cfg.vx_cmd = float(args.speed_sp)
    if hasattr(env_cfg, "wind_enabled"):
        has_static_wind = (abs(float(args.wind_x_mps)) > 1.0e-6) or (abs(float(args.wind_y_mps)) > 1.0e-6)
        has_ou_gust = bool(args.wind_ou) and (
            (float(args.wind_ou_sigma_x_mps) > 1.0e-6) or (float(args.wind_ou_sigma_y_mps) > 1.0e-6)
        )
        enable_wind = bool(args.random_wind) or has_static_wind or has_ou_gust
        env_cfg.wind_enabled = bool(enable_wind)
        env_cfg.wind_xy_mps = (float(args.wind_x_mps), float(args.wind_y_mps))
        env_cfg.randomize_wind = bool(args.random_wind)
        env_cfg.wind_x_range_mps = (float(args.wind_x_min_mps), float(args.wind_x_max_mps))
        env_cfg.wind_y_range_mps = (float(args.wind_y_min_mps), float(args.wind_y_max_mps))
        if hasattr(env_cfg, "wind_ou_enabled"):
            env_cfg.wind_ou_enabled = bool(args.wind_ou)
        if hasattr(env_cfg, "wind_ou_tau_s"):
            env_cfg.wind_ou_tau_s = float(args.wind_ou_tau_s)
        if hasattr(env_cfg, "wind_ou_sigma_xy_mps"):
            env_cfg.wind_ou_sigma_xy_mps = (float(args.wind_ou_sigma_x_mps), float(args.wind_ou_sigma_y_mps))
        if hasattr(env_cfg, "wind_ou_clip_to_range"):
            env_cfg.wind_ou_clip_to_range = bool(args.wind_ou_clip_to_range)
    if hasattr(env_cfg, "enable_wing_aero") and hasattr(env_cfg, "enable_tail_aero"):
        env_cfg.enable_wing_aero = bool(args.aero_mode in ("full", "wings_only"))
        env_cfg.enable_tail_aero = bool(args.aero_mode in ("full", "tail_only"))
    if args.delaurier_enable_separation is not None and hasattr(env_cfg, "delaurier_enable_separation"):
        env_cfg.delaurier_enable_separation = bool(args.delaurier_enable_separation)
    if args.delaurier_induced_drag_efficiency is not None and hasattr(env_cfg, "delaurier_induced_drag_efficiency"):
        env_cfg.delaurier_induced_drag_efficiency = float(args.delaurier_induced_drag_efficiency)
    if args.fuselage_drag_cda is not None and hasattr(env_cfg, "fuselage_drag_cda"):
        env_cfg.fuselage_drag_cda = float(args.fuselage_drag_cda)
    env = gym.make(args.task, cfg=env_cfg)

    env.reset()
    controller_cfg = PX4LikeLoiterControllerCfg(
        circle_center_xy=(float(args.loiter_center_x), float(args.loiter_center_y)),
        loiter_radius_m=float(args.loiter_radius_m),
        loiter_clockwise=bool(args.loiter_clockwise),
        wind_xy=(float(args.wind_x_mps), float(args.wind_y_mps)),
        control_dt_s=float(env_step_dt),
        height_sp_m=float(args.height_sp),
        pitch_trim_deg=float(args.pitch_trim_deg),
        max_roll_deg=float(args.max_roll_deg),
        roll_kp=float(args.roll_kp),
        roll_kd=float(args.roll_kd),
        pitch_kp=float(args.pitch_kp),
        pitch_kd=float(args.pitch_kd),
        yaw_kp=float(args.yaw_kp),
        yaw_kd=float(args.yaw_kd),
        height_kp=float(args.height_kp),
        height_rate_kd=float(args.height_rate_kd),
        max_pitch_up_deg=float(args.max_pitch_up_deg),
        max_pitch_down_deg=float(args.max_pitch_down_deg),
        freq_trim_hz=float(args.freq_trim_hz),
        min_flap_hz=float(env.unwrapped.cfg.min_flap_hz),
        max_flap_hz=float(env.unwrapped.cfg.max_flap_hz),
        enable_tecs=bool(args.enable_tecs),
        tecs_max_climb_rate_mps=float(args.tecs_max_climb_rate_mps),
        tecs_min_sink_rate_mps=float(args.tecs_min_sink_rate_mps),
        tecs_altitude_error_gain=float(args.tecs_altitude_error_gain),
        tecs_airspeed_error_gain=float(args.tecs_airspeed_error_gain),
        tecs_pitch_speed_weight=float(args.tecs_pitch_speed_weight),
        tecs_pitch_damping_gain=float(args.tecs_pitch_damping_gain),
        tecs_integrator_gain_pitch=float(args.tecs_integrator_gain_pitch),
        tecs_throttle_damping_gain=float(args.tecs_throttle_damping_gain),
        tecs_integrator_gain_throttle=float(args.tecs_integrator_gain_throttle),
        tecs_ste_rate_time_const_s=float(args.tecs_ste_rate_time_const_s),
        tecs_tas_min_mps=float(args.tecs_tas_min_mps),
        tecs_tas_error_percentage=float(args.tecs_tas_error_percentage),
        tecs_detect_underspeed=bool(args.tecs_detect_underspeed),
        tecs_altitude_filter_tau_s=float(args.tecs_altitude_filter_tau_s),
        tecs_altitude_rate_filter_tau_s=float(args.tecs_altitude_rate_filter_tau_s),
        tecs_pitch_sp_filter_tau_s=float(args.tecs_pitch_sp_filter_tau_s),
        tecs_pitch_sp_rate_limit_deg_s=float(args.tecs_pitch_sp_rate_limit_deg_s),
        tecs_throttle_sp_filter_tau_s=float(args.tecs_throttle_sp_filter_tau_s),
        tecs_altitude_hold_error_band_m=float(args.tecs_altitude_hold_error_band_m),
        tecs_altitude_capture_error_m=float(args.tecs_altitude_capture_error_m),
        tecs_altitude_capture_time_const_s=float(args.tecs_altitude_capture_time_const_s),
        tecs_airspeed_error_gain_capture_scale=float(args.tecs_airspeed_error_gain_capture_scale),
        tecs_pitch_speed_weight_capture=float(args.tecs_pitch_speed_weight_capture),
        tecs_capture_extra_climb_rate_mps=float(args.tecs_capture_extra_climb_rate_mps),
        tecs_capture_extra_sink_rate_mps=float(args.tecs_capture_extra_sink_rate_mps),
        inner_pitch_lpf_tau_s=float(args.inner_pitch_lpf_tau_s),
        inner_pitch_rate_lpf_tau_s=float(args.inner_pitch_rate_lpf_tau_s),
        inner_elevon_pitch_rate_limit_per_s=float(args.inner_elevon_pitch_rate_limit_per_s),
        inner_elevon_roll_rate_limit_per_s=float(args.inner_elevon_roll_rate_limit_per_s),
        inner_pitch_ki=float(args.inner_pitch_ki),
        inner_pitch_integrator_limit=float(args.inner_pitch_integrator_limit),
        inner_pitch_integrator_leak_per_s=float(args.inner_pitch_integrator_leak_per_s),
        enable_speed_hold=bool(args.enable_speed_hold),
        speed_sp_mps=float(args.speed_sp),
        speed_kp_hz_per_mps=float(args.speed_kp_hz_per_mps),
        freq_height_kp_hz_per_m=float(args.freq_height_kp_hz_per_m),
        freq_height_rate_kd_hz_per_mps=float(args.freq_height_rate_kd_hz_per_mps),
    )
    controller = PX4LikeLoiterController(controller_cfg, device=env.unwrapped.device)
    controller.reset()

    resets = 0
    episode_id = 0
    traj_rows: list[dict[str, float]] = []
    abs_radial_err_hist: list[float] = []
    abs_height_err_hist: list[float] = []
    abs_track_err_hist: list[float] = []
    speed_hist: list[float] = []

    for step in range(int(args.steps)):
        robot = env.unwrapped._robot
        env_origins = env.unwrapped.scene.env_origins
        pos_local = (robot.data.root_pos_w - env_origins).clone()
        vel_w = robot.data.root_lin_vel_w.clone()
        quat_w = robot.data.root_quat_w.clone()
        roll, pitch, yaw = euler_xyz_from_quat(quat_w)
        ang_vel_b = robot.data.root_ang_vel_b.clone()
        if hasattr(env.unwrapped, "_wind_w") and (env.unwrapped._wind_w is not None):
            wind_w = env.unwrapped._wind_w.clone()
        else:
            wind_w = torch.zeros_like(vel_w)
        vel_air_w = vel_w - wind_w

        actions, diag = controller.compute_actions(
            pos_local=pos_local,
            ground_vel_local=vel_w,
            wind_vel_local=wind_w[:, 0:2],
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            ang_vel_body=ang_vel_b,
        )

        _, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        if done.any():
            resets += int(done.sum().item())
            episode_id += int(done.sum().item())
            env.reset()
            controller.reset()

        idx = 0
        speed = float(torch.linalg.norm(vel_w[idx]).item())
        airspeed = float(torch.linalg.norm(vel_air_w[idx]).item())
        track_err_abs = float(torch.abs(diag["signed_track_error"][idx]).item())
        radial_err_abs = float(torch.abs(diag["radial_error_m"][idx]).item())
        height_err_abs = abs(float(args.height_sp) - float(pos_local[idx, 2].item()))
        speed_hist.append(speed)
        abs_track_err_hist.append(track_err_abs)
        abs_radial_err_hist.append(radial_err_abs)
        abs_height_err_hist.append(height_err_abs)

        traj_rows.append(
            {
                "episode_id": float(episode_id),
                "step": float(step),
                "t": float(step) * env_step_dt,
                "x": float(pos_local[idx, 0].item()),
                "y": float(pos_local[idx, 1].item()),
                "z": float(pos_local[idx, 2].item()),
                "vx": float(vel_w[idx, 0].item()),
                "vy": float(vel_w[idx, 1].item()),
                "vz": float(vel_w[idx, 2].item()),
                "speed": speed,
                "airspeed": airspeed,
                "wind_x_mps": float(wind_w[idx, 0].item()),
                "wind_y_mps": float(wind_w[idx, 1].item()),
                "wind_z_mps": float(wind_w[idx, 2].item()),
                "roll_deg": float(torch.rad2deg(roll[idx]).item()),
                "pitch_deg": float(torch.rad2deg(pitch[idx]).item()),
                "yaw_deg": float(torch.rad2deg(yaw[idx]).item()),
                "course_sp_deg": float(torch.rad2deg(diag["course_sp"][idx]).item()),
                "heading_sp_deg": float(torch.rad2deg(diag["heading_sp"][idx]).item()),
                "roll_sp_deg": float(torch.rad2deg(diag["roll_sp"][idx]).item()),
                "pitch_sp_deg": float(torch.rad2deg(diag["pitch_sp"][idx]).item()),
                "height_err_m": float(args.height_sp) - float(pos_local[idx, 2].item()),
                "track_error_m": float(diag["signed_track_error"][idx].item()),
                "radial_error_m": float(diag["radial_error_m"][idx].item()),
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

        if args.print_every > 0 and (step % int(args.print_every) == 0):
            print(
                f"[step {step:05d}] radial={traj_rows[-1]['radial_error_m']:.3f} m, "
                f"height_err={traj_rows[-1]['height_err_m']:.3f} m, reward={traj_rows[-1]['reward']:.3f}"
            )

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_path = run_dir / "trajectory_env0.csv"
    summary_path = run_dir / "summary.json"

    with traj_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        writer.writeheader()
        writer.writerows(traj_rows)

    wind_x_tensor = torch.tensor([row["wind_x_mps"] for row in traj_rows])
    wind_y_tensor = torch.tensor([row["wind_y_mps"] for row in traj_rows])
    summary = {
        "task": args.task,
        "mission_mode": "loiter_circle",
        "num_envs": int(args.num_envs),
        "steps": int(args.steps),
        "resets": int(resets),
        "env_dt_s": float(env_step_dt),
        "aero_mode": str(args.aero_mode),
        "enable_tecs": bool(args.enable_tecs),
        "enable_speed_hold": bool(args.enable_speed_hold),
        "speed_sp_mps": float(args.speed_sp),
        "height_sp_m": float(args.height_sp),
        "loiter_center_x": float(args.loiter_center_x),
        "loiter_center_y": float(args.loiter_center_y),
        "loiter_radius_m": float(args.loiter_radius_m),
        "loiter_clockwise": bool(args.loiter_clockwise),
        "wind_x_mps": float(args.wind_x_mps),
        "wind_y_mps": float(args.wind_y_mps),
        "random_wind": bool(args.random_wind),
        "wind_x_range_mps": [float(args.wind_x_min_mps), float(args.wind_x_max_mps)],
        "wind_y_range_mps": [float(args.wind_y_min_mps), float(args.wind_y_max_mps)],
        "wind_ou_enabled": bool(args.wind_ou),
        "wind_ou_tau_s": float(args.wind_ou_tau_s),
        "wind_ou_sigma_x_mps": float(args.wind_ou_sigma_x_mps),
        "wind_ou_sigma_y_mps": float(args.wind_ou_sigma_y_mps),
        "wind_ou_clip_to_range": bool(args.wind_ou_clip_to_range),
        "wind_x_logged_mean_mps": float(torch.mean(wind_x_tensor).item()),
        "wind_x_logged_std_mps": float(torch.std(wind_x_tensor, unbiased=False).item()),
        "wind_y_logged_mean_mps": float(torch.mean(wind_y_tensor).item()),
        "wind_y_logged_std_mps": float(torch.std(wind_y_tensor, unbiased=False).item()),
        "mean_abs_track_error_m": float(sum(abs_track_err_hist) / max(len(abs_track_err_hist), 1)),
        "p95_abs_track_error_m": _quantile(abs_track_err_hist, 0.95),
        "mean_abs_radial_error_m": float(sum(abs_radial_err_hist) / max(len(abs_radial_err_hist), 1)),
        "p95_abs_radial_error_m": _quantile(abs_radial_err_hist, 0.95),
        "mean_abs_height_error_m": float(sum(abs_height_err_hist) / max(len(abs_height_err_hist), 1)),
        "p95_abs_height_error_m": _quantile(abs_height_err_hist, 0.95),
        "mean_speed_mps": float(sum(speed_hist) / max(len(speed_hist), 1)),
        "output_dir": str(run_dir),
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    simulation_app.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

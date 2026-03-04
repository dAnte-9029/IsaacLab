"""Run PX4-like straight-line control on the flapping DeLaurier environment.

Example:
  ./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py \
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
    parser = argparse.ArgumentParser(description="PX4-like straight-line control for FlappingBot.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0",
        help="Gym task id.",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--line_length", type=float, default=120.0)
    parser.add_argument("--height_sp", type=float, default=10.0)
    parser.add_argument("--pitch_trim_deg", type=float, default=10.0)
    parser.add_argument("--height_kp", type=float, default=0.06)
    parser.add_argument("--height_rate_kd", type=float, default=0.02)
    parser.add_argument("--max_pitch_up_deg", type=float, default=25.0)
    parser.add_argument("--max_pitch_down_deg", type=float, default=20.0)
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
    parser.add_argument("--inner_pitch_lpf_tau_s", type=float, default=0.12)
    parser.add_argument("--inner_pitch_rate_lpf_tau_s", type=float, default=0.1)
    parser.add_argument("--inner_elevon_pitch_rate_limit_per_s", type=float, default=2.0)
    parser.add_argument("--inner_elevon_roll_rate_limit_per_s", type=float, default=6.0)
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
    parser.add_argument("--out_dir", type=Path, default=Path("logs/flapping_px4/straight_line"))
    parser.add_argument("--print_every", type=int, default=250)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


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
        from flapping_bot.px4_like import PX4LikeStraightLineController, PX4LikeStraightLineControllerCfg
    except ModuleNotFoundError:
        from flapping_bot.flapping_bot.px4_like import PX4LikeStraightLineController, PX4LikeStraightLineControllerCfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.randomize_commands = False
    env_cfg.height_cmd = float(args.height_sp)
    env_cfg.action_space = 4
    # PX4-like control already produces smooth commands; disable RL-oriented action smoothing to avoid extra lag.
    if hasattr(env_cfg, "act_lpf_tau_s"):
        env_cfg.act_lpf_tau_s = 0.0
    if hasattr(env_cfg, "act_rate_limit_per_s"):
        env_cfg.act_rate_limit_per_s = 0.0
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = float(args.episode_length_s)
    else:
        env_step_dt = float(env_cfg.sim.dt) * float(env_cfg.decimation)
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(args.steps) * env_step_dt + 1.0)
    if args.enable_speed_hold:
        env_cfg.vx_cmd = float(args.speed_sp)
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
    controller_cfg = PX4LikeStraightLineControllerCfg(
        line_start_xy=(0.0, 0.0),
        line_end_xy=(float(args.line_length), 0.0),
        control_dt_s=float(env_step_dt),
        height_sp_m=float(args.height_sp),
        pitch_trim_deg=float(args.pitch_trim_deg),
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
        inner_pitch_lpf_tau_s=float(args.inner_pitch_lpf_tau_s),
        inner_pitch_rate_lpf_tau_s=float(args.inner_pitch_rate_lpf_tau_s),
        inner_elevon_pitch_rate_limit_per_s=float(args.inner_elevon_pitch_rate_limit_per_s),
        inner_elevon_roll_rate_limit_per_s=float(args.inner_elevon_roll_rate_limit_per_s),
        enable_speed_hold=bool(args.enable_speed_hold),
        speed_sp_mps=float(args.speed_sp),
        speed_kp_hz_per_mps=float(args.speed_kp_hz_per_mps),
        freq_height_kp_hz_per_m=float(args.freq_height_kp_hz_per_m),
        freq_height_rate_kd_hz_per_mps=float(args.freq_height_rate_kd_hz_per_mps),
    )
    controller = PX4LikeStraightLineController(controller_cfg, device=env.unwrapped.device)
    controller.reset()

    resets = 0
    episode_id = 0
    traj_rows: list[dict[str, float]] = []
    abs_track_err_hist: list[float] = []
    abs_course_err_deg_hist: list[float] = []
    env_step_dt = float(env_cfg.sim.dt) * float(env_cfg.decimation)
    mass_total = float(env.unwrapped._robot.data.default_mass[0].sum().item())

    for step in range(int(args.steps)):
        # Snapshot state *before* env.step() to keep logs consistent (robot.data tensors are updated in-place).
        robot = env.unwrapped._robot
        env_origins = env.unwrapped.scene.env_origins
        pos_local = (robot.data.root_pos_w - env_origins).clone()
        vel_w = robot.data.root_lin_vel_w.clone()
        vel_body = robot.data.root_lin_vel_b.clone()
        quat_w = robot.data.root_quat_w.clone()
        roll, pitch, yaw = euler_xyz_from_quat(quat_w)
        ang_vel_b = robot.data.root_ang_vel_b.clone()
        vel_body_from_w = quat_apply_inverse(quat_w, vel_w)
        vel_b_mismatch = torch.linalg.norm(vel_body - vel_body_from_w, dim=1)

        actions, diag = controller.compute_actions(
            pos_local=pos_local,
            ground_vel_local=vel_w,
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

        # Log env-0 trajectory for quick diagnostics.
        idx = 0
        course_err_deg = float(torch.rad2deg(torch.abs(diag["course_err"][idx])).item())
        track_err_abs = float(torch.abs(diag["signed_track_error"][idx]).item())
        abs_course_err_deg_hist.append(course_err_deg)
        abs_track_err_hist.append(track_err_abs)
        vx = float(vel_w[idx, 0].item())
        vy = float(vel_w[idx, 1].item())
        vz = float(vel_w[idx, 2].item())
        speed = float(torch.linalg.norm(vel_w[idx]).item())
        wing_f_b = env.unwrapped._debug_last_wing_force_b[idx].clone()
        tail_f_b = env.unwrapped._debug_last_tail_force_b[idx].clone()
        total_f_b = env.unwrapped._debug_last_force_b[idx].clone()
        total_tau_b = env.unwrapped._debug_last_torque_b[idx].clone()
        ax_pred = float((total_f_b[0] / mass_total).item())
        traj_rows.append(
            {
                "episode_id": float(episode_id),
                "step": float(step),
                "t": float(step) * env_step_dt,
                "x": float(pos_local[idx, 0].item()),
                "y": float(pos_local[idx, 1].item()),
                "z": float(pos_local[idx, 2].item()),
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "speed": speed,
                "vx_b": float(vel_body[idx, 0].item()),
                "vy_b": float(vel_body[idx, 1].item()),
                "vz_b": float(vel_body[idx, 2].item()),
                "vx_b_from_w": float(vel_body_from_w[idx, 0].item()),
                "vy_b_from_w": float(vel_body_from_w[idx, 1].item()),
                "vz_b_from_w": float(vel_body_from_w[idx, 2].item()),
                "vel_b_mismatch": float(vel_b_mismatch[idx].item()),
                "roll_deg": float(torch.rad2deg(roll[idx]).item()),
                "pitch_deg": float(torch.rad2deg(pitch[idx]).item()),
                "yaw_deg": float(torch.rad2deg(yaw[idx]).item()),
                "course_sp_deg": float(torch.rad2deg(diag["course_sp"][idx]).item()),
                "roll_sp_deg": float(torch.rad2deg(diag["roll_sp"][idx]).item()),
                "pitch_sp_deg": float(torch.rad2deg(diag["pitch_sp"][idx]).item()),
                "height_err_m": float(args.height_sp) - float(pos_local[idx, 2].item()),
                "track_error_m": float(diag["signed_track_error"][idx].item()),
                "course_error_deg": course_err_deg,
                "freq_hz": float(diag["freq_hz"][idx].item()),
                "tecs_tas_sp": float(diag["tecs_tas_sp"][idx].item()) if "tecs_tas_sp" in diag else float("nan"),
                "tecs_tas": float(diag["tecs_tas"][idx].item()) if "tecs_tas" in diag else float("nan"),
                "tecs_tas_rate": float(diag["tecs_tas_rate"][idx].item()) if "tecs_tas_rate" in diag else float("nan"),
                "tecs_altitude_filt": float(diag["tecs_altitude_filt"][idx].item())
                if "tecs_altitude_filt" in diag
                else float("nan"),
                "tecs_altitude_rate_filt": float(diag["tecs_altitude_rate_filt"][idx].item())
                if "tecs_altitude_rate_filt" in diag
                else float("nan"),
                "tecs_ste_rate_sp": float(diag["tecs_ste_rate_sp"][idx].item()) if "tecs_ste_rate_sp" in diag else float("nan"),
                "tecs_ste_rate_est": float(diag["tecs_ste_rate_est"][idx].item()) if "tecs_ste_rate_est" in diag else float("nan"),
                "tecs_seb_rate_sp": float(diag["tecs_seb_rate_sp"][idx].item()) if "tecs_seb_rate_sp" in diag else float("nan"),
                "tecs_seb_rate_est": float(diag["tecs_seb_rate_est"][idx].item()) if "tecs_seb_rate_est" in diag else float("nan"),
                "tecs_ratio_underspeed": float(diag["tecs_ratio_underspeed"][idx].item())
                if "tecs_ratio_underspeed" in diag
                else float("nan"),
                "tecs_throttle_sp": float(diag["tecs_throttle_sp"][idx].item())
                if "tecs_throttle_sp" in diag
                else float("nan"),
                "tecs_pitch_sp_deg": float(torch.rad2deg(diag["tecs_pitch_sp"][idx]).item())
                if "tecs_pitch_sp" in diag
                else float("nan"),
                "tecs_pitch_integ": float(diag["tecs_pitch_integ"][idx].item()) if "tecs_pitch_integ" in diag else float("nan"),
                "tecs_throttle_integ": float(diag["tecs_throttle_integ"][idx].item())
                if "tecs_throttle_integ" in diag
                else float("nan"),
                "pitch_meas_filt_deg": float(torch.rad2deg(diag["pitch_meas_filt"][idx]).item())
                if "pitch_meas_filt" in diag
                else float("nan"),
                "pitch_rate_filt_dps": float(torch.rad2deg(diag["pitch_rate_filt"][idx]).item())
                if "pitch_rate_filt" in diag
                else float("nan"),
                "action_elevon_pitch_raw": float(diag["action_elevon_pitch_raw"][idx].item())
                if "action_elevon_pitch_raw" in diag
                else float("nan"),
                "action_elevon_roll_raw": float(diag["action_elevon_roll_raw"][idx].item())
                if "action_elevon_roll_raw" in diag
                else float("nan"),
                "aero_wing_Fx_b": float(wing_f_b[0].item()),
                "aero_wing_Fy_b": float(wing_f_b[1].item()),
                "aero_wing_Fz_b": float(wing_f_b[2].item()),
                "aero_tail_Fx_b": float(tail_f_b[0].item()),
                "aero_tail_Fy_b": float(tail_f_b[1].item()),
                "aero_tail_Fz_b": float(tail_f_b[2].item()),
                "aero_total_Fx_b": float(total_f_b[0].item()),
                "aero_total_Fy_b": float(total_f_b[1].item()),
                "aero_total_Fz_b": float(total_f_b[2].item()),
                "aero_total_Tx_b": float(total_tau_b[0].item()),
                "aero_total_Ty_b": float(total_tau_b[1].item()),
                "aero_total_Tz_b": float(total_tau_b[2].item()),
                "ax_pred_mps2": ax_pred,
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
                f"[step {step:05d}] y={traj_rows[-1]['y']:.3f} m, "
                f"course_err={course_err_deg:.2f} deg, reward={traj_rows[-1]['reward']:.3f}"
            )

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_path = run_dir / "trajectory_env0.csv"
    summary_path = run_dir / "summary.json"

    with traj_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        writer.writeheader()
        writer.writerows(traj_rows)

    if abs_track_err_hist:
        track_tensor = torch.tensor(abs_track_err_hist)
        course_tensor = torch.tensor(abs_course_err_deg_hist)
        summary = {
            "task": args.task,
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "resets": int(resets),
            "env_dt_s": float(env_step_dt),
            "mass_total_kg": float(mass_total),
            "aero_mode": str(args.aero_mode),
            "enable_tecs": bool(args.enable_tecs),
            "enable_speed_hold": bool(args.enable_speed_hold),
            "speed_sp_mps": float(args.speed_sp),
            "height_sp_m": float(args.height_sp),
            "pitch_trim_deg": float(args.pitch_trim_deg),
            "freq_trim_hz": float(args.freq_trim_hz),
            "tecs_max_climb_rate_mps": float(args.tecs_max_climb_rate_mps),
            "tecs_min_sink_rate_mps": float(args.tecs_min_sink_rate_mps),
            "tecs_altitude_error_gain": float(args.tecs_altitude_error_gain),
            "tecs_airspeed_error_gain": float(args.tecs_airspeed_error_gain),
            "tecs_pitch_speed_weight": float(args.tecs_pitch_speed_weight),
            "tecs_pitch_damping_gain": float(args.tecs_pitch_damping_gain),
            "tecs_integrator_gain_pitch": float(args.tecs_integrator_gain_pitch),
            "tecs_throttle_damping_gain": float(args.tecs_throttle_damping_gain),
            "tecs_integrator_gain_throttle": float(args.tecs_integrator_gain_throttle),
            "tecs_ste_rate_time_const_s": float(args.tecs_ste_rate_time_const_s),
            "tecs_tas_min_mps": float(args.tecs_tas_min_mps),
            "tecs_tas_error_percentage": float(args.tecs_tas_error_percentage),
            "tecs_detect_underspeed": bool(args.tecs_detect_underspeed),
            "tecs_altitude_filter_tau_s": float(args.tecs_altitude_filter_tau_s),
            "tecs_altitude_rate_filter_tau_s": float(args.tecs_altitude_rate_filter_tau_s),
            "tecs_pitch_sp_filter_tau_s": float(args.tecs_pitch_sp_filter_tau_s),
            "tecs_pitch_sp_rate_limit_deg_s": float(args.tecs_pitch_sp_rate_limit_deg_s),
            "tecs_throttle_sp_filter_tau_s": float(args.tecs_throttle_sp_filter_tau_s),
            "inner_pitch_lpf_tau_s": float(args.inner_pitch_lpf_tau_s),
            "inner_pitch_rate_lpf_tau_s": float(args.inner_pitch_rate_lpf_tau_s),
            "inner_elevon_pitch_rate_limit_per_s": float(args.inner_elevon_pitch_rate_limit_per_s),
            "inner_elevon_roll_rate_limit_per_s": float(args.inner_elevon_roll_rate_limit_per_s),
            "rudder_max_deg": float(getattr(env_cfg, "rudder_max_deg", 25.0)),
            "elevon_max_deg": float(getattr(env_cfg, "elevon_max_deg", 25.0)),
            "elevon_trim_deg": float(getattr(env_cfg, "elevon_trim_deg", 0.0)),
            "elevon_pitch_mix": float(getattr(env_cfg, "elevon_pitch_mix", 1.0)),
            "elevon_roll_mix": float(getattr(env_cfg, "elevon_roll_mix", 1.0)),
            "delaurier_enable_separation": bool(getattr(env_cfg, "delaurier_enable_separation", False)),
            "delaurier_induced_drag_efficiency": float(getattr(env_cfg, "delaurier_induced_drag_efficiency", 0.0)),
            "fuselage_drag_cda": float(getattr(env_cfg, "fuselage_drag_cda", 0.0)),
            "mean_abs_track_error_m": float(torch.mean(track_tensor).item()),
            "p95_abs_track_error_m": float(torch.quantile(track_tensor, 0.95).item()),
            "mean_abs_course_error_deg": float(torch.mean(course_tensor).item()),
            "p95_abs_course_error_deg": float(torch.quantile(course_tensor, 0.95).item()),
            "output_dir": str(run_dir),
        }
    else:
        summary = {
            "task": args.task,
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "resets": int(resets),
            "output_dir": str(run_dir),
        }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    simulation_app.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

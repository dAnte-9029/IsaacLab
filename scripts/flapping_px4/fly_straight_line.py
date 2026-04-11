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


def _resolve_initial_elevon_actions(args: argparse.Namespace, env_cfg) -> tuple[float, float]:
    if not bool(getattr(args, "seed_controller_from_env_reset", False)):
        return 0.0, 0.0
    elevon_limit_deg = max(float(getattr(env_cfg, "elevon_max_deg", 0.0)), 1.0e-6)
    return (
        float(getattr(env_cfg, "reset_elevon_pitch_deg", 0.0)) / elevon_limit_deg,
        float(getattr(env_cfg, "reset_elevon_roll_deg", 0.0)) / elevon_limit_deg,
    )


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
    parser.add_argument(
        "--state_source",
        type=str,
        choices=("truth", "estimated", "compare"),
        default="truth",
        help="State source for control: truth, estimated, or compare (control=estimated + log both).",
    )
    parser.add_argument("--sensor_noise_scale", type=float, default=1.0, help="Scale factor for sensor noise std.")
    parser.add_argument("--sensor_bias_scale", type=float, default=1.0, help="Scale factor for sensor bias std.")
    parser.add_argument("--sensor_delay_scale", type=float, default=1.0, help="Scale factor for sensor delays.")
    parser.add_argument("--estimator_attitude_gain", type=float, default=0.05, help="Accel correction gain for roll/pitch.")
    parser.add_argument(
        "--estimator_attitude_gain_min",
        type=float,
        default=0.0,
        help="Minimum scale for accel-based attitude correction under gating.",
    )
    parser.add_argument(
        "--estimator_accel_gate_sigma_mps2",
        type=float,
        default=1.25,
        help="Accel-norm gating sigma for attitude correction.",
    )
    parser.add_argument(
        "--estimator_accel_gate_gyro_dps",
        type=float,
        default=90.0,
        help="Gyro-rate gating threshold (deg/s) for attitude correction.",
    )
    parser.add_argument(
        "--estimator_attitude_max_pitch_deg",
        type=float,
        default=85.0,
        help="Max absolute pitch allowed in attitude integration.",
    )
    parser.add_argument("--estimator_yaw_gain", type=float, default=0.08, help="Mag correction gain for yaw.")
    parser.add_argument("--estimator_wind_tau_s", type=float, default=1.5, help="LPF time constant for wind estimate.")
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
    parser.add_argument("--freq_trim_hz", type=float, default=2.5)
    parser.add_argument(
        "--enable_tecs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable PX4-like TECS for longitudinal control (pitch + flapping frequency).",
    )
    parser.add_argument("--tecs_max_climb_rate_mps", type=float, default=3.0)
    parser.add_argument("--tecs_min_sink_rate_mps", type=float, default=2.0)
    parser.add_argument("--tecs_altitude_error_gain", type=float, default=3.0)
    parser.add_argument("--tecs_airspeed_error_gain", type=float, default=0.8)
    parser.add_argument("--tecs_pitch_speed_weight", type=float, default=0.35)
    parser.add_argument("--tecs_pitch_damping_gain", type=float, default=0.26)
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
    parser.add_argument("--tecs_altitude_hold_error_band_m", type=float, default=0.05)
    parser.add_argument("--tecs_altitude_capture_error_m", type=float, default=0.8)
    parser.add_argument("--tecs_altitude_capture_time_const_s", type=float, default=0.45)
    parser.add_argument("--tecs_altitude_capture_release_error_m", type=float, default=0.03)
    parser.add_argument("--tecs_altitude_capture_release_time_s", type=float, default=0.35)
    parser.add_argument("--tecs_altitude_capture_persistence_gain", type=float, default=1.0)
    parser.add_argument("--tecs_airspeed_error_gain_capture_scale", type=float, default=0.35)
    parser.add_argument("--tecs_pitch_speed_weight_capture", type=float, default=0.10)
    parser.add_argument("--tecs_capture_extra_climb_rate_mps", type=float, default=1.4)
    parser.add_argument("--tecs_capture_extra_sink_rate_mps", type=float, default=0.2)
    parser.add_argument("--inner_pitch_lpf_tau_s", type=float, default=0.12)
    parser.add_argument("--inner_pitch_rate_lpf_tau_s", type=float, default=0.1)
    parser.add_argument("--inner_pitch_cycle_mean_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inner_pitch_cycle_mean_tau_s", type=float, default=0.30)
    parser.add_argument("--inner_pitch_rate_cycle_mean_tau_s", type=float, default=0.24)
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
    parser.add_argument(
        "--seed_controller_from_env_reset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize the controller's internal elevon state from env reset trim to mimic teacher startup.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("logs/flapping_px4/straight_line"))
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
        from flapping_bot.px4_like import (
            PX4LikeStraightLineController,
            PX4LikeStraightLineControllerCfg,
            SensorStateEstimator,
            SensorSuiteCfg,
            StateEstimatorCfg,
        )
    except ModuleNotFoundError:
        from flapping_bot.flapping_bot.px4_like import (
            PX4LikeStraightLineController,
            PX4LikeStraightLineControllerCfg,
            SensorStateEstimator,
            SensorSuiteCfg,
            StateEstimatorCfg,
        )

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.randomize_commands = False
    env_cfg.height_cmd = float(args.height_sp)
    env_cfg.action_space = 4
    env_step_dt = float(env_cfg.sim.dt) * float(env_cfg.decimation)
    # PX4-like control already produces smooth commands; disable RL-oriented action smoothing to avoid extra lag.
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
    initial_elevon_pitch_action, initial_elevon_roll_action = _resolve_initial_elevon_actions(args, env.unwrapped.cfg)
    controller_cfg = PX4LikeStraightLineControllerCfg(
        line_start_xy=(0.0, 0.0),
        line_end_xy=(float(args.line_length), 0.0),
        wind_xy=(float(args.wind_x_mps), float(args.wind_y_mps)),
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
        tecs_altitude_hold_error_band_m=float(args.tecs_altitude_hold_error_band_m),
        tecs_altitude_capture_error_m=float(args.tecs_altitude_capture_error_m),
        tecs_altitude_capture_time_const_s=float(args.tecs_altitude_capture_time_const_s),
        tecs_altitude_capture_release_error_m=float(args.tecs_altitude_capture_release_error_m),
        tecs_altitude_capture_release_time_s=float(args.tecs_altitude_capture_release_time_s),
        tecs_altitude_capture_persistence_gain=float(args.tecs_altitude_capture_persistence_gain),
        tecs_airspeed_error_gain_capture_scale=float(args.tecs_airspeed_error_gain_capture_scale),
        tecs_pitch_speed_weight_capture=float(args.tecs_pitch_speed_weight_capture),
        tecs_capture_extra_climb_rate_mps=float(args.tecs_capture_extra_climb_rate_mps),
        tecs_capture_extra_sink_rate_mps=float(args.tecs_capture_extra_sink_rate_mps),
        inner_pitch_lpf_tau_s=float(args.inner_pitch_lpf_tau_s),
        inner_pitch_rate_lpf_tau_s=float(args.inner_pitch_rate_lpf_tau_s),
        inner_pitch_cycle_mean_enabled=bool(args.inner_pitch_cycle_mean_enabled),
        inner_pitch_cycle_mean_tau_s=float(args.inner_pitch_cycle_mean_tau_s),
        inner_pitch_rate_cycle_mean_tau_s=float(args.inner_pitch_rate_cycle_mean_tau_s),
        inner_elevon_pitch_rate_limit_per_s=float(args.inner_elevon_pitch_rate_limit_per_s),
        inner_elevon_roll_rate_limit_per_s=float(args.inner_elevon_roll_rate_limit_per_s),
        inner_pitch_ki=float(args.inner_pitch_ki),
        inner_pitch_integrator_limit=float(args.inner_pitch_integrator_limit),
        inner_pitch_integrator_leak_per_s=float(args.inner_pitch_integrator_leak_per_s),
        initial_elevon_pitch_action=initial_elevon_pitch_action,
        initial_elevon_roll_action=initial_elevon_roll_action,
        enable_speed_hold=bool(args.enable_speed_hold),
        speed_sp_mps=float(args.speed_sp),
        speed_kp_hz_per_mps=float(args.speed_kp_hz_per_mps),
        freq_height_kp_hz_per_m=float(args.freq_height_kp_hz_per_m),
        freq_height_rate_kd_hz_per_mps=float(args.freq_height_rate_kd_hz_per_mps),
    )
    controller = PX4LikeStraightLineController(controller_cfg, device=env.unwrapped.device)
    controller.reset()

    state_estimator: SensorStateEstimator | None = None
    if args.state_source in ("estimated", "compare"):
        sensor_cfg = SensorSuiteCfg(
            gps_delay_s=0.10 * float(args.sensor_delay_scale),
            baro_delay_s=0.04 * float(args.sensor_delay_scale),
            airspeed_delay_s=0.03 * float(args.sensor_delay_scale),
            mag_delay_s=0.03 * float(args.sensor_delay_scale),
        )
        estimator_cfg = StateEstimatorCfg(
            roll_pitch_accel_gain=float(args.estimator_attitude_gain),
            roll_pitch_accel_gain_min=float(args.estimator_attitude_gain_min),
            roll_pitch_accel_gate_sigma_mps2=float(args.estimator_accel_gate_sigma_mps2),
            roll_pitch_accel_gate_gyro_dps=float(args.estimator_accel_gate_gyro_dps),
            attitude_max_pitch_deg=float(args.estimator_attitude_max_pitch_deg),
            yaw_mag_gain=float(args.estimator_yaw_gain),
            wind_lpf_tau_s=float(args.estimator_wind_tau_s),
        )
        state_estimator = SensorStateEstimator(
            sensor_cfg=sensor_cfg,
            estimator_cfg=estimator_cfg,
            num_envs=int(args.num_envs),
            device=env.unwrapped.device,
            control_dt_s=float(env_step_dt),
            noise_scale=float(args.sensor_noise_scale),
            bias_scale=float(args.sensor_bias_scale),
        )
        robot_init = env.unwrapped._robot
        env_origins_init = env.unwrapped.scene.env_origins
        pos_init = (robot_init.data.root_pos_w - env_origins_init).clone()
        vel_init = robot_init.data.root_lin_vel_w.clone()
        quat_init = robot_init.data.root_quat_w.clone()
        roll_init, pitch_init, yaw_init = euler_xyz_from_quat(quat_init)
        if hasattr(env.unwrapped, "_wind_w") and (env.unwrapped._wind_w is not None):
            wind_init = env.unwrapped._wind_w.clone()
        else:
            wind_init = torch.zeros_like(vel_init)
        airspeed_init = torch.linalg.norm(vel_init - wind_init, dim=1)
        state_estimator.reset(
            pos_local_true=pos_init,
            vel_local_true=vel_init,
            roll_true=roll_init,
            pitch_true=pitch_init,
            yaw_true=yaw_init,
            airspeed_true=airspeed_init,
            wind_local_true=wind_init,
        )

    resets = 0
    episode_id = 0
    traj_rows: list[dict[str, float]] = []
    abs_track_err_hist: list[float] = []
    abs_course_err_deg_hist: list[float] = []
    est_pos_xy_err_hist: list[float] = []
    est_vel_xyz_err_hist: list[float] = []
    est_yaw_err_deg_hist: list[float] = []
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
        if hasattr(env.unwrapped, "_wind_w") and (env.unwrapped._wind_w is not None):
            wind_w = env.unwrapped._wind_w.clone()
        else:
            wind_w = torch.zeros_like(vel_w)
        wind_b = quat_apply_inverse(quat_w, wind_w)
        vel_air_w = vel_w - wind_w
        vel_air_b = vel_body_from_w - wind_b

        airspeed_true = torch.linalg.norm(vel_air_w, dim=1)
        est_state: dict[str, torch.Tensor] | None = None
        est_diag: dict[str, torch.Tensor] = {}
        if state_estimator is not None:
            est_state, est_diag = state_estimator.step(
                pos_local_true=pos_local,
                vel_local_true=vel_w,
                roll_true=roll,
                pitch_true=pitch,
                yaw_true=yaw,
                ang_vel_body_true=ang_vel_b,
                airspeed_true=airspeed_true,
            )

        if args.state_source == "truth":
            ctrl_pos = pos_local
            ctrl_vel = vel_w
            ctrl_roll = roll
            ctrl_pitch = pitch
            ctrl_yaw = yaw
            ctrl_ang_vel = ang_vel_b
            ctrl_wind_xy = wind_w[:, 0:2]
        else:
            assert est_state is not None
            ctrl_pos = est_state["pos_local"]
            ctrl_vel = est_state["ground_vel_local"]
            ctrl_roll = est_state["roll"]
            ctrl_pitch = est_state["pitch"]
            ctrl_yaw = est_state["yaw"]
            ctrl_ang_vel = est_state["ang_vel_body"]
            ctrl_wind_xy = est_state["wind_xy"]

        actions, diag = controller.compute_actions(
            pos_local=ctrl_pos,
            ground_vel_local=ctrl_vel,
            wind_vel_local=ctrl_wind_xy,
            roll=ctrl_roll,
            pitch=ctrl_pitch,
            yaw=ctrl_yaw,
            ang_vel_body=ctrl_ang_vel,
        )

        _, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        if done.any():
            resets += int(done.sum().item())
            episode_id += int(done.sum().item())
            env.reset()
            controller.reset()
            if state_estimator is not None:
                robot_reset = env.unwrapped._robot
                env_origins_reset = env.unwrapped.scene.env_origins
                pos_reset = (robot_reset.data.root_pos_w - env_origins_reset).clone()
                vel_reset = robot_reset.data.root_lin_vel_w.clone()
                quat_reset = robot_reset.data.root_quat_w.clone()
                roll_reset, pitch_reset, yaw_reset = euler_xyz_from_quat(quat_reset)
                if hasattr(env.unwrapped, "_wind_w") and (env.unwrapped._wind_w is not None):
                    wind_reset = env.unwrapped._wind_w.clone()
                else:
                    wind_reset = torch.zeros_like(vel_reset)
                airspeed_reset = torch.linalg.norm(vel_reset - wind_reset, dim=1)
                state_estimator.reset(
                    pos_local_true=pos_reset,
                    vel_local_true=vel_reset,
                    roll_true=roll_reset,
                    pitch_true=pitch_reset,
                    yaw_true=yaw_reset,
                    airspeed_true=airspeed_reset,
                    wind_local_true=wind_reset,
                )

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
        airspeed = float(torch.linalg.norm(vel_air_w[idx]).item())
        beta_rad = torch.atan2(
            vel_air_b[idx, 1],
            torch.clamp(torch.sqrt(vel_air_b[idx, 0] ** 2 + vel_air_b[idx, 2] ** 2), min=1.0e-6),
        )
        wing_f_b = env.unwrapped._debug_last_wing_force_b[idx].clone()
        tail_f_b = env.unwrapped._debug_last_tail_force_b[idx].clone()
        total_f_b = env.unwrapped._debug_last_force_b[idx].clone()
        total_tau_b = env.unwrapped._debug_last_torque_b[idx].clone()
        ax_pred = float((total_f_b[0] / mass_total).item())

        est_pos_xy_err = float("nan")
        est_vel_xyz_err = float("nan")
        est_yaw_err_deg = float("nan")
        x_est = float("nan")
        y_est = float("nan")
        z_est = float("nan")
        vx_est = float("nan")
        vy_est = float("nan")
        vz_est = float("nan")
        wind_x_est = float("nan")
        wind_y_est = float("nan")
        speed_est = float("nan")
        airspeed_est = float("nan")
        if est_state is not None:
            pos_est = est_state["pos_local"]
            vel_est = est_state["ground_vel_local"]
            x_est = float(pos_est[idx, 0].item())
            y_est = float(pos_est[idx, 1].item())
            z_est = float(pos_est[idx, 2].item())
            vx_est = float(vel_est[idx, 0].item())
            vy_est = float(vel_est[idx, 1].item())
            vz_est = float(vel_est[idx, 2].item())
            wind_x_est = float(est_state["wind_xy"][idx, 0].item())
            wind_y_est = float(est_state["wind_xy"][idx, 1].item())
            speed_est = float(torch.linalg.norm(vel_est[idx]).item())
            airspeed_est = float(est_state["airspeed"][idx].item())
            est_pos_xy_err = float(torch.linalg.norm(pos_est[idx, 0:2] - pos_local[idx, 0:2]).item())
            est_vel_xyz_err = float(torch.linalg.norm(vel_est[idx] - vel_w[idx]).item())
            yaw_err = torch.atan2(torch.sin(est_state["yaw"][idx] - yaw[idx]), torch.cos(est_state["yaw"][idx] - yaw[idx]))
            est_yaw_err_deg = float(torch.abs(torch.rad2deg(yaw_err)).item())
            est_pos_xy_err_hist.append(est_pos_xy_err)
            est_vel_xyz_err_hist.append(est_vel_xyz_err)
            est_yaw_err_deg_hist.append(est_yaw_err_deg)
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
                "airspeed": airspeed,
                "vx_b": float(vel_body[idx, 0].item()),
                "vy_b": float(vel_body[idx, 1].item()),
                "vz_b": float(vel_body[idx, 2].item()),
                "air_vx_b": float(vel_air_b[idx, 0].item()),
                "air_vy_b": float(vel_air_b[idx, 1].item()),
                "air_vz_b": float(vel_air_b[idx, 2].item()),
                "beta_deg": float(torch.rad2deg(beta_rad).item()),
                "wind_x_mps": float(wind_w[idx, 0].item()),
                "wind_y_mps": float(wind_w[idx, 1].item()),
                "wind_z_mps": float(wind_w[idx, 2].item()),
                "vx_b_from_w": float(vel_body_from_w[idx, 0].item()),
                "vy_b_from_w": float(vel_body_from_w[idx, 1].item()),
                "vz_b_from_w": float(vel_body_from_w[idx, 2].item()),
                "vel_b_mismatch": float(vel_b_mismatch[idx].item()),
                "roll_deg": float(torch.rad2deg(roll[idx]).item()),
                "pitch_deg": float(torch.rad2deg(pitch[idx]).item()),
                "yaw_deg": float(torch.rad2deg(yaw[idx]).item()),
                "x_est": x_est,
                "y_est": y_est,
                "z_est": z_est,
                "vx_est": vx_est,
                "vy_est": vy_est,
                "vz_est": vz_est,
                "speed_est": speed_est,
                "airspeed_est": airspeed_est,
                "wind_x_est_mps": wind_x_est,
                "wind_y_est_mps": wind_y_est,
                "est_pos_xy_err_m": est_pos_xy_err,
                "est_vel_xyz_err_mps": est_vel_xyz_err,
                "est_yaw_err_deg": est_yaw_err_deg,
                "state_source": float(0 if args.state_source == "truth" else (1 if args.state_source == "estimated" else 2)),
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
                "tecs_capture_blend": float(diag["tecs_capture_blend"][idx].item())
                if "tecs_capture_blend" in diag
                else float("nan"),
                "tecs_capture_blend_raw": float(diag["tecs_capture_blend_raw"][idx].item())
                if "tecs_capture_blend_raw" in diag
                else float("nan"),
                "tecs_capture_active": float(diag["tecs_capture_active"][idx].item())
                if "tecs_capture_active" in diag
                else float("nan"),
                "tecs_capture_release_timer_s": float(diag["tecs_capture_release_timer_s"][idx].item())
                if "tecs_capture_release_timer_s" in diag
                else float("nan"),
                "tecs_height_err_raw": float(diag["tecs_height_err_raw"][idx].item())
                if "tecs_height_err_raw" in diag
                else float("nan"),
                "tecs_altitude_rate_sp_hold": float(diag["tecs_altitude_rate_sp_hold"][idx].item())
                if "tecs_altitude_rate_sp_hold" in diag
                else float("nan"),
                "tecs_altitude_rate_sp_capture": float(diag["tecs_altitude_rate_sp_capture"][idx].item())
                if "tecs_altitude_rate_sp_capture" in diag
                else float("nan"),
                "tecs_ste_rate_sp": float(diag["tecs_ste_rate_sp"][idx].item()) if "tecs_ste_rate_sp" in diag else float("nan"),
                "tecs_ste_rate_est": float(diag["tecs_ste_rate_est"][idx].item()) if "tecs_ste_rate_est" in diag else float("nan"),
                "tecs_ste_rate_capture_bias": float(diag["tecs_ste_rate_capture_bias"][idx].item())
                if "tecs_ste_rate_capture_bias" in diag
                else float("nan"),
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
                "pitch_err_filt_deg": float(torch.rad2deg(diag["pitch_err_filt"][idx]).item())
                if "pitch_err_filt" in diag
                else float("nan"),
                "action_elevon_pitch_raw": float(diag["action_elevon_pitch_raw"][idx].item())
                if "action_elevon_pitch_raw" in diag
                else float("nan"),
                "action_elevon_pitch_integ": float(diag["action_elevon_pitch_integ"][idx].item())
                if "action_elevon_pitch_integ" in diag
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
                "sensor_gps_x": float(est_diag["gps_x"][idx].item()) if "gps_x" in est_diag else float("nan"),
                "sensor_gps_y": float(est_diag["gps_y"][idx].item()) if "gps_y" in est_diag else float("nan"),
                "sensor_gps_z": float(est_diag["gps_z"][idx].item()) if "gps_z" in est_diag else float("nan"),
                "sensor_baro_alt": float(est_diag["baro_alt"][idx].item()) if "baro_alt" in est_diag else float("nan"),
                "sensor_airspeed": float(est_diag["airspeed_meas"][idx].item()) if "airspeed_meas" in est_diag else float("nan"),
                "sensor_mag_yaw_deg": float(torch.rad2deg(est_diag["mag_yaw"][idx]).item())
                if "mag_yaw" in est_diag
                else float("nan"),
                "sensor_gyro_x_dps": float(torch.rad2deg(est_diag["gyro_x"][idx]).item()) if "gyro_x" in est_diag else float("nan"),
                "sensor_gyro_y_dps": float(torch.rad2deg(est_diag["gyro_y"][idx]).item()) if "gyro_y" in est_diag else float("nan"),
                "sensor_gyro_z_dps": float(torch.rad2deg(est_diag["gyro_z"][idx]).item()) if "gyro_z" in est_diag else float("nan"),
                "sensor_accel_x_mps2": float(est_diag["accel_x"][idx].item()) if "accel_x" in est_diag else float("nan"),
                "sensor_accel_y_mps2": float(est_diag["accel_y"][idx].item()) if "accel_y" in est_diag else float("nan"),
                "sensor_accel_z_mps2": float(est_diag["accel_z"][idx].item()) if "accel_z" in est_diag else float("nan"),
                "sensor_att_corr_gain": float(est_diag["att_corr_gain"][idx].item())
                if "att_corr_gain" in est_diag
                else float("nan"),
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
        wind_x_tensor = torch.tensor([row["wind_x_mps"] for row in traj_rows])
        wind_y_tensor = torch.tensor([row["wind_y_mps"] for row in traj_rows])
        summary = {
            "task": args.task,
            "mission_mode": "straight_line",
            "state_source": str(args.state_source),
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
            "sensor_noise_scale": float(args.sensor_noise_scale),
            "sensor_bias_scale": float(args.sensor_bias_scale),
            "sensor_delay_scale": float(args.sensor_delay_scale),
            "estimator_attitude_gain": float(args.estimator_attitude_gain),
            "estimator_attitude_gain_min": float(args.estimator_attitude_gain_min),
            "estimator_accel_gate_sigma_mps2": float(args.estimator_accel_gate_sigma_mps2),
            "estimator_accel_gate_gyro_dps": float(args.estimator_accel_gate_gyro_dps),
            "estimator_attitude_max_pitch_deg": float(args.estimator_attitude_max_pitch_deg),
            "estimator_yaw_gain": float(args.estimator_yaw_gain),
            "estimator_wind_tau_s": float(args.estimator_wind_tau_s),
            "wind_x_logged_mean_mps": float(torch.mean(wind_x_tensor).item()),
            "wind_x_logged_std_mps": float(torch.std(wind_x_tensor, unbiased=False).item()),
            "wind_x_logged_min_mps": float(torch.min(wind_x_tensor).item()),
            "wind_x_logged_max_mps": float(torch.max(wind_x_tensor).item()),
            "wind_y_logged_mean_mps": float(torch.mean(wind_y_tensor).item()),
            "wind_y_logged_std_mps": float(torch.std(wind_y_tensor, unbiased=False).item()),
            "wind_y_logged_min_mps": float(torch.min(wind_y_tensor).item()),
            "wind_y_logged_max_mps": float(torch.max(wind_y_tensor).item()),
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
            "tecs_altitude_hold_error_band_m": float(args.tecs_altitude_hold_error_band_m),
            "tecs_altitude_capture_error_m": float(args.tecs_altitude_capture_error_m),
            "tecs_altitude_capture_time_const_s": float(args.tecs_altitude_capture_time_const_s),
            "tecs_altitude_capture_release_error_m": float(args.tecs_altitude_capture_release_error_m),
            "tecs_altitude_capture_release_time_s": float(args.tecs_altitude_capture_release_time_s),
            "tecs_altitude_capture_persistence_gain": float(args.tecs_altitude_capture_persistence_gain),
            "tecs_airspeed_error_gain_capture_scale": float(args.tecs_airspeed_error_gain_capture_scale),
            "tecs_pitch_speed_weight_capture": float(args.tecs_pitch_speed_weight_capture),
            "tecs_capture_extra_climb_rate_mps": float(args.tecs_capture_extra_climb_rate_mps),
            "tecs_capture_extra_sink_rate_mps": float(args.tecs_capture_extra_sink_rate_mps),
            "inner_pitch_lpf_tau_s": float(args.inner_pitch_lpf_tau_s),
            "inner_pitch_rate_lpf_tau_s": float(args.inner_pitch_rate_lpf_tau_s),
            "inner_pitch_cycle_mean_enabled": bool(args.inner_pitch_cycle_mean_enabled),
            "inner_pitch_cycle_mean_tau_s": float(args.inner_pitch_cycle_mean_tau_s),
            "inner_pitch_rate_cycle_mean_tau_s": float(args.inner_pitch_rate_cycle_mean_tau_s),
            "inner_elevon_pitch_rate_limit_per_s": float(args.inner_elevon_pitch_rate_limit_per_s),
            "inner_elevon_roll_rate_limit_per_s": float(args.inner_elevon_roll_rate_limit_per_s),
            "inner_pitch_ki": float(args.inner_pitch_ki),
            "inner_pitch_integrator_limit": float(args.inner_pitch_integrator_limit),
            "inner_pitch_integrator_leak_per_s": float(args.inner_pitch_integrator_leak_per_s),
            "seed_controller_from_env_reset": bool(args.seed_controller_from_env_reset),
            "initial_elevon_pitch_action": float(initial_elevon_pitch_action),
            "initial_elevon_roll_action": float(initial_elevon_roll_action),
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
            "mission_mode": "straight_line",
            "state_source": str(args.state_source),
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "resets": int(resets),
            "output_dir": str(run_dir),
        }

    if est_pos_xy_err_hist:
        summary["mean_est_pos_xy_err_m"] = float(sum(est_pos_xy_err_hist) / len(est_pos_xy_err_hist))
        summary["p95_est_pos_xy_err_m"] = _quantile(est_pos_xy_err_hist, 0.95)
    if est_vel_xyz_err_hist:
        summary["mean_est_vel_xyz_err_mps"] = float(sum(est_vel_xyz_err_hist) / len(est_vel_xyz_err_hist))
        summary["p95_est_vel_xyz_err_mps"] = _quantile(est_vel_xyz_err_hist, 0.95)
    if est_yaw_err_deg_hist:
        summary["mean_est_yaw_err_deg"] = float(sum(est_yaw_err_deg_hist) / len(est_yaw_err_deg_hist))
        summary["p95_est_yaw_err_deg"] = _quantile(est_yaw_err_deg_hist, 0.95)

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    simulation_app.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

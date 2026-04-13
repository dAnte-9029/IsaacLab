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
import math
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
    parser.add_argument("--seed", type=int, default=None, help="Environment RNG seed for deterministic rollouts.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--auto_extend_steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-extend runtime to include warmup + minimum loiter turns.",
    )
    parser.add_argument("--min_loiter_turns", type=float, default=1.0, help="Minimum desired loiter turns for evaluation.")
    parser.add_argument("--metrics_warmup_s", type=float, default=5.0, help="Ignore first N seconds in steady-state metrics.")
    parser.add_argument("--height_sp", type=float, default=10.0)
    parser.add_argument(
        "--spawn_on_circle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset on loiter circle with tangent heading to remove long capture transient.",
    )
    parser.add_argument(
        "--spawn_on_circle_speed_mps",
        type=float,
        default=None,
        help="Initial tangent speed when spawn_on_circle is enabled. Defaults to speed_sp/vx_cmd.",
    )
    parser.add_argument(
        "--state_source",
        type=str,
        choices=("truth", "estimated", "compare"),
        default="truth",
        help="State source for control: truth, estimated, or compare (control=estimated + log both).",
    )
    parser.add_argument(
        "--teacher_state_source",
        type=str,
        choices=("truth", "estimated"),
        default=None,
        help="Teacher runtime state source. Defaults to truth for truth-control, estimated otherwise.",
    )
    parser.add_argument(
        "--imu_source",
        type=str,
        choices=("synthetic", "isaacsim"),
        default="synthetic",
        help="IMU provider to declare in the env runtime contract.",
    )
    parser.add_argument("--sensor_noise_scale", type=float, default=1.0, help="Scale factor for sensor noise std.")
    parser.add_argument("--sensor_bias_scale", type=float, default=1.0, help="Scale factor for sensor bias std.")
    parser.add_argument("--sensor_delay_scale", type=float, default=1.0, help="Scale factor for sensor delays.")
    parser.add_argument("--estimator_attitude_gain", type=float, default=0.05, help="Accel correction gain for roll/pitch.")
    parser.add_argument(
        "--estimator_attitude_correction_mode",
        type=str,
        choices=("gravity_vector", "px4_gravity", "euler_lpf", "euler", "legacy"),
        default="gravity_vector",
        help="Roll/pitch accel correction mode.",
    )
    parser.add_argument(
        "--estimator_accel_hard_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hard-disable accel attitude correction outside the PX4-style accel/gyro gate.",
    )
    parser.add_argument("--estimator_accel_gate_low_g", type=float, default=0.9)
    parser.add_argument("--estimator_accel_gate_high_g", type=float, default=1.1)
    parser.add_argument("--estimator_accel_gate_lpf_tau_s", type=float, default=0.25)
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
    parser.add_argument("--estimator_gps_pos_tau_s", type=float, default=0.18, help="GPS position LPF tau (s).")
    parser.add_argument("--estimator_gps_vel_tau_s", type=float, default=0.12, help="GPS velocity LPF tau (s).")
    parser.add_argument(
        "--estimator_gps_delay_compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compensate GPS measurement delay with delayed GPS velocity.",
    )
    parser.add_argument("--estimator_baro_alt_tau_s", type=float, default=0.18, help="Baro altitude LPF tau (s).")
    parser.add_argument("--estimator_baro_rate_tau_s", type=float, default=0.15, help="Baro altitude-rate LPF tau (s).")
    parser.add_argument(
        "--estimator_baro_delay_compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compensate baro delay with estimated climb/sink rate.",
    )
    parser.add_argument("--estimator_airspeed_tau_s", type=float, default=0.12, help="Airspeed LPF tau (s).")
    parser.add_argument("--estimator_wind_tau_s", type=float, default=1.5, help="LPF time constant for wind estimate.")
    parser.add_argument("--estimator_imu_propagation_gain", type=float, default=0.30, help="IMU propagation blend gain [0..1].")
    parser.add_argument(
        "--estimator_imu_accel_world_tau_s",
        type=float,
        default=0.06,
        help="LPF tau for world-frame IMU acceleration (s).",
    )
    parser.add_argument(
        "--estimator_imu_accel_world_clip_mps2",
        type=float,
        default=30.0,
        help="Norm clip for world-frame IMU acceleration (m/s^2).",
    )
    parser.add_argument("--loiter_center_x", type=float, default=40.0)
    parser.add_argument("--loiter_center_y", type=float, default=0.0)
    parser.add_argument("--loiter_radius_m", type=float, default=40.0)
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
    parser.add_argument("--roll_kp", type=float, default=4.5)
    parser.add_argument("--roll_kd", type=float, default=0.85)
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
    parser.add_argument("--tecs_load_factor_use_roll_sp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inner_pitch_lpf_tau_s", type=float, default=0.12)
    parser.add_argument("--inner_pitch_rate_lpf_tau_s", type=float, default=0.1)
    parser.add_argument("--inner_pitch_cycle_mean_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inner_pitch_cycle_mean_tau_s", type=float, default=0.30)
    parser.add_argument("--inner_pitch_rate_cycle_mean_tau_s", type=float, default=0.24)
    parser.add_argument("--inner_elevon_pitch_rate_limit_per_s", type=float, default=2.0)
    parser.add_argument("--inner_elevon_roll_rate_limit_per_s", type=float, default=6.0)
    parser.add_argument("--profile_max_roll_deg", type=float, default=None)
    parser.add_argument("--profile_roll_kd", type=float, default=None)
    parser.add_argument("--profile_inner_elevon_roll_rate_limit_per_s", type=float, default=None)
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
        "--total_mass_kg_override",
        type=float,
        default=None,
        help="Runtime override for vehicle total mass without editing the asset files.",
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


def _resolve_state_source_selection(args: argparse.Namespace):
    try:
        from flapping_bot.direct.flapping_bot.state_source_contract import resolve_runtime_state_source_selection
    except ModuleNotFoundError:
        from flapping_bot.flapping_bot.direct.flapping_bot.state_source_contract import resolve_runtime_state_source_selection
    return resolve_runtime_state_source_selection(
        state_source=str(args.state_source),
        teacher_state_source=args.teacher_state_source,
        imu_source=str(args.imu_source),
    )


def _build_imu_measurement(
    imu_provider,
    *,
    quat_w,
    vel_w,
    prev_vel_w,
    ang_vel_b,
    dt: float,
    quat_apply_inverse_fn,
):
    vel_acc_world = (vel_w - prev_vel_w) / max(float(dt), 1.0e-6)
    specific_force_world = vel_acc_world.clone()
    specific_force_world[:, 2] = specific_force_world[:, 2] + 9.81
    specific_force_body = quat_apply_inverse_fn(quat_w, specific_force_world)
    return imu_provider.build_from_truth(
        ang_vel_body=ang_vel_b,
        specific_force_body=specific_force_body,
    )


def _apply_runtime_mass_override_arg(env_cfg, args: argparse.Namespace) -> None:
    total_mass_kg_override = getattr(args, "total_mass_kg_override", None)
    if total_mass_kg_override is None:
        return
    env_cfg.total_mass_kg_override = float(total_mass_kg_override)


def _resolve_logged_mass_total_kg(env) -> float:
    mass_total = getattr(env.unwrapped, "_mass_total", None)
    if mass_total is not None:
        return float(mass_total[0].item())
    return float(env.unwrapped._robot.data.default_mass[0].sum().item())


def _create_isaacsim_imu_sensor(imu_provider, env, *, env_step_dt: float):
    if getattr(imu_provider, "backend_name", "") != "isaacsim":
        return None

    try:
        from flapping_bot.px4_like import IsaacSimImuSensorSpec
    except ModuleNotFoundError:
        from flapping_bot.flapping_bot.px4_like import IsaacSimImuSensorSpec

    robot = env.unwrapped._robot
    body_name = str(robot.body_names[0]) if len(robot.body_names) > 0 else "base_link"
    prim_path = f"{env.unwrapped.cfg.robot.prim_path}/{body_name}"
    sensor = imu_provider.create_sensor(
        IsaacSimImuSensorSpec(
            prim_path=prim_path,
            update_period=float(env_step_dt),
        )
    )
    if hasattr(sensor, "is_initialized") and not sensor.is_initialized:
        sensor._initialize_impl()
        sensor._is_initialized = True
    sensor.reset()
    sensor.update(dt=float(env_step_dt), force_recompute=True)
    return sensor


def main():
    args = _parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    state_source_selection = _resolve_state_source_selection(args)

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    try:
        from flapping_bot.px4_like import (
            PX4LikeLoiterController,
            PX4LikeLoiterControllerCfg,
            SensorStateEstimator,
            SensorSuiteCfg,
            StateEstimatorCfg,
            apply_controller_tuning_profile,
            build_imu_provider,
            resolve_controller_tuning_profile,
        )
    except ModuleNotFoundError:
        from flapping_bot.flapping_bot.px4_like import (
            PX4LikeLoiterController,
            PX4LikeLoiterControllerCfg,
            SensorStateEstimator,
            SensorSuiteCfg,
            StateEstimatorCfg,
            apply_controller_tuning_profile,
            build_imu_provider,
            resolve_controller_tuning_profile,
        )

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = None if args.seed is None else int(args.seed)
    env_cfg.randomize_commands = False
    if hasattr(env_cfg, "teacher_state_source"):
        env_cfg.teacher_state_source = state_source_selection.teacher_state_source
    if hasattr(env_cfg, "policy_state_source"):
        env_cfg.policy_state_source = state_source_selection.policy_state_source
    if hasattr(env_cfg, "imu_source"):
        env_cfg.imu_source = state_source_selection.imu_source
    if hasattr(env_cfg, "teacher_guidance_use_wind_truth"):
        env_cfg.teacher_guidance_use_wind_truth = bool(
            bool(getattr(env_cfg, "teacher_guidance_use_wind_truth"))
            and state_source_selection.teacher_state_source == "truth"
        )
    env_cfg.height_cmd = float(args.height_sp)
    env_cfg.action_space = 4
    env_step_dt = float(env_cfg.sim.dt) * float(env_cfg.decimation)
    if hasattr(env_cfg, "act_lpf_tau_s"):
        env_cfg.act_lpf_tau_s = 0.0
    if hasattr(env_cfg, "act_rate_limit_per_s"):
        env_cfg.act_rate_limit_per_s = 0.0
    requested_steps = int(args.steps)
    effective_steps = requested_steps
    if bool(args.auto_extend_steps):
        min_turns = max(float(args.min_loiter_turns), 0.0)
        warmup_s = max(float(args.metrics_warmup_s), 0.0)
        if min_turns > 0.0:
            ref_speed = max(float(args.speed_sp), 1.0)
            min_turn_time_s = min_turns * (2.0 * math.pi * float(args.loiter_radius_m)) / ref_speed
            min_steps_needed = int(math.ceil((warmup_s + min_turn_time_s) / env_step_dt))
            if min_steps_needed > effective_steps:
                effective_steps = min_steps_needed
                print(
                    f"[loiter] Auto-extend steps from {requested_steps} to {effective_steps} "
                    f"for {min_turns:.2f} turns + {warmup_s:.1f}s warmup."
                )
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = float(args.episode_length_s)
    else:
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(effective_steps) * env_step_dt + 1.0)
    if hasattr(env_cfg, "terminate_abs_y"):
        loiter_y_needed = abs(float(args.loiter_center_y)) + float(args.loiter_radius_m) + 15.0
        env_cfg.terminate_abs_y = max(float(env_cfg.terminate_abs_y), loiter_y_needed)
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
    _apply_runtime_mass_override_arg(env_cfg, args)
    env = gym.make(args.task, cfg=env_cfg)

    def _reset_on_loiter_entry() -> None:
        if not bool(args.spawn_on_circle):
            return
        robot = env.unwrapped._robot
        env_origins = env.unwrapped.scene.env_origins
        root_state = robot.data.root_state_w.clone()
        num_envs = root_state.shape[0]
        device = root_state.device

        entry_x = float(args.loiter_center_x) - float(args.loiter_radius_m)
        entry_y = float(args.loiter_center_y)
        root_state[:, 0] = env_origins[:, 0] + entry_x
        root_state[:, 1] = env_origins[:, 1] + entry_y
        root_state[:, 2] = env_origins[:, 2] + float(args.height_sp)

        pitch_reset = -math.radians(float(getattr(env_cfg, "reset_pitch_deg", 0.0)))
        yaw_entry = (0.5 * math.pi) if bool(args.loiter_clockwise) else (-0.5 * math.pi)
        roll_tensor = torch.zeros((num_envs,), device=device)
        pitch_tensor = torch.full((num_envs,), pitch_reset, device=device)
        yaw_tensor = torch.full((num_envs,), yaw_entry, device=device)
        root_state[:, 3:7] = quat_from_euler_xyz(roll=roll_tensor, pitch=pitch_tensor, yaw=yaw_tensor)

        if args.spawn_on_circle_speed_mps is not None:
            init_speed = max(float(args.spawn_on_circle_speed_mps), 0.0)
        elif bool(args.enable_speed_hold):
            init_speed = max(float(args.speed_sp), 0.0)
        else:
            init_speed = max(float(getattr(env_cfg, "vx_cmd", args.speed_sp)), 0.0)
        root_state[:, 7] = init_speed * math.cos(yaw_entry)
        root_state[:, 8] = init_speed * math.sin(yaw_entry)
        root_state[:, 9] = 0.0
        root_state[:, 10:13] = 0.0
        robot.write_root_state_to_sim(root_state)
        if hasattr(env.unwrapped, "_spawn_root_state") and (env.unwrapped._spawn_root_state is not None):
            env.unwrapped._spawn_root_state[:] = root_state

    env.reset()
    _reset_on_loiter_entry()
    mass_total = _resolve_logged_mass_total_kg(env)
    controller_tuning_profile = resolve_controller_tuning_profile(
        controller_state_source=state_source_selection.controller_state_source
    )
    controller_kwargs = apply_controller_tuning_profile(
        dict(
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
            tecs_altitude_capture_release_error_m=float(args.tecs_altitude_capture_release_error_m),
            tecs_altitude_capture_release_time_s=float(args.tecs_altitude_capture_release_time_s),
            tecs_altitude_capture_persistence_gain=float(args.tecs_altitude_capture_persistence_gain),
            tecs_airspeed_error_gain_capture_scale=float(args.tecs_airspeed_error_gain_capture_scale),
            tecs_pitch_speed_weight_capture=float(args.tecs_pitch_speed_weight_capture),
            tecs_capture_extra_climb_rate_mps=float(args.tecs_capture_extra_climb_rate_mps),
            tecs_capture_extra_sink_rate_mps=float(args.tecs_capture_extra_sink_rate_mps),
            tecs_load_factor_use_roll_sp=bool(args.tecs_load_factor_use_roll_sp),
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
            enable_speed_hold=bool(args.enable_speed_hold),
            speed_sp_mps=float(args.speed_sp),
            speed_kp_hz_per_mps=float(args.speed_kp_hz_per_mps),
            freq_height_kp_hz_per_m=float(args.freq_height_kp_hz_per_m),
            freq_height_rate_kd_hz_per_mps=float(args.freq_height_rate_kd_hz_per_mps),
        ),
        controller_state_source=state_source_selection.controller_state_source,
        controller_kind="loiter",
        explicit_overrides={
            key: value
            for key, value in {
                "max_roll_deg": args.profile_max_roll_deg,
                "roll_kd": args.profile_roll_kd,
                "inner_elevon_roll_rate_limit_per_s": args.profile_inner_elevon_roll_rate_limit_per_s,
            }.items()
            if value is not None
        },
    )
    controller_cfg = PX4LikeLoiterControllerCfg(**controller_kwargs)
    controller = PX4LikeLoiterController(controller_cfg, device=env.unwrapped.device)
    controller.reset()

    circle_center_xy = torch.tensor([float(args.loiter_center_x), float(args.loiter_center_y)], device=env.unwrapped.device)
    state_estimator: SensorStateEstimator | None = None
    imu_provider = None
    imu_sensor = None
    imu_measurement_mode = "disabled"
    prev_vel_for_imu = None
    if state_source_selection.controller_state_source in ("estimated", "compare"):
        imu_provider = build_imu_provider(state_source_selection.imu_source)
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
            roll_pitch_accel_hard_gate=bool(args.estimator_accel_hard_gate),
            roll_pitch_accel_gate_low_g=float(args.estimator_accel_gate_low_g),
            roll_pitch_accel_gate_high_g=float(args.estimator_accel_gate_high_g),
            roll_pitch_accel_gate_lpf_tau_s=float(args.estimator_accel_gate_lpf_tau_s),
            roll_pitch_accel_correction_mode=str(args.estimator_attitude_correction_mode),
            attitude_max_pitch_deg=float(args.estimator_attitude_max_pitch_deg),
            yaw_mag_gain=float(args.estimator_yaw_gain),
            gps_pos_lpf_tau_s=float(args.estimator_gps_pos_tau_s),
            gps_vel_lpf_tau_s=float(args.estimator_gps_vel_tau_s),
            gps_delay_compensation=bool(args.estimator_gps_delay_compensation),
            baro_alt_lpf_tau_s=float(args.estimator_baro_alt_tau_s),
            baro_rate_lpf_tau_s=float(args.estimator_baro_rate_tau_s),
            baro_delay_compensation=bool(args.estimator_baro_delay_compensation),
            airspeed_lpf_tau_s=float(args.estimator_airspeed_tau_s),
            wind_lpf_tau_s=float(args.estimator_wind_tau_s),
            imu_propagation_gain=float(args.estimator_imu_propagation_gain),
            imu_accel_world_lpf_tau_s=float(args.estimator_imu_accel_world_tau_s),
            imu_accel_world_clip_mps2=float(args.estimator_imu_accel_world_clip_mps2),
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
        imu_sensor = _create_isaacsim_imu_sensor(imu_provider, env, env_step_dt=env_step_dt)
        imu_measurement_mode = "isaacsim_live" if imu_sensor is not None else "synthetic_truth_derived"
        prev_vel_for_imu = vel_init.clone()

    resets = 0
    episode_id = 0
    traj_rows: list[dict[str, float]] = []
    abs_radial_err_hist: list[float] = []
    abs_height_err_hist: list[float] = []
    abs_track_err_hist: list[float] = []
    speed_hist: list[float] = []
    abs_radial_err_post_hist: list[float] = []
    abs_height_err_post_hist: list[float] = []
    abs_track_err_post_hist: list[float] = []
    speed_post_hist: list[float] = []
    est_pos_xy_err_hist: list[float] = []
    est_vel_xyz_err_hist: list[float] = []
    est_yaw_err_deg_hist: list[float] = []
    est_pos_xy_err_post_hist: list[float] = []
    est_vel_xyz_err_post_hist: list[float] = []
    est_yaw_err_deg_post_hist: list[float] = []
    warmup_s = max(float(args.metrics_warmup_s), 0.0)
    prev_angle = None
    unwrapped_delta = 0.0

    for step in range(int(effective_steps)):
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
        airspeed_true = torch.linalg.norm(vel_air_w, dim=1)

        est_state: dict[str, torch.Tensor] | None = None
        est_diag: dict[str, torch.Tensor] = {}
        if state_estimator is not None:
            assert imu_provider is not None
            if imu_sensor is not None:
                imu_sensor.update(dt=float(env_step_dt), force_recompute=True)
                imu_measurement = imu_provider.build_from_sensor(imu_sensor)
            else:
                assert prev_vel_for_imu is not None
                imu_measurement = _build_imu_measurement(
                    imu_provider,
                    quat_w=quat_w,
                    vel_w=vel_w,
                    prev_vel_w=prev_vel_for_imu,
                    ang_vel_b=ang_vel_b,
                    dt=env_step_dt,
                    quat_apply_inverse_fn=quat_apply_inverse,
                )
            est_state, est_diag = state_estimator.step(
                pos_local_true=pos_local,
                vel_local_true=vel_w,
                roll_true=roll,
                pitch_true=pitch,
                yaw_true=yaw,
                ang_vel_body_true=ang_vel_b,
                airspeed_true=airspeed_true,
                imu_measurement=imu_measurement,
            )
            prev_vel_for_imu = vel_w.clone()

        if state_source_selection.controller_state_source == "truth":
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
            _reset_on_loiter_entry()
            controller.reset()
            prev_angle = None
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
                if imu_sensor is not None:
                    imu_sensor.reset()
                    imu_sensor.update(dt=float(env_step_dt), force_recompute=True)
                prev_vel_for_imu = vel_reset.clone()

        idx = 0
        speed = float(torch.linalg.norm(vel_w[idx]).item())
        airspeed = float(airspeed_true[idx].item())
        radial_error_true = torch.linalg.norm(pos_local[:, 0:2] - circle_center_xy.view(1, 2), dim=1) - float(args.loiter_radius_m)
        radial_err_abs = float(torch.abs(radial_error_true[idx]).item())
        track_err_abs = radial_err_abs
        height_err_abs = abs(float(args.height_sp) - float(pos_local[idx, 2].item()))
        speed_hist.append(speed)
        abs_track_err_hist.append(track_err_abs)
        abs_radial_err_hist.append(radial_err_abs)
        abs_height_err_hist.append(height_err_abs)

        t_now = float(step) * env_step_dt
        if t_now >= warmup_s:
            speed_post_hist.append(speed)
            abs_track_err_post_hist.append(track_err_abs)
            abs_radial_err_post_hist.append(radial_err_abs)
            abs_height_err_post_hist.append(height_err_abs)

        angle_now = math.atan2(
            float(pos_local[idx, 1].item()) - float(args.loiter_center_y),
            float(pos_local[idx, 0].item()) - float(args.loiter_center_x),
        )
        if prev_angle is not None:
            delta = (angle_now - prev_angle + math.pi) % (2.0 * math.pi) - math.pi
            unwrapped_delta += float(delta)
        prev_angle = angle_now

        est_pos_xy_err = float("nan")
        est_vel_xyz_err = float("nan")
        est_yaw_err_deg = float("nan")
        x_est = float("nan")
        y_est = float("nan")
        z_est = float("nan")
        vx_est = float("nan")
        vy_est = float("nan")
        vz_est = float("nan")
        speed_est = float("nan")
        airspeed_est = float("nan")
        wind_x_est = float("nan")
        wind_y_est = float("nan")
        roll_est_deg = float("nan")
        pitch_est_deg = float("nan")
        yaw_est_deg = float("nan")
        if est_state is not None:
            pos_est = est_state["pos_local"]
            vel_est = est_state["ground_vel_local"]
            x_est = float(pos_est[idx, 0].item())
            y_est = float(pos_est[idx, 1].item())
            z_est = float(pos_est[idx, 2].item())
            vx_est = float(vel_est[idx, 0].item())
            vy_est = float(vel_est[idx, 1].item())
            vz_est = float(vel_est[idx, 2].item())
            speed_est = float(torch.linalg.norm(vel_est[idx]).item())
            airspeed_est = float(est_state["airspeed"][idx].item())
            wind_x_est = float(est_state["wind_xy"][idx, 0].item())
            wind_y_est = float(est_state["wind_xy"][idx, 1].item())
            roll_est_deg = float(torch.rad2deg(est_state["roll"][idx]).item())
            pitch_est_deg = float(torch.rad2deg(est_state["pitch"][idx]).item())
            yaw_est_deg = float(torch.rad2deg(est_state["yaw"][idx]).item())
            est_pos_xy_err = float(torch.linalg.norm(pos_est[idx, 0:2] - pos_local[idx, 0:2]).item())
            est_vel_xyz_err = float(torch.linalg.norm(vel_est[idx] - vel_w[idx]).item())
            yaw_err = torch.atan2(torch.sin(est_state["yaw"][idx] - yaw[idx]), torch.cos(est_state["yaw"][idx] - yaw[idx]))
            est_yaw_err_deg = float(torch.abs(torch.rad2deg(yaw_err)).item())
            est_pos_xy_err_hist.append(est_pos_xy_err)
            est_vel_xyz_err_hist.append(est_vel_xyz_err)
            est_yaw_err_deg_hist.append(est_yaw_err_deg)
            if t_now >= warmup_s:
                est_pos_xy_err_post_hist.append(est_pos_xy_err)
                est_vel_xyz_err_post_hist.append(est_vel_xyz_err)
                est_yaw_err_deg_post_hist.append(est_yaw_err_deg)

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
                "roll_est_deg": roll_est_deg,
                "pitch_est_deg": pitch_est_deg,
                "yaw_est_deg": yaw_est_deg,
                "est_pos_xy_err_m": est_pos_xy_err,
                "est_vel_xyz_err_mps": est_vel_xyz_err,
                "est_yaw_err_deg": est_yaw_err_deg,
                "state_source": float(
                    0
                    if state_source_selection.controller_state_source == "truth"
                    else (1 if state_source_selection.controller_state_source == "estimated" else 2)
                ),
                "teacher_state_source": str(state_source_selection.teacher_state_source),
                "policy_state_source": str(state_source_selection.policy_state_source),
                "imu_source": str(state_source_selection.imu_source),
                "course_sp_deg": float(torch.rad2deg(diag["course_sp"][idx]).item()),
                "heading_sp_deg": float(torch.rad2deg(diag["heading_sp"][idx]).item()),
                "roll_sp_deg": float(torch.rad2deg(diag["roll_sp"][idx]).item()),
                "pitch_sp_deg": float(torch.rad2deg(diag["pitch_sp"][idx]).item()),
                "height_err_m": float(args.height_sp) - float(pos_local[idx, 2].item()),
                "track_error_m": float(radial_error_true[idx].item()),
                "radial_error_m": float(radial_error_true[idx].item()),
                "radial_error_ctrl_m": float(diag["radial_error_m"][idx].item()),
                "freq_hz": float(diag["freq_hz"][idx].item()),
                "tecs_tas_sp": float(diag["tecs_tas_sp"][idx].item()) if "tecs_tas_sp" in diag else float("nan"),
                "tecs_tas": float(diag["tecs_tas"][idx].item()) if "tecs_tas" in diag else float("nan"),
                "tecs_throttle_sp": float(diag["tecs_throttle_sp"][idx].item())
                if "tecs_throttle_sp" in diag
                else float("nan"),
                "tecs_pitch_sp_deg": float(torch.rad2deg(diag["tecs_pitch_sp"][idx]).item())
                if "tecs_pitch_sp" in diag
                else float("nan"),
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
                "sensor_accel_norm_mps2": float(est_diag["accel_norm"][idx].item())
                if "accel_norm" in est_diag
                else float("nan"),
                "sensor_accel_gate_lpf_norm_mps2": float(est_diag["accel_gate_lpf_norm"][idx].item())
                if "accel_gate_lpf_norm" in est_diag
                else float("nan"),
                "sensor_att_corr_gain": float(est_diag["att_corr_gain"][idx].item())
                if "att_corr_gain" in est_diag
                else float("nan"),
                "sensor_att_corr_scale": float(est_diag["att_corr_scale"][idx].item())
                if "att_corr_scale" in est_diag
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
        "state_source": str(state_source_selection.controller_state_source),
        "teacher_state_source": str(state_source_selection.teacher_state_source),
        "policy_state_source": str(state_source_selection.policy_state_source),
        "imu_source": str(state_source_selection.imu_source),
        "imu_measurement_mode": str(imu_measurement_mode),
        "num_envs": int(args.num_envs),
        "env_seed": None if args.seed is None else int(args.seed),
        "steps_requested": int(requested_steps),
        "steps": int(effective_steps),
        "resets": int(resets),
        "env_dt_s": float(env_step_dt),
        "mass_total_kg": float(mass_total),
        "aero_mode": str(args.aero_mode),
        "enable_tecs": bool(args.enable_tecs),
        "enable_speed_hold": bool(args.enable_speed_hold),
        "speed_sp_mps": float(args.speed_sp),
        "height_sp_m": float(args.height_sp),
        "loiter_center_x": float(args.loiter_center_x),
        "loiter_center_y": float(args.loiter_center_y),
        "loiter_radius_m": float(args.loiter_radius_m),
        "loiter_clockwise": bool(args.loiter_clockwise),
        "terminate_abs_y_m": float(getattr(env.unwrapped.cfg, "terminate_abs_y", float("nan"))),
        "spawn_on_circle": bool(args.spawn_on_circle),
        "spawn_on_circle_speed_mps": float(args.spawn_on_circle_speed_mps)
        if args.spawn_on_circle_speed_mps is not None
        else None,
        "metrics_warmup_s": float(warmup_s),
        "min_loiter_turns": float(args.min_loiter_turns),
        "auto_extend_steps": bool(args.auto_extend_steps),
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
        "controller_tuning_profile": str(controller_tuning_profile),
        "controller_max_roll_deg": float(controller_cfg.max_roll_deg),
        "controller_roll_kp": float(controller_cfg.roll_kp),
        "controller_roll_kd": float(controller_cfg.roll_kd),
        "controller_guidance_period_s": float(controller_cfg.guidance_period_s),
        "controller_guidance_damping": float(controller_cfg.guidance_damping),
        "controller_guidance_roll_time_const_s": float(controller_cfg.guidance_roll_time_const_s),
        "controller_heading_p_gain": float(controller_cfg.heading_p_gain),
        "controller_inner_elevon_pitch_rate_limit_per_s": float(controller_cfg.inner_elevon_pitch_rate_limit_per_s),
        "controller_inner_elevon_roll_rate_limit_per_s": float(controller_cfg.inner_elevon_roll_rate_limit_per_s),
        "controller_tecs_load_factor_use_roll_sp": bool(controller_cfg.tecs_load_factor_use_roll_sp),
        "sensor_noise_scale": float(args.sensor_noise_scale),
        "sensor_bias_scale": float(args.sensor_bias_scale),
        "sensor_delay_scale": float(args.sensor_delay_scale),
        "estimator_attitude_gain": float(args.estimator_attitude_gain),
        "estimator_attitude_correction_mode": str(args.estimator_attitude_correction_mode),
        "estimator_attitude_gain_min": float(args.estimator_attitude_gain_min),
        "estimator_accel_hard_gate": bool(args.estimator_accel_hard_gate),
        "estimator_accel_gate_low_g": float(args.estimator_accel_gate_low_g),
        "estimator_accel_gate_high_g": float(args.estimator_accel_gate_high_g),
        "estimator_accel_gate_lpf_tau_s": float(args.estimator_accel_gate_lpf_tau_s),
        "estimator_accel_gate_sigma_mps2": float(args.estimator_accel_gate_sigma_mps2),
        "estimator_accel_gate_gyro_dps": float(args.estimator_accel_gate_gyro_dps),
        "estimator_attitude_max_pitch_deg": float(args.estimator_attitude_max_pitch_deg),
        "estimator_yaw_gain": float(args.estimator_yaw_gain),
        "estimator_gps_pos_tau_s": float(args.estimator_gps_pos_tau_s),
        "estimator_gps_vel_tau_s": float(args.estimator_gps_vel_tau_s),
        "estimator_gps_delay_compensation": bool(args.estimator_gps_delay_compensation),
        "estimator_baro_alt_tau_s": float(args.estimator_baro_alt_tau_s),
        "estimator_baro_rate_tau_s": float(args.estimator_baro_rate_tau_s),
        "estimator_baro_delay_compensation": bool(args.estimator_baro_delay_compensation),
        "estimator_airspeed_tau_s": float(args.estimator_airspeed_tau_s),
        "estimator_wind_tau_s": float(args.estimator_wind_tau_s),
        "estimator_imu_propagation_gain": float(args.estimator_imu_propagation_gain),
        "estimator_imu_accel_world_tau_s": float(args.estimator_imu_accel_world_tau_s),
        "estimator_imu_accel_world_clip_mps2": float(args.estimator_imu_accel_world_clip_mps2),
        "mean_abs_track_error_m": float(sum(abs_track_err_hist) / max(len(abs_track_err_hist), 1)),
        "p95_abs_track_error_m": _quantile(abs_track_err_hist, 0.95),
        "mean_abs_radial_error_m": float(sum(abs_radial_err_hist) / max(len(abs_radial_err_hist), 1)),
        "p95_abs_radial_error_m": _quantile(abs_radial_err_hist, 0.95),
        "mean_abs_height_error_m": float(sum(abs_height_err_hist) / max(len(abs_height_err_hist), 1)),
        "p95_abs_height_error_m": _quantile(abs_height_err_hist, 0.95),
        "mean_speed_mps": float(sum(speed_hist) / max(len(speed_hist), 1)),
        "turns_completed": float(abs(unwrapped_delta) / (2.0 * math.pi)),
        "output_dir": str(run_dir),
    }
    if abs_radial_err_post_hist:
        summary["mean_abs_radial_error_post_warmup_m"] = float(
            sum(abs_radial_err_post_hist) / max(len(abs_radial_err_post_hist), 1)
        )
        summary["p95_abs_radial_error_post_warmup_m"] = _quantile(abs_radial_err_post_hist, 0.95)
    if abs_track_err_post_hist:
        summary["mean_abs_track_error_post_warmup_m"] = float(
            sum(abs_track_err_post_hist) / max(len(abs_track_err_post_hist), 1)
        )
        summary["p95_abs_track_error_post_warmup_m"] = _quantile(abs_track_err_post_hist, 0.95)
    if abs_height_err_post_hist:
        summary["mean_abs_height_error_post_warmup_m"] = float(
            sum(abs_height_err_post_hist) / max(len(abs_height_err_post_hist), 1)
        )
        summary["p95_abs_height_error_post_warmup_m"] = _quantile(abs_height_err_post_hist, 0.95)
    if speed_post_hist:
        summary["mean_speed_post_warmup_mps"] = float(sum(speed_post_hist) / max(len(speed_post_hist), 1))
    if est_pos_xy_err_hist:
        summary["mean_est_pos_xy_err_m"] = float(sum(est_pos_xy_err_hist) / len(est_pos_xy_err_hist))
        summary["p95_est_pos_xy_err_m"] = _quantile(est_pos_xy_err_hist, 0.95)
    if est_vel_xyz_err_hist:
        summary["mean_est_vel_xyz_err_mps"] = float(sum(est_vel_xyz_err_hist) / len(est_vel_xyz_err_hist))
        summary["p95_est_vel_xyz_err_mps"] = _quantile(est_vel_xyz_err_hist, 0.95)
    if est_yaw_err_deg_hist:
        summary["mean_est_yaw_err_deg"] = float(sum(est_yaw_err_deg_hist) / len(est_yaw_err_deg_hist))
        summary["p95_est_yaw_err_deg"] = _quantile(est_yaw_err_deg_hist, 0.95)
    if est_pos_xy_err_post_hist:
        summary["mean_est_pos_xy_err_post_warmup_m"] = float(sum(est_pos_xy_err_post_hist) / len(est_pos_xy_err_post_hist))
        summary["p95_est_pos_xy_err_post_warmup_m"] = _quantile(est_pos_xy_err_post_hist, 0.95)
    if est_vel_xyz_err_post_hist:
        summary["mean_est_vel_xyz_err_post_warmup_mps"] = float(
            sum(est_vel_xyz_err_post_hist) / len(est_vel_xyz_err_post_hist)
        )
        summary["p95_est_vel_xyz_err_post_warmup_mps"] = _quantile(est_vel_xyz_err_post_hist, 0.95)
    if est_yaw_err_deg_post_hist:
        summary["mean_est_yaw_err_post_warmup_deg"] = float(sum(est_yaw_err_deg_post_hist) / len(est_yaw_err_deg_post_hist))
        summary["p95_est_yaw_err_post_warmup_deg"] = _quantile(est_yaw_err_deg_post_hist, 0.95)

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    simulation_app.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

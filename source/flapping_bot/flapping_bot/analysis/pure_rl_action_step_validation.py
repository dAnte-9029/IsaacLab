"""CPU-native action step-response validation for the direct-surface PureRL task."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

from flapping_bot.direct.flapping_bot.action_contract import (
    frequency_hz_to_normalized_action,
    joint_position_to_normalized_action,
)


FIXED_ROOT_CASE_NAMES = (
    "frequency",
    "rudder_10deg",
    "rudder_30deg",
    "rudder_soft_limit",
    "left_elevon_10deg",
    "left_elevon_30deg",
    "left_elevon_soft_limit",
    "right_elevon_10deg",
    "right_elevon_30deg",
    "right_elevon_soft_limit",
)
FREE_ROOT_CASE_NAMES = ("baseline", "rudder_plus_5deg", "left_plus_5deg", "right_plus_5deg")
_TAIL_CASE_SPECS = (
    ("rudder", 10.0),
    ("rudder", 30.0),
    ("rudder", None),
    ("left_elevon", 10.0),
    ("left_elevon", 30.0),
    ("left_elevon", None),
    ("right_elevon", 10.0),
    ("right_elevon", 30.0),
    ("right_elevon", None),
)
_FREQUENCY_LEVELS_HZ = (0.0, 2.66958, 5.0, 2.66958, 0.0)
_TAIL_LEVEL_SIGNS = (0.0, 1.0, 0.0, -1.0, 0.0)


@dataclass(frozen=True)
class PureRLActionStepResult:
    """Provenance-independent summary and aligned numeric traces."""

    summary: dict[str, object]
    traces: dict[str, np.ndarray]


def sequence_level(time_s: float, *, dwell_s: float, levels: tuple[float, ...]) -> float:
    """Return the zero-order-held sequence level at nonnegative time."""

    time_value = float(time_s)
    dwell_value = float(dwell_s)
    if time_value < 0.0:
        raise ValueError("time_s must be nonnegative.")
    if dwell_value <= 0.0:
        raise ValueError("dwell_s must be positive.")
    if not levels:
        raise ValueError("levels must not be empty.")
    index = min(int(time_value / dwell_value), len(levels) - 1)
    return float(levels[index])


def summarize_step_response(
    time_s: np.ndarray,
    target: np.ndarray,
    actual: np.ndarray,
    *,
    settling_fraction: float = 0.02,
) -> dict[str, object]:
    """Summarize each detected scalar command transition.

    Rise time is the elapsed 10--90 percent crossing time. Settling time is the
    first sample after which the response remains within the larger of two
    percent of step amplitude or `1e-6` until the next command transition.
    """

    t = np.asarray(time_s, dtype=np.float64)
    command = np.asarray(target, dtype=np.float64)
    response = np.asarray(actual, dtype=np.float64)
    if t.ndim != 1 or command.shape != t.shape or response.shape != t.shape:
        raise ValueError("time_s, target and actual must be aligned 1D arrays.")
    if t.size < 2 or not np.all(np.diff(t) > 0.0):
        raise ValueError("time_s must contain at least two strictly increasing samples.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(command)) or not np.all(np.isfinite(response)):
        raise ValueError("Step-response inputs must be finite.")
    if not 0.0 < float(settling_fraction) < 1.0:
        raise ValueError("settling_fraction must be in (0, 1).")

    transition_indices = np.flatnonzero(np.abs(np.diff(command)) > 1.0e-10) + 1
    transitions: list[dict[str, float | None]] = []
    for transition_number, start_index in enumerate(transition_indices):
        end_index = (
            int(transition_indices[transition_number + 1])
            if transition_number + 1 < transition_indices.size
            else int(t.size)
        )
        previous_target = float(command[start_index - 1])
        next_target = float(command[start_index])
        amplitude = next_target - previous_target
        if abs(amplitude) <= 1.0e-12:
            continue
        segment = response[start_index:end_index]
        segment_time = t[start_index:end_index]
        normalized = (segment - previous_target) / amplitude

        def _first_crossing(level: float) -> float | None:
            matches = np.flatnonzero(normalized >= level)
            if matches.size == 0:
                return None
            return float(segment_time[int(matches[0])] - t[start_index])

        t10 = _first_crossing(0.1)
        t90 = _first_crossing(0.9)
        rise_time = None if t10 is None or t90 is None else max(0.0, t90 - t10)
        tolerance = max(abs(amplitude) * float(settling_fraction), 1.0e-6)
        in_band = np.abs(segment - next_target) <= tolerance
        settling_time: float | None = None
        for offset in range(in_band.size):
            if bool(np.all(in_band[offset:])):
                settling_time = float(segment_time[offset] - t[start_index])
                break
        overshoot = np.maximum((segment - next_target) * math.copysign(1.0, amplitude), 0.0)
        steady_count = max(1, int(math.ceil(0.2 * segment.size)))
        steady_error = float(np.mean(segment[-steady_count:]) - next_target)
        transitions.append(
            {
                "time_s": float(t[start_index]),
                "from": previous_target,
                "to": next_target,
                "rise_time_10_90_s": rise_time,
                "settling_time_2pct_s": settling_time,
                "overshoot_percent": 100.0 * float(np.max(overshoot)) / abs(amplitude),
                "steady_state_error": steady_error,
            }
        )

    rate = np.diff(response) / np.diff(t)
    return {
        "transition_count": len(transitions),
        "max_abs_rate_per_s": float(np.max(np.abs(rate))),
        "transitions": transitions,
    }


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _stack_trace(trace: dict[str, list[np.ndarray | float]]) -> dict[str, np.ndarray]:
    return {name: np.asarray(values, dtype=np.float64) for name, values in trace.items()}


def _configure_robot_asset(cfg: object, *, asset_path: Path | None, usd_dir: Path | None) -> None:
    if asset_path is None and usd_dir is None:
        return
    spawn = cfg.robot.spawn
    replacements: dict[str, str] = {}
    if asset_path is not None:
        replacements["asset_path"] = str(Path(asset_path).resolve())
    if usd_dir is not None:
        replacements["usd_dir"] = str(Path(usd_dir).resolve())
    cfg.robot = cfg.robot.replace(spawn=spawn.replace(**replacements))


def _make_cfg(
    *,
    num_envs: int,
    fixed_root: bool,
    enable_aerodynamics: bool,
    airspeed_mps: float,
    asset_path: Path | None,
    usd_dir: Path | None,
):
    from flapping_bot.direct.flapping_bot.straight_flight_env import (
        FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
    )

    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg()
    cfg.seed = 0
    cfg.scene.num_envs = int(num_envs)
    cfg.scene.env_spacing = 5.0
    cfg.sim.device = "cpu"
    cfg.sim.dt = 1.0 / 480.0
    cfg.sim.render_interval = 8
    cfg.decimation = 8
    cfg.freeze_steps_after_reset = 0
    cfg.episode_length_s = 20.0
    cfg.teacher_guidance_enabled = False
    cfg.randomize_commands = False
    cfg.wind_ou_enabled = False
    cfg.wind_curriculum_enabled = False
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    cfg.fuselage_drag_cda = 0.0
    cfg.reset_pitch_deg = 0.0
    cfg.reset_flap_hz = 0.0
    cfg.reset_rudder_deg = 0.0
    cfg.reset_elevon_pitch_deg = 0.0
    cfg.reset_elevon_roll_deg = 0.0
    cfg.reset_forward_speed_mps = 0.0 if fixed_root else float(airspeed_mps)
    cfg.enable_wing_aero = bool(enable_aerodynamics)
    cfg.enable_tail_aero = bool(enable_aerodynamics)
    if not enable_aerodynamics:
        # The formal native per-wing coupling mode deliberately requires an
        # enabled DeLaurier wrench. This diagnostic has no wing load to apply.
        cfg.wing_aero_coupling_mode = "commanded_base_equivalent"
    cfg.wind_enabled = bool(fixed_root and enable_aerodynamics)
    cfg.wind_xy_mps = (-float(airspeed_mps), 0.0) if cfg.wind_enabled else (0.0, 0.0)
    cfg.sim.gravity = (0.0, 0.0, 0.0) if fixed_root else (0.0, 0.0, -9.81)
    articulation_props = cfg.robot.spawn.articulation_props.replace(fix_root_link=bool(fixed_root))
    cfg.robot = cfg.robot.replace(
        spawn=cfg.robot.spawn.replace(
            articulation_props=articulation_props,
            rigid_props=cfg.robot.spawn.rigid_props.replace(
                disable_gravity=bool(fixed_root),
                retain_accelerations=False,
            ),
        )
    )
    _configure_robot_asset(cfg, asset_path=asset_path, usd_dir=usd_dir)
    return cfg


def _tail_limit_tuple(env: object, channel_name: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    if channel_name == "rudder":
        local_index = env._IDX_RUDDER
        action_index = 1
    elif channel_name == "left_elevon":
        local_index = env._IDX_LEFT_TAIL
        action_index = 2
    elif channel_name == "right_elevon":
        local_index = env._IDX_RIGHT_TAIL
        action_index = 3
    else:
        raise ValueError(f"Unknown tail channel: {channel_name}")
    return (
        env._joint_lower_limits[local_index],
        env._joint_upper_limits[local_index],
        action_index,
    )


def _physical_tail_target_rad(
    sign: float,
    *,
    amplitude_deg: float | None,
    lower_limit_rad: torch.Tensor,
    upper_limit_rad: torch.Tensor,
) -> torch.Tensor:
    if sign == 0.0:
        return torch.zeros((), device=lower_limit_rad.device, dtype=lower_limit_rad.dtype)
    limit = upper_limit_rad if sign > 0.0 else lower_limit_rad
    if amplitude_deg is None:
        return limit
    requested = math.copysign(math.radians(float(amplitude_deg)), sign)
    return torch.clamp(
        torch.as_tensor(requested, device=limit.device, dtype=limit.dtype),
        min=lower_limit_rad,
        max=upper_limit_rad,
    )


def _fixed_root_actions(env: object, time_s: float, dwell_s: float) -> torch.Tensor:
    actions = torch.zeros((len(FIXED_ROOT_CASE_NAMES), 4), device=env.device)
    frequency_targets = torch.full(
        (len(FIXED_ROOT_CASE_NAMES),),
        2.66958,
        device=env.device,
    )
    frequency_targets[0] = sequence_level(
        time_s,
        dwell_s=dwell_s,
        levels=_FREQUENCY_LEVELS_HZ,
    )
    actions[:, 0] = frequency_hz_to_normalized_action(
        frequency_targets,
        minimum_frequency_hz=float(env.cfg.min_flap_hz),
        maximum_frequency_hz=float(env.cfg.max_flap_hz),
    )
    sign = sequence_level(time_s, dwell_s=dwell_s, levels=_TAIL_LEVEL_SIGNS)
    for env_index, (channel_name, amplitude_deg) in enumerate(_TAIL_CASE_SPECS, start=1):
        lower, upper, action_index = _tail_limit_tuple(env, channel_name)
        target_rad = _physical_tail_target_rad(
            sign,
            amplitude_deg=amplitude_deg,
            lower_limit_rad=lower,
            upper_limit_rad=upper,
        )
        actions[env_index, action_index] = joint_position_to_normalized_action(
            target_rad,
            lower_limit_rad=lower,
            upper_limit_rad=upper,
        )
    return actions


def _advance_one_physics_step(env: object) -> None:
    env._sim_step_counter += 1
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)


def run_fixed_root_action_step_job(
    *,
    enable_aerodynamics: bool,
    dwell_s: float = 0.75,
    airspeed_mps: float = 7.0,
    asset_path: Path | None = None,
    usd_dir: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> PureRLActionStepResult:
    """Run the fixed-root A or B direct-action step-response matrix."""

    if dwell_s <= 0.0:
        raise ValueError("dwell_s must be positive.")
    from flapping_bot.direct.flapping_bot.straight_flight_env import FlappingBotStraightFlightEnv

    cfg = _make_cfg(
        num_envs=len(FIXED_ROOT_CASE_NAMES),
        fixed_root=True,
        enable_aerodynamics=bool(enable_aerodynamics),
        airspeed_mps=float(airspeed_mps),
        asset_path=asset_path,
        usd_dir=usd_dir,
    )
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        total_steps = int(math.ceil(dwell_s * len(_FREQUENCY_LEVELS_HZ) / env.physics_dt))
        policy_decimation = 8
        actions = _fixed_root_actions(env, 0.0, dwell_s)
        trace: dict[str, list[np.ndarray | float]] = {
            "time_s": [],
            "target_frequency_hz": [],
            "actual_frequency_hz": [],
            "tail_command_rad": [],
            "tail_actual_rad": [],
            "tail_velocity_rad_s": [],
            "tail_force_b_n": [],
            "tail_moment_b_about_base_com_nm": [],
            "wing_tracking_error_rad": [],
            "wing_sync_error_rad": [],
        }
        left_wing_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_wing_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        tail_local_indices = [env._IDX_RUDDER, env._IDX_LEFT_TAIL, env._IDX_RIGHT_TAIL]
        tail_joint_ids = [int(env._joint_ids[index]) for index in tail_local_indices]
        for step in range(total_steps):
            if step % policy_decimation == 0:
                actions = _fixed_root_actions(env, step * env.physics_dt, dwell_s)
                env._pre_physics_step(actions)
            if progress_callback is not None and step % max(total_steps // 4, 1) == 0:
                mode_label = "aero" if enable_aerodynamics else "no-aero"
                progress_callback(
                    f"fixed-root {mode_label} {100 * step / total_steps:.0f}%"
                )
            _advance_one_physics_step(env)
            joint_position = env._robot.data.joint_pos
            left_position = joint_position[:, left_wing_id] - float(env._wing_mid_L)
            right_position = -(joint_position[:, right_wing_id] - float(env._wing_mid_R))
            reference = float(env._wing_amp) * torch.sin(env._ideal_inverse_phase_state.phase_rad)
            tail_command = torch.stack(
                (env._rudder_cmd, env._left_elevon_cmd, env._right_elevon_cmd), dim=1
            )
            finite_tensors = (
                env._robot.data.root_state_w,
                env._robot.data.joint_pos,
                env._robot.data.joint_vel,
                env._freq,
                env._debug_last_tail_force_b,
                env._debug_last_tail_moment_b_about_base_com_nm,
            )
            if any(not bool(torch.all(torch.isfinite(value))) for value in finite_tensors):
                raise RuntimeError(f"Nonfinite fixed-root action response at physics step {step}.")
            trace["time_s"].append((step + 1) * env.physics_dt)
            trace["target_frequency_hz"].append(_to_numpy(env._phase_target_frequency_hz))
            trace["actual_frequency_hz"].append(_to_numpy(env._freq))
            trace["tail_command_rad"].append(_to_numpy(tail_command))
            trace["tail_actual_rad"].append(_to_numpy(joint_position[:, tail_joint_ids]))
            trace["tail_velocity_rad_s"].append(_to_numpy(env._robot.data.joint_vel[:, tail_joint_ids]))
            trace["tail_force_b_n"].append(_to_numpy(env._debug_last_tail_force_b))
            trace["tail_moment_b_about_base_com_nm"].append(
                _to_numpy(env._debug_last_tail_moment_b_about_base_com_nm)
            )
            trace["wing_tracking_error_rad"].append(_to_numpy(left_position - reference))
            trace["wing_sync_error_rad"].append(_to_numpy(left_position - right_position))

        arrays = _stack_trace(trace)
        time_s = arrays["time_s"]
        cases: list[dict[str, object]] = []
        frequency_response = summarize_step_response(
            time_s,
            arrays["target_frequency_hz"][:, 0],
            arrays["actual_frequency_hz"][:, 0],
        )
        cases.append({"name": "frequency", "response": frequency_response})
        for env_index, (channel_name, _) in enumerate(_TAIL_CASE_SPECS, start=1):
            channel_index = {"rudder": 0, "left_elevon": 1, "right_elevon": 2}[channel_name]
            response = summarize_step_response(
                time_s,
                arrays["tail_command_rad"][:, env_index, channel_index],
                arrays["tail_actual_rad"][:, env_index, channel_index],
            )
            cases.append({"name": FIXED_ROOT_CASE_NAMES[env_index], "response": response})

        frequency_transitions = frequency_response["transitions"]
        frequency_response_accepted = bool(
            len(frequency_transitions) == 4
            and all(
                transition["rise_time_10_90_s"] is not None
                and transition["settling_time_2pct_s"] is not None
                and float(transition["settling_time_2pct_s"]) <= 0.30
                and float(transition["overshoot_percent"]) <= 5.0
                and abs(float(transition["steady_state_error"])) <= 0.02
                for transition in frequency_transitions
            )
        )
        tail_response_accepted = bool(
            all(
                len(case["response"]["transitions"]) == 4
                and all(
                    transition["rise_time_10_90_s"] is not None
                    and transition["settling_time_2pct_s"] is not None
                    and float(transition["settling_time_2pct_s"]) <= 0.50
                    and float(transition["overshoot_percent"]) <= 5.0
                    and abs(float(transition["steady_state_error"])) <= math.radians(0.25)
                    for transition in case["response"]["transitions"]
                )
                for case in cases[1:]
            )
        )

        soft_lower = _to_numpy(env._joint_lower_limits[tail_local_indices])
        soft_upper = _to_numpy(env._joint_upper_limits[tail_local_indices])
        tail_actual = arrays["tail_actual_rad"]
        limit_violation_rad = np.maximum(
            np.maximum(soft_lower[None, None, :] - tail_actual, 0.0),
            np.maximum(tail_actual - soft_upper[None, None, :], 0.0),
        )
        max_tracking_deg = math.degrees(float(np.max(np.abs(arrays["wing_tracking_error_rad"]))))
        max_sync_deg = math.degrees(float(np.max(np.abs(arrays["wing_sync_error_rad"]))))
        max_limit_violation_deg = math.degrees(float(np.max(limit_violation_rad)))
        aero_checks: dict[str, object] = {}
        if enable_aerodynamics:
            window = max(1, int(round(0.15 / env.physics_dt)))
            neutral_end = int(round(dwell_s / env.physics_dt))
            plus_end = int(round(2.0 * dwell_s / env.physics_dt))
            minus_end = int(round(4.0 * dwell_s / env.physics_dt))
            rudder_force_neutral = np.mean(
                arrays["tail_force_b_n"][neutral_end - window:neutral_end, 1, :], axis=0
            )
            rudder_moment_neutral = np.mean(
                arrays["tail_moment_b_about_base_com_nm"][neutral_end - window:neutral_end, 1, :], axis=0
            )
            rudder_force_plus = np.mean(
                arrays["tail_force_b_n"][plus_end - window:plus_end, 1, :], axis=0
            ) - rudder_force_neutral
            rudder_force_minus = np.mean(
                arrays["tail_force_b_n"][minus_end - window:minus_end, 1, :], axis=0
            ) - rudder_force_neutral
            rudder_moment_plus = np.mean(
                arrays["tail_moment_b_about_base_com_nm"][plus_end - window:plus_end, 1, :], axis=0
            ) - rudder_moment_neutral
            rudder_moment_minus = np.mean(
                arrays["tail_moment_b_about_base_com_nm"][minus_end - window:minus_end, 1, :], axis=0
            ) - rudder_moment_neutral
            left_moment_neutral = np.mean(
                arrays["tail_moment_b_about_base_com_nm"][neutral_end - window:neutral_end, 4, :], axis=0
            )
            right_moment_neutral = np.mean(
                arrays["tail_moment_b_about_base_com_nm"][neutral_end - window:neutral_end, 7, :], axis=0
            )
            left_moment_plus = np.mean(
                arrays["tail_moment_b_about_base_com_nm"][plus_end - window:plus_end, 4, :], axis=0
            ) - left_moment_neutral
            right_moment_plus = np.mean(
                arrays["tail_moment_b_about_base_com_nm"][plus_end - window:plus_end, 7, :], axis=0
            ) - right_moment_neutral
            aero_checks = {
                "rudder_positive_fy_negative": bool(rudder_force_plus[1] < 0.0),
                "rudder_negative_fy_positive": bool(rudder_force_minus[1] > 0.0),
                "rudder_positive_mz_positive": bool(rudder_moment_plus[2] > 0.0),
                "rudder_negative_mz_negative": bool(rudder_moment_minus[2] < 0.0),
                "left_right_positive_roll_moments_opposed": bool(left_moment_plus[0] * right_moment_plus[0] < 0.0),
                "left_right_positive_pitch_moments_same_sign": bool(left_moment_plus[1] * right_moment_plus[1] > 0.0),
                "rudder_positive_force_b_n": rudder_force_plus.tolist(),
                "rudder_negative_force_b_n": rudder_force_minus.tolist(),
                "rudder_positive_moment_b_nm": rudder_moment_plus.tolist(),
                "rudder_negative_moment_b_nm": rudder_moment_minus.tolist(),
                "left_positive_moment_b_nm": left_moment_plus.tolist(),
                "right_positive_moment_b_nm": right_moment_plus.tolist(),
            }
        all_aero_checks = all(value for value in aero_checks.values() if isinstance(value, bool))
        accepted = bool(
            frequency_response_accepted
            and tail_response_accepted
            and max_tracking_deg < 0.1
            and max_sync_deg < 0.1
            and max_limit_violation_deg < 0.25
            and (not enable_aerodynamics or all_aero_checks)
        )
        summary: dict[str, object] = {
            "schema_version": "pure_rl_direct_action_step_v1",
            "job_type": "fixed_root_action_step",
            "mode": "aero" if enable_aerodynamics else "no_aero",
            "fixed_root": True,
            "physics_hz": 480.0,
            "policy_hz": 60.0,
            "policy_decimation": policy_decimation,
            "dwell_s": float(dwell_s),
            "duration_s": float(total_steps * env.physics_dt),
            "airspeed_mps": float(airspeed_mps if enable_aerodynamics else 0.0),
            "outer_action_lpf_tau_s": 0.0,
            "outer_action_rate_limit_per_s": 0.0,
            "tail_aero_deflection_source": str(cfg.tail_aero_deflection_source),
            "tail_wrench_application_body": "base_link",
            "tail_wrench_reference": "base_com",
            "tail_hinge_aero_load_applied": False,
            "tail_soft_lower_rad": soft_lower.tolist(),
            "tail_soft_upper_rad": soft_upper.tolist(),
            "max_tail_soft_limit_violation_deg": max_limit_violation_deg,
            "max_wing_tracking_error_deg": max_tracking_deg,
            "max_wing_sync_error_deg": max_sync_deg,
            "response_acceptance": {
                "frequency_response_accepted": frequency_response_accepted,
                "tail_response_accepted": tail_response_accepted,
                "maximum_frequency_settling_s": 0.30,
                "maximum_tail_settling_s": 0.50,
                "maximum_overshoot_percent": 5.0,
                "maximum_frequency_steady_error_hz": 0.02,
                "maximum_tail_steady_error_deg": 0.25,
            },
            "aero_sign_and_symmetry_checks": aero_checks,
            "cases": cases,
            "all_cases_accepted": accepted,
        }
        return PureRLActionStepResult(summary=summary, traces=arrays)
    finally:
        import omni.physx

        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()


def run_free_root_action_smoke_job(
    *,
    duration_s: float = 1.0,
    airspeed_mps: float = 7.0,
    asset_path: Path | None = None,
    usd_dir: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> PureRLActionStepResult:
    """Run a bounded free-root direct-surface pulse smoke from the trim seed."""

    if duration_s <= 0.0 or duration_s > 1.0:
        raise ValueError("duration_s must be in (0, 1].")
    from flapping_bot.direct.flapping_bot.straight_flight_env import FlappingBotStraightFlightEnv

    cfg = _make_cfg(
        num_envs=len(FREE_ROOT_CASE_NAMES),
        fixed_root=False,
        enable_aerodynamics=True,
        airspeed_mps=float(airspeed_mps),
        asset_path=asset_path,
        usd_dir=usd_dir,
    )
    cfg.reset_pitch_deg = 11.70944
    cfg.reset_flap_hz = 2.66958
    cfg.reset_rudder_deg = 0.46
    cfg.reset_elevon_pitch_deg = -6.96703
    cfg.reset_elevon_roll_deg = -3.05
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        base_action = env._act_cmd.detach().clone()
        base_tail_position = torch.stack(
            (env._rudder_cmd, env._left_elevon_cmd, env._right_elevon_cmd), dim=1
        ).detach().clone()
        total_steps = int(math.ceil(duration_s / env.physics_dt))
        trace: dict[str, list[np.ndarray | float]] = {
            "time_s": [],
            "root_position_w_m": [],
            "root_linear_velocity_w_mps": [],
            "root_angular_velocity_b_rad_s": [],
            "tail_command_rad": [],
            "tail_actual_rad": [],
            "tail_force_b_n": [],
            "tail_moment_b_about_base_com_nm": [],
        }
        pulse_delta = math.radians(5.0)
        tail_specs = ((1, "rudder"), (2, "left_elevon"), (3, "right_elevon"))
        tail_joint_ids = [
            int(env._joint_ids[env._IDX_RUDDER]),
            int(env._joint_ids[env._IDX_LEFT_TAIL]),
            int(env._joint_ids[env._IDX_RIGHT_TAIL]),
        ]
        actions = base_action
        for step in range(total_steps):
            time_s = step * env.physics_dt
            if step % 8 == 0:
                actions = base_action.clone()
                if 0.25 <= time_s < 0.5:
                    for env_index, channel_name in tail_specs:
                        lower, upper, action_index = _tail_limit_tuple(env, channel_name)
                        channel_index = {
                            "rudder": 0,
                            "left_elevon": 1,
                            "right_elevon": 2,
                        }[channel_name]
                        base_position = base_tail_position[env_index, channel_index]
                        target = torch.clamp(base_position + pulse_delta, min=lower, max=upper)
                        actions[env_index, action_index] = joint_position_to_normalized_action(
                            target,
                            lower_limit_rad=lower,
                            upper_limit_rad=upper,
                        )
                env._pre_physics_step(actions)
            if progress_callback is not None and step % max(total_steps // 4, 1) == 0:
                progress_callback(f"free-root smoke {100 * step / total_steps:.0f}%")
            _advance_one_physics_step(env)
            tail_command = torch.stack(
                (env._rudder_cmd, env._left_elevon_cmd, env._right_elevon_cmd), dim=1
            )
            finite_tensors = (
                env._robot.data.root_state_w,
                env._robot.data.joint_pos,
                env._robot.data.joint_vel,
                env._debug_last_force_b,
                env._debug_last_torque_b,
            )
            if any(not bool(torch.all(torch.isfinite(value))) for value in finite_tensors):
                raise RuntimeError(f"Nonfinite free-root action smoke at physics step {step}.")
            trace["time_s"].append((step + 1) * env.physics_dt)
            trace["root_position_w_m"].append(_to_numpy(env._robot.data.root_pos_w))
            trace["root_linear_velocity_w_mps"].append(_to_numpy(env._robot.data.root_lin_vel_w))
            trace["root_angular_velocity_b_rad_s"].append(_to_numpy(env._robot.data.root_ang_vel_b))
            trace["tail_command_rad"].append(_to_numpy(tail_command))
            trace["tail_actual_rad"].append(_to_numpy(env._robot.data.joint_pos[:, tail_joint_ids]))
            trace["tail_force_b_n"].append(_to_numpy(env._debug_last_tail_force_b))
            trace["tail_moment_b_about_base_com_nm"].append(
                _to_numpy(env._debug_last_tail_moment_b_about_base_com_nm)
            )
        arrays = _stack_trace(trace)
        linear_speed = np.linalg.norm(arrays["root_linear_velocity_w_mps"], axis=2)
        angular_speed = np.linalg.norm(arrays["root_angular_velocity_b_rad_s"], axis=2)
        max_linear_speed = float(np.max(linear_speed))
        max_angular_speed = float(np.max(angular_speed))
        pulse_slice = (arrays["time_s"] >= 0.35) & (arrays["time_s"] < 0.5)
        baseline_moment = np.mean(arrays["tail_moment_b_about_base_com_nm"][pulse_slice, 0, :], axis=0)
        rudder_delta = np.mean(arrays["tail_moment_b_about_base_com_nm"][pulse_slice, 1, :], axis=0) - baseline_moment
        left_delta = np.mean(arrays["tail_moment_b_about_base_com_nm"][pulse_slice, 2, :], axis=0) - baseline_moment
        right_delta = np.mean(arrays["tail_moment_b_about_base_com_nm"][pulse_slice, 3, :], axis=0) - baseline_moment
        response_checks = {
            "positive_rudder_increases_yaw_moment": bool(rudder_delta[2] > 0.0),
            "left_right_roll_moment_increments_opposed": bool(left_delta[0] * right_delta[0] < 0.0),
            "left_right_pitch_moment_increments_same_sign": bool(left_delta[1] * right_delta[1] > 0.0),
        }
        summary: dict[str, object] = {
            "schema_version": "pure_rl_direct_action_step_v1",
            "job_type": "free_root_action_smoke",
            "fixed_root": False,
            "physics_hz": 480.0,
            "policy_hz": 60.0,
            "duration_s": float(total_steps * env.physics_dt),
            "airspeed_mps": float(airspeed_mps),
            "reset_seed": {
                "pitch_deg_nose_up": 11.70944,
                "frequency_hz": 2.66958,
                "rudder_deg": 0.46,
                "elevon_pitch_deg": -6.96703,
                "elevon_roll_deg": -3.05,
                "periodic_orbit_validated": False,
            },
            "max_root_linear_speed_mps": max_linear_speed,
            "max_root_angular_speed_rad_s": max_angular_speed,
            "tail_response_checks": response_checks,
            "tail_hinge_aero_load_applied": False,
            "stability_claim": "finite_and_numerically_bounded_only",
            "all_cases_accepted": bool(
                max_linear_speed < 100.0
                and max_angular_speed < 1000.0
                and all(response_checks.values())
            ),
        }
        return PureRLActionStepResult(summary=summary, traces=arrays)
    finally:
        import omni.physx

        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()


def run_pure_rl_action_step_job(
    *,
    mode: Literal["no_aero", "aero", "free_smoke"],
    dwell_s: float = 0.75,
    duration_s: float = 1.0,
    airspeed_mps: float = 7.0,
    asset_path: Path | None = None,
    usd_dir: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> PureRLActionStepResult:
    """Dispatch one fresh-process action validation mode."""

    if mode == "no_aero":
        return run_fixed_root_action_step_job(
            enable_aerodynamics=False,
            dwell_s=dwell_s,
            airspeed_mps=airspeed_mps,
            asset_path=asset_path,
            usd_dir=usd_dir,
            progress_callback=progress_callback,
        )
    if mode == "aero":
        return run_fixed_root_action_step_job(
            enable_aerodynamics=True,
            dwell_s=dwell_s,
            airspeed_mps=airspeed_mps,
            asset_path=asset_path,
            usd_dir=usd_dir,
            progress_callback=progress_callback,
        )
    if mode == "free_smoke":
        return run_free_root_action_smoke_job(
            duration_s=duration_s,
            airspeed_mps=airspeed_mps,
            asset_path=asset_path,
            usd_dir=usd_dir,
            progress_callback=progress_callback,
        )
    raise ValueError(f"Unsupported mode: {mode}")


__all__ = [
    "FIXED_ROOT_CASE_NAMES",
    "FREE_ROOT_CASE_NAMES",
    "PureRLActionStepResult",
    "run_fixed_root_action_step_job",
    "run_free_root_action_smoke_job",
    "run_pure_rl_action_step_job",
    "sequence_level",
    "summarize_step_response",
]

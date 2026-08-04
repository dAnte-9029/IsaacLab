"""Headless validation workflow for the native holonomic wing mechanism."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg,
    FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg,
)

from .multibody_inertial_validation import fit_fundamental_harmonic
from .multibody_aero_validation import compute_impulse_momentum_closure

_STATE_SIGNALS = (
    "q_actual_rad",
    "qdot_actual_rad_s",
    "qdd_physx_rad_s2",
)
_AERO_SIGNALS = (
    "wing_force_b_x_n",
    "wing_force_b_z_n",
    "wing_moment_b_y_nm",
)
_STRIP_COMPONENTS = (
    "dN_c",
    "dN_a",
    "dT_s",
    "dD_camber",
    "dD_f",
    "dM_ac",
    "dM_a",
)
_MAX_MECHANISM_ERROR_DEG = 0.1
_MAX_RELATIVE_LINEAR_CLOSURE_ERROR = 1.0e-3
_MAX_RELATIVE_ANGULAR_CLOSURE_ERROR = 5.0e-2


def _frequency_actions(
    frequencies_hz: Sequence[float],
    *,
    maximum_frequency_hz: float,
    device: str,
) -> torch.Tensor:
    frequencies = torch.as_tensor(frequencies_hz, dtype=torch.float32, device=device)
    actions = torch.zeros((len(frequencies_hz), 4), dtype=torch.float32, device=device)
    actions[:, 0] = 2.0 * frequencies / float(maximum_frequency_hz) - 1.0
    return actions


def _make_cfg(
    *,
    dt_s: float,
    device: str,
    num_envs: int,
    enable_aerodynamics: bool,
    airspeed_mps: float,
    solver_position_iterations: int | None,
    solver_velocity_iterations: int | None,
    fixed_root: bool,
):
    if enable_aerodynamics:
        cfg = FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg()
    else:
        cfg = FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.scene.env_spacing = 5.0
    cfg.sim.device = str(device)
    cfg.sim.dt = float(dt_s)
    cfg.sim.render_interval = 1
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.decimation = 1
    cfg.episode_length_s = 20.0
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 0.0
    cfg.reset_flap_hz = 0.0
    cfg.enable_tail_aero = False
    cfg.fuselage_drag_cda = 0.0
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    cfg.randomize_commands = False
    cfg.wind_enabled = bool(enable_aerodynamics)
    cfg.wind_xy_mps = (-float(airspeed_mps), 0.0)
    cfg.wind_ou_enabled = False
    cfg.delaurier_store_strip_diagnostics = bool(enable_aerodynamics)
    articulation_props = cfg.robot.spawn.articulation_props.replace(
        fix_root_link=bool(fixed_root),
    )
    if solver_position_iterations is not None:
        articulation_props = articulation_props.replace(
            solver_position_iteration_count=int(solver_position_iterations),
        )
    if solver_velocity_iterations is not None:
        articulation_props = articulation_props.replace(
            solver_velocity_iteration_count=int(solver_velocity_iterations),
        )
    cfg.robot = cfg.robot.replace(
        spawn=cfg.robot.spawn.replace(
            rigid_props=cfg.robot.spawn.rigid_props.replace(
                disable_gravity=True,
                # Aerodynamic wrenches are recomputed and applied once per physics step.
                # Retaining them would accumulate the PhysX force buffer across steps.
                retain_accelerations=False,
            ),
            articulation_props=articulation_props,
        ),
    )
    return cfg


def _empty_case_trace() -> dict[str, list[float]]:
    return {
        "state_time_s": [],
        "aero_time_s": [],
        "frequency_hz": [],
        "tracking_error_rad": [],
        "velocity_error_rad_s": [],
        "sync_error_rad": [],
        **{name: [] for name in _STATE_SIGNALS},
        **{name: [] for name in _AERO_SIGNALS},
        **{f"component_{name}": [] for name in _STRIP_COMPONENTS},
    }


def _fit(time_s: list[float], values: list[float], frequency_hz: float) -> dict[str, float]:
    return fit_fundamental_harmonic(
        np.asarray(time_s, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        frequency_hz=float(frequency_hz),
    ).to_dict()


def _component_sums(strip_loads: object, env_index: int) -> dict[str, torch.Tensor]:
    wing_slice = slice(2 * env_index, 2 * env_index + 2)
    return {
        name: getattr(strip_loads, name)[wing_slice].sum(dim=1)
        for name in _STRIP_COMPONENTS
    }


def _peak_force_diagnostic(
    *,
    env: FlappingBotStraightFlightEnv,
    env_index: int,
    wing_index: int,
    time_s: float,
    phase_rad: float,
    frequency_before_hz: float,
) -> dict[str, float | int | str]:
    strip_loads = env._debug_last_delaurier_strip_loads
    strip_wrench = env._debug_last_delaurier_strip_wrench
    if strip_loads is None or strip_wrench is None:
        return {}
    batch_index = 2 * env_index + wing_index
    result: dict[str, float | int | str] = {
        "time_s": float(time_s),
        "phase_rad": float(phase_rad),
        "phase_deg": math.degrees(float(phase_rad) % (2.0 * math.pi)),
        "frequency_before_hz": float(frequency_before_hz),
        "frequency_after_hz": float(env._freq[env_index].item()),
        "phase_acceleration_rad_s2": float(
            env._phase_acceleration_rad_s2[env_index].item()
        ),
        "wing_index": int(wing_index),
        "wing": "left" if wing_index == 0 else "right",
        "force_link_norm_n": float(
            torch.linalg.vector_norm(
                env._debug_last_wing_force_link_n[env_index, wing_index]
            ).item()
        ),
        "q_aero_rad": float(
            env._debug_last_wing_aero_position_rad[env_index, wing_index].item()
        ),
        "qdot_aero_rad_s": float(
            env._debug_last_wing_aero_velocity_rad_s[env_index, wing_index].item()
        ),
        "qdd_aero_rad_s2": float(
            env._debug_last_wing_aero_acceleration_rad_s2[env_index, wing_index].item()
        ),
        "force_normal_wang_n": float(strip_wrench.force_normal_wang[batch_index, 1].item()),
        "force_chordwise_wang_n": float(
            strip_wrench.force_chordwise_wang[batch_index, 2].item()
        ),
    }
    for name in _STRIP_COMPONENTS:
        values = getattr(strip_loads, name)[batch_index]
        result[f"{name}_sum"] = float(values.sum().item())
        result[f"{name}_max_abs_strip"] = float(torch.max(torch.abs(values)).item())
    if strip_loads.alpha is not None:
        result["alpha_max_abs_deg"] = math.degrees(
            float(torch.max(torch.abs(strip_loads.alpha[batch_index])).item())
        )
    if strip_loads.alpha_prime is not None:
        result["alpha_prime_max_abs_deg"] = math.degrees(
            float(torch.max(torch.abs(strip_loads.alpha_prime[batch_index])).item())
        )
    if strip_loads.alpha_le is not None:
        result["alpha_le_max_abs_deg"] = math.degrees(
            float(torch.max(torch.abs(strip_loads.alpha_le[batch_index])).item())
        )
    if strip_loads.reduced_frequency is not None:
        result["reduced_frequency_min"] = float(
            torch.min(strip_loads.reduced_frequency[batch_index]).item()
        )
        result["reduced_frequency_max"] = float(
            torch.max(strip_loads.reduced_frequency[batch_index]).item()
        )
    twist = env._debug_last_delaurier_twist_kinematics
    if twist is not None:
        for name, values in (
            ("theta_max_abs_deg", twist.theta),
            ("theta_dot_max_abs_deg_s", twist.theta_dot),
            ("theta_ddot_max_abs_deg_s2", twist.theta_ddot),
        ):
            result[name] = math.degrees(
                float(torch.max(torch.abs(values[batch_index])).item())
            )
    return result


def run_native_holonomic_frequency_job(
    *,
    frequencies_hz: Sequence[float] = (2.0, 3.0, 4.0, 5.0),
    dt_s: float = 1.0 / 480.0,
    device: str = "cpu",
    enable_aerodynamics: bool = False,
    airspeed_mps: float = 8.0,
    total_cycles: int = 4,
    measurement_cycles: int = 2,
    solver_position_iterations: int | None = None,
    solver_velocity_iterations: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run fixed-root frequency cases with correctly aligned state timestamps."""

    frequencies = tuple(float(value) for value in frequencies_hz)
    if not frequencies or any(value <= 0.0 for value in frequencies):
        raise ValueError("frequencies_hz must contain positive values.")
    if total_cycles < measurement_cycles or measurement_cycles < 1:
        raise ValueError("Use total_cycles >= measurement_cycles >= 1.")
    if solver_position_iterations is not None and solver_position_iterations < 1:
        raise ValueError("solver_position_iterations must be positive.")
    if solver_velocity_iterations is not None and solver_velocity_iterations < 1:
        raise ValueError("solver_velocity_iterations must be positive.")
    cfg = _make_cfg(
        dt_s=float(dt_s),
        device=str(device),
        num_envs=len(frequencies),
        enable_aerodynamics=bool(enable_aerodynamics),
        airspeed_mps=float(airspeed_mps),
        solver_position_iterations=solver_position_iterations,
        solver_velocity_iterations=solver_velocity_iterations,
        fixed_root=True,
    )
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        actions = _frequency_actions(
            frequencies,
            maximum_frequency_hz=cfg.max_flap_hz,
            device=env.device,
        )
        env._pre_physics_step(actions)
        traces = [_empty_case_trace() for _ in frequencies]
        global_peak_force_n = np.zeros(len(frequencies), dtype=np.float64)
        steady_peak_force_n = np.zeros(len(frequencies), dtype=np.float64)
        peak_diagnostics: list[dict[str, float | int | str]] = [
            {} for _ in frequencies
        ]
        duration_s = max(float(total_cycles) / value for value in frequencies)
        total_steps = int(math.ceil(duration_s / float(dt_s)))
        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])

        for step in range(total_steps):
            if progress_callback is not None and step % max(total_steps // 4, 1) == 0:
                progress_callback(
                    f"native aero={enable_aerodynamics} device={device} "
                    f"dt={dt_s:.9f}s {100.0 * step / total_steps:.0f}%"
                )
            aero_time_s = step * float(dt_s)
            phase_before = env._phase.detach().clone()
            frequency_before = env._freq.detach().clone()
            env._sim_step_counter += 1
            env._apply_action()

            force_norm = torch.linalg.vector_norm(
                env._debug_last_wing_force_link_n,
                dim=-1,
            )
            for index, frequency_hz in enumerate(frequencies):
                peak_value, peak_wing = torch.max(force_norm[index], dim=0)
                peak_value_float = float(peak_value.item())
                if peak_value_float > global_peak_force_n[index]:
                    global_peak_force_n[index] = peak_value_float
                    peak_diagnostics[index] = _peak_force_diagnostic(
                        env=env,
                        env_index=index,
                        wing_index=int(peak_wing.item()),
                        time_s=aero_time_s,
                        phase_rad=float(phase_before[index].item()),
                        frequency_before_hz=float(frequency_before[index].item()),
                    )
                measurement_start_s = duration_s - float(measurement_cycles) / frequency_hz
                if aero_time_s < measurement_start_s:
                    continue
                steady_peak_force_n[index] = max(
                    steady_peak_force_n[index],
                    peak_value_float,
                )
                trace = traces[index]
                trace["aero_time_s"].append(aero_time_s)
                trace["wing_force_b_x_n"].append(
                    float(env._debug_last_wing_force_b[index, 0].item())
                )
                trace["wing_force_b_z_n"].append(
                    float(env._debug_last_wing_force_b[index, 2].item())
                )
                trace["wing_moment_b_y_nm"].append(
                    float(env._debug_last_torque_b[index, 1].item())
                )
                if enable_aerodynamics:
                    strip_loads = env._debug_last_delaurier_strip_loads
                    if strip_loads is None:
                        raise RuntimeError("Expected DeLaurier strip diagnostics.")
                    component_sums = _component_sums(strip_loads, index)
                    for name, values in component_sums.items():
                        trace[f"component_{name}"].extend(
                            float(value) for value in values.detach().cpu().tolist()
                        )

            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            state_time_s = (step + 1) * float(dt_s)
            left_position = env._robot.data.joint_pos[:, left_joint_id] - float(env._wing_mid_L)
            right_position = -(
                env._robot.data.joint_pos[:, right_joint_id] - float(env._wing_mid_R)
            )
            q_actual = 0.5 * (left_position + right_position)
            qdot_actual = 0.5 * (
                env._robot.data.joint_vel[:, left_joint_id]
                - env._robot.data.joint_vel[:, right_joint_id]
            )
            qdd_physx = 0.5 * (
                env._robot.data.joint_acc[:, left_joint_id]
                - env._robot.data.joint_acc[:, right_joint_id]
            )
            phase = env._ideal_inverse_phase_state.phase_rad
            frequency = env._ideal_inverse_phase_state.frequency_hz
            q_reference = env._wing_amp * torch.sin(phase)
            qdot_reference = env._wing_amp * 2.0 * math.pi * frequency * torch.cos(phase)
            sync_error = torch.abs(left_position - right_position)

            finite_tensors = (
                q_actual,
                qdot_actual,
                qdd_physx,
                env._debug_last_wing_force_link_n,
                env._debug_last_wing_moment_link_about_com_nm,
            )
            if any(not bool(torch.all(torch.isfinite(value))) for value in finite_tensors):
                raise RuntimeError(f"Nonfinite native mechanism state at step {step}.")
            for index, frequency_hz in enumerate(frequencies):
                measurement_start_s = duration_s - float(measurement_cycles) / frequency_hz
                if state_time_s < measurement_start_s:
                    continue
                trace = traces[index]
                trace["state_time_s"].append(state_time_s)
                trace["frequency_hz"].append(float(frequency[index].item()))
                trace["q_actual_rad"].append(float(q_actual[index].item()))
                trace["qdot_actual_rad_s"].append(float(qdot_actual[index].item()))
                trace["qdd_physx_rad_s2"].append(float(qdd_physx[index].item()))
                trace["tracking_error_rad"].append(
                    float(torch.abs(q_actual[index] - q_reference[index]).item())
                )
                trace["velocity_error_rad_s"].append(
                    float(torch.abs(qdot_actual[index] - qdot_reference[index]).item())
                )
                trace["sync_error_rad"].append(float(sync_error[index].item()))

        cases: list[dict[str, object]] = []
        for index, frequency_hz in enumerate(frequencies):
            trace = traces[index]
            harmonics = {
                name: _fit(trace["state_time_s"], trace[name], frequency_hz)
                for name in _STATE_SIGNALS
            }
            harmonics.update(
                {
                    name: _fit(trace["aero_time_s"], trace[name], frequency_hz)
                    for name in _AERO_SIGNALS
                }
            )
            component_summary: dict[str, dict[str, float]] = {}
            if enable_aerodynamics:
                for name in _STRIP_COMPONENTS:
                    values = np.asarray(trace[f"component_{name}"], dtype=np.float64)
                    component_summary[name] = {
                        "max_abs_sum": float(np.max(np.abs(values))),
                        "rms_sum": float(np.sqrt(np.mean(np.square(values)))),
                    }
            max_tracking_error_deg = math.degrees(max(trace["tracking_error_rad"]))
            max_sync_error_deg = math.degrees(max(trace["sync_error_rad"]))
            accepted = max_tracking_error_deg < 0.1 and max_sync_error_deg < 0.1
            cases.append(
                {
                    "frequency_hz": frequency_hz,
                    "dt_s": float(dt_s),
                    "device": str(device),
                    "enable_aerodynamics": bool(enable_aerodynamics),
                    "airspeed_mps": float(airspeed_mps) if enable_aerodynamics else 0.0,
                    "stable": True,
                    "accepted": accepted,
                    "sample_count": len(trace["state_time_s"]),
                    "mean_actual_frequency_hz": float(np.mean(trace["frequency_hz"])),
                    "max_tracking_error_deg": max_tracking_error_deg,
                    "rms_tracking_error_deg": math.degrees(
                        float(np.sqrt(np.mean(np.square(trace["tracking_error_rad"]))))
                    ),
                    "max_velocity_error_rad_s": max(trace["velocity_error_rad_s"]),
                    "rms_velocity_error_rad_s": float(
                        np.sqrt(np.mean(np.square(trace["velocity_error_rad_s"])))
                    ),
                    "max_sync_error_deg": max_sync_error_deg,
                    "global_peak_single_wing_force_n": float(global_peak_force_n[index]),
                    "steady_peak_single_wing_force_n": float(steady_peak_force_n[index]),
                    "harmonics": harmonics,
                    "strip_component_summary": component_summary,
                    "global_peak_force_diagnostic": peak_diagnostics[index],
                }
            )
        return {
            "schema_version": "native_holonomic_frequency_v3",
            "job_type": "frequency",
            "dt_s": float(dt_s),
            "device": str(device),
            "enable_aerodynamics": bool(enable_aerodynamics),
            "solver_position_iterations": int(
                cfg.robot.spawn.articulation_props.solver_position_iteration_count
            ),
            "solver_velocity_iterations": int(
                cfg.robot.spawn.articulation_props.solver_velocity_iteration_count
            ),
            "all_cases_accepted": all(bool(case["accepted"]) for case in cases),
            "cases": cases,
        }
    finally:
        import omni.physx

        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()


def _mass_inertia_tensors(
    env: FlappingBotStraightFlightEnv,
) -> tuple[torch.Tensor, torch.Tensor]:
    masses = env._robot.root_physx_view.get_masses()[0].to(env.device)
    inertias = env._robot.root_physx_view.get_inertias()[0].reshape(-1, 3, 3)
    return masses, inertias.to(env.device)


def _total_momentum(
    env: FlappingBotStraightFlightEnv,
    masses_kg: torch.Tensor,
    inertias_kg_m2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = env._robot.data.body_com_pos_w
    linear_velocities = env._robot.data.body_com_lin_vel_w
    angular_velocities = env._robot.data.body_com_ang_vel_w
    mass_weights = masses_kg.view(1, -1, 1)
    system_com = torch.sum(mass_weights * positions, dim=1) / masses_kg.sum()
    linear_momentum = torch.sum(mass_weights * linear_velocities, dim=1)
    rotations = matrix_from_quat(env._robot.data.body_com_quat_w)
    inertia_w = rotations @ inertias_kg_m2.unsqueeze(0) @ rotations.transpose(-1, -2)
    spin = torch.einsum("nbij,nbj->nbi", inertia_w, angular_velocities)
    orbital = torch.linalg.cross(
        positions - system_com.unsqueeze(1),
        mass_weights * linear_velocities,
        dim=-1,
    )
    angular_momentum = torch.sum(spin + orbital, dim=1)
    return system_com, linear_momentum, angular_momentum


def _external_wrench_about_system_com(
    env: FlappingBotStraightFlightEnv,
    system_com_w_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    body_quat_w = env._robot.data.body_link_quat_w
    forces_local = env._robot._external_force_b
    moments_local_about_com = env._robot._external_torque_b
    forces_w = quat_apply(
        body_quat_w.reshape(-1, 4),
        forces_local.reshape(-1, 3),
    ).reshape_as(forces_local)
    moments_w_about_body_com = quat_apply(
        body_quat_w.reshape(-1, 4),
        moments_local_about_com.reshape(-1, 3),
    ).reshape_as(moments_local_about_com)
    moments_w_about_system_com = moments_w_about_body_com + torch.linalg.cross(
        env._robot.data.body_com_pos_w - system_com_w_m.unsqueeze(1),
        forces_w,
        dim=-1,
    )
    return forces_w.sum(dim=1), moments_w_about_system_com.sum(dim=1)


def run_native_holonomic_free_aero_job(
    *,
    frequencies_hz: Sequence[float] = (2.0, 5.0),
    dt_s: float = 1.0 / 480.0,
    device: str = "cpu",
    airspeed_mps: float = 8.0,
    duration_s: float = 1.0,
    wing_link_load_mode: str = "full_wing_link_wrench",
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run a released-base aerodynamic impulse-momentum closure gate."""

    frequencies = tuple(float(value) for value in frequencies_hz)
    if not frequencies or any(value <= 0.0 for value in frequencies):
        raise ValueError("frequencies_hz must contain positive values.")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")
    cfg = _make_cfg(
        dt_s=float(dt_s),
        device=str(device),
        num_envs=len(frequencies),
        enable_aerodynamics=True,
        airspeed_mps=float(airspeed_mps),
        solver_position_iterations=None,
        solver_velocity_iterations=None,
        fixed_root=False,
    )
    cfg.wing_link_aero_load_mode = str(wing_link_load_mode)
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        actions = _frequency_actions(
            frequencies,
            maximum_frequency_hz=cfg.max_flap_hz,
            device=env.device,
        )
        env._pre_physics_step(actions)
        masses, inertias = _mass_inertia_tensors(env)
        total_steps = int(math.ceil(float(duration_s) / float(dt_s)))
        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        momentum_linear: list[list[np.ndarray]] = [[] for _ in frequencies]
        momentum_angular: list[list[np.ndarray]] = [[] for _ in frequencies]
        force_traces: list[list[np.ndarray]] = [[] for _ in frequencies]
        moment_traces: list[list[np.ndarray]] = [[] for _ in frequencies]
        max_tracking = np.zeros(len(frequencies), dtype=np.float64)
        max_sync = np.zeros(len(frequencies), dtype=np.float64)
        max_force = np.zeros(len(frequencies), dtype=np.float64)
        max_root_linear_speed = np.zeros(len(frequencies), dtype=np.float64)
        max_root_angular_speed = np.zeros(len(frequencies), dtype=np.float64)
        previous_root_state = env._robot.data.root_state_w.detach().clone()

        for step in range(total_steps):
            if progress_callback is not None and step % max(total_steps // 4, 1) == 0:
                progress_callback(
                    f"native free aero device={device} dt={dt_s:.9f}s "
                    f"{100.0 * step / total_steps:.0f}%"
                )
            env._sim_step_counter += 1
            env._apply_action()
            pre_force_norm = torch.linalg.vector_norm(
                env._debug_last_wing_force_link_n,
                dim=-1,
            )
            pre_root_linear_speed = torch.linalg.vector_norm(
                env._robot.data.root_lin_vel_w,
                dim=1,
            )
            pre_root_angular_speed = torch.linalg.vector_norm(
                env._robot.data.root_ang_vel_w,
                dim=1,
            )
            bounded_metric = torch.maximum(
                pre_force_norm.amax(dim=1) / 100.0,
                torch.maximum(
                    pre_root_linear_speed / 100.0,
                    pre_root_angular_speed / 1000.0,
                ),
            )
            if bool(torch.any(bounded_metric > 1.0)):
                diagnostic_env = int(torch.argmax(bounded_metric).item())
                peak_wing = int(torch.argmax(pre_force_norm[diagnostic_env]).item())
                diagnostic = _peak_force_diagnostic(
                    env=env,
                    env_index=diagnostic_env,
                    wing_index=peak_wing,
                    time_s=step * float(dt_s),
                    phase_rad=float(env._phase[diagnostic_env].item()),
                    frequency_before_hz=float(env._freq[diagnostic_env].item()),
                )
                root_quat = env._robot.data.root_quat_w[diagnostic_env : diagnostic_env + 1]
                wind_body = quat_apply_inverse(
                    root_quat,
                    env._wind_w[diagnostic_env : diagnostic_env + 1],
                )[0]
                air_velocity_body = (
                    env._robot.data.root_lin_vel_b[diagnostic_env] - wind_body
                )
                raise RuntimeError(
                    "Unbounded native free-body aerodynamic response "
                    f"at step {step}, t={step * float(dt_s):.9f}s, "
                    f"frequency_target_hz={frequencies[diagnostic_env]}, "
                    f"root_linear_speed_mps={float(pre_root_linear_speed[diagnostic_env].item())}, "
                    f"root_angular_speed_rad_s={float(pre_root_angular_speed[diagnostic_env].item())}, "
                    "root_linear_velocity_w_mps="
                    f"{env._robot.data.root_lin_vel_w[diagnostic_env].cpu().tolist()}, "
                    "root_angular_velocity_w_rad_s="
                    f"{env._robot.data.root_ang_vel_w[diagnostic_env].cpu().tolist()}, "
                    f"air_velocity_body_mps={air_velocity_body.cpu().tolist()}, "
                    f"net_wing_force_body_n={env._debug_last_wing_force_b[diagnostic_env].cpu().tolist()}, "
                    f"net_wing_moment_body_nm={env._debug_last_torque_b[diagnostic_env].cpu().tolist()}, "
                    f"peak_single_wing_force_n={float(pre_force_norm[diagnostic_env, peak_wing].item())}, "
                    f"load_mode={wing_link_load_mode}, diagnostic={diagnostic}"
                )
            system_com_pre, linear_pre, angular_pre = _total_momentum(
                env,
                masses,
                inertias,
            )
            force_w, moment_w = _external_wrench_about_system_com(
                env,
                system_com_pre,
            )
            if step == 0:
                for index in range(len(frequencies)):
                    momentum_linear[index].append(linear_pre[index].detach().cpu().numpy())
                    momentum_angular[index].append(angular_pre[index].detach().cpu().numpy())
            for index in range(len(frequencies)):
                force_traces[index].append(force_w[index].detach().cpu().numpy())
                moment_traces[index].append(moment_w[index].detach().cpu().numpy())

            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            _, linear_post, angular_post = _total_momentum(env, masses, inertias)
            left_position = env._robot.data.joint_pos[:, left_joint_id] - float(env._wing_mid_L)
            right_position = -(
                env._robot.data.joint_pos[:, right_joint_id] - float(env._wing_mid_R)
            )
            reference = env._wing_amp * torch.sin(env._ideal_inverse_phase_state.phase_rad)
            tracking = torch.abs(left_position - reference)
            sync = torch.abs(left_position - right_position)
            wing_force = torch.linalg.vector_norm(
                env._debug_last_wing_force_link_n,
                dim=-1,
            ).amax(dim=1)
            finite_tensors = {
                "root_state_w": env._robot.data.root_state_w,
                "joint_pos": env._robot.data.joint_pos,
                "joint_vel": env._robot.data.joint_vel,
                "wing_force_link_n": env._debug_last_wing_force_link_n,
                "wing_moment_link_nm": env._debug_last_wing_moment_link_about_com_nm,
            }
            invalid_names = [
                name
                for name, value in finite_tensors.items()
                if not bool(torch.all(torch.isfinite(value)))
            ]
            if invalid_names:
                invalid_envs = sorted(
                    {
                        int(index)
                        for value in finite_tensors.values()
                        for index in torch.nonzero(
                            ~torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1),
                            as_tuple=False,
                        ).flatten().tolist()
                    }
                )
                diagnostic_env = invalid_envs[0]
                raise RuntimeError(
                    "Nonfinite native free-body state "
                    f"at step {step}, t={(step + 1) * float(dt_s):.9f}s, "
                    f"invalid={invalid_names}, envs={invalid_envs}, "
                    f"frequency_target_hz={frequencies[diagnostic_env]}, "
                    f"frequency_actual_hz={float(env._freq[diagnostic_env].item())}, "
                    f"phase_rad={float(env._phase[diagnostic_env].item())}, "
                    f"previous_root_state={previous_root_state[diagnostic_env].cpu().tolist()}, "
                    f"joint_pos={env._robot.data.joint_pos[diagnostic_env].cpu().tolist()}, "
                    f"joint_vel={env._robot.data.joint_vel[diagnostic_env].cpu().tolist()}, "
                    "last_wing_force_link_n="
                    f"{env._debug_last_wing_force_link_n[diagnostic_env].cpu().tolist()}, "
                    "last_wing_moment_link_nm="
                    f"{env._debug_last_wing_moment_link_about_com_nm[diagnostic_env].cpu().tolist()}"
                )
            previous_root_state.copy_(env._robot.data.root_state_w)
            for index in range(len(frequencies)):
                momentum_linear[index].append(linear_post[index].detach().cpu().numpy())
                momentum_angular[index].append(angular_post[index].detach().cpu().numpy())
                max_tracking[index] = max(max_tracking[index], float(tracking[index].item()))
                max_sync[index] = max(max_sync[index], float(sync[index].item()))
                max_force[index] = max(max_force[index], float(wing_force[index].item()))
                max_root_linear_speed[index] = max(
                    max_root_linear_speed[index],
                    float(torch.linalg.vector_norm(env._robot.data.root_lin_vel_w[index]).item()),
                )
                max_root_angular_speed[index] = max(
                    max_root_angular_speed[index],
                    float(torch.linalg.vector_norm(env._robot.data.root_ang_vel_w[index]).item()),
                )

        cases: list[dict[str, object]] = []
        for index, frequency_hz in enumerate(frequencies):
            max_tracking_deg = math.degrees(max_tracking[index])
            max_sync_deg = math.degrees(max_sync[index])
            momentum_closure = compute_impulse_momentum_closure(
                linear_momentum_w_kg_m_s=np.asarray(momentum_linear[index]),
                angular_momentum_about_com_w_kg_m2_s=np.asarray(momentum_angular[index]),
                external_force_w_n=np.asarray(force_traces[index]),
                external_moment_about_com_w_nm=np.asarray(moment_traces[index]),
                dt_s=float(dt_s),
            )
            momentum_closure_accepted = (
                momentum_closure["relative_linear_closure_error"]
                < _MAX_RELATIVE_LINEAR_CLOSURE_ERROR
                and momentum_closure["relative_angular_closure_error"]
                < _MAX_RELATIVE_ANGULAR_CLOSURE_ERROR
            )
            cases.append(
                {
                    "frequency_hz": frequency_hz,
                    "dt_s": float(dt_s),
                    "duration_s": total_steps * float(dt_s),
                    "airspeed_mps": float(airspeed_mps),
                    "stable": True,
                    "accepted": (
                        max_tracking_deg < _MAX_MECHANISM_ERROR_DEG
                        and max_sync_deg < _MAX_MECHANISM_ERROR_DEG
                        and momentum_closure_accepted
                    ),
                    "momentum_closure_accepted": momentum_closure_accepted,
                    "max_tracking_error_deg": max_tracking_deg,
                    "max_sync_error_deg": max_sync_deg,
                    "max_single_wing_force_n": float(max_force[index]),
                    "max_root_linear_speed_mps": float(max_root_linear_speed[index]),
                    "max_root_angular_speed_rad_s": float(max_root_angular_speed[index]),
                    "momentum_closure": momentum_closure,
                }
            )
        return {
            "schema_version": "native_holonomic_free_aero_v2",
            "job_type": "free_aero",
            "device": str(device),
            "dt_s": float(dt_s),
            "wing_link_load_mode": str(wing_link_load_mode),
            "solver_position_iterations": int(
                cfg.robot.spawn.articulation_props.solver_position_iteration_count
            ),
            "solver_velocity_iterations": int(
                cfg.robot.spawn.articulation_props.solver_velocity_iteration_count
            ),
            "acceptance_thresholds": {
                "max_mechanism_error_deg": _MAX_MECHANISM_ERROR_DEG,
                "max_relative_linear_closure_error": _MAX_RELATIVE_LINEAR_CLOSURE_ERROR,
                "max_relative_angular_closure_error": _MAX_RELATIVE_ANGULAR_CLOSURE_ERROR,
            },
            "all_cases_accepted": all(bool(case["accepted"]) for case in cases),
            "cases": cases,
        }
    finally:
        import omni.physx

        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()


__all__ = [
    "run_native_holonomic_frequency_job",
    "run_native_holonomic_free_aero_job",
]

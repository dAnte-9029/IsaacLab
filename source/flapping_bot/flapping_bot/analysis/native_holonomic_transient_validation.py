"""Continuous-frequency validation for the native holonomic wing plant."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch


TRANSIENT_PROFILE_NAMES = ("slow_ramp", "fast_ramp", "sinusoidal")
_MAX_MECHANISM_ERROR_DEG = 0.1
_MAX_NUMERIC_ROOT_LINEAR_SPEED_MPS = 100.0
_MAX_NUMERIC_ROOT_ANGULAR_SPEED_RAD_S = 1000.0


@dataclass(frozen=True)
class NativeHolonomicTransientResult:
    """Summary plus aligned numeric traces from one fresh-process job."""

    summary: dict[str, object]
    traces: dict[str, np.ndarray]


def native_transient_frequency_targets_hz(time_s: float) -> tuple[float, float, float]:
    """Return the 2--5 Hz slow-ramp, fast-ramp and sinusoidal targets."""

    time_value = float(time_s)
    if time_value < 0.0:
        raise ValueError("time_s must be nonnegative.")
    slow_ramp = 2.0 + 3.0 * min(time_value / 1.0, 1.0)
    fast_ramp = 2.0 + 3.0 * min(time_value / 0.25, 1.0)
    sinusoidal = 3.5 - 1.5 * math.cos(2.0 * math.pi * time_value)
    return slow_ramp, fast_ramp, sinusoidal


def _stack_trace(trace: dict[str, list[np.ndarray | float]]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in trace.items()
    }


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _to_numpy_copy(value: torch.Tensor) -> np.ndarray:
    """Copy a tensor sample so later Isaac Lab buffer updates cannot mutate it."""

    return value.detach().cpu().numpy().copy()


def run_native_holonomic_transient_job(
    *,
    dt_s: float,
    fixed_root: bool,
    duration_s: float | None = None,
    airspeed_mps: float = 8.0,
    device: str = "cpu",
    progress_callback: Callable[[str], None] | None = None,
) -> NativeHolonomicTransientResult:
    """Run three continuous-frequency profiles without a flight controller.

    The fixed-root job uses an 8 m/s wind, zero gravity and no tail load. The
    free-root job starts at 8 m/s in still air and enables gravity plus neutral
    tail aerodynamics. Both jobs keep the prescribed-acceleration DeLaurier
    wrench as the applied plant path while recording an actual-acceleration
    shadow wrench and a common-coordinate multibody inverse-dynamics estimate.
    """

    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError("dt_s must be positive.")
    if str(device) != "cpu":
        raise ValueError("The native holonomic constraint requires CPU PhysX.")
    resolved_duration_s = float(duration_s if duration_s is not None else (2.0 if fixed_root else 1.0))
    if resolved_duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")

    from flapping_bot.direct.flapping_bot.straight_flight_env import (
        FlappingBotStraightFlightEnv,
    )

    from .native_holonomic_validation import _make_cfg

    cfg = _make_cfg(
        dt_s=dt,
        device=str(device),
        num_envs=len(TRANSIENT_PROFILE_NAMES),
        enable_aerodynamics=True,
        airspeed_mps=float(airspeed_mps),
        solver_position_iterations=16,
        solver_velocity_iterations=4,
        fixed_root=bool(fixed_root),
    )
    cfg.native_holonomic_load_diagnostics = True
    cfg.delaurier_shadow_actual_acceleration = True
    cfg.delaurier_store_strip_diagnostics = False
    cfg.reset_flap_hz = 2.0
    cfg.elevon_trim_deg = 0.0
    cfg.tail_elevator_bias_deg = 0.0
    if fixed_root:
        cfg.sim.gravity = (0.0, 0.0, 0.0)
        cfg.reset_forward_speed_mps = 0.0
        cfg.enable_tail_aero = False
        cfg.wind_enabled = True
        cfg.wind_xy_mps = (-float(airspeed_mps), 0.0)
        disable_gravity = True
    else:
        cfg.sim.gravity = (0.0, 0.0, -9.81)
        cfg.reset_forward_speed_mps = float(airspeed_mps)
        cfg.enable_tail_aero = True
        cfg.wind_enabled = False
        cfg.wind_xy_mps = (0.0, 0.0)
        disable_gravity = False
    cfg.robot = cfg.robot.replace(
        spawn=cfg.robot.spawn.replace(
            rigid_props=cfg.robot.spawn.rigid_props.replace(
                disable_gravity=disable_gravity,
                retain_accelerations=False,
            ),
        ),
    )

    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        initial_root_position_w = env._robot.data.root_pos_w.detach().clone()
        total_steps = int(math.ceil(resolved_duration_s / dt))
        trace: dict[str, list[np.ndarray | float]] = {
            "aero_time_s": [],
            "state_time_s": [],
            "target_frequency_hz": [],
            "mechanism_frequency_hz": [],
            "frequency_rate_hz_s": [],
            "phase_rad": [],
            "q_reference_rad": [],
            "q_actual_rad": [],
            "qdot_reference_rad_s": [],
            "qdot_actual_rad_s": [],
            "qdd_prescribed_rad_s2": [],
            "qdd_physx_rad_s2": [],
            "sync_error_rad": [],
            "applied_wing_force_b_n": [],
            "applied_wing_moment_b_nm": [],
            "shadow_actual_accel_wing_force_b_n": [],
            "shadow_actual_accel_wing_moment_b_nm": [],
            "common_inertia_kg_m2": [],
            "constraint_inertia_torque_estimate_nm": [],
            "constraint_bias_torque_estimate_nm": [],
            "constraint_external_load_torque_estimate_nm": [],
            "constraint_total_torque_estimate_nm": [],
            "constraint_power_estimate_w": [],
            "root_position_w_m": [],
            "root_linear_velocity_w_mps": [],
            "root_angular_velocity_w_rad_s": [],
        }

        for step in range(total_steps):
            if progress_callback is not None and step % max(total_steps // 4, 1) == 0:
                progress_callback(
                    f"native transient fixed_root={fixed_root} dt={dt:.9f}s "
                    f"{100.0 * step / total_steps:.0f}%"
                )
            aero_time_s = step * dt
            targets = torch.tensor(
                native_transient_frequency_targets_hz(aero_time_s),
                dtype=torch.float32,
                device=env.device,
            )
            actions = torch.zeros((len(TRANSIENT_PROFILE_NAMES), 4), device=env.device)
            actions[:, 0] = 2.0 * targets / float(cfg.max_flap_hz) - 1.0
            env._pre_physics_step(actions)
            env._sim_step_counter += 1
            env._apply_action()

            trace["aero_time_s"].append(aero_time_s)
            trace["target_frequency_hz"].append(_to_numpy_copy(targets))
            trace["qdd_prescribed_rad_s2"].append(_to_numpy_copy(env._qdd_cmd))
            trace["applied_wing_force_b_n"].append(_to_numpy_copy(env._debug_last_wing_force_b))
            trace["applied_wing_moment_b_nm"].append(
                _to_numpy_copy(env._debug_last_wing_moment_b_about_base_com_nm)
            )
            trace["shadow_actual_accel_wing_force_b_n"].append(
                _to_numpy_copy(env._debug_last_shadow_actual_accel_wing_force_b_n)
            )
            trace["shadow_actual_accel_wing_moment_b_nm"].append(
                _to_numpy_copy(env._debug_last_shadow_actual_accel_wing_moment_b_nm)
            )
            trace["common_inertia_kg_m2"].append(
                _to_numpy_copy(env._debug_last_native_common_inertia_kg_m2)
            )
            trace["constraint_inertia_torque_estimate_nm"].append(
                _to_numpy_copy(env._debug_last_native_inertia_torque_nm)
            )
            trace["constraint_bias_torque_estimate_nm"].append(
                _to_numpy_copy(env._debug_last_native_bias_torque_nm)
            )
            trace["constraint_external_load_torque_estimate_nm"].append(
                _to_numpy_copy(env._debug_last_native_external_load_torque_nm)
            )
            trace["constraint_total_torque_estimate_nm"].append(
                _to_numpy_copy(env._debug_last_native_constraint_torque_estimate_nm)
            )
            trace["constraint_power_estimate_w"].append(
                _to_numpy_copy(env._debug_last_native_constraint_power_estimate_w)
            )

            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

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
            mechanism_frequency = env._ideal_inverse_phase_state.frequency_hz
            q_reference = env._wing_amp * torch.sin(phase)
            qdot_reference = (
                env._wing_amp
                * 2.0
                * math.pi
                * mechanism_frequency
                * torch.cos(phase)
            )
            finite_tensors = (
                env._robot.data.root_state_w,
                env._robot.data.joint_pos,
                env._robot.data.joint_vel,
                env._debug_last_wing_force_link_n,
                env._debug_last_shadow_actual_accel_wing_force_b_n,
                env._debug_last_native_constraint_torque_estimate_nm,
                env._debug_last_native_constraint_power_estimate_w,
            )
            if any(not bool(torch.all(torch.isfinite(value))) for value in finite_tensors):
                raise RuntimeError(f"Nonfinite native transient state at step {step}.")

            trace["state_time_s"].append((step + 1) * dt)
            trace["mechanism_frequency_hz"].append(_to_numpy_copy(mechanism_frequency))
            trace["frequency_rate_hz_s"].append(
                _to_numpy_copy(env._phase_acceleration_rad_s2 / (2.0 * math.pi))
            )
            trace["phase_rad"].append(_to_numpy_copy(phase))
            trace["q_reference_rad"].append(_to_numpy_copy(q_reference))
            trace["q_actual_rad"].append(_to_numpy_copy(q_actual))
            trace["qdot_reference_rad_s"].append(_to_numpy_copy(qdot_reference))
            trace["qdot_actual_rad_s"].append(_to_numpy_copy(qdot_actual))
            trace["qdd_physx_rad_s2"].append(_to_numpy_copy(qdd_physx))
            trace["sync_error_rad"].append(_to_numpy_copy(torch.abs(left_position - right_position)))
            trace["root_position_w_m"].append(_to_numpy_copy(env._robot.data.root_pos_w))
            trace["root_linear_velocity_w_mps"].append(
                _to_numpy_copy(env._robot.data.root_lin_vel_w)
            )
            trace["root_angular_velocity_w_rad_s"].append(
                _to_numpy_copy(env._robot.data.root_ang_vel_w)
            )

        arrays = _stack_trace(trace)
        force_delta = (
            arrays["shadow_actual_accel_wing_force_b_n"]
            - arrays["applied_wing_force_b_n"]
        )
        moment_delta = (
            arrays["shadow_actual_accel_wing_moment_b_nm"]
            - arrays["applied_wing_moment_b_nm"]
        )
        root_displacement = arrays["root_position_w_m"] - _to_numpy_copy(initial_root_position_w)[None, :, :]
        cases: list[dict[str, object]] = []
        for index, profile_name in enumerate(TRANSIENT_PROFILE_NAMES):
            tracking_error = arrays["q_actual_rad"][:, index] - arrays["q_reference_rad"][:, index]
            sync_error = arrays["sync_error_rad"][:, index]
            frequency_error = (
                arrays["mechanism_frequency_hz"][:, index]
                - arrays["target_frequency_hz"][:, index]
            )
            force_delta_norm = np.linalg.norm(force_delta[:, index, :], axis=1)
            moment_delta_norm = np.linalg.norm(moment_delta[:, index, :], axis=1)
            root_linear_speed = np.linalg.norm(arrays["root_linear_velocity_w_mps"][:, index, :], axis=1)
            root_angular_speed = np.linalg.norm(arrays["root_angular_velocity_w_rad_s"][:, index, :], axis=1)
            max_tracking_error_deg = math.degrees(float(np.max(np.abs(tracking_error))))
            max_sync_error_deg = math.degrees(float(np.max(np.abs(sync_error))))
            numerically_bounded = bool(
                np.max(root_linear_speed) < _MAX_NUMERIC_ROOT_LINEAR_SPEED_MPS
                and np.max(root_angular_speed) < _MAX_NUMERIC_ROOT_ANGULAR_SPEED_RAD_S
            )
            cases.append(
                {
                    "profile": profile_name,
                    "accepted": bool(
                        max_tracking_error_deg < _MAX_MECHANISM_ERROR_DEG
                        and max_sync_error_deg < _MAX_MECHANISM_ERROR_DEG
                        and numerically_bounded
                    ),
                    "numerically_bounded": numerically_bounded,
                    "max_tracking_error_deg": max_tracking_error_deg,
                    "rms_tracking_error_deg": math.degrees(_rms(tracking_error)),
                    "max_sync_error_deg": max_sync_error_deg,
                    "rms_target_frequency_error_hz": _rms(frequency_error),
                    "max_abs_target_frequency_error_hz": float(np.max(np.abs(frequency_error))),
                    "rms_shadow_force_delta_n": _rms(force_delta_norm),
                    "peak_shadow_force_delta_n": float(np.max(force_delta_norm)),
                    "rms_shadow_moment_delta_nm": _rms(moment_delta_norm),
                    "peak_shadow_moment_delta_nm": float(np.max(moment_delta_norm)),
                    "rms_constraint_torque_estimate_nm": _rms(
                        arrays["constraint_total_torque_estimate_nm"][:, index]
                    ),
                    "peak_abs_constraint_torque_estimate_nm": float(
                        np.max(np.abs(arrays["constraint_total_torque_estimate_nm"][:, index]))
                    ),
                    "rms_constraint_power_estimate_w": _rms(
                        arrays["constraint_power_estimate_w"][:, index]
                    ),
                    "peak_abs_constraint_power_estimate_w": float(
                        np.max(np.abs(arrays["constraint_power_estimate_w"][:, index]))
                    ),
                    "max_root_displacement_m": float(
                        np.max(np.linalg.norm(root_displacement[:, index, :], axis=1))
                    ),
                    "max_root_linear_speed_mps": float(np.max(root_linear_speed)),
                    "max_root_angular_speed_rad_s": float(np.max(root_angular_speed)),
                }
            )

        summary: dict[str, object] = {
            "schema_version": "native_holonomic_transient_v1",
            "job_type": "continuous_frequency_transient",
            "fixed_root": bool(fixed_root),
            "dt_s": dt,
            "duration_s": total_steps * dt,
            "device": str(device),
            "airspeed_mps": float(airspeed_mps),
            "profile_names": list(TRANSIENT_PROFILE_NAMES),
            "applied_aerodynamic_acceleration_source": "prescribed_acceleration",
            "shadow_aerodynamic_acceleration_source": "actual_joint_acceleration",
            "constraint_load_quantity": "multibody_inverse_dynamics_estimate",
            "constraint_load_is_physx_multiplier": False,
            "constraint_load_is_motor_shaft_quantity": False,
            "flight_controller_enabled": False,
            "stability_claim": "finite_and_numerically_bounded_only",
            "solver_position_iterations": 16,
            "solver_velocity_iterations": 4,
            "acceptance_thresholds": {
                "max_mechanism_error_deg": _MAX_MECHANISM_ERROR_DEG,
                "max_numeric_root_linear_speed_mps": _MAX_NUMERIC_ROOT_LINEAR_SPEED_MPS,
                "max_numeric_root_angular_speed_rad_s": _MAX_NUMERIC_ROOT_ANGULAR_SPEED_RAD_S,
            },
            "all_cases_accepted": all(bool(case["accepted"]) for case in cases),
            "cases": cases,
        }
        return NativeHolonomicTransientResult(summary=summary, traces=arrays)
    finally:
        import omni.physx

        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()


__all__ = [
    "NativeHolonomicTransientResult",
    "TRANSIENT_PROFILE_NAMES",
    "native_transient_frequency_targets_hz",
    "run_native_holonomic_transient_job",
]

"""Headless PhysX benchmark for measured-wing inertial coupling."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context
from isaaclab.utils.math import matrix_from_quat

from flapping_bot.assets import IdealCoupledFlappingBotCfg, apply_hard_opposed_wing_mimic
from flapping_bot.physics import (
    MEASURED_WING_NEUTRAL_DIHEDRAL_RAD,
    build_measured_wing_multibody_tensors,
    compute_amini_sfwm_inertial_response,
    measured_flapping_bot_amini_params,
)

from .multibody_inertial_validation import (
    compute_total_momentum_about_system_com_w,
    fit_fundamental_harmonic,
    wrapped_phase_difference_deg,
)

_STROKE_AMPLITUDE_RAD = math.radians(30.0)


def _harmonic_dict(time_s: np.ndarray, values: np.ndarray, frequency_hz: float) -> dict[str, float]:
    return fit_fundamental_harmonic(time_s, values, frequency_hz=frequency_hz).to_dict()


def _run_case(
    *,
    robot: Articulation,
    sim,
    dt_s: float,
    frequency_hz: float,
    total_cycles: int,
    measurement_cycles: int,
    masses_kg: torch.Tensor,
    inertias_kg_m2: torch.Tensor,
    base_body_id: int,
    left_joint_id: int,
    right_joint_id: int,
) -> dict[str, object]:
    params = measured_flapping_bot_amini_params()
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_state[:, 7:]))
    robot.write_joint_state_to_sim(
        torch.zeros_like(robot.data.default_joint_pos),
        torch.zeros_like(robot.data.default_joint_vel),
    )
    robot.reset()
    sim.forward()
    robot.update(dt_s)

    duration_s = total_cycles / frequency_hz
    total_steps = int(math.ceil(duration_s / dt_s))
    measurement_start_s = (total_cycles - measurement_cycles) / frequency_hz
    base_com_b = robot.data.body_com_pos_b[0, base_body_id].clone()
    point_o_b = torch.tensor(params.point_o_in_base_flu_m, device=robot.device, dtype=base_com_b.dtype)
    point_o_from_base_com_b = point_o_b - base_com_b

    traces: dict[str, list[float]] = {
        name: []
        for name in (
            "time_s",
            "q_ref_rad",
            "qd_ref_rad_s",
            "qdd_ref_rad_s2",
            "q_actual_rad",
            "qd_actual_rad_s",
            "qdd_actual_rad_s2",
            "sync_error_rad",
            "point_o_vertical_acceleration_frd_m_s2",
            "base_pitch_angular_acceleration_frd_rad_s2",
            "amini_vertical_acceleration_frd_m_s2",
            "amini_pitch_angular_acceleration_frd_rad_s2",
            "driver_torque_nm",
            "linear_momentum_norm_kg_m_s",
            "angular_momentum_norm_kg_m2_s",
            "linear_internal_scale_kg_m_s",
            "angular_internal_scale_kg_m2_s",
        )
    }
    inertia_matrices = inertias_kg_m2.reshape(-1, 3, 3).to(device=robot.device)
    masses = masses_kg.to(device=robot.device)

    for step in range(total_steps):
        target_time_s = (step + 1) * dt_s
        omega = 2.0 * math.pi * frequency_hz
        phase = omega * target_time_s
        q_ref = _STROKE_AMPLITUDE_RAD * math.sin(phase)
        qd_ref = _STROKE_AMPLITUDE_RAD * omega * math.cos(phase)
        qdd_ref = -_STROKE_AMPLITUDE_RAD * omega**2 * math.sin(phase)
        robot.set_joint_position_target(
            torch.tensor([[q_ref]], device=robot.device),
            joint_ids=[left_joint_id],
        )
        robot.set_joint_velocity_target(
            torch.tensor([[qd_ref]], device=robot.device),
            joint_ids=[left_joint_id],
        )
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt_s)

        if target_time_s < measurement_start_s:
            # Access every step so Isaac Lab's finite-difference joint
            # acceleration cache retains the correct previous velocity.
            _ = robot.data.joint_acc
            continue

        q_left = robot.data.joint_pos[0, left_joint_id]
        q_right = robot.data.joint_pos[0, right_joint_id]
        qd_left = robot.data.joint_vel[0, left_joint_id]
        qd_right = robot.data.joint_vel[0, right_joint_id]
        qdd_left = robot.data.joint_acc[0, left_joint_id]
        qdd_right = robot.data.joint_acc[0, right_joint_id]
        q_actual = 0.5 * (q_left - q_right)
        qd_actual = 0.5 * (qd_left - qd_right)
        qdd_actual = 0.5 * (qdd_left - qdd_right)

        base_link_rotation_w = matrix_from_quat(robot.data.body_link_quat_w[0, base_body_id])
        point_o_from_base_com_w = base_link_rotation_w @ point_o_from_base_com_b
        base_omega_w = robot.data.body_com_ang_vel_w[0, base_body_id]
        base_alpha_w = robot.data.body_com_ang_acc_w[0, base_body_id]
        point_o_acceleration_w = (
            robot.data.body_com_lin_acc_w[0, base_body_id]
            + torch.cross(base_alpha_w, point_o_from_base_com_w, dim=-1)
            + torch.cross(
                base_omega_w,
                torch.cross(base_omega_w, point_o_from_base_com_w, dim=-1),
                dim=-1,
            )
        )
        point_o_acceleration_flu = base_link_rotation_w.transpose(-1, -2) @ point_o_acceleration_w
        base_alpha_flu = base_link_rotation_w.transpose(-1, -2) @ base_alpha_w

        amini = compute_amini_sfwm_inertial_response(
            gamma_rad=MEASURED_WING_NEUTRAL_DIHEDRAL_RAD + q_actual,
            gamma_dot_rad_s=qd_actual,
            gamma_ddot_rad_s2=qdd_actual,
            params=params,
        )
        _, linear_momentum_w, angular_momentum_w = compute_total_momentum_about_system_com_w(
            masses_kg=masses,
            inertia_about_com_principal_kg_m2=inertia_matrices,
            com_positions_w_m=robot.data.body_com_pos_w[0],
            com_quaternions_wxyz=robot.data.body_com_quat_w[0],
            com_linear_velocities_w_m_s=robot.data.body_com_lin_vel_w[0],
            angular_velocities_w_rad_s=robot.data.body_com_ang_vel_w[0],
        )
        linear_internal_scale = torch.sum(
            masses * torch.linalg.vector_norm(robot.data.body_com_lin_vel_w[0], dim=-1)
        )
        radius_from_system_com = robot.data.body_com_pos_w[0] - torch.sum(
            masses[:, None] * robot.data.body_com_pos_w[0],
            dim=0,
        ) / masses.sum()
        angular_internal_scale = torch.sum(
            masses
            * torch.linalg.vector_norm(radius_from_system_com, dim=-1)
            * torch.linalg.vector_norm(robot.data.body_com_lin_vel_w[0], dim=-1)
        ) + torch.sum(
            torch.linalg.matrix_norm(inertia_matrices, dim=(-2, -1))
            * torch.linalg.vector_norm(robot.data.body_com_ang_vel_w[0], dim=-1)
        )

        values = {
            "time_s": target_time_s,
            "q_ref_rad": q_ref,
            "qd_ref_rad_s": qd_ref,
            "qdd_ref_rad_s2": qdd_ref,
            "q_actual_rad": float(q_actual.item()),
            "qd_actual_rad_s": float(qd_actual.item()),
            "qdd_actual_rad_s2": float(qdd_actual.item()),
            "sync_error_rad": float((q_left + q_right).item()),
            "point_o_vertical_acceleration_frd_m_s2": float((-point_o_acceleration_flu[2]).item()),
            "base_pitch_angular_acceleration_frd_rad_s2": float((-base_alpha_flu[1]).item()),
            "amini_vertical_acceleration_frd_m_s2": float(
                amini.vertical_acceleration_frd_m_s2.item()
            ),
            "amini_pitch_angular_acceleration_frd_rad_s2": float(
                amini.pitch_angular_acceleration_frd_rad_s2.item()
            ),
            "driver_torque_nm": float(robot.data.applied_torque[0, left_joint_id].item()),
            "linear_momentum_norm_kg_m_s": float(torch.linalg.vector_norm(linear_momentum_w).item()),
            "angular_momentum_norm_kg_m2_s": float(torch.linalg.vector_norm(angular_momentum_w).item()),
            "linear_internal_scale_kg_m_s": float(linear_internal_scale.item()),
            "angular_internal_scale_kg_m2_s": float(angular_internal_scale.item()),
        }
        for name, value in values.items():
            traces[name].append(value)

    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in traces.items()}
    time_s = arrays["time_s"]
    harmonic_names = (
        "q_ref_rad",
        "qd_ref_rad_s",
        "qdd_ref_rad_s2",
        "q_actual_rad",
        "qd_actual_rad_s",
        "qdd_actual_rad_s2",
        "point_o_vertical_acceleration_frd_m_s2",
        "base_pitch_angular_acceleration_frd_rad_s2",
        "amini_vertical_acceleration_frd_m_s2",
        "amini_pitch_angular_acceleration_frd_rad_s2",
        "driver_torque_nm",
    )
    harmonics = {
        name: _harmonic_dict(time_s, arrays[name], frequency_hz)
        for name in harmonic_names
    }
    physx_vertical = harmonics["point_o_vertical_acceleration_frd_m_s2"]
    amini_vertical = harmonics["amini_vertical_acceleration_frd_m_s2"]
    physx_pitch = harmonics["base_pitch_angular_acceleration_frd_rad_s2"]
    amini_pitch = harmonics["amini_pitch_angular_acceleration_frd_rad_s2"]
    linear_scale = max(float(np.max(arrays["linear_internal_scale_kg_m_s"])), 1.0e-12)
    angular_scale = max(float(np.max(arrays["angular_internal_scale_kg_m2_s"])), 1.0e-12)
    return {
        "frequency_hz": frequency_hz,
        "dt_s": dt_s,
        "total_cycles": total_cycles,
        "measurement_cycles": measurement_cycles,
        "sample_count": int(time_s.size),
        "max_sync_error_deg": math.degrees(float(np.max(np.abs(arrays["sync_error_rad"])))),
        "max_tracking_error_deg": math.degrees(
            float(np.max(np.abs(arrays["q_actual_rad"] - arrays["q_ref_rad"])))
        ),
        "max_velocity_tracking_error_rad_s": float(
            np.max(np.abs(arrays["qd_actual_rad_s"] - arrays["qd_ref_rad_s"]))
        ),
        "max_acceleration_tracking_error_rad_s2": float(
            np.max(np.abs(arrays["qdd_actual_rad_s2"] - arrays["qdd_ref_rad_s2"]))
        ),
        "max_driver_torque_nm": float(np.max(np.abs(arrays["driver_torque_nm"]))),
        "max_linear_momentum_norm_kg_m_s": float(
            np.max(arrays["linear_momentum_norm_kg_m_s"])
        ),
        "max_angular_momentum_norm_kg_m2_s": float(
            np.max(arrays["angular_momentum_norm_kg_m2_s"])
        ),
        "relative_linear_momentum_residual": float(
            np.max(arrays["linear_momentum_norm_kg_m_s"]) / linear_scale
        ),
        "relative_angular_momentum_residual": float(
            np.max(arrays["angular_momentum_norm_kg_m2_s"]) / angular_scale
        ),
        "harmonics": harmonics,
        "physx_to_amini_vertical_amplitude_ratio": (
            physx_vertical["amplitude"] / max(amini_vertical["amplitude"], 1.0e-12)
        ),
        "physx_minus_amini_vertical_phase_deg": wrapped_phase_difference_deg(
            physx_vertical["phase_rad"],
            amini_vertical["phase_rad"],
        ),
        "physx_to_amini_pitch_amplitude_ratio": (
            physx_pitch["amplitude"] / max(amini_pitch["amplitude"], 1.0e-12)
        ),
        "physx_minus_amini_pitch_phase_deg": wrapped_phase_difference_deg(
            physx_pitch["phase_rad"],
            amini_pitch["phase_rad"],
        ),
    }


def _frequency_scaling(cases: list[dict[str, object]]) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for dt_s in sorted({float(case["dt_s"]) for case in cases}):
        group = sorted(
            (case for case in cases if float(case["dt_s"]) == dt_s),
            key=lambda case: float(case["frequency_hz"]),
        )
        if len(group) < 2:
            continue
        frequencies = np.asarray([float(case["frequency_hz"]) for case in group])
        for signal_name in (
            "point_o_vertical_acceleration_frd_m_s2",
            "base_pitch_angular_acceleration_frd_rad_s2",
            "amini_vertical_acceleration_frd_m_s2",
            "amini_pitch_angular_acceleration_frd_rad_s2",
        ):
            amplitudes = np.asarray(
                [float(case["harmonics"][signal_name]["amplitude"]) for case in group]
            )
            slope, intercept = np.polyfit(np.log(frequencies), np.log(amplitudes), 1)
            predicted = intercept + slope * np.log(frequencies)
            residual = np.log(amplitudes) - predicted
            total = np.log(amplitudes) - np.mean(np.log(amplitudes))
            r_squared = 1.0 - float(np.sum(residual**2) / max(np.sum(total**2), 1.0e-15))
            normalized = amplitudes / frequencies**2
            results.append(
                {
                    "dt_s": dt_s,
                    "signal": signal_name,
                    "log_log_frequency_exponent": float(slope),
                    "log_log_r_squared": r_squared,
                    "amplitude_over_f_squared_cv": float(
                        np.std(normalized) / max(np.mean(normalized), 1.0e-15)
                    ),
                }
            )
    return results


def _time_step_convergence(cases: list[dict[str, object]]) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    reference_dt = min(float(case["dt_s"]) for case in cases)
    for frequency_hz in sorted({float(case["frequency_hz"]) for case in cases}):
        group = [case for case in cases if float(case["frequency_hz"]) == frequency_hz]
        reference = next(case for case in group if float(case["dt_s"]) == reference_dt)
        for case in group:
            for signal_name in (
                "q_actual_rad",
                "point_o_vertical_acceleration_frd_m_s2",
                "base_pitch_angular_acceleration_frd_rad_s2",
            ):
                harmonic = case["harmonics"][signal_name]
                reference_harmonic = reference["harmonics"][signal_name]
                results.append(
                    {
                        "frequency_hz": frequency_hz,
                        "dt_s": float(case["dt_s"]),
                        "reference_dt_s": reference_dt,
                        "signal": signal_name,
                        "relative_amplitude_error": abs(
                            float(harmonic["amplitude"]) - float(reference_harmonic["amplitude"])
                        )
                        / max(float(reference_harmonic["amplitude"]), 1.0e-12),
                        "phase_error_deg": abs(
                            wrapped_phase_difference_deg(
                                float(harmonic["phase_rad"]),
                                float(reference_harmonic["phase_rad"]),
                            )
                        ),
                    }
                )
    return results


def run_physx_multibody_inertial_benchmark(
    *,
    frequencies_hz: Sequence[float] = (2.0, 3.0, 4.0, 5.0),
    time_steps_s: Sequence[float] = (1.0 / 240.0, 1.0 / 480.0, 1.0e-3),
    total_cycles: int = 8,
    measurement_cycles: int = 4,
) -> dict[str, object]:
    """Run the no-gravity/no-aerodynamics frequency and time-step matrix."""

    if total_cycles <= measurement_cycles or measurement_cycles < 2:
        raise ValueError("Use total_cycles > measurement_cycles >= 2.")
    cases: list[dict[str, object]] = []
    for dt_index, dt_s in enumerate(time_steps_s):
        with build_simulation_context(
            device="cpu",
            dt=float(dt_s),
            auto_add_lighting=False,
            gravity_enabled=False,
        ) as sim:
            sim._app_control_on_stop_handle = None
            prim_path = f"/World/PhysxMultibodyInertialBenchmark{dt_index}"
            robot = Articulation(
                IdealCoupledFlappingBotCfg.replace(
                    prim_path=prim_path,
                    spawn=IdealCoupledFlappingBotCfg.spawn.replace(
                        rigid_props=IdealCoupledFlappingBotCfg.spawn.rigid_props.replace(
                            disable_gravity=True,
                            retain_accelerations=True,
                        ),
                    ),
                )
            )
            apply_hard_opposed_wing_mimic(
                robot.stage,
                articulation_root_path=prim_path,
            )
            sim.reset()
            robot.update(float(dt_s))
            env_ids = torch.arange(robot.num_instances, device="cpu", dtype=torch.int64)
            masses, inertias, coms = build_measured_wing_multibody_tensors(
                body_names=robot.body_names,
                masses_kg=robot.root_physx_view.get_masses().clone(),
                inertias_kg_m2=robot.root_physx_view.get_inertias().clone(),
                com_poses_link=robot.root_physx_view.get_coms().clone(),
            )
            robot.root_physx_view.set_masses(masses, env_ids)
            robot.root_physx_view.set_inertias(inertias, env_ids)
            robot.root_physx_view.set_coms(coms, env_ids)
            sim.forward()
            left_joint_id, right_joint_id = (
                int(index)
                for index in robot.find_joints(["left_wing", "right_wing"], preserve_order=True)[0]
            )
            base_body_id = int(robot.find_bodies(["base_link"], preserve_order=True)[0][0])
            for frequency_hz in frequencies_hz:
                cases.append(
                    _run_case(
                        robot=robot,
                        sim=sim,
                        dt_s=float(dt_s),
                        frequency_hz=float(frequency_hz),
                        total_cycles=total_cycles,
                        measurement_cycles=measurement_cycles,
                        masses_kg=masses[0],
                        inertias_kg_m2=inertias[0],
                        base_body_id=base_body_id,
                        left_joint_id=left_joint_id,
                        right_joint_id=right_joint_id,
                    )
                )
    return {
        "schema_version": "physx_multibody_inertial_benchmark_v1",
        "stroke_amplitude_deg": math.degrees(_STROKE_AMPLITUDE_RAD),
        "frequencies_hz": [float(value) for value in frequencies_hz],
        "time_steps_s": [float(value) for value in time_steps_s],
        "total_cycles": total_cycles,
        "measurement_cycles": measurement_cycles,
        "amini_parameters": {
            **measured_flapping_bot_amini_params().__dict__,
            "constant_pitch_inertia_kg_m2": measured_flapping_bot_amini_params().constant_pitch_inertia_kg_m2,
            "wing_to_weight_ratio": measured_flapping_bot_amini_params().wing_to_weight_ratio,
        },
        "cases": cases,
        "frequency_scaling": _frequency_scaling(cases),
        "time_step_convergence": _time_step_convergence(cases),
    }

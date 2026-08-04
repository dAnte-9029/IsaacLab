"""Headless PhysX validation for measured-wing aerodynamic coupling."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg,
    FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledDeLaurierEnvCfg,
    FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg,
    FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg,
    FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledDeLaurierEnvCfg,
)
from flapping_bot.physics import (
    ACTUAL_JOINT_ACCELERATION,
    ACTUAL_PER_WING_LINK,
    COMMANDED_BASE_EQUIVALENT,
    FULL_WING_LINK_WRENCH,
    IDEAL_TORQUE_PER_WING_LINK,
    IDEAL_INVERSE_DYNAMICS_PER_WING_LINK,
    PRESCRIBED_PER_WING_LINK,
    SINUSOIDAL_PHASE_PER_WING_LINK,
    map_opposed_joint_states_to_physical_wing_kinematics,
)

from .multibody_aero_validation import (
    compare_fixed_mode_cases,
    compute_impulse_momentum_closure,
    compute_symmetric_wrench_residuals,
    summarize_time_step_convergence,
)
from .multibody_inertial_validation import fit_fundamental_harmonic, wrapped_phase_difference_deg

_CONTROL_PERIOD_S = 1.0 / 120.0
_FREE_SIGNAL_NAMES = (
    "q_actual_rad",
    "qdot_actual_rad_s",
    "qdd_actual_rad_s2",
    "wing_force_b_x_n",
    "wing_force_b_z_n",
    "wing_moment_b_y_nm",
    "base_acceleration_b_z_m_s2",
    "base_pitch_acceleration_b_y_rad_s2",
)
_FIXED_SIGNAL_NAMES = (
    "q_actual_rad",
    "qdot_actual_rad_s",
    "qdd_physx_rad_s2",
    "qdd_aero_input_rad_s2",
    "aero_hinge_torque_left_nm",
    "aero_hinge_torque_right_nm",
    "aero_hinge_power_w",
    "driver_torque_left_nm",
    "driver_power_w",
    "actual_frequency_hz",
    "wing_force_b_x_n",
    "wing_force_b_y_n",
    "wing_force_b_z_n",
    "wing_moment_b_x_nm",
    "wing_moment_b_y_nm",
    "wing_moment_b_z_nm",
)


class _NumericalInstabilityError(RuntimeError):
    """Internal fail-fast signal carrying the first invalid physics step."""

    def __init__(
        self,
        *,
        label: str,
        step: int,
        invalid: Sequence[str],
        diagnostics: dict[str, float],
    ):
        self.label = label
        self.step = int(step)
        self.invalid = tuple(invalid)
        self.diagnostics = dict(diagnostics)
        super().__init__(
            f"{label} became nonfinite at physics step {step}: {list(invalid)}; "
            f"finite max-abs diagnostics={diagnostics}"
        )


def _harmonic_dict(
    time_s: np.ndarray,
    values: np.ndarray,
    frequency_hz: float,
) -> dict[str, float]:
    return fit_fundamental_harmonic(
        time_s,
        values,
        frequency_hz=frequency_hz,
    ).to_dict()


def _frequency_action(
    frequencies_hz: Sequence[float],
    *,
    min_frequency_hz: float,
    max_frequency_hz: float,
    device: str,
) -> torch.Tensor:
    frequencies = torch.as_tensor(frequencies_hz, dtype=torch.float32, device=device)
    normalized = (
        2.0
        * (frequencies - float(min_frequency_hz))
        / (float(max_frequency_hz) - float(min_frequency_hz))
        - 1.0
    )
    actions = torch.zeros((len(frequencies_hz), 4), dtype=torch.float32, device=device)
    actions[:, 0] = normalized
    return actions


def _make_cfg(
    *,
    dt_s: float,
    num_envs: int,
    fixed_root: bool,
    coupling_mode: str,
    acceleration_source: str = ACTUAL_JOINT_ACCELERATION,
    wing_link_load_mode: str = FULL_WING_LINK_WRENCH,
    reset_frequency_hz: float | None = None,
) -> (
    FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg
    | FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledDeLaurierEnvCfg
    | FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg
    | FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg
    | FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledDeLaurierEnvCfg
):
    decimation = int(round(_CONTROL_PERIOD_S / float(dt_s)))
    if decimation < 1 or not math.isclose(
        decimation * float(dt_s),
        _CONTROL_PERIOD_S,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("dt_s must divide the fixed 1/120 s control period.")
    if coupling_mode == PRESCRIBED_PER_WING_LINK:
        cfg = FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg()
    elif coupling_mode == IDEAL_TORQUE_PER_WING_LINK:
        cfg = FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledDeLaurierEnvCfg()
    elif coupling_mode == SINUSOIDAL_PHASE_PER_WING_LINK:
        cfg = (
            FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg()
        )
    elif coupling_mode == IDEAL_INVERSE_DYNAMICS_PER_WING_LINK:
        cfg = (
            FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledDeLaurierEnvCfg()
        )
    else:
        cfg = FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.scene.env_spacing = 5.0
    cfg.sim.device = "cpu"
    cfg.sim.dt = float(dt_s)
    cfg.sim.render_interval = decimation
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.decimation = decimation
    cfg.episode_length_s = 20.0
    cfg.enable_tail_aero = False
    cfg.fuselage_drag_cda = 0.0
    cfg.freeze_steps_after_reset = 0
    cfg.reset_pitch_deg = 0.0
    cfg.reset_forward_speed_mps = 0.0 if fixed_root else 8.0
    if reset_frequency_hz is not None:
        cfg.reset_flap_hz = float(reset_frequency_hz)
    cfg.reset_rudder_deg = 0.0
    cfg.reset_elevon_pitch_deg = 0.0
    cfg.reset_elevon_roll_deg = 0.0
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    cfg.randomize_commands = False
    cfg.wind_enabled = False
    cfg.wind_ou_enabled = False
    cfg.delaurier_store_strip_diagnostics = False
    cfg.wing_aero_coupling_mode = coupling_mode
    cfg.wing_aero_acceleration_source = acceleration_source
    cfg.wing_link_aero_load_mode = wing_link_load_mode
    cfg.robot = cfg.robot.replace(
        spawn=cfg.robot.spawn.replace(
            rigid_props=cfg.robot.spawn.rigid_props.replace(
                disable_gravity=True,
                retain_accelerations=False,
            ),
            articulation_props=cfg.robot.spawn.articulation_props.replace(
                fix_root_link=bool(fixed_root),
            ),
        ),
    )
    return cfg


def _mass_inertia_tensors(
    env: FlappingBotStraightFlightEnv,
) -> tuple[torch.Tensor, torch.Tensor]:
    masses = env._robot.root_physx_view.get_masses()[0].clone()
    inertias = env._robot.root_physx_view.get_inertias()[0].clone().reshape(-1, 3, 3)
    return masses, inertias


def _total_momentum_batch(
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
    inertia_w = (
        rotations
        @ inertias_kg_m2.unsqueeze(0)
        @ rotations.transpose(-1, -2)
    )
    spin = torch.einsum("nbij,nbj->nbi", inertia_w, angular_velocities)
    orbital = torch.linalg.cross(
        positions - system_com.unsqueeze(1),
        mass_weights * linear_velocities,
        dim=-1,
    )
    angular_momentum = torch.sum(spin + orbital, dim=1)
    return system_com, linear_momentum, angular_momentum


def _external_wrench_about_system_com_w(
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


def _base_accelerations_b(
    env: FlappingBotStraightFlightEnv,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_body_id = int(env._base_body_ids[0])
    base_quat_w = env._robot.data.body_link_quat_w[:, base_body_id]
    linear_b = quat_apply_inverse(
        base_quat_w,
        env._robot.data.body_com_lin_acc_w[:, base_body_id],
    )
    angular_b = quat_apply_inverse(
        base_quat_w,
        env._robot.data.body_com_ang_acc_w[:, base_body_id],
    )
    return linear_b, angular_b


def _computed_aero_hinge_torque_and_power(
    env: FlappingBotStraightFlightEnv,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return computed wing hinge torque and joint power in URDF signs.

    The torque has shape ``(N, 2)`` in N m, ordered left then right. Total
    power has shape ``(N,)`` in W. Positive power means the computed
    aerodynamic wrench injects mechanical energy into the two wing joints.
    This remains a computation diagnostic when the wrench is applied at the
    base rather than at the wing links.
    """

    wing_body_ids = [int(index) for index in env._wing_body_ids]
    wing_joint_ids = [
        int(env._joint_ids[env._IDX_LEFT_WING]),
        int(env._joint_ids[env._IDX_RIGHT_WING]),
    ]
    wing_quat_w = env._robot.data.body_link_quat_w[:, wing_body_ids]
    force_w = quat_apply(
        wing_quat_w.reshape(-1, 4),
        env._debug_last_wing_force_link_n.reshape(-1, 3),
    ).reshape_as(env._debug_last_wing_force_link_n)
    moment_about_com_w = quat_apply(
        wing_quat_w.reshape(-1, 4),
        env._debug_last_wing_moment_link_about_com_nm.reshape(-1, 3),
    ).reshape_as(env._debug_last_wing_moment_link_about_com_nm)
    moment_about_hinge_w = moment_about_com_w + torch.linalg.cross(
        env._robot.data.body_com_pos_w[:, wing_body_ids]
        - env._robot.data.body_link_pos_w[:, wing_body_ids],
        force_w,
        dim=-1,
    )
    hinge_axis_link = torch.zeros_like(force_w)
    hinge_axis_link[..., 0] = 1.0
    hinge_axis_w = quat_apply(
        wing_quat_w.reshape(-1, 4),
        hinge_axis_link.reshape(-1, 3),
    ).reshape_as(hinge_axis_link)
    hinge_torque_nm = torch.sum(moment_about_hinge_w * hinge_axis_w, dim=-1)
    hinge_power_w = torch.sum(
        hinge_torque_nm * env._robot.data.joint_vel[:, wing_joint_ids],
        dim=1,
    )
    return hinge_torque_nm, hinge_power_w


def _empty_trace(signal_names: Sequence[str]) -> dict[str, list[float]]:
    return {"time_s": [], **{name: [] for name in signal_names}}


def _require_finite(
    *,
    label: str,
    step: int,
    tensors: dict[str, torch.Tensor],
) -> None:
    invalid = [
        name for name, value in tensors.items() if not bool(torch.all(torch.isfinite(value)))
    ]
    if invalid:
        diagnostics = {
            name: float(torch.nan_to_num(torch.abs(value), nan=0.0).max().item())
            for name, value in tensors.items()
        }
        raise _NumericalInstabilityError(
            label=label,
            step=step,
            invalid=invalid,
            diagnostics=diagnostics,
        )


def _require_bounded(
    *,
    label: str,
    step: int,
    tensors_and_limits: dict[str, tuple[torch.Tensor, float]],
) -> None:
    diagnostics = {
        name: float(torch.max(torch.abs(value)).item())
        for name, (value, _) in tensors_and_limits.items()
    }
    exceeded = [
        name
        for name, (_, limit) in tensors_and_limits.items()
        if diagnostics[name] > float(limit)
    ]
    if exceeded:
        raise _NumericalInstabilityError(
            label=f"{label} exceeded numerical validity bounds",
            step=step,
            invalid=exceeded,
            diagnostics=diagnostics,
        )


def _analyze_trace(
    trace: dict[str, list[float]],
    *,
    frequency_hz: float,
    signal_names: Sequence[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    means: dict[str, float] = {}
    harmonics: dict[str, dict[str, float]] = {}
    for signal_name in signal_names:
        values = np.asarray(trace[signal_name], dtype=np.float64)
        means[signal_name] = float(np.mean(values))
        harmonics[signal_name] = _harmonic_dict(time_s, values, frequency_hz)
    return means, harmonics


def _run_free_cases(
    *,
    dt_s: float,
    frequencies_hz: Sequence[float],
    total_cycles: int,
    measurement_cycles: int,
    progress_callback: Callable[[str], None] | None,
) -> list[dict[str, object]]:
    cfg = _make_cfg(
        dt_s=dt_s,
        num_envs=len(frequencies_hz),
        fixed_root=False,
        coupling_mode=ACTUAL_PER_WING_LINK,
    )
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        actions = _frequency_action(
            frequencies_hz,
            min_frequency_hz=cfg.min_flap_hz,
            max_frequency_hz=cfg.max_flap_hz,
            device=env.device,
        )
        env._pre_physics_step(actions)
        masses, inertias = _mass_inertia_tensors(env)
        traces = [_empty_trace(_FREE_SIGNAL_NAMES) for _ in frequencies_hz]
        linear_momentum_traces: list[list[np.ndarray]] = [
            [] for _ in frequencies_hz
        ]
        angular_momentum_traces: list[list[np.ndarray]] = [
            [] for _ in frequencies_hz
        ]
        force_traces: list[list[np.ndarray]] = [[] for _ in frequencies_hz]
        moment_traces: list[list[np.ndarray]] = [[] for _ in frequencies_hz]
        startup_max_qdd_error = np.zeros(len(frequencies_hz), dtype=np.float64)
        startup_max_force = np.zeros(len(frequencies_hz), dtype=np.float64)
        steady_max_qdd_error = np.zeros(len(frequencies_hz), dtype=np.float64)
        steady_max_force = np.zeros(len(frequencies_hz), dtype=np.float64)
        max_sync_error = np.zeros(len(frequencies_hz), dtype=np.float64)
        max_tracking_error = np.zeros(len(frequencies_hz), dtype=np.float64)

        duration_s = max(float(total_cycles) / float(f) for f in frequencies_hz)
        total_steps = int(math.ceil(duration_s / float(dt_s)))
        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        wing_joint_ids = [left_joint_id, right_joint_id]
        for step in range(total_steps):
            if progress_callback is not None and step % max(total_steps // 4, 1) == 0:
                progress_callback(
                    "free "
                    f"f={float(frequencies_hz[0]):g}Hz dt={float(dt_s):.9f}s "
                    f"{100.0 * step / total_steps:.0f}%"
            )
            target_time_s = (step + 1) * float(dt_s)
            env._sim_step_counter += 1
            env._apply_action()
            system_com_pre, linear_pre, angular_pre = _total_momentum_batch(
                env,
                masses,
                inertias,
            )
            force_w, moment_w = _external_wrench_about_system_com_w(
                env,
                system_com_pre,
            )
            actual_q = 0.5 * (
                env._robot.data.joint_pos[:, left_joint_id]
                - float(env._wing_mid_L)
                - env._robot.data.joint_pos[:, right_joint_id]
                + float(env._wing_mid_R)
            )
            actual_qd = 0.5 * (
                env._robot.data.joint_vel[:, left_joint_id]
                - env._robot.data.joint_vel[:, right_joint_id]
            )
            actual_qdd = torch.mean(env._debug_last_wing_aero_acceleration_rad_s2, dim=1)
            qdd_error = torch.abs(actual_qdd - env._qdd_cmd)
            wing_force_norm = torch.linalg.vector_norm(
                env._debug_last_wing_force_link_n,
                dim=-1,
            ).amax(dim=1)
            sync_error = torch.abs(
                env._robot.data.joint_pos[:, left_joint_id]
                + env._robot.data.joint_pos[:, right_joint_id]
            )
            tracking_error = torch.abs(actual_q - env._q_cmd)
            _require_finite(
                label="free-body pre-step state",
                step=step,
                tensors={
                    "root_state_w": env._robot.data.root_state_w,
                    "joint_position": env._robot.data.joint_pos,
                    "joint_velocity": env._robot.data.joint_vel,
                    "wing_acceleration_input": actual_qdd,
                    "wing_force_link": env._debug_last_wing_force_link_n,
                    "wing_moment_link": env._debug_last_wing_moment_link_about_com_nm,
                    "external_force_buffer": env._robot._external_force_b,
                    "external_torque_buffer": env._robot._external_torque_b,
                },
            )
            _require_bounded(
                label="free-body pre-step state",
                step=step,
                tensors_and_limits={
                    "root_linear_velocity_m_s": (
                        env._robot.data.root_lin_vel_w,
                        30.0,
                    ),
                    "root_angular_velocity_rad_s": (
                        env._robot.data.root_ang_vel_w,
                        30.0,
                    ),
                    "wing_joint_position_rad": (
                        env._robot.data.joint_pos[:, wing_joint_ids],
                        0.8,
                    ),
                    "wing_joint_velocity_rad_s": (
                        env._robot.data.joint_vel[:, wing_joint_ids],
                        50.0,
                    ),
                    "wing_force_link_n": (
                        env._debug_last_wing_force_link_n,
                        200.0,
                    ),
                    "wing_moment_link_nm": (
                        env._debug_last_wing_moment_link_about_com_nm,
                        50.0,
                    ),
                    "net_wing_force_b_n": (
                        env._debug_last_wing_force_b,
                        400.0,
                    ),
                    "net_wing_moment_b_nm": (
                        env._debug_last_torque_b,
                        100.0,
                    ),
                },
            )

            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            _require_finite(
                label="free-body post-step state",
                step=step,
                tensors={
                    "root_state_w": env._robot.data.root_state_w,
                    "joint_position": env._robot.data.joint_pos,
                    "joint_velocity": env._robot.data.joint_vel,
                    "pre_step_wing_acceleration_input": actual_qdd,
                    "pre_step_wing_force_link": env._debug_last_wing_force_link_n,
                    "pre_step_wing_moment_link": env._debug_last_wing_moment_link_about_com_nm,
                },
            )
            _require_bounded(
                label="free-body post-step state",
                step=step,
                tensors_and_limits={
                    "root_linear_velocity_m_s": (
                        env._robot.data.root_lin_vel_w,
                        30.0,
                    ),
                    "root_angular_velocity_rad_s": (
                        env._robot.data.root_ang_vel_w,
                        30.0,
                    ),
                    "wing_joint_position_rad": (
                        env._robot.data.joint_pos[:, wing_joint_ids],
                        0.8,
                    ),
                    "wing_joint_velocity_rad_s": (
                        env._robot.data.joint_vel[:, wing_joint_ids],
                        50.0,
                    ),
                },
            )
            _, linear_post, angular_post = _total_momentum_batch(
                env,
                masses,
                inertias,
            )
            base_linear_acc_b, base_angular_acc_b = _base_accelerations_b(env)

            for index, frequency_hz in enumerate(frequencies_hz):
                startup_end_s = 2.0 / float(frequency_hz)
                measurement_start_s = duration_s - float(measurement_cycles) / float(
                    frequency_hz
                )
                max_sync_error[index] = max(
                    max_sync_error[index],
                    float(sync_error[index].item()),
                )
                max_tracking_error[index] = max(
                    max_tracking_error[index],
                    float(tracking_error[index].item()),
                )
                if target_time_s <= startup_end_s:
                    startup_max_qdd_error[index] = max(
                        startup_max_qdd_error[index],
                        float(qdd_error[index].item()),
                    )
                    startup_max_force[index] = max(
                        startup_max_force[index],
                        float(wing_force_norm[index].item()),
                    )
                if target_time_s < measurement_start_s:
                    continue
                steady_max_qdd_error[index] = max(
                    steady_max_qdd_error[index],
                    float(qdd_error[index].item()),
                )
                steady_max_force[index] = max(
                    steady_max_force[index],
                    float(wing_force_norm[index].item()),
                )
                trace = traces[index]
                trace["time_s"].append(target_time_s)
                trace["q_actual_rad"].append(float(actual_q[index].item()))
                trace["qdot_actual_rad_s"].append(float(actual_qd[index].item()))
                trace["qdd_actual_rad_s2"].append(float(actual_qdd[index].item()))
                trace["wing_force_b_x_n"].append(
                    float(env._debug_last_wing_force_b[index, 0].item())
                )
                trace["wing_force_b_z_n"].append(
                    float(env._debug_last_wing_force_b[index, 2].item())
                )
                trace["wing_moment_b_y_nm"].append(
                    float(env._debug_last_torque_b[index, 1].item())
                )
                trace["base_acceleration_b_z_m_s2"].append(
                    float(base_linear_acc_b[index, 2].item())
                )
                trace["base_pitch_acceleration_b_y_rad_s2"].append(
                    float(base_angular_acc_b[index, 1].item())
                )
                if not linear_momentum_traces[index]:
                    linear_momentum_traces[index].append(
                        linear_pre[index].detach().cpu().numpy()
                    )
                    angular_momentum_traces[index].append(
                        angular_pre[index].detach().cpu().numpy()
                    )
                force_traces[index].append(force_w[index].detach().cpu().numpy())
                moment_traces[index].append(moment_w[index].detach().cpu().numpy())
                linear_momentum_traces[index].append(
                    linear_post[index].detach().cpu().numpy()
                )
                angular_momentum_traces[index].append(
                    angular_post[index].detach().cpu().numpy()
                )

        cases: list[dict[str, object]] = []
        for index, frequency_hz in enumerate(frequencies_hz):
            means, harmonics = _analyze_trace(
                traces[index],
                frequency_hz=float(frequency_hz),
                signal_names=_FREE_SIGNAL_NAMES,
            )
            closure = compute_impulse_momentum_closure(
                linear_momentum_w_kg_m_s=np.asarray(
                    linear_momentum_traces[index]
                ),
                angular_momentum_about_com_w_kg_m2_s=np.asarray(
                    angular_momentum_traces[index]
                ),
                external_force_w_n=np.asarray(force_traces[index]),
                external_moment_about_com_w_nm=np.asarray(moment_traces[index]),
                dt_s=float(dt_s),
            )
            cases.append(
                {
                    "frequency_hz": float(frequency_hz),
                    "dt_s": float(dt_s),
                    "stable": True,
                    "sample_count": len(traces[index]["time_s"]),
                    "max_sync_error_deg": math.degrees(max_sync_error[index]),
                    "max_tracking_error_deg": math.degrees(
                        max_tracking_error[index]
                    ),
                    "startup_max_qdd_input_error_rad_s2": float(
                        startup_max_qdd_error[index]
                    ),
                    "steady_max_qdd_input_error_rad_s2": float(
                        steady_max_qdd_error[index]
                    ),
                    "startup_max_single_wing_force_n": float(
                        startup_max_force[index]
                    ),
                    "steady_max_single_wing_force_n": float(
                        steady_max_force[index]
                    ),
                    "means": means,
                    "harmonics": harmonics,
                    "momentum_closure": closure,
                }
            )
        return cases
    finally:
        env.close()


def _run_free_impulse_closure_case(
    *,
    dt_s: float,
    frequency_hz: float,
    duration_s: float,
    progress_callback: Callable[[str], None] | None,
) -> dict[str, object]:
    cfg = _make_cfg(
        dt_s=dt_s,
        num_envs=1,
        fixed_root=False,
        coupling_mode=ACTUAL_PER_WING_LINK,
    )
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        actions = _frequency_action(
            (frequency_hz,),
            min_frequency_hz=cfg.min_flap_hz,
            max_frequency_hz=cfg.max_flap_hz,
            device=env.device,
        )
        env._pre_physics_step(actions)
        masses, inertias = _mass_inertia_tensors(env)
        total_steps = int(math.ceil(float(duration_s) / float(dt_s)))
        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        wing_joint_ids = [left_joint_id, right_joint_id]
        linear_momentum_trace: list[np.ndarray] = []
        angular_momentum_trace: list[np.ndarray] = []
        force_trace: list[np.ndarray] = []
        moment_trace: list[np.ndarray] = []
        max_sync_error_rad = 0.0
        max_tracking_error_rad = 0.0
        max_qdd_error_rad_s2 = 0.0
        max_single_wing_force_n = 0.0

        for step in range(total_steps):
            if progress_callback is not None and step == 0:
                progress_callback(
                    f"free impulse f={frequency_hz:g}Hz dt={dt_s:.9f}s"
                )
            env._sim_step_counter += 1
            env._apply_action()
            system_com_pre, linear_pre, angular_pre = _total_momentum_batch(
                env,
                masses,
                inertias,
            )
            force_w, moment_w = _external_wrench_about_system_com_w(
                env,
                system_com_pre,
            )
            actual_q = torch.mean(env._debug_last_wing_aero_position_rad, dim=1)
            actual_qdd = torch.mean(
                env._debug_last_wing_aero_acceleration_rad_s2,
                dim=1,
            )
            sync_error = torch.abs(
                env._robot.data.joint_pos[:, left_joint_id]
                + env._robot.data.joint_pos[:, right_joint_id]
            )
            tracking_error = torch.abs(actual_q - env._q_cmd)
            qdd_error = torch.abs(actual_qdd - env._qdd_cmd)
            single_wing_force = torch.linalg.vector_norm(
                env._debug_last_wing_force_link_n,
                dim=-1,
            ).amax(dim=1)
            _require_finite(
                label="free impulse pre-step state",
                step=step,
                tensors={
                    "root_state_w": env._robot.data.root_state_w,
                    "joint_position": env._robot.data.joint_pos,
                    "joint_velocity": env._robot.data.joint_vel,
                    "wing_force_link": env._debug_last_wing_force_link_n,
                    "wing_moment_link": env._debug_last_wing_moment_link_about_com_nm,
                },
            )
            if not linear_momentum_trace:
                linear_momentum_trace.append(linear_pre[0].detach().cpu().numpy())
                angular_momentum_trace.append(
                    angular_pre[0].detach().cpu().numpy()
                )
            force_trace.append(force_w[0].detach().cpu().numpy())
            moment_trace.append(moment_w[0].detach().cpu().numpy())

            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            _, linear_post, angular_post = _total_momentum_batch(
                env,
                masses,
                inertias,
            )
            _require_finite(
                label="free impulse post-step state",
                step=step,
                tensors={
                    "root_state_w": env._robot.data.root_state_w,
                    "joint_position": env._robot.data.joint_pos,
                    "joint_velocity": env._robot.data.joint_vel,
                },
            )
            linear_momentum_trace.append(linear_post[0].detach().cpu().numpy())
            angular_momentum_trace.append(
                angular_post[0].detach().cpu().numpy()
            )
            max_sync_error_rad = max(
                max_sync_error_rad,
                float(sync_error[0].item()),
            )
            max_tracking_error_rad = max(
                max_tracking_error_rad,
                float(tracking_error[0].item()),
            )
            max_qdd_error_rad_s2 = max(
                max_qdd_error_rad_s2,
                float(qdd_error[0].item()),
            )
            max_single_wing_force_n = max(
                max_single_wing_force_n,
                float(single_wing_force[0].item()),
            )

        return {
            "frequency_hz": float(frequency_hz),
            "dt_s": float(dt_s),
            "stable": True,
            "duration_s": total_steps * float(dt_s),
            "sample_count": total_steps,
            "max_sync_error_deg": math.degrees(max_sync_error_rad),
            "max_tracking_error_deg": math.degrees(max_tracking_error_rad),
            "max_qdd_input_error_rad_s2": max_qdd_error_rad_s2,
            "max_single_wing_force_n": max_single_wing_force_n,
            "momentum_closure": compute_impulse_momentum_closure(
                linear_momentum_w_kg_m_s=np.asarray(linear_momentum_trace),
                angular_momentum_about_com_w_kg_m2_s=np.asarray(
                    angular_momentum_trace
                ),
                external_force_w_n=np.asarray(force_trace),
                external_moment_about_com_w_nm=np.asarray(moment_trace),
                dt_s=float(dt_s),
            ),
        }
    finally:
        env.close()


def _fixed_case_definitions(
    *,
    frequencies_hz: Sequence[float],
    airspeeds_mps: Sequence[float],
    angles_of_attack_deg: Sequence[float],
) -> list[dict[str, float]]:
    return [
        {
            "frequency_hz": float(frequency_hz),
            "airspeed_mps": float(airspeed_mps),
            "angle_of_attack_deg": float(angle_of_attack_deg),
        }
        for angle_of_attack_deg in angles_of_attack_deg
        for airspeed_mps in airspeeds_mps
        for frequency_hz in frequencies_hz
    ]


def _run_fixed_scan(
    *,
    dt_s: float,
    coupling_mode: str,
    acceleration_source: str,
    wing_link_load_mode: str,
    frequencies_hz: Sequence[float],
    airspeeds_mps: Sequence[float],
    angles_of_attack_deg: Sequence[float],
    total_cycles: int,
    measurement_cycles: int,
    progress_callback: Callable[[str], None] | None,
) -> list[dict[str, object]]:
    unique_frequencies = {float(value) for value in frequencies_hz}
    if (
        coupling_mode in {
            SINUSOIDAL_PHASE_PER_WING_LINK,
            IDEAL_INVERSE_DYNAMICS_PER_WING_LINK,
        }
        and len(unique_frequencies) != 1
    ):
        raise ValueError(
            "Phase-driven per-wing modes require one frequency per Isaac process "
            "so the periodic mechanism state can be initialized consistently."
        )
    definitions = _fixed_case_definitions(
        frequencies_hz=frequencies_hz,
        airspeeds_mps=airspeeds_mps,
        angles_of_attack_deg=angles_of_attack_deg,
    )
    cfg = _make_cfg(
        dt_s=dt_s,
        num_envs=len(definitions),
        fixed_root=True,
        coupling_mode=coupling_mode,
        acceleration_source=acceleration_source,
        wing_link_load_mode=wing_link_load_mode,
        reset_frequency_hz=(
            next(iter(unique_frequencies))
            if coupling_mode
            in {
                SINUSOIDAL_PHASE_PER_WING_LINK,
                IDEAL_INVERSE_DYNAMICS_PER_WING_LINK,
            }
            else None
        ),
    )
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        case_frequencies = [case["frequency_hz"] for case in definitions]
        actions = _frequency_action(
            case_frequencies,
            min_frequency_hz=cfg.min_flap_hz,
            max_frequency_hz=cfg.max_flap_hz,
            device=env.device,
        )
        desired_air_velocity_b = torch.zeros(
            (len(definitions), 3),
            dtype=torch.float32,
            device=env.device,
        )
        for index, case in enumerate(definitions):
            alpha_rad = math.radians(case["angle_of_attack_deg"])
            desired_air_velocity_b[index, 0] = case["airspeed_mps"] * math.cos(
                alpha_rad
            )
            desired_air_velocity_b[index, 2] = -case["airspeed_mps"] * math.sin(
                alpha_rad
            )
        env._wind_w.copy_(
            quat_apply(env._robot.data.root_quat_w, -desired_air_velocity_b)
        )
        env._wind_mean_w.copy_(env._wind_w)
        env._pre_physics_step(actions)
        traces = [_empty_trace(_FIXED_SIGNAL_NAMES) for _ in definitions]
        max_sync_error = np.zeros(len(definitions), dtype=np.float64)
        max_tracking_error = np.zeros(len(definitions), dtype=np.float64)
        max_driver_torque_nm = np.zeros(len(definitions), dtype=np.float64)
        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        wing_joint_ids = [left_joint_id, right_joint_id]
        duration_s = max(float(total_cycles) / float(f) for f in frequencies_hz)
        total_steps = int(math.ceil(duration_s / float(dt_s)))

        for step in range(total_steps):
            if progress_callback is not None and step % max(total_steps // 4, 1) == 0:
                progress_callback(
                    f"fixed mode={coupling_mode} dt={float(dt_s):.9f}s "
                    f"{100.0 * step / total_steps:.0f}%"
            )
            target_time_s = (step + 1) * float(dt_s)
            env._sim_step_counter += 1
            env._apply_action()
            physical_wing_kinematics = (
                map_opposed_joint_states_to_physical_wing_kinematics(
                    joint_position_rad=env._robot.data.joint_pos[:, wing_joint_ids],
                    joint_velocity_rad_s=env._robot.data.joint_vel[:, wing_joint_ids],
                    joint_acceleration_rad_s2=env._robot.data.joint_acc[:, wing_joint_ids],
                    left_joint_mid_rad=float(env._wing_mid_L),
                    right_joint_mid_rad=float(env._wing_mid_R),
                )
            )
            actual_q = torch.mean(
                physical_wing_kinematics.position_rad,
                dim=1,
            )
            actual_qd = torch.mean(
                physical_wing_kinematics.velocity_rad_s,
                dim=1,
            )
            aero_input_qdd = torch.mean(
                env._debug_last_wing_aero_acceleration_rad_s2,
                dim=1,
            )
            physx_qdd = torch.mean(
                physical_wing_kinematics.acceleration_rad_s2,
                dim=1,
            )
            aero_hinge_torque_nm, aero_hinge_power_w = (
                _computed_aero_hinge_torque_and_power(env)
            )
            driver_torque_nm = env._robot.data.applied_torque[:, left_joint_id]
            driver_power_w = (
                driver_torque_nm
                * env._robot.data.joint_vel[:, left_joint_id]
            )
            sync_error = torch.abs(
                physical_wing_kinematics.position_rad[:, 0]
                - physical_wing_kinematics.position_rad[:, 1]
            )
            tracking_error = torch.abs(actual_q - env._q_cmd)
            _require_finite(
                label=f"fixed-body {coupling_mode} pre-step state",
                step=step,
                tensors={
                    "joint_position": env._robot.data.joint_pos,
                    "joint_velocity": env._robot.data.joint_vel,
                    "wing_acceleration_input": aero_input_qdd,
                    "wing_physx_acceleration": physx_qdd,
                    "aero_hinge_torque": aero_hinge_torque_nm,
                    "aero_hinge_power": aero_hinge_power_w,
                    "driver_torque": driver_torque_nm,
                    "driver_power": driver_power_w,
                    "actual_frequency_hz": env._freq,
                    "wing_force_link": env._debug_last_wing_force_link_n,
                    "wing_moment_link": env._debug_last_wing_moment_link_about_com_nm,
                    "external_force_buffer": env._robot._external_force_b,
                    "external_torque_buffer": env._robot._external_torque_b,
                },
            )
            _require_bounded(
                label=f"fixed-body {coupling_mode} pre-step state",
                step=step,
                tensors_and_limits={
                    "wing_joint_position_rad": (
                        env._robot.data.joint_pos[:, wing_joint_ids],
                        0.8,
                    ),
                    "wing_joint_velocity_rad_s": (
                        env._robot.data.joint_vel[:, wing_joint_ids],
                        50.0,
                    ),
                    "wing_force_link_n": (
                        env._debug_last_wing_force_link_n,
                        200.0,
                    ),
                    "wing_moment_link_nm": (
                        env._debug_last_wing_moment_link_about_com_nm,
                        50.0,
                    ),
                    "wing_physx_acceleration_rad_s2": (physx_qdd, 50_000.0),
                    "wing_aero_acceleration_input_rad_s2": (
                        aero_input_qdd,
                        50_000.0,
                    ),
                    "computed_aero_hinge_power_w": (
                        aero_hinge_power_w,
                        1_000_000.0,
                    ),
                    "driver_torque_nm": (driver_torque_nm, 2_000.0),
                    "actual_frequency_hz": (env._freq, 25.0),
                },
            )
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            _require_finite(
                label=f"fixed-body {coupling_mode} post-step state",
                step=step,
                tensors={
                    "joint_position": env._robot.data.joint_pos,
                    "joint_velocity": env._robot.data.joint_vel,
                },
            )
            _require_bounded(
                label=f"fixed-body {coupling_mode} post-step state",
                step=step,
                tensors_and_limits={
                    "wing_joint_position_rad": (
                        env._robot.data.joint_pos[:, wing_joint_ids],
                        0.8,
                    ),
                    "wing_joint_velocity_rad_s": (
                        env._robot.data.joint_vel[:, wing_joint_ids],
                        50.0,
                    ),
                },
            )

            for index, case in enumerate(definitions):
                max_sync_error[index] = max(
                    max_sync_error[index],
                    float(sync_error[index].item()),
                )
                max_tracking_error[index] = max(
                    max_tracking_error[index],
                    float(tracking_error[index].item()),
                )
                max_driver_torque_nm[index] = max(
                    max_driver_torque_nm[index],
                    abs(float(driver_torque_nm[index].item())),
                )
                measurement_start_s = duration_s - float(measurement_cycles) / case[
                    "frequency_hz"
                ]
                if target_time_s < measurement_start_s:
                    continue
                trace = traces[index]
                trace["time_s"].append(target_time_s)
                trace["q_actual_rad"].append(float(actual_q[index].item()))
                trace["qdot_actual_rad_s"].append(float(actual_qd[index].item()))
                trace["qdd_physx_rad_s2"].append(float(physx_qdd[index].item()))
                trace["qdd_aero_input_rad_s2"].append(
                    float(aero_input_qdd[index].item())
                )
                trace["aero_hinge_torque_left_nm"].append(
                    float(aero_hinge_torque_nm[index, 0].item())
                )
                trace["aero_hinge_torque_right_nm"].append(
                    float(aero_hinge_torque_nm[index, 1].item())
                )
                trace["aero_hinge_power_w"].append(
                    float(aero_hinge_power_w[index].item())
                )
                trace["driver_torque_left_nm"].append(
                    float(driver_torque_nm[index].item())
                )
                trace["driver_power_w"].append(
                    float(driver_power_w[index].item())
                )
                trace["actual_frequency_hz"].append(
                    float(env._freq[index].item())
                )
                for axis, suffix in enumerate(("x", "y", "z")):
                    trace[f"wing_force_b_{suffix}_n"].append(
                        float(env._debug_last_wing_force_b[index, axis].item())
                    )
                    trace[f"wing_moment_b_{suffix}_nm"].append(
                        float(env._debug_last_torque_b[index, axis].item())
                    )

        cases: list[dict[str, object]] = []
        for index, definition in enumerate(definitions):
            means, harmonics = _analyze_trace(
                traces[index],
                frequency_hz=definition["frequency_hz"],
                signal_names=_FIXED_SIGNAL_NAMES,
            )
            force_b = np.column_stack(
                [
                    np.asarray(traces[index][f"wing_force_b_{axis}_n"])
                    for axis in ("x", "y", "z")
                ]
            )
            moment_b = np.column_stack(
                [
                    np.asarray(traces[index][f"wing_moment_b_{axis}_nm"])
                    for axis in ("x", "y", "z")
                ]
            )
            cases.append(
                {
                    **definition,
                    "dt_s": float(dt_s),
                    "coupling_mode": coupling_mode,
                    "acceleration_source": acceleration_source,
                    "wing_link_load_mode": wing_link_load_mode,
                    "stable": True,
                    "sample_count": len(traces[index]["time_s"]),
                    "max_sync_error_deg": math.degrees(max_sync_error[index]),
                    "max_tracking_error_deg": math.degrees(
                        max_tracking_error[index]
                    ),
                    "max_driver_torque_nm": float(
                        max_driver_torque_nm[index]
                    ),
                    "minimum_actual_frequency_hz": float(
                        np.min(traces[index]["actual_frequency_hz"])
                    ),
                    "maximum_actual_frequency_hz": float(
                        np.max(traces[index]["actual_frequency_hz"])
                    ),
                    "means": means,
                    "harmonics": harmonics,
                    "symmetry": compute_symmetric_wrench_residuals(
                        force_b_n=force_b,
                        moment_b_about_base_com_nm=moment_b,
                    ),
                }
            )
        return cases
    finally:
        env.close()


def run_free_multibody_aero_job(
    *,
    frequency_hz: float,
    dt_s: float,
    duration_s: float = 0.1,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run one isolated short free-body impulse-closure worker."""

    try:
        cases = [
            _run_free_impulse_closure_case(
                dt_s=float(dt_s),
                frequency_hz=float(frequency_hz),
                duration_s=float(duration_s),
                progress_callback=progress_callback,
            )
        ]
    except _NumericalInstabilityError as error:
        cases = [
            {
                "frequency_hz": float(frequency_hz),
                "dt_s": float(dt_s),
                "stable": False,
                "failure_step": error.step,
                "failure_time_s": (error.step + 1) * float(dt_s),
                "failure_label": error.label,
                "invalid_tensors": list(error.invalid),
                "failure_diagnostics": error.diagnostics,
            }
        ]
    return {
        "schema_version": "physx_multibody_aero_worker_v1",
        "job_type": "free",
        "free_cases": cases,
        "fixed_cases": [],
    }


def run_fixed_multibody_aero_job(
    *,
    dt_s: float,
    coupling_mode: str,
    acceleration_source: str = ACTUAL_JOINT_ACCELERATION,
    wing_link_load_mode: str = FULL_WING_LINK_WRENCH,
    frequencies_hz: Sequence[float] = (2.0, 3.0, 4.0, 5.0),
    airspeeds_mps: Sequence[float] = (4.0, 6.0, 8.0),
    angles_of_attack_deg: Sequence[float] = (-5.0, 0.0, 5.0),
    total_cycles: int = 8,
    measurement_cycles: int = 4,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run one isolated fixed-body coupling-mode worker."""

    if total_cycles < measurement_cycles or measurement_cycles < 1:
        raise ValueError("Use total_cycles >= measurement_cycles >= 1.")
    definitions = _fixed_case_definitions(
        frequencies_hz=frequencies_hz,
        airspeeds_mps=airspeeds_mps,
        angles_of_attack_deg=angles_of_attack_deg,
    )
    try:
        cases = _run_fixed_scan(
            dt_s=float(dt_s),
            coupling_mode=coupling_mode,
            acceleration_source=acceleration_source,
            wing_link_load_mode=wing_link_load_mode,
            frequencies_hz=frequencies_hz,
            airspeeds_mps=airspeeds_mps,
            angles_of_attack_deg=angles_of_attack_deg,
            total_cycles=int(total_cycles),
            measurement_cycles=int(measurement_cycles),
            progress_callback=progress_callback,
        )
    except _NumericalInstabilityError as error:
        cases = [
            {
                **definition,
                "dt_s": float(dt_s),
                "coupling_mode": coupling_mode,
                "acceleration_source": acceleration_source,
                "wing_link_load_mode": wing_link_load_mode,
                "stable": False,
                "failure_step": error.step,
                "failure_time_s": (error.step + 1) * float(dt_s),
                "failure_label": error.label,
                "invalid_tensors": list(error.invalid),
                "failure_diagnostics": error.diagnostics,
            }
            for definition in definitions
        ]
    return {
        "schema_version": "physx_multibody_aero_worker_v1",
        "job_type": "fixed",
        "free_cases": [],
        "fixed_cases": cases,
    }


def run_physx_multibody_aero_benchmark(
    *,
    frequencies_hz: Sequence[float] = (2.0, 3.0, 4.0, 5.0),
    time_steps_s: Sequence[float] = (1.0 / 240.0, 1.0 / 480.0),
    airspeeds_mps: Sequence[float] = (4.0, 6.0, 8.0),
    angles_of_attack_deg: Sequence[float] = (-5.0, 0.0, 5.0),
    total_cycles: int = 8,
    measurement_cycles: int = 4,
    run_fixed_baseline: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run free-body convergence/closure and fixed-body aerodynamic matrices."""

    if total_cycles <= measurement_cycles or measurement_cycles < 2:
        raise ValueError("Use total_cycles > measurement_cycles >= 2.")
    if not frequencies_hz or not time_steps_s:
        raise ValueError("At least one frequency and time step are required.")
    free_cases: list[dict[str, object]] = []
    for dt_s in time_steps_s:
        for frequency_hz in frequencies_hz:
            try:
                free_cases.extend(
                    _run_free_cases(
                        dt_s=float(dt_s),
                        frequencies_hz=(float(frequency_hz),),
                        total_cycles=total_cycles,
                        measurement_cycles=measurement_cycles,
                        progress_callback=progress_callback,
                    )
                )
            except _NumericalInstabilityError as error:
                free_cases.append(
                    {
                        "frequency_hz": float(frequency_hz),
                        "dt_s": float(dt_s),
                        "stable": False,
                        "failure_step": error.step,
                        "failure_time_s": (error.step + 1) * float(dt_s),
                        "failure_label": error.label,
                        "invalid_tensors": list(error.invalid),
                        "failure_diagnostics": error.diagnostics,
                    }
                )
    stable_free_cases = [
        case for case in free_cases if bool(case.get("stable", True))
    ]
    convergence = summarize_time_step_convergence(
        stable_free_cases,
        signal_names=_FREE_SIGNAL_NAMES,
    )

    wind_tunnel_dt_s = min(float(value) for value in time_steps_s)
    fixed_cases = _run_fixed_scan(
        dt_s=wind_tunnel_dt_s,
        coupling_mode=ACTUAL_PER_WING_LINK,
        acceleration_source=ACTUAL_JOINT_ACCELERATION,
        wing_link_load_mode=FULL_WING_LINK_WRENCH,
        frequencies_hz=frequencies_hz,
        airspeeds_mps=airspeeds_mps,
        angles_of_attack_deg=angles_of_attack_deg,
        total_cycles=total_cycles,
        measurement_cycles=measurement_cycles,
        progress_callback=progress_callback,
    )
    if run_fixed_baseline:
        fixed_cases.extend(
            _run_fixed_scan(
                dt_s=wind_tunnel_dt_s,
                coupling_mode=COMMANDED_BASE_EQUIVALENT,
                acceleration_source=ACTUAL_JOINT_ACCELERATION,
                wing_link_load_mode=FULL_WING_LINK_WRENCH,
                frequencies_hz=frequencies_hz,
                airspeeds_mps=airspeeds_mps,
                angles_of_attack_deg=angles_of_attack_deg,
                total_cycles=total_cycles,
                measurement_cycles=measurement_cycles,
                progress_callback=progress_callback,
            )
        )
    return {
        "schema_version": "physx_multibody_aero_benchmark_v1",
        "control_period_s": _CONTROL_PERIOD_S,
        "frequencies_hz": [float(value) for value in frequencies_hz],
        "time_steps_s": [float(value) for value in time_steps_s],
        "airspeeds_mps": [float(value) for value in airspeeds_mps],
        "angles_of_attack_deg": [
            float(value) for value in angles_of_attack_deg
        ],
        "total_cycles": int(total_cycles),
        "measurement_cycles": int(measurement_cycles),
        "wind_tunnel_dt_s": wind_tunnel_dt_s,
        "free_cases": free_cases,
        "time_step_convergence": convergence,
        "fixed_cases": fixed_cases,
        "fixed_mode_comparison": compare_fixed_mode_cases(
            fixed_cases,
            actual_mode=ACTUAL_PER_WING_LINK,
            baseline_mode=COMMANDED_BASE_EQUIVALENT,
            signal_names=(
                "q_actual_rad",
                "wing_force_b_x_n",
                "wing_force_b_z_n",
                "wing_moment_b_y_nm",
            ),
        )
        if run_fixed_baseline
        else [],
    }


__all__ = [
    "run_fixed_multibody_aero_job",
    "run_free_multibody_aero_job",
    "run_physx_multibody_aero_benchmark",
]

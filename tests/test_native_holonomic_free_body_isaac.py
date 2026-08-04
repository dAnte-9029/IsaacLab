"""Headless free-body momentum gate for the native holonomic mechanism."""

from __future__ import annotations

import math
from pathlib import Path

from isaaclab.app import AppLauncher

_EXTENSION_ROOT = (
    Path(__file__).resolve().parents[1]
    / "source/flapping_bot/native_extensions/omni.flapping_bot.holonomic_constraint"
)
simulation_app = AppLauncher(
    headless=True,
    kit_args=f"--ext-folder {str(_EXTENSION_ROOT.parent)} --enable omni.flapping_bot.holonomic_constraint",
).app

import omni.physx
import numpy as np
import pytest
import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg,
)
from flapping_bot.analysis.multibody_aero_validation import (
    compute_impulse_momentum_closure,
)


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
    inertia_w = inertias_kg_m2.unsqueeze(0)
    inertia_w = rotations @ inertia_w @ rotations.transpose(-1, -2)
    spin = torch.einsum("nbij,nbj->nbi", inertia_w, angular_velocities)
    orbital = torch.linalg.cross(
        positions - system_com.unsqueeze(1),
        mass_weights * linear_velocities,
        dim=-1,
    )
    angular_momentum = torch.sum(spin + orbital, dim=1)
    return system_com, linear_momentum, angular_momentum


@pytest.mark.isaacsim_ci
def test_native_holonomic_free_body_conserves_system_momentum() -> None:
    dt_denominator = 480
    cfg = FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = "cpu"
    cfg.sim.dt = 1.0 / float(dt_denominator)
    cfg.decimation = 1
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.robot.spawn.rigid_props.disable_gravity = True
    cfg.robot.spawn.rigid_props.retain_accelerations = False
    cfg.robot.spawn.articulation_props.fix_root_link = False
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 0.0
    cfg.reset_flap_hz = 0.0
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        masses = env._robot.root_physx_view.get_masses()[0].to(env.device)
        print(f"native holonomic body masses: {masses.cpu().tolist()}, total={float(masses.sum().item())}")
        inertias = (
            env._robot.root_physx_view.get_inertias()[0]
            .reshape(-1, 3, 3)
            .to(env.device)
        )
        initial_com, initial_linear, initial_angular = _total_momentum(
            env,
            masses,
            inertias,
        )
        actions = torch.zeros((2, 4), device=env.device)
        actions[:, 0] = 2.0 * torch.tensor((2.0, 5.0), device=env.device) / cfg.max_flap_hz - 1.0
        env._pre_physics_step(actions)
        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        max_tracking_error = 0.0
        max_sync_error = 0.0
        max_com_drift = 0.0
        max_linear_drift = 0.0
        max_angular_drift = 0.0

        for _ in range(int(round(1.5 / cfg.sim.dt))):
            env._sim_step_counter += 1
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            system_com, linear_momentum, angular_momentum = _total_momentum(
                env,
                masses,
                inertias,
            )
            left_position = env._robot.data.joint_pos[:, left_joint_id] - float(
                env._wing_mid_L
            )
            right_position = -(
                env._robot.data.joint_pos[:, right_joint_id] - float(env._wing_mid_R)
            )
            reference = env._wing_amp * torch.sin(
                env._ideal_inverse_phase_state.phase_rad
            )
            max_tracking_error = max(
                max_tracking_error,
                float(torch.max(torch.abs(left_position - reference)).item()),
            )
            max_sync_error = max(
                max_sync_error,
                float(torch.max(torch.abs(left_position - right_position)).item()),
            )
            max_com_drift = max(
                max_com_drift,
                float(torch.max(torch.linalg.vector_norm(system_com - initial_com, dim=1)).item()),
            )
            max_linear_drift = max(
                max_linear_drift,
                float(
                    torch.max(
                        torch.linalg.vector_norm(linear_momentum - initial_linear, dim=1)
                    ).item()
                ),
            )
            max_angular_drift = max(
                max_angular_drift,
                float(
                    torch.max(
                        torch.linalg.vector_norm(angular_momentum - initial_angular, dim=1)
                    ).item()
                ),
            )

        loaded_linear_momentum: list[np.ndarray] = []
        loaded_angular_momentum: list[np.ndarray] = []
        applied_force: list[np.ndarray] = []
        applied_moment: list[np.ndarray] = []
        loaded_body_id = int(env._base_body_ids[0])
        known_force_link = torch.zeros((2, 1, 3), device=env.device)
        known_force_link[:, 0, 0] = 1.0
        known_torque_link = torch.zeros_like(known_force_link)
        loaded_steps = int(round(0.2 / cfg.sim.dt))
        for step in range(loaded_steps):
            env._sim_step_counter += 1
            env._apply_action()
            system_com_pre, linear_pre, angular_pre = _total_momentum(
                env,
                masses,
                inertias,
            )
            loaded_body_quat_w = env._robot.data.body_link_quat_w[:, loaded_body_id]
            known_force_w = quat_apply(loaded_body_quat_w, known_force_link[:, 0])
            known_torque_w = quat_apply(loaded_body_quat_w, known_torque_link[:, 0])
            moment_about_system_com = known_torque_w + torch.linalg.cross(
                env._robot.data.body_com_pos_w[:, loaded_body_id] - system_com_pre,
                known_force_w,
                dim=-1,
            )
            if step == 0:
                loaded_linear_momentum.append(linear_pre[0].detach().cpu().numpy())
                loaded_angular_momentum.append(angular_pre[0].detach().cpu().numpy())
            applied_force.append(known_force_w[0].detach().cpu().numpy())
            applied_moment.append(moment_about_system_com[0].detach().cpu().numpy())
            env._robot.set_external_force_and_torque(
                forces=known_force_link,
                torques=known_torque_link,
                body_ids=[loaded_body_id],
                is_global=False,
            )
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            _, linear_post, angular_post = _total_momentum(env, masses, inertias)
            loaded_linear_momentum.append(linear_post[0].detach().cpu().numpy())
            loaded_angular_momentum.append(angular_post[0].detach().cpu().numpy())

        loaded_closure = compute_impulse_momentum_closure(
            linear_momentum_w_kg_m_s=np.asarray(loaded_linear_momentum),
            angular_momentum_about_com_w_kg_m2_s=np.asarray(loaded_angular_momentum),
            external_force_w_n=np.asarray(applied_force),
            external_moment_about_com_w_nm=np.asarray(applied_moment),
            dt_s=float(cfg.sim.dt),
        )
        print(f"native holonomic known-load closure: {loaded_closure}")

        print(
            "native holonomic free-body gate:"
            f" dt=1/{dt_denominator}s,"
            f" max_tracking={math.degrees(max_tracking_error):.9f}deg,"
            f" max_sync={math.degrees(max_sync_error):.9f}deg,"
            f" max_com_drift={max_com_drift:.6e}m,"
            f" max_linear_drift={max_linear_drift:.6e}kg*m/s,"
            f" max_angular_drift={max_angular_drift:.6e}kg*m^2/s"
            f", known_load_linear_closure={loaded_closure['relative_linear_closure_error']:.6e}"
            f", known_load_angular_closure={loaded_closure['relative_angular_closure_error']:.6e}"
        )
        assert torch.all(torch.isfinite(env._robot.data.root_state_w))
        assert max_tracking_error < math.radians(0.1)
        assert max_sync_error < math.radians(0.1)
        assert max_com_drift < 1.0e-4
        assert max_linear_drift < 1.0e-4
        assert max_angular_drift < 5.0e-4
        assert loaded_closure["max_linear_residual_kg_m_s"] < 1.0e-4
        assert loaded_closure["max_angular_residual_kg_m2_s"] < 5.0e-4
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()

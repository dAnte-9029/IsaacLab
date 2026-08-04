"""Headless fixed-root gate for the native holonomic wing mechanism."""

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

import numpy as np
import omni.physx
import pytest
import torch

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg,
)


@pytest.mark.isaacsim_ci
def test_native_holonomic_mechanism_tracks_physx_joint_state() -> None:
    cfg = FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = "cpu"
    cfg.sim.dt = 1.0 / 480.0
    cfg.decimation = 1
    cfg.robot.spawn.rigid_props.disable_gravity = True
    cfg.robot.spawn.articulation_props.fix_root_link = True
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 0.0
    cfg.reset_flap_hz = 2.0
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        assert int(env._native_holonomic.get_active_joint_count()) == 2
        assert "passive_wings" in env._robot.actuators
        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        expected_reset_velocity = env._wing_amp * 2.0 * math.pi * cfg.reset_flap_hz
        dof_velocity = env._robot.root_physx_view.get_dof_velocities()
        torch.testing.assert_close(
            dof_velocity[:, left_joint_id],
            torch.full_like(dof_velocity[:, left_joint_id], expected_reset_velocity),
        )
        torch.testing.assert_close(
            dof_velocity[:, right_joint_id],
            torch.full_like(dof_velocity[:, right_joint_id], -expected_reset_velocity),
        )
        target_frequency_hz = torch.tensor((2.0, 5.0), device=env.device)
        actions = torch.zeros((2, 4), device=env.device)
        actions[:, 0] = 2.0 * target_frequency_hz / cfg.max_flap_hz - 1.0
        env._pre_physics_step(actions)

        tracking_errors: list[float] = []
        velocity_errors: list[float] = []
        sync_errors: list[float] = []
        frequencies: list[np.ndarray] = []
        total_steps = int(round(1.25 / cfg.sim.dt))
        discard_steps = int(round(0.50 / cfg.sim.dt))
        for step in range(total_steps):
            env._sim_step_counter += 1
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            if step < discard_steps:
                continue

            left_position = env._robot.data.joint_pos[:, left_joint_id] - float(env._wing_mid_L)
            right_position = -(
                env._robot.data.joint_pos[:, right_joint_id] - float(env._wing_mid_R)
            )
            left_velocity = env._robot.data.joint_vel[:, left_joint_id]
            next_phase = env._ideal_inverse_phase_state.phase_rad
            next_frequency = env._ideal_inverse_phase_state.frequency_hz
            next_position_reference = env._wing_amp * torch.sin(next_phase)
            next_velocity_reference = (
                env._wing_amp * 2.0 * math.pi * next_frequency * torch.cos(next_phase)
            )
            tracking_errors.append(
                float(torch.max(torch.abs(left_position - next_position_reference)).item())
            )
            velocity_errors.append(
                float(torch.max(torch.abs(left_velocity - next_velocity_reference)).item())
            )
            sync_errors.append(float(torch.max(torch.abs(left_position - right_position)).item()))
            frequencies.append(env._freq.detach().cpu().numpy().copy())

        debug_state = env._native_holonomic.get_debug_state()
        mean_frequencies = np.mean(np.stack(frequencies), axis=0)
        print(
            "native holonomic robot gate:"
            f" mean_frequencies={mean_frequencies.tolist()}Hz,"
            f" max_tracking={math.degrees(max(tracking_errors)):.9f}deg,"
            f" max_velocity_error={max(velocity_errors):.6e}rad/s,"
            f" max_sync={math.degrees(max(sync_errors)):.9f}deg,"
            f" solver={debug_state}"
        )
        assert np.isfinite(tracking_errors).all()
        assert np.isfinite(velocity_errors).all()
        assert np.max(np.abs(mean_frequencies - target_frequency_hz.cpu().numpy())) < 1.0e-4
        assert max(tracking_errors) < math.radians(0.1)
        assert max(sync_errors) < math.radians(0.1)
        assert int(debug_state["solver_prep_count"]) > total_steps * cfg.scene.num_envs

        env.reset()
        transition_tracking_errors: list[float] = []
        transition_sync_errors: list[float] = []
        maximum_phase_increment = 2.0 * math.pi * cfg.max_flap_hz * cfg.sim.dt
        previous_phase = env._ideal_inverse_phase_state.phase_rad.detach().clone()
        for target_hz in (0.0, 2.0, 5.0, 3.0, 0.0):
            transition_actions = torch.zeros((2, 4), device=env.device)
            transition_actions[:, 0] = 2.0 * target_hz / cfg.max_flap_hz - 1.0
            env._pre_physics_step(transition_actions)
            segment_steps = int(round(0.20 / cfg.sim.dt))
            for _ in range(segment_steps):
                env._sim_step_counter += 1
                env._apply_action()
                current_phase = env._ideal_inverse_phase_state.phase_rad.detach().clone()
                phase_increment = current_phase - previous_phase
                assert bool(torch.all(phase_increment >= -1.0e-7))
                assert bool(torch.all(phase_increment <= maximum_phase_increment + 1.0e-6))
                previous_phase = current_phase
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)

                left_position = env._robot.data.joint_pos[:, left_joint_id] - float(
                    env._wing_mid_L
                )
                right_position = -(
                    env._robot.data.joint_pos[:, right_joint_id] - float(env._wing_mid_R)
                )
                position_reference = env._wing_amp * torch.sin(current_phase)
                transition_tracking_errors.append(
                    float(torch.max(torch.abs(left_position - position_reference)).item())
                )
                transition_sync_errors.append(
                    float(torch.max(torch.abs(left_position - right_position)).item())
                )

        print(
            "native holonomic transition gate:"
            " schedule=[0,2,5,3,0]Hz,"
            f" max_tracking={math.degrees(max(transition_tracking_errors)):.9f}deg,"
            f" max_sync={math.degrees(max(transition_sync_errors)):.9f}deg"
        )
        assert np.isfinite(transition_tracking_errors).all()
        assert np.isfinite(transition_sync_errors).all()
        assert max(transition_tracking_errors) < math.radians(0.1)
        assert max(transition_sync_errors) < math.radians(0.1)
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()

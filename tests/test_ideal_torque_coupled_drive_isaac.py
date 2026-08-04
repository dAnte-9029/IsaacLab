"""Headless kinematic-consistency gate for the ideal torque wing drive."""

from __future__ import annotations

from _delaurier_isaac_app import simulation_app

import math

import numpy as np
import pytest
import torch

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledEnvCfg,
)


@pytest.mark.isaacsim_ci
def test_ideal_torque_drive_has_consistent_position_and_velocity() -> None:
    cfg = FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    cfg.sim.dt = 1.0 / 480.0
    cfg.decimation = 4
    cfg.robot.spawn.rigid_props.disable_gravity = True
    cfg.robot.spawn.articulation_props.fix_root_link = True
    cfg.enable_wing_aero = False
    cfg.enable_tail_aero = False
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 0.0
    cfg.reset_flap_hz = 5.0
    cfg.min_flap_hz = 2.0
    cfg.max_flap_hz = 5.0
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        actions = torch.zeros((1, 4), device=env.device)
        actions[:, 0] = 1.0
        env._pre_physics_step(actions)
        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        positions: list[float] = []
        velocities: list[float] = []
        sync_errors: list[float] = []
        tracking_errors: list[float] = []
        torques: list[float] = []
        for step in range(6 * 96):
            env._sim_step_counter += 1
            env._apply_action()
            reference_position = float(env._q_cmd[0].item())
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            if step < 2 * 96:
                continue
            q_left = float(env._robot.data.joint_pos[0, left_joint_id].item()) - float(env._wing_mid_L)
            q_right = -(
                float(env._robot.data.joint_pos[0, right_joint_id].item()) - float(env._wing_mid_R)
            )
            positions.append(0.5 * (q_left + q_right))
            velocities.append(
                0.5
                * float(
                    (
                        env._robot.data.joint_vel[0, left_joint_id]
                        - env._robot.data.joint_vel[0, right_joint_id]
                    ).item()
                )
            )
            sync_errors.append(abs(q_left - q_right))
            tracking_errors.append(abs(positions[-1] - reference_position))
            torques.append(abs(float(env._robot.data.applied_torque[0, left_joint_id].item())))

        position = np.asarray(positions)
        velocity = np.asarray(velocities)
        finite_difference_velocity = (position[2:] - position[:-2]) / (2.0 * cfg.sim.dt)
        velocity_error = velocity[1:-1] - finite_difference_velocity
        relative_velocity_rms_error = float(
            np.sqrt(np.mean(velocity_error**2))
            / max(np.sqrt(np.mean(finite_difference_velocity**2)), 1.0e-12)
        )
        print(
            "ideal torque drive:"
            f" max_tracking={math.degrees(max(tracking_errors)):.6f}deg,"
            f" max_sync={math.degrees(max(sync_errors)):.6f}deg,"
            f" q_amp={0.5 * (position.max() - position.min()):.6f}rad,"
            f" qd_amp={0.5 * (velocity.max() - velocity.min()):.6f}rad/s,"
            f" velocity_rms_error={100.0 * relative_velocity_rms_error:.3f}%,"
            f" max_torque={max(torques):.6f}Nm"
        )
        assert np.isfinite(position).all()
        assert np.isfinite(velocity).all()
        assert max(sync_errors) < math.radians(0.1)
        assert max(tracking_errors) < math.radians(5.0)
        assert 0.5 * (velocity.max() - velocity.min()) > 10.0
        assert relative_velocity_rms_error < 0.05
    finally:
        env.close()

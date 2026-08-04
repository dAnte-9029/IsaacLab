"""Headless fixed-root gate for the nonlinear sinusoidal phase mechanism."""

from __future__ import annotations

from _delaurier_isaac_app import simulation_app

import math

import numpy as np
import pytest
import torch

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledEnvCfg,
)


@pytest.mark.isaacsim_ci
def test_sinusoidal_phase_drive_has_consistent_physx_joint_state() -> None:
    cfg = FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    cfg.sim.dt = 1.0 / 480.0
    cfg.decimation = 1
    cfg.robot.spawn.rigid_props.disable_gravity = True
    cfg.robot.spawn.articulation_props.fix_root_link = True
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 0.0
    cfg.reset_flap_hz = 0.0
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        total_steps = int(round(3.0 / cfg.sim.dt))
        discard_steps = int(round(1.0 / cfg.sim.dt))
        for target_frequency_hz in (2.0, 5.0):
            actions = torch.zeros((1, 4), device=env.device)
            actions[:, 0] = (
                2.0 * target_frequency_hz / cfg.max_flap_hz - 1.0
            )
            env._pre_physics_step(actions)
            positions: list[float] = []
            velocities: list[float] = []
            references: list[float] = []
            frequencies: list[float] = []
            sync_errors: list[float] = []
            efforts: list[float] = []
            for step in range(total_steps):
                env._sim_step_counter += 1
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
                if step < discard_steps:
                    continue
                q_left = (
                    float(env._robot.data.joint_pos[0, left_joint_id].item())
                    - float(env._wing_mid_L)
                )
                q_right = -(
                    float(env._robot.data.joint_pos[0, right_joint_id].item())
                    - float(env._wing_mid_R)
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
                references.append(
                    float(
                        (
                            env._wing_amp
                            * torch.sin(
                                env._sinusoidal_phase_drive_state.phase_rad[0]
                            )
                        ).item()
                    )
                )
                frequencies.append(float(env._freq[0].item()))
                sync_errors.append(abs(q_left - q_right))
                efforts.append(
                    abs(
                        float(
                            env._robot.data.applied_torque[
                                0,
                                left_joint_id,
                            ].item()
                        )
                    )
                )

            position = np.asarray(positions)
            velocity = np.asarray(velocities)
            reference = np.asarray(references)
            frequency = np.asarray(frequencies)
            finite_difference_velocity = (position[2:] - position[:-2]) / (
                2.0 * cfg.sim.dt
            )
            velocity_error = velocity[1:-1] - finite_difference_velocity
            relative_velocity_rms_error = float(
                np.sqrt(np.mean(velocity_error**2))
                / max(
                    np.sqrt(np.mean(finite_difference_velocity**2)),
                    1.0e-12,
                )
            )
            max_tracking_error = float(np.max(np.abs(position - reference)))
            mean_frequency = float(np.mean(frequency))
            print(
                "sinusoidal phase drive:"
                f" target_frequency={target_frequency_hz:.6f}Hz,"
                f" mean_frequency={mean_frequency:.6f}Hz,"
                f" final_frequency={frequency[-1]:.6f}Hz,"
                f" min_frequency={frequency.min():.6f}Hz,"
                f" max_frequency={frequency.max():.6f}Hz,"
                f" max_tracking={math.degrees(max_tracking_error):.6f}deg,"
                f" max_sync={math.degrees(max(sync_errors)):.6f}deg,"
                f" velocity_rms_error={100.0 * relative_velocity_rms_error:.3f}%,"
                f" max_effort={max(efforts):.6f}Nm"
            )
            assert np.isfinite(position).all()
            assert np.isfinite(velocity).all()
            assert np.isfinite(frequency).all()
            assert max(sync_errors) < math.radians(0.1)
            assert max_tracking_error < math.radians(0.5)
            assert relative_velocity_rms_error < 0.05
            assert (
                abs(mean_frequency - target_frequency_hz) / target_frequency_hz
                < 0.02
            )
            assert max(efforts) <= cfg.sinusoidal_effort_limit_nm + 1.0e-5
    finally:
        env.close()

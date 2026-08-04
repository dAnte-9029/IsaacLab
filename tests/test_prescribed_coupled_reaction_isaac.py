"""Headless inertial-reaction gate for the prescribed wing mechanism."""

from __future__ import annotations

"""Launch Isaac Sim before importing Isaac Lab simulation modules."""

from _delaurier_isaac_app import simulation_app

"""The remaining imports require a running Isaac Sim application."""

import math

import pytest
import torch

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledEnvCfg,
)


@pytest.mark.isaacsim_ci
def test_prescribed_mechanism_transmits_wing_inertia_to_free_base() -> None:
    """Require prescribed massive-wing motion to react on the floating body."""

    cfg = FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    cfg.robot.spawn.rigid_props.disable_gravity = True
    cfg.enable_wing_aero = False
    cfg.enable_tail_aero = False
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 0.0
    cfg.reset_flap_hz = 2.0
    cfg.min_flap_hz = 2.0
    cfg.max_flap_hz = 5.0
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        base_body_id = int(env._base_body_ids[0])
        max_vertical_speed_m_s = 0.0
        max_pitch_rate_rad_s = 0.0
        max_wing_acceleration_rad_s2 = 0.0
        for _ in range(60):
            env.step(torch.zeros((1, 4), device=env.device))
            max_vertical_speed_m_s = max(
                max_vertical_speed_m_s,
                abs(float(env._robot.data.body_link_lin_vel_w[0, base_body_id, 2].item())),
            )
            max_pitch_rate_rad_s = max(
                max_pitch_rate_rad_s,
                abs(float(env._robot.data.body_link_ang_vel_w[0, base_body_id, 1].item())),
            )
            left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
            right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
            physical_acceleration = 0.5 * (
                env._robot.data.joint_acc[0, left_joint_id]
                - env._robot.data.joint_acc[0, right_joint_id]
            )
            max_wing_acceleration_rad_s2 = max(
                max_wing_acceleration_rad_s2,
                abs(float(physical_acceleration.item())),
            )

        print(
            "prescribed inertial reaction:"
            f" max_base_vz={max_vertical_speed_m_s:.9f}m/s,"
            f" max_base_pitch_rate={max_pitch_rate_rad_s:.9f}rad/s,"
            f" max_physx_qdd={max_wing_acceleration_rad_s2:.9f}rad/s^2,"
            f" expected_qdd_amplitude={math.radians(30.0) * (4.0 * math.pi) ** 2:.9f}rad/s^2"
        )
        assert max_wing_acceleration_rad_s2 > 1.0
        assert max_vertical_speed_m_s > 1.0e-3
        assert max_pitch_rate_rad_s > 1.0e-3
    finally:
        env.close()

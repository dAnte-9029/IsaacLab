"""Headless smoke test for the prescribed coupled-wing mechanism."""

from __future__ import annotations

"""Launch Isaac Sim before importing Isaac Lab simulation modules."""

from _delaurier_isaac_app import simulation_app

"""The remaining imports require a running Isaac Sim application."""

import math

import pytest
import torch

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg,
)
from flapping_bot.physics import PRESCRIBED_PER_WING_LINK


@pytest.mark.isaacsim_ci
def test_prescribed_mechanism_initializes_and_moves_passive_wings() -> None:
    """Exercise the prescribed constraints with per-wing aerodynamic loading."""

    cfg = FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    cfg.robot.spawn.rigid_props.disable_gravity = True
    cfg.robot.spawn.articulation_props.fix_root_link = True
    cfg.enable_wing_aero = True
    cfg.enable_tail_aero = False
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 8.0
    cfg.reset_flap_hz = 2.0
    cfg.min_flap_hz = 2.0
    cfg.max_flap_hz = 5.0
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        assert "passive_wings" in env._robot.actuators
        assert "wing_driver" not in env._robot.actuators
        assert env.cfg.wing_aero_coupling_mode == PRESCRIBED_PER_WING_LINK

        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        max_tracking_error = 0.0
        max_sync_error = 0.0
        for _ in range(30):
            env.step(torch.zeros((1, 4), device=env.device))
            q_left = env._robot.data.joint_pos[0, left_joint_id]
            q_right = env._robot.data.joint_pos[0, right_joint_id]
            max_tracking_error = max(
                max_tracking_error,
                float(torch.abs(q_left - env._q_cmd[0]).item()),
            )
            max_sync_error = max(
                max_sync_error,
                float(torch.abs(q_left + q_right).item()),
            )
            assert torch.isfinite(q_left)
            assert torch.isfinite(q_right)

        print(
            "prescribed moving-limit smoke:"
            f" max_tracking={math.degrees(max_tracking_error):.6f}deg,"
            f" max_sync={math.degrees(max_sync_error):.6f}deg"
        )
        assert max_tracking_error < math.radians(5.0)
        assert max_sync_error < math.radians(0.1)
        assert torch.isfinite(env._debug_last_wing_force_link_n).all()
        assert torch.linalg.vector_norm(env._debug_last_wing_force_link_n) > 0.0
    finally:
        env.close()

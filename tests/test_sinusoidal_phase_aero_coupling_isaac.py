"""Headless wiring gate for sinusoidal phase per-wing aerodynamics."""

from __future__ import annotations

from _delaurier_isaac_app import simulation_app

import pytest
import torch

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg,
)
from flapping_bot.physics import SINUSOIDAL_PHASE_PER_WING_LINK


@pytest.mark.isaacsim_ci
def test_sinusoidal_phase_aero_writes_actual_state_wrenches_to_wing_links() -> None:
    cfg = (
        FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg()
    )
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    cfg.sim.dt = 1.0 / 480.0
    cfg.decimation = 1
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.reset_flap_hz = 2.0
    cfg.reset_forward_speed_mps = 0.0
    cfg.freeze_steps_after_reset = 0
    cfg.robot = cfg.robot.replace(
        spawn=cfg.robot.spawn.replace(
            rigid_props=cfg.robot.spawn.rigid_props.replace(
                disable_gravity=True,
                retain_accelerations=False,
            ),
            articulation_props=cfg.robot.spawn.articulation_props.replace(
                fix_root_link=True,
            ),
        ),
    )
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        env._wind_w[:, 0] = -8.0
        env._wind_mean_w.copy_(env._wind_w)
        actions = torch.zeros((1, 4), device=env.device)
        actions[:, 0] = 2.0 * 2.0 / cfg.max_flap_hz - 1.0
        env._pre_physics_step(actions)
        env._sim_step_counter += 1
        env._apply_action()

        base_body_id = int(env._base_body_ids[0])
        wing_body_ids = [int(index) for index in env._wing_body_ids]
        assert env.cfg.wing_aero_coupling_mode == SINUSOIDAL_PHASE_PER_WING_LINK
        torch.testing.assert_close(
            env._robot._external_force_b[:, base_body_id],
            torch.zeros((1, 3), device=env.device),
        )
        torch.testing.assert_close(
            env._robot._external_torque_b[:, base_body_id],
            torch.zeros((1, 3), device=env.device),
        )
        torch.testing.assert_close(
            env._robot._external_force_b[:, wing_body_ids],
            env._debug_last_wing_force_link_n,
        )
        torch.testing.assert_close(
            env._robot._external_torque_b[:, wing_body_ids],
            env._debug_last_wing_moment_link_about_com_nm,
        )
        torch.testing.assert_close(
            env._debug_last_wing_aero_position_rad,
            torch.zeros_like(env._debug_last_wing_aero_position_rad),
            atol=1.0e-6,
            rtol=0.0,
        )
        expected_velocity = env._wing_amp * 2.0 * torch.pi * 2.0
        torch.testing.assert_close(
            env._debug_last_wing_aero_velocity_rad_s,
            torch.full_like(
                env._debug_last_wing_aero_velocity_rad_s,
                expected_velocity,
            ),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        assert torch.all(torch.isfinite(env._debug_last_wing_aero_acceleration_rad_s2))
        assert torch.linalg.vector_norm(env._debug_last_wing_force_link_n) > 1.0e-4
    finally:
        env.close()

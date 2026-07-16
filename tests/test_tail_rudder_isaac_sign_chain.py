"""Headless action-to-rudder-to-PhysX yaw sign contract."""

from __future__ import annotations

"""Launch Isaac Sim before importing environment modules."""

from _delaurier_isaac_app import simulation_app

"""The remaining imports require a running Isaac Sim application."""

import math

import pytest
import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightEnvCfg,
)
from flapping_bot.px4_like.straight_line_controller import PX4LikeStraightLineControllerCfg


@pytest.mark.isaacsim_ci
def test_rudder_action_sign_reaches_tail_wrench_and_physx_yaw_rate() -> None:
    """Check mirrored pulses and a short legacy/no-P/corrected-P comparison."""

    cfg = FlappingBotStraightFlightEnvCfg()
    cfg.seed = 0
    cfg.scene.num_envs = 6
    cfg.scene.env_spacing = 3.0
    cfg.sim.device = "cpu"
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.enable_wing_aero = False
    cfg.enable_tail_aero = True
    cfg.freeze_steps_after_reset = 0
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    cfg.teacher_guidance_enabled = False
    cfg.wind_enabled = False
    cfg.randomize_wind = False
    cfg.reset_pitch_deg = 0.0
    cfg.reset_elevon_pitch_deg = 0.0
    cfg.reset_elevon_roll_deg = 0.0
    cfg.reset_rudder_deg = 0.0
    cfg.reset_forward_speed_mps = 8.0

    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        actions = torch.zeros((6, 4), device=env.device)
        actions[:3, 1] = torch.tensor([0.0, 0.1, -0.1], device=env.device)

        # Environments 3:6 start with the same +5 deg yaw-minus-air-heading
        # error. They compare the legacy positive P sign, no P, and the
        # production-corrected negative P sign.
        ab_env_ids = torch.arange(3, 6, device=env.device)
        ab_root_state = env._robot.data.root_state_w[ab_env_ids].clone()
        yaw_initial = torch.full((3,), math.radians(5.0), device=env.device)
        zero_angle = torch.zeros_like(yaw_initial)
        ab_root_state[:, 3:7] = quat_from_euler_xyz(zero_angle, zero_angle, yaw_initial)
        ab_root_state[:, 7:10] = torch.tensor([8.0, 0.0, 0.0], device=env.device)
        ab_root_state[:, 10:13] = 0.0
        env._robot.write_root_state_to_sim(ab_root_state, env_ids=ab_env_ids)
        env.sim.forward()
        env._robot.update(env.physics_dt)

        def alignment_error() -> torch.Tensor:
            _, _, yaw = euler_xyz_from_quat(env._robot.data.root_quat_w)
            heading = torch.atan2(env._robot.data.root_lin_vel_w[:, 1], env._robot.data.root_lin_vel_w[:, 0])
            return torch.atan2(torch.sin(yaw - heading), torch.cos(yaw - heading))

        initial_alignment_error = alignment_error()[ab_env_ids].detach().cpu()
        controller_cfg = PX4LikeStraightLineControllerCfg()
        yaw_kp = float(controller_cfg.yaw_kp)
        yaw_kd = float(controller_cfg.yaw_kd)

        for _ in range(12):
            error = alignment_error()
            yaw_rate = env._robot.data.root_ang_vel_b[:, 2]
            actions[3, 1] = yaw_kp * error[3] - yaw_kd * yaw_rate[3]
            actions[4, 1] = -yaw_kd * yaw_rate[4]
            actions[5, 1] = -yaw_kp * error[5] - yaw_kd * yaw_rate[5]
            env.step(actions)

        executed_rudder = env._debug_last_exec_rudder_rad[:3].detach().cpu()
        tail_force_b = env._debug_last_tail_force_b[:3].detach().cpu()
        tail_moment_b = env._debug_last_torque_b[:3].detach().cpu()
        yaw_rate_b = env._robot.data.root_ang_vel_b[:3, 2].detach().cpu()
        controlled_joint_position = env._robot.data.joint_pos[:3, env._joint_ids].detach().cpu()
        rudder_joint_position = controlled_joint_position[:, env._IDX_RUDDER]

        expected_rudder = math.radians(cfg.rudder_max_deg) * torch.tensor([0.0, 0.1, -0.1])
        torch.testing.assert_close(executed_rudder, expected_rudder, atol=1.0e-6, rtol=0.0)
        assert abs(float(rudder_joint_position[0])) < 2.0e-5
        assert rudder_joint_position[1] > 0.0 > rudder_joint_position[2]
        joint_tracking_ratio = torch.abs(rudder_joint_position[1:] / expected_rudder[1:])
        assert torch.all((joint_tracking_ratio > 0.8) & (joint_tracking_ratio < 1.2))

        delta_fy = tail_force_b[:, 1] - tail_force_b[0, 1]
        delta_mz = tail_moment_b[:, 2] - tail_moment_b[0, 2]
        delta_yaw_rate = yaw_rate_b - yaw_rate_b[0]
        assert delta_fy[1] < 0.0 < delta_fy[2]
        assert delta_mz[1] > 0.0 > delta_mz[2]
        assert delta_yaw_rate[1] > 0.0 > delta_yaw_rate[2]

        final_alignment_error = alignment_error()[ab_env_ids].detach().cpu()
        final_yaw_rate = env._robot.data.root_ang_vel_b[ab_env_ids, 2].detach().cpu()
        assert torch.all(final_alignment_error > 0.0)
        assert final_alignment_error[2] < final_alignment_error[1] < final_alignment_error[0]
        assert final_yaw_rate[2] < final_yaw_rate[1] < final_yaw_rate[0] < 0.0

        print(
            "[RUDDER_SIGN_CHAIN] "
            f"executed_rudder_deg={torch.rad2deg(executed_rudder).tolist()}; "
            f"joint_rudder_deg={torch.rad2deg(rudder_joint_position).tolist()}; "
            f"joint_tracking_ratio={joint_tracking_ratio.tolist()}; "
            f"tail_fy_N={tail_force_b[:, 1].tolist()}; "
            f"tail_mz_Nm={tail_moment_b[:, 2].tolist()}; "
            f"yaw_rate_rad_s={yaw_rate_b.tolist()}; "
            f"delta_yaw_rate_rad_s={delta_yaw_rate.tolist()}; "
            "ab_cases=[legacy_positive_P,no_P,production_corrected_P]; "
            f"ab_initial_error_deg={torch.rad2deg(initial_alignment_error).tolist()}; "
            f"ab_final_error_deg={torch.rad2deg(final_alignment_error).tolist()}; "
            f"ab_final_yaw_rate_rad_s={final_yaw_rate.tolist()}"
        )
    finally:
        env.close()

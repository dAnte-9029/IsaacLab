"""Headless integration test for actual-motion per-wing aerodynamic coupling."""

from __future__ import annotations

"""Launch Isaac Sim before importing Isaac Lab simulation modules."""

from _delaurier_isaac_app import simulation_app

"""The remaining imports require a running Isaac Sim application."""

import math

import pytest
import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg,
)
from flapping_bot.physics import ACTUAL_PER_WING_LINK


@pytest.mark.isaacsim_ci
def test_actual_per_wing_aero_uses_link_wrenches_without_base_duplication() -> None:
    cfg = FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.robot = cfg.robot.replace(
        spawn=cfg.robot.spawn.replace(
            rigid_props=cfg.robot.spawn.rigid_props.replace(disable_gravity=True),
        )
    )
    cfg.enable_tail_aero = False
    cfg.fuselage_drag_cda = 0.0
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 8.0
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        actions = torch.zeros((1, 4), device=env.device)
        env._pre_physics_step(actions)
        env._apply_action()

        assert env.cfg.wing_aero_coupling_mode == ACTUAL_PER_WING_LINK
        base_body_id = int(env._base_body_ids[0])
        wing_body_ids = [int(index) for index in env._wing_body_ids]
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
        assert float(torch.linalg.vector_norm(env._debug_last_wing_force_link_n).item()) > 1.0e-4

        wing_quat_w = env._robot.data.body_link_quat_w[:, wing_body_ids].reshape(-1, 4)
        wing_force_w = quat_apply(
            wing_quat_w,
            env._debug_last_wing_force_link_n.reshape(-1, 3),
        ).reshape(1, 2, 3)
        wing_moment_about_com_w = quat_apply(
            wing_quat_w,
            env._debug_last_wing_moment_link_about_com_nm.reshape(-1, 3),
        ).reshape(1, 2, 3)
        wing_com_pos_w = env._robot.data.body_com_pos_w[:, wing_body_ids]
        base_com_pos_w = env._robot.data.body_com_pos_w[:, base_body_id]
        net_force_w = wing_force_w.sum(dim=1)
        net_moment_about_base_com_w = (
            wing_moment_about_com_w
            + torch.linalg.cross(
                wing_com_pos_w - base_com_pos_w.unsqueeze(1),
                wing_force_w,
                dim=-1,
            )
        ).sum(dim=1)
        root_quat_w = env._robot.data.root_quat_w
        torch.testing.assert_close(
            quat_apply_inverse(root_quat_w, net_force_w),
            env._debug_last_wing_force_b,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            quat_apply_inverse(root_quat_w, net_moment_about_base_com_w),
            env._debug_last_torque_b,
            atol=2.0e-5,
            rtol=2.0e-5,
        )

        max_sync_error_rad = 0.0
        max_wing_force_n = 0.0
        max_wing_moment_nm = 0.0
        max_mean_acceleration_input_error_rad_s2 = 0.0
        steady_max_wing_force_n = 0.0
        steady_max_mean_acceleration_input_error_rad_s2 = 0.0
        acceleration_error_peak_step = -1
        acceleration_at_error_peak_rad_s2 = 0.0
        commanded_acceleration_at_error_peak_rad_s2 = 0.0
        termination_count = 0
        for step in range(120):
            _, _, terminated, truncated, _ = env.step(actions)
            termination_count += int(torch.count_nonzero(terminated | truncated).item())
            left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
            right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
            sync_error = (
                env._robot.data.joint_pos[:, left_joint_id]
                + env._robot.data.joint_pos[:, right_joint_id]
            )
            max_sync_error_rad = max(
                max_sync_error_rad,
                float(torch.max(torch.abs(sync_error)).item()),
            )
            max_wing_force_n = max(
                max_wing_force_n,
                float(torch.max(torch.linalg.vector_norm(env._debug_last_wing_force_link_n, dim=-1)).item()),
            )
            max_wing_moment_nm = max(
                max_wing_moment_nm,
                float(
                    torch.max(
                        torch.linalg.vector_norm(
                            env._debug_last_wing_moment_link_about_com_nm,
                            dim=-1,
                        )
                    ).item()
                ),
            )
            acceleration_error = float(
                env.extras["log"]["WingAero/mean_abs_acceleration_input_error_rad_s2"]
            )
            if acceleration_error > max_mean_acceleration_input_error_rad_s2:
                max_mean_acceleration_input_error_rad_s2 = acceleration_error
                acceleration_error_peak_step = step
                acceleration_at_error_peak_rad_s2 = float(
                    torch.max(torch.abs(env._debug_last_wing_aero_acceleration_rad_s2)).item()
                )
                commanded_acceleration_at_error_peak_rad_s2 = float(
                    torch.max(torch.abs(env._qdd_cmd)).item()
                )
            if step >= 60:
                steady_max_wing_force_n = max(
                    steady_max_wing_force_n,
                    float(
                        torch.max(
                            torch.linalg.vector_norm(
                                env._debug_last_wing_force_link_n,
                                dim=-1,
                            )
                        ).item()
                    ),
                )
                steady_max_mean_acceleration_input_error_rad_s2 = max(
                    steady_max_mean_acceleration_input_error_rad_s2,
                    acceleration_error,
                )
            assert torch.all(torch.isfinite(env._debug_last_wing_aero_position_rad))
            assert torch.all(torch.isfinite(env._debug_last_wing_aero_velocity_rad_s))
            assert torch.all(torch.isfinite(env._debug_last_wing_aero_acceleration_rad_s2))
            assert torch.all(torch.isfinite(env._debug_last_wing_force_link_n))
            assert torch.all(torch.isfinite(env._debug_last_wing_moment_link_about_com_nm))
        assert math.degrees(max_sync_error_rad) < 0.1
        assert max_wing_force_n > 1.0e-4
        assert "WingAero/mean_abs_position_input_error_rad" in env.extras["log"]
        print(
            "[MULTIBODY_WING_AERO]"
            f" max_sync_error_deg={math.degrees(max_sync_error_rad):.6f};"
            f" max_wing_force_n={max_wing_force_n:.6f};"
            f" max_wing_moment_about_com_nm={max_wing_moment_nm:.6f};"
            " max_mean_acceleration_input_error_rad_s2="
            f"{max_mean_acceleration_input_error_rad_s2:.6f};"
            f" acceleration_error_peak_step={acceleration_error_peak_step};"
            f" actual_abs_acceleration_at_peak_rad_s2={acceleration_at_error_peak_rad_s2:.6f};"
            " commanded_abs_acceleration_at_peak_rad_s2="
            f"{commanded_acceleration_at_error_peak_rad_s2:.6f};"
            f" steady_max_wing_force_n={steady_max_wing_force_n:.6f};"
            " steady_max_mean_acceleration_input_error_rad_s2="
            f"{steady_max_mean_acceleration_input_error_rad_s2:.6f};"
            f" termination_count={termination_count}"
        )
    finally:
        env.close()

"""Headless uncontrolled-flight smoke gate for the native holonomic plant."""

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
import pytest
import torch

from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg,
)


@pytest.mark.isaacsim_ci
def test_native_holonomic_uncontrolled_flight_remains_numerically_bounded() -> None:
    cfg = FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    cfg.sim.dt = 1.0 / 480.0
    cfg.decimation = 1
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 8.0
    cfg.reset_flap_hz = 4.0
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    cfg.randomize_commands = False
    cfg.enable_tail_aero = True
    cfg.wind_enabled = False
    cfg.wind_ou_enabled = False
    cfg.robot = cfg.robot.replace(
        spawn=cfg.robot.spawn.replace(
            rigid_props=cfg.robot.spawn.rigid_props.replace(
                disable_gravity=False,
                retain_accelerations=False,
            ),
            articulation_props=cfg.robot.spawn.articulation_props.replace(
                fix_root_link=False,
            ),
        ),
    )
    assert cfg.enable_tail_aero
    assert cfg.fuselage_drag_cda == 0.0
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        target_frequency_hz = 4.0
        actions = torch.zeros((1, 4), device=env.device)
        actions[:, 0] = 2.0 * target_frequency_hz / cfg.max_flap_hz - 1.0
        env._pre_physics_step(actions)
        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        max_tracking_error = 0.0
        max_sync_error = 0.0
        max_root_linear_speed = 0.0
        max_root_angular_speed = 0.0
        max_single_wing_force = 0.0
        max_root_velocity_step = 0.0
        previous_root_velocity = env._robot.data.root_vel_w.clone()

        for _ in range(int(round(1.0 / cfg.sim.dt))):
            env._sim_step_counter += 1
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

            finite_tensors = (
                env._robot.data.root_state_w,
                env._robot.data.joint_pos,
                env._robot.data.joint_vel,
                env._debug_last_wing_force_link_n,
                env._debug_last_wing_moment_link_about_com_nm,
            )
            assert all(bool(torch.all(torch.isfinite(value))) for value in finite_tensors)
            left_position = env._robot.data.joint_pos[:, left_joint_id] - float(env._wing_mid_L)
            right_position = -(
                env._robot.data.joint_pos[:, right_joint_id] - float(env._wing_mid_R)
            )
            reference = env._wing_amp * torch.sin(env._ideal_inverse_phase_state.phase_rad)
            max_tracking_error = max(
                max_tracking_error,
                float(torch.max(torch.abs(left_position - reference)).item()),
            )
            max_sync_error = max(
                max_sync_error,
                float(torch.max(torch.abs(left_position - right_position)).item()),
            )
            max_root_linear_speed = max(
                max_root_linear_speed,
                float(torch.linalg.vector_norm(env._robot.data.root_lin_vel_w, dim=1).max().item()),
            )
            max_root_angular_speed = max(
                max_root_angular_speed,
                float(torch.linalg.vector_norm(env._robot.data.root_ang_vel_w, dim=1).max().item()),
            )
            max_single_wing_force = max(
                max_single_wing_force,
                float(
                    torch.linalg.vector_norm(env._debug_last_wing_force_link_n, dim=-1)
                    .max()
                    .item()
                ),
            )
            max_root_velocity_step = max(
                max_root_velocity_step,
                float(
                    torch.linalg.vector_norm(
                        env._robot.data.root_vel_w - previous_root_velocity,
                        dim=1,
                    )
                    .max()
                    .item()
                ),
            )
            previous_root_velocity.copy_(env._robot.data.root_vel_w)

        print(
            "native holonomic uncontrolled-flight smoke:"
            f" max_tracking={math.degrees(max_tracking_error):.9f}deg,"
            f" max_sync={math.degrees(max_sync_error):.9f}deg,"
            f" max_root_linear_speed={max_root_linear_speed:.6f}m/s,"
            f" max_root_angular_speed={max_root_angular_speed:.6f}rad/s,"
            f" max_single_wing_force={max_single_wing_force:.6f}N,"
            f" max_root_velocity_step={max_root_velocity_step:.6f}"
        )
        assert max_tracking_error < math.radians(0.1)
        assert max_sync_error < math.radians(0.1)
        assert max_root_linear_speed < 100.0
        assert max_root_angular_speed < 1000.0
        assert max_single_wing_force < 100.0
        assert max_root_velocity_step < 100.0
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()

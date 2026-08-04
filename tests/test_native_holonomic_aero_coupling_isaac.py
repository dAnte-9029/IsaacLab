"""Headless aerodynamic-load gate for the native holonomic mechanism."""

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
from flapping_bot.physics import NATIVE_HOLONOMIC_PER_WING_LINK


@pytest.mark.isaacsim_ci
def test_native_holonomic_mechanism_rejects_per_wing_aerodynamic_loads() -> None:
    cfg = FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    cfg.sim.dt = 1.0 / 480.0
    cfg.decimation = 1
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.freeze_steps_after_reset = 0
    cfg.reset_forward_speed_mps = 0.0
    cfg.reset_flap_hz = 0.0
    cfg.act_lpf_tau_s = 0.0
    cfg.act_rate_limit_per_s = 0.0
    cfg.wind_enabled = True
    cfg.wind_xy_mps = (-8.0, 0.0)
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
        assert env.cfg.wing_aero_coupling_mode == NATIVE_HOLONOMIC_PER_WING_LINK
        assert int(env._native_holonomic.get_active_joint_count()) == 1
        target_frequency_hz = 5.0
        actions = torch.zeros((1, 4), device=env.device)
        actions[:, 0] = 2.0 * target_frequency_hz / cfg.max_flap_hz - 1.0
        env._pre_physics_step(actions)

        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        max_tracking_error = 0.0
        max_sync_error = 0.0
        max_wing_force = 0.0
        total_steps = int(round(0.75 / cfg.sim.dt))
        discard_steps = int(round(0.25 / cfg.sim.dt))
        for step in range(total_steps):
            env._sim_step_counter += 1
            env._apply_action()
            max_wing_force = max(
                max_wing_force,
                float(torch.linalg.vector_norm(env._debug_last_wing_force_link_n, dim=-1).max()),
            )
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            if step < discard_steps:
                continue

            left_position = env._robot.data.joint_pos[:, left_joint_id] - float(env._wing_mid_L)
            right_position = -(
                env._robot.data.joint_pos[:, right_joint_id] - float(env._wing_mid_R)
            )
            reference = env._wing_amp * torch.sin(env._ideal_inverse_phase_state.phase_rad)
            max_tracking_error = max(
                max_tracking_error,
                float(torch.max(torch.abs(left_position - reference))),
            )
            max_sync_error = max(
                max_sync_error,
                float(torch.max(torch.abs(left_position - right_position))),
            )

        print(
            "native holonomic aero gate:"
            f" max_tracking={math.degrees(max_tracking_error):.9f}deg,"
            f" max_sync={math.degrees(max_sync_error):.9f}deg,"
            f" max_wing_force={max_wing_force:.6f}N"
        )
        assert max_wing_force > 1.0e-4
        assert max_tracking_error < math.radians(0.1)
        assert max_sync_error < math.radians(0.1)
        assert torch.all(torch.isfinite(env._debug_last_wing_force_link_n))
        assert torch.all(torch.isfinite(env._debug_last_wing_moment_link_about_com_nm))
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()

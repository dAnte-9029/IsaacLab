"""Headless integration test for the formal ideal coupled-wing asset adapter."""

from __future__ import annotations

"""Launch Isaac Sim before importing Isaac Lab simulation modules."""

from _delaurier_isaac_app import simulation_app

"""The remaining imports require a running Isaac Sim application."""

import math

import pytest
from pxr import PhysxSchema, UsdPhysics
import torch

from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context

from flapping_bot.assets import IdealCoupledFlappingBotCfg, apply_hard_opposed_wing_mimic
from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg,
)
from flapping_bot.physics import build_measured_wing_multibody_tensors


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    ("frequency_hz", "dt_s"),
    [
        (2.0, 1.0 / 240.0),
        (5.0, 1.0 / 240.0),
    ],
)
def test_formal_asset_tracks_one_driver_and_preserves_hard_opposed_coupling(
    frequency_hz: float,
    dt_s: float,
) -> None:
    """Run the real robot asset with measured masses and no per-step state writes."""

    with build_simulation_context(
        device="cpu",
        dt=dt_s,
        auto_add_lighting=False,
        gravity_enabled=False,
    ) as sim:
        sim._app_control_on_stop_handle = None
        robot = Articulation(
            IdealCoupledFlappingBotCfg.replace(
                prim_path="/World/IdealCoupledFlappingBot",
                spawn=IdealCoupledFlappingBotCfg.spawn.replace(
                    rigid_props=IdealCoupledFlappingBotCfg.spawn.rigid_props.replace(disable_gravity=True),
                ),
            )
        )
        right_joint_path = apply_hard_opposed_wing_mimic(
            robot.stage,
            articulation_root_path="/World/IdealCoupledFlappingBot",
        )
        right_joint_prim = robot.stage.GetPrimAtPath(right_joint_path)
        mimic_api = PhysxSchema.PhysxMimicJointAPI(right_joint_prim, UsdPhysics.Tokens.rotX)
        assert mimic_api.GetGearingAttr().Get() == pytest.approx(1.0)
        assert mimic_api.GetOffsetAttr().Get() == pytest.approx(0.0)
        assert str(mimic_api.GetReferenceJointRel().GetTargets()[0]).endswith("/left_wing")

        sim.reset()
        robot.update(dt_s)
        env_ids = torch.arange(robot.num_instances, device="cpu", dtype=torch.int64)
        masses, inertias, coms = build_measured_wing_multibody_tensors(
            body_names=robot.body_names,
            masses_kg=robot.root_physx_view.get_masses().clone(),
            inertias_kg_m2=robot.root_physx_view.get_inertias().clone(),
            com_poses_link=robot.root_physx_view.get_coms().clone(),
        )
        robot.root_physx_view.set_masses(masses, env_ids)
        robot.root_physx_view.set_inertias(inertias, env_ids)
        robot.root_physx_view.set_coms(coms, env_ids)

        left_joint_id, right_joint_id = (
            int(index)
            for index in robot.find_joints(["left_wing", "right_wing"], preserve_order=True)[0]
        )
        base_body_id = int(robot.find_bodies(["base_link"], preserve_order=True)[0][0])
        robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
        robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_state[:, 7:]))
        robot.write_joint_state_to_sim(
            torch.zeros_like(robot.data.default_joint_pos),
            torch.zeros_like(robot.data.default_joint_vel),
        )
        robot.reset()
        sim.forward()

        steps_per_cycle = round(1.0 / (frequency_hz * dt_s))
        total_steps = 4 * steps_per_cycle
        measurement_start = 2 * steps_per_cycle
        sync_errors: list[float] = []
        tracking_errors: list[float] = []
        base_vertical_speeds: list[float] = []
        base_pitch_rates: list[float] = []
        applied_driver_torques: list[float] = []

        for step in range(total_steps):
            # The target is consumed by the following simulation step.
            phase_rad = 2.0 * math.pi * frequency_hz * (step + 1) * dt_s
            q_ref = math.radians(30.0) * math.sin(phase_rad)
            qd_ref = math.radians(30.0) * 2.0 * math.pi * frequency_hz * math.cos(phase_rad)
            robot.set_joint_position_target(
                torch.tensor([[q_ref]], device=robot.device),
                joint_ids=[left_joint_id],
            )
            robot.set_joint_velocity_target(
                torch.tensor([[qd_ref]], device=robot.device),
                joint_ids=[left_joint_id],
            )
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt_s)

            if step >= measurement_start:
                q_left = float(robot.data.joint_pos[0, left_joint_id].item())
                q_right = float(robot.data.joint_pos[0, right_joint_id].item())
                sync_errors.append(q_left + q_right)
                tracking_errors.append(q_left - q_ref)
                base_vertical_speeds.append(float(robot.data.body_link_lin_vel_w[0, base_body_id, 2].item()))
                base_pitch_rates.append(float(robot.data.body_link_ang_vel_w[0, base_body_id, 1].item()))
                applied_driver_torques.append(float(robot.data.applied_torque[0, left_joint_id].item()))

        max_sync_error = max(abs(value) for value in sync_errors)
        max_tracking_error = max(abs(value) for value in tracking_errors)
        max_driver_torque = max(abs(value) for value in applied_driver_torques)
        max_vertical_speed = max(abs(value) for value in base_vertical_speeds)
        max_pitch_rate = max(abs(value) for value in base_pitch_rates)
        print(
            f"frequency={frequency_hz:.1f}Hz,"
            f" max_sync={math.degrees(max_sync_error):.6f}deg,"
            f" max_tracking={math.degrees(max_tracking_error):.6f}deg,"
            f" max_driver_torque={max_driver_torque:.6f}Nm,"
            f" max_vertical_speed={max_vertical_speed:.6f}m/s,"
            f" max_pitch_rate={max_pitch_rate:.6f}rad/s"
        )
        assert max_sync_error < math.radians(0.1)
        assert max_tracking_error < math.radians(0.5)
        assert max_driver_torque < 999.0
        assert max_driver_torque > 1.0e-3
        assert max_vertical_speed > 1.0e-3
        assert max_pitch_rate > 1.0e-3


@pytest.mark.isaacsim_ci
def test_formal_environment_initializes_measured_ideal_coupled_variant() -> None:
    """Exercise the actual scene setup, schema adapter and plant dispatch."""

    cfg = FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cpu"
    cfg.enable_wing_aero = False
    cfg.enable_tail_aero = False
    cfg.freeze_steps_after_reset = 0
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        torch.testing.assert_close(
            env._mass_total,
            torch.tensor([0.90415], device=env.device),
            atol=1.0e-7,
            rtol=0.0,
        )
        assert env.cfg.use_kinematic_joint_override is False
        assert "wing_driver" in env._robot.actuators
        assert "passive_right_wing" in env._robot.actuators

        # One environment step exercises the formal target path. Synchronization
        # accuracy is covered by the clean 2 Hz and 5 Hz dynamics cases above;
        # simulation-context prototypes are not fully removed between cases.
        env.step(torch.zeros((1, 4), device=env.device))
        q_left = env._robot.data.joint_pos[0, int(env._joint_ids[env._IDX_LEFT_WING])]
        q_right = env._robot.data.joint_pos[0, int(env._joint_ids[env._IDX_RIGHT_WING])]
        assert torch.isfinite(q_left)
        assert torch.isfinite(q_right)
    finally:
        env.close()

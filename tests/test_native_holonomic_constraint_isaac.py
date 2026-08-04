"""Headless smoke gate for the native PhysX wing trajectory constraint."""

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

import omni.kit.app
import omni.physx
import omni.physics.tensors as physx
import omni.usd
import pytest
import torch
from omni.physx.scripts import physicsUtils
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

from isaaclab.sim import SimulationCfg, SimulationContext


def _enable_extension() -> tuple[object, object]:
    manager = omni.kit.app.get_app().get_extension_manager()
    assert manager.is_extension_enabled("omni.flapping_bot.holonomic_constraint")
    from omni.flapping_bot.holonomic_constraint import _native

    return manager, _native


@pytest.mark.isaacsim_ci
def test_native_constraint_tracks_and_preserves_free_pair_momentum() -> None:
    manager, native = _enable_extension()
    sim = SimulationContext(SimulationCfg(dt=1.0 / 240.0, device="cpu", gravity=(0.0, 0.0, 0.0)))
    try:
        stage = omni.usd.get_context().get_stage()

        body0 = physicsUtils.add_rigid_box(stage, "/World/body0", size=Gf.Vec3f(0.1))
        body1 = physicsUtils.add_rigid_box(stage, "/World/body1", size=Gf.Vec3f(0.1))
        for body in (body0, body1):
            UsdPhysics.MassAPI.Apply(body).CreateMassAttr(1.0)
            PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateAngularDampingAttr(0.0)

        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/fixed_body0")
        fixed.CreateBody1Rel().SetTargets([body0.GetPath()])

        revolute = UsdPhysics.RevoluteJoint.Define(stage, "/World/revolute")
        revolute.CreateBody0Rel().SetTargets([body0.GetPath()])
        revolute.CreateBody1Rel().SetTargets([body1.GetPath()])
        revolute.CreateAxisAttr(UsdGeom.Tokens.x)

        trajectory_path = "/World/wing_trajectory"
        trajectory_prim = stage.DefinePrim(trajectory_path, native.get_joint_type_name())
        assert trajectory_prim.IsA(UsdPhysics.Joint)
        trajectory = UsdPhysics.Joint(trajectory_prim)
        trajectory.CreateBody0Rel().SetTargets([body0.GetPath()])
        trajectory.CreateBody1Rel().SetTargets([body1.GetPath()])
        trajectory.CreateExcludeFromArticulationAttr(True)
        trajectory.GetPrim().CreateAttribute("flappingBot:constraintType", Sdf.ValueTypeNames.String).Set(
            "wingTrajectory"
        )

        free_body0 = physicsUtils.add_rigid_box(
            stage,
            "/World/free_body0",
            size=Gf.Vec3f(0.1),
            position=Gf.Vec3f(0.0, 1.0, 0.0),
        )
        free_body1 = physicsUtils.add_rigid_box(
            stage,
            "/World/free_body1",
            size=Gf.Vec3f(0.1),
            position=Gf.Vec3f(0.0, 1.0, 0.0),
        )
        for body in (free_body0, free_body1):
            UsdPhysics.MassAPI.Apply(body).CreateMassAttr(1.0)
            PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateAngularDampingAttr(0.0)
            PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateLinearDampingAttr(0.0)

        free_revolute = UsdPhysics.RevoluteJoint.Define(stage, "/World/free_revolute")
        free_revolute.CreateBody0Rel().SetTargets([free_body0.GetPath()])
        free_revolute.CreateBody1Rel().SetTargets([free_body1.GetPath()])
        free_revolute.CreateAxisAttr(UsdGeom.Tokens.x)

        free_trajectory_path = "/World/free_wing_trajectory"
        free_trajectory_prim = stage.DefinePrim(
            free_trajectory_path,
            native.get_joint_type_name(),
        )
        free_trajectory = UsdPhysics.Joint(free_trajectory_prim)
        free_trajectory.CreateBody0Rel().SetTargets([free_body0.GetPath()])
        free_trajectory.CreateBody1Rel().SetTargets([free_body1.GetPath()])
        free_trajectory.CreateExcludeFromArticulationAttr(True)
        free_trajectory.GetPrim().CreateAttribute(
            "flappingBot:constraintType",
            Sdf.ValueTypeNames.String,
        ).Set("wingTrajectory")

        target_rad = math.radians(20.0)
        native.set_targets(
            [trajectory_path, free_trajectory_path],
            [target_rad, 0.0],
            [0.0, 0.0],
        )
        sim.reset()
        assert native.get_active_joint_count() == 2

        tensor_sim = physx.create_simulation_view("torch")
        tensor_sim.set_subspace_roots("/")
        free_bodies = tensor_sim.create_rigid_body_view("/World/free_body*")
        assert free_bodies.count == 2

        maximum_linear_momentum = 0.0
        maximum_angular_momentum = 0.0
        maximum_linear_impulse_residual = 0.0
        maximum_angular_impulse_residual = 0.0
        box_inertia_kg_m2 = 1.0 * (0.1**2 + 0.1**2) / 12.0
        dt_s = 1.0 / 240.0
        amplitude_rad = math.radians(20.0)
        frequency_hz = 2.0
        external_force_n = torch.zeros((2, 3), dtype=torch.float32)
        external_force_n[1, 0] = 1.0
        external_torque_nm = torch.zeros((2, 3), dtype=torch.float32)
        external_torque_nm[1, 0] = 0.1
        force_indices = torch.tensor([1], dtype=torch.int32)
        last_free_target = 0.0

        for step in range(240):
            phase = 2.0 * math.pi * frequency_hz * step * dt_s
            free_target = amplitude_rad * math.sin(phase)
            free_velocity = amplitude_rad * 2.0 * math.pi * frequency_hz * math.cos(phase)
            last_free_target = free_target
            native.set_targets(
                [trajectory_path, free_trajectory_path],
                [target_rad, free_target],
                [0.0, free_velocity],
            )
            free_bodies.apply_forces_and_torques_at_position(
                external_force_n,
                external_torque_nm,
                None,
                force_indices,
                True,
            )
            sim.step(render=False)
            transforms = free_bodies.get_transforms()
            velocities = free_bodies.get_velocities()
            positions = transforms[:, :3]
            linear_velocities = velocities[:, :3]
            angular_velocities = velocities[:, 3:]
            system_com = torch.mean(positions, dim=0)
            linear_momentum = torch.sum(linear_velocities, dim=0)
            angular_momentum = box_inertia_kg_m2 * torch.sum(
                angular_velocities,
                dim=0,
            ) + torch.sum(
                torch.linalg.cross(
                    positions - system_com,
                    linear_velocities,
                    dim=-1,
                ),
                dim=0,
            )
            maximum_linear_momentum = max(
                maximum_linear_momentum,
                float(torch.linalg.vector_norm(linear_momentum).item()),
            )
            maximum_angular_momentum = max(
                maximum_angular_momentum,
                float(torch.linalg.vector_norm(angular_momentum).item()),
            )
            expected_linear_momentum = torch.tensor(
                ((step + 1) * dt_s, 0.0, 0.0),
                dtype=linear_momentum.dtype,
            )
            expected_angular_momentum = torch.tensor(
                (0.1 * (step + 1) * dt_s, 0.0, 0.0),
                dtype=angular_momentum.dtype,
            )
            maximum_linear_impulse_residual = max(
                maximum_linear_impulse_residual,
                float(
                    torch.linalg.vector_norm(
                        linear_momentum - expected_linear_momentum
                    ).item()
                ),
            )
            maximum_angular_impulse_residual = max(
                maximum_angular_impulse_residual,
                float(
                    torch.linalg.vector_norm(
                        angular_momentum - expected_angular_momentum
                    ).item()
                ),
            )

        solver_state = native.get_debug_state()
        angle_rad = float(solver_state["actual_position_rad"])
        print(
            "native holonomic constraint:"
            f" active={native.get_active_joint_count()},"
            f" target={math.degrees(last_free_target):.6f}deg,"
            f" actual={math.degrees(angle_rad):.6f}deg,"
            f" free_pair_max_linear_momentum={maximum_linear_momentum:.6e}kg*m/s,"
            f" free_pair_max_angular_momentum={maximum_angular_momentum:.6e}kg*m^2/s,"
            f" max_linear_impulse_residual={maximum_linear_impulse_residual:.6e}kg*m/s,"
            f" max_angular_impulse_residual={maximum_angular_impulse_residual:.6e}kg*m^2/s,"
            f" solver={solver_state}"
        )
        assert math.isfinite(angle_rad)
        assert abs(angle_rad - last_free_target) < math.radians(0.25)
        assert maximum_linear_impulse_residual < 1.0e-4
        assert maximum_angular_impulse_residual < 1.0e-4
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        release_state = native.get_debug_state()
        assert native.get_active_joint_count() == 0
        assert int(release_state["release_exit_count"]) == 2
        native.shutdown()
        SimulationContext.clear_instance()
        manager.set_extension_enabled_immediate("omni.flapping_bot.holonomic_constraint", False)

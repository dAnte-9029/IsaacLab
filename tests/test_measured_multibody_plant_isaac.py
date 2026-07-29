"""Headless PhysX readback test for the measured-wing mass partition."""

from __future__ import annotations

"""Launch Isaac Sim before importing Isaac Lab simulation modules."""

from _delaurier_isaac_app import simulation_app

"""The remaining imports require a running Isaac Sim application."""

import pytest
import torch

from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context

from flapping_bot.assets import FlappingBotCfg
from flapping_bot.physics import (
    MEASURED_MULTIBODY_TOTAL_MASS_KG,
    build_measured_wing_multibody_tensors,
)


@pytest.mark.isaacsim_ci
def test_physx_accepts_and_reads_back_measured_wing_multibody_properties() -> None:
    """Apply the project-local partition to the real articulation and read it back."""

    with build_simulation_context(device="cpu", auto_add_lighting=False, gravity_enabled=False) as sim:
        sim._app_control_on_stop_handle = None
        robot = Articulation(FlappingBotCfg.replace(prim_path="/World/MeasuredMultibodyRobot"))
        sim.reset()
        robot.update(sim.cfg.dt)

        env_ids = torch.arange(robot.num_instances, device="cpu", dtype=torch.int64)
        masses = robot.root_physx_view.get_masses().clone()
        inertias = robot.root_physx_view.get_inertias().clone()
        coms = robot.root_physx_view.get_coms().clone()
        expected_masses, expected_inertias, expected_coms = build_measured_wing_multibody_tensors(
            body_names=robot.body_names,
            masses_kg=masses,
            inertias_kg_m2=inertias,
            com_poses_link=coms,
        )

        robot.root_physx_view.set_masses(expected_masses, env_ids)
        robot.root_physx_view.set_inertias(expected_inertias, env_ids)
        robot.root_physx_view.set_coms(expected_coms, env_ids)
        sim.forward()

        applied_masses = robot.root_physx_view.get_masses()
        applied_inertias = robot.root_physx_view.get_inertias()
        applied_coms = robot.root_physx_view.get_coms()
        torch.testing.assert_close(applied_masses, expected_masses, atol=1.0e-7, rtol=0.0)
        torch.testing.assert_close(applied_inertias, expected_inertias, atol=1.0e-8, rtol=0.0)
        # PhysX normalizes the stored COM quaternion on readback in float32.
        torch.testing.assert_close(applied_coms, expected_coms, atol=2.0e-7, rtol=0.0)
        torch.testing.assert_close(
            applied_masses.sum(dim=1),
            torch.full(
                (robot.num_instances,),
                MEASURED_MULTIBODY_TOTAL_MASS_KG,
                dtype=applied_masses.dtype,
                device=applied_masses.device,
            ),
            atol=1.0e-7,
            rtol=0.0,
        )

        body_ids, _ = robot.find_bodies(
            ["base_link", "left_wing", "right_wing", "left_tail", "right_tail", "rudder"],
            preserve_order=True,
        )
        body_id_by_name = dict(
            zip(
                ("base_link", "left_wing", "right_wing", "left_tail", "right_tail", "rudder"),
                (int(body_id) for body_id in body_ids),
                strict=True,
            )
        )
        assert applied_masses[0, body_id_by_name["base_link"]].item() == pytest.approx(0.78231, abs=1.0e-7)
        assert applied_masses[0, body_id_by_name["left_wing"]].item() == pytest.approx(0.06077, abs=1.0e-7)
        assert applied_masses[0, body_id_by_name["right_wing"]].item() == pytest.approx(0.06077, abs=1.0e-7)
        for link_name in ("left_tail", "right_tail", "rudder"):
            assert applied_masses[0, body_id_by_name[link_name]].item() == pytest.approx(1.0e-4, abs=1.0e-9)

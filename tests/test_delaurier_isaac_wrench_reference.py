"""Headless Isaac Sim reference tests for a DeLaurier wing-origin wrench.

This module intentionally launches Isaac Sim.  It checks the final local
base-link force/torque passed to ``Articulation.set_external_force_and_torque``
and a one-step signed dynamics response; it is not a pure Python unit test.
"""

from __future__ import annotations

"""Launch Isaac Sim before importing Isaac Lab simulation modules."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""The remaining imports require a running Isaac Sim application."""

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from flapping_bot.assets import FlappingBotCfg
from flapping_bot.physics.qsm_delaurier1993 import transform_wang_wrench_to_link


# The tolerance is for Isaac's float32 external-wrench buffer, not the
# analytical force/moment calculation.  The one-step response only checks its
# sign because this is a multi-body articulation, not a single rigid body.
_BUFFER_ATOL = 2.0e-5
_DYNAMICS_DIRECTION_EPS = 1.0e-8

# Shape: (1, 3, 3). These are the exact Wang-to-link matrices configured by
# FlappingBotStraightFlightEnv. The test's expected moment is still evaluated
# independently from positions and force with an explicit cross product.
_WANG_TO_LEFT_LINK = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
_WANG_TO_RIGHT_LINK = torch.tensor([[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])


def _make_reference_robot() -> Articulation:
    """Spawn the project articulation with gravity disabled for wrench checks."""

    # Gravity is disabled at SimulationContext scope. Keeping the original
    # UrdfFileCfg untouched prevents the converter cache from rewriting the
    # checked-in robot conversion metadata during this test.
    robot_cfg = FlappingBotCfg.replace(prim_path="/World/ReferenceRobot")
    return Articulation(robot_cfg)


def _reset_robot(robot: Articulation) -> None:
    """Reset root and joints to a zero-velocity, identity-orientation state."""

    root_state = robot.data.default_root_state.clone()
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot.device)
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    robot.reset()


def _wing_origin_wrench_expected_in_base_link(
    robot: Articulation,
    *,
    wing_name: str,
    force_wang: torch.Tensor,
    free_moment_wang: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hand-close a wing-origin wrench and express it in the base-link frame.

    Args:
        robot: Initialized one-instance flapping articulation.
        wing_name: ``left_wing`` or ``right_wing``.
        force_wang: Shape ``(1, 3)`` in N, Wang frame, applied at wing-link
            origin.  The test deliberately uses no legacy quarter-chord point.
        free_moment_wang: Shape ``(1, 3)`` in N m, Wang frame, about the same
            wing-link origin.

    Returns:
        Final force and torque in the base-link local frame, then the same
        target force and torque in world frame.  The torque is about base COM.
    """

    base_body_ids, _ = robot.find_bodies(["base_link"], preserve_order=True)
    wing_body_ids, _ = robot.find_bodies([wing_name], preserve_order=True)
    assert len(base_body_ids) == 1 and len(wing_body_ids) == 1
    base_body_id, wing_body_id = int(base_body_ids[0]), int(wing_body_ids[0])

    wang_to_link = _WANG_TO_LEFT_LINK if wing_name == "left_wing" else _WANG_TO_RIGHT_LINK
    wang_to_link = wang_to_link.to(device=robot.device, dtype=robot.data.root_quat_w.dtype)
    force_link, free_moment_link = transform_wang_wrench_to_link(force_wang, free_moment_wang, wang_to_link)

    # ``body_pos_w`` is the exact wing-root/link-origin field used by the
    # environment.  ``root_com_pos_w`` is intentionally different from the
    # base-link origin for this URDF and is the required wrench reference.
    wing_origin_w = robot.data.body_pos_w[:, wing_body_id, :]
    wing_link_quat_w = robot.data.body_quat_w[:, wing_body_id, :]
    base_com_w = robot.data.root_com_pos_w
    base_link_quat_w = robot.data.body_link_quat_w[:, base_body_id, :]
    torch.testing.assert_close(base_link_quat_w, robot.data.root_quat_w, atol=_BUFFER_ATOL, rtol=0.0)
    assert not torch.allclose(base_com_w, robot.data.body_link_pos_w[:, base_body_id, :], atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(wing_origin_w, robot.data.body_link_pos_w[:, wing_body_id, :], atol=_BUFFER_ATOL, rtol=0.0)

    force_w = quat_apply(wing_link_quat_w, force_link)
    free_moment_w = quat_apply(wing_link_quat_w, free_moment_link)

    # Independent reference closure in world frame: M_G = M_O + (p_O-p_G) x F.
    expected_moment_w_about_base_com = free_moment_w + torch.linalg.cross(wing_origin_w - base_com_w, force_w)
    expected_force_base_link = quat_apply_inverse(base_link_quat_w, force_w)
    expected_moment_base_link = quat_apply_inverse(base_link_quat_w, expected_moment_w_about_base_com)
    return expected_force_base_link, expected_moment_base_link, force_w, expected_moment_w_about_base_com


def _apply_and_check_reference_wrench(
    robot: Articulation,
    sim: sim_utils.SimulationContext,
    *,
    label: str,
    expected_force_base_link: torch.Tensor,
    expected_moment_base_link: torch.Tensor,
    expected_force_w: torch.Tensor,
    expected_moment_w: torch.Tensor,
    require_linear_response: bool,
    require_angular_response: bool,
) -> None:
    """Apply one complete base-COM wrench and verify Isaac's input/direction."""

    base_body_ids, _ = robot.find_bodies(["base_link"], preserve_order=True)
    assert len(base_body_ids) == 1
    base_body_id = int(base_body_ids[0])
    robot.set_external_force_and_torque(
        forces=expected_force_base_link.unsqueeze(1),
        torques=expected_moment_base_link.unsqueeze(1),
        body_ids=base_body_ids,
        is_global=False,
    )

    actual_force_base_link = robot._external_force_b[:, base_body_id, :]
    actual_moment_base_link = robot._external_torque_b[:, base_body_id, :]
    force_input_max_abs_error = torch.max(torch.abs(actual_force_base_link - expected_force_base_link)).item()
    moment_input_max_abs_error = torch.max(torch.abs(actual_moment_base_link - expected_moment_base_link)).item()
    torch.testing.assert_close(
        actual_force_base_link,
        expected_force_base_link,
        atol=_BUFFER_ATOL,
        rtol=0.0,
        msg=f"{label}: expected/actual force in base_link frame differ",
    )
    torch.testing.assert_close(
        actual_moment_base_link,
        expected_moment_base_link,
        atol=_BUFFER_ATOL,
        rtol=0.0,
        msg=f"{label}: expected/actual moment about base COM differ",
    )
    assert robot._use_global_wrench_frame is False, f"{label}: expected base-link-local articulation wrench"

    linear_velocity_before_w = robot.data.root_com_lin_vel_w.clone()
    angular_velocity_before_w = robot.data.root_com_ang_vel_w.clone()
    robot.set_joint_position_target(robot.data.default_joint_pos)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.cfg.dt)
    delta_linear_velocity_w = robot.data.root_com_lin_vel_w - linear_velocity_before_w
    delta_angular_velocity_w = robot.data.root_com_ang_vel_w - angular_velocity_before_w

    # A free floating articulation has internal joint reactions, so this is a
    # deliberate sign/direction check rather than an invalid single-rigid-body
    # I*alpha equality. The input-buffer equality above is the exact wrench
    # reference test; this confirms PhysX actually reacts to it.
    if require_linear_response:
        assert torch.sum(delta_linear_velocity_w * expected_force_w).item() > _DYNAMICS_DIRECTION_EPS, (
            f"{label}: expected linear response along force; force_w={expected_force_w}, "
            f"delta_v_w={delta_linear_velocity_w}"
        )
    if require_angular_response:
        assert torch.sum(delta_angular_velocity_w * expected_moment_w).item() > _DYNAMICS_DIRECTION_EPS, (
            f"{label}: expected angular response along moment; moment_w={expected_moment_w}, "
            f"delta_omega_w={delta_angular_velocity_w}"
        )
    print(
        "[REFERENCE_WRENCH] "
        f"label={label}; dt_s={sim.cfg.dt}; "
        f"expected_force_base_link={expected_force_base_link.tolist()}; "
        f"actual_force_base_link={actual_force_base_link.tolist()}; "
        f"expected_moment_base_com={expected_moment_base_link.tolist()}; "
        f"actual_moment_base_com={actual_moment_base_link.tolist()}; "
        f"force_input_max_abs_error={force_input_max_abs_error:.3e}; "
        f"moment_input_max_abs_error={moment_input_max_abs_error:.3e}; "
        f"delta_linear_velocity_w={delta_linear_velocity_w.tolist()}; "
        f"delta_angular_velocity_w={delta_angular_velocity_w.tolist()}"
    )


@pytest.mark.isaacsim_ci
def test_delaurier_isaac_reference_wrenches() -> None:
    """Validate two force directions, both reflected couples, and combined wrenches."""

    with build_simulation_context(device="cpu", auto_add_lighting=False, gravity_enabled=False) as sim:
        sim._app_control_on_stop_handle = None
        robot = _make_reference_robot()
        sim.reset()
        robot.update(sim.cfg.dt)
        assert robot.is_initialized

        zero_moment_wang = torch.zeros((1, 3), device=robot.device)
        # Case A: two non-collinear force directions at the left wing origin;
        # there is no free couple in either subcase.
        for force_wang in (
            torch.tensor([[0.0, 0.0, 80.0]], device=robot.device),
            torch.tensor([[0.0, 65.0, 0.0]], device=robot.device),
        ):
            _reset_robot(robot)
            expected = _wing_origin_wrench_expected_in_base_link(
                robot,
                wing_name="left_wing",
                force_wang=force_wang,
                free_moment_wang=zero_moment_wang,
            )
            _apply_and_check_reference_wrench(
                robot,
                sim,
                label=f"single-force left wing force_wang={force_wang.tolist()}",
                expected_force_base_link=expected[0],
                expected_moment_base_link=expected[1],
                expected_force_w=expected[2],
                expected_moment_w=expected[3],
                require_linear_response=True,
                require_angular_response=True,
            )

        # Case B: free couple only. With F=0, base-COM translation must leave
        # the couple unchanged. Both sides exercise the right-wing reflection.
        for wing_name in ("left_wing", "right_wing"):
            _reset_robot(robot)
            free_moment_wang = torch.tensor([[25.0, 0.0, 0.0]], device=robot.device)
            expected = _wing_origin_wrench_expected_in_base_link(
                robot,
                wing_name=wing_name,
                force_wang=torch.zeros((1, 3), device=robot.device),
                free_moment_wang=free_moment_wang,
            )
            torch.testing.assert_close(expected[1], quat_apply_inverse(robot.data.root_quat_w, expected[3]), atol=_BUFFER_ATOL, rtol=0.0)
            _apply_and_check_reference_wrench(
                robot,
                sim,
                label=f"single-free-couple {wing_name}",
                expected_force_base_link=expected[0],
                expected_moment_base_link=expected[1],
                expected_force_w=expected[2],
                expected_moment_w=expected[3],
                require_linear_response=False,
                require_angular_response=True,
            )

        # Case C: a force and a free couple together. The explicit reference
        # contains exactly one r_O/G x F term, preventing legacy quarter-chord
        # closure from being silently added a second time.
        for wing_name in ("left_wing", "right_wing"):
            _reset_robot(robot)
            force_wang = torch.tensor([[0.0, 0.0, 45.0]], device=robot.device)
            free_moment_wang = torch.tensor([[18.0, 0.0, 0.0]], device=robot.device)
            expected = _wing_origin_wrench_expected_in_base_link(
                robot,
                wing_name=wing_name,
                force_wang=force_wang,
                free_moment_wang=free_moment_wang,
            )
            _apply_and_check_reference_wrench(
                robot,
                sim,
                label=f"combined force-plus-couple {wing_name}",
                expected_force_base_link=expected[0],
                expected_moment_base_link=expected[1],
                expected_force_w=expected[2],
                expected_moment_w=expected[3],
                require_linear_response=True,
                require_angular_response=True,
            )

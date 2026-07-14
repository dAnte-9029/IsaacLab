"""Headless PhysX contract between engineering phase and real wing-link pose."""

from __future__ import annotations

"""Launch Isaac Sim before importing Isaac Lab simulation modules."""

from _delaurier_isaac_app import simulation_app

"""The remaining imports require a running Isaac Sim application."""

import math
from dataclasses import dataclass

import pytest
import torch

from isaaclab.assets import Articulation
from isaaclab.sim import build_simulation_context
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from flapping_bot.assets import FlappingBotCfg
from flapping_bot.direct.flapping_bot.startup_phase import map_symmetric_flap_coordinate_to_joint_space
from flapping_bot.physics.delaurier_twist import resolve_delaurier_phase


_POSE_ATOL = 2.0e-5
_DIRECTION_EPS = 1.0e-4
_STROKE_AMPLITUDE_RAD = math.radians(20.0)
_PHASE_RATE_RAD_S = 5.0
_PROBE_SPAN_M = 0.65
_BODY_MIRROR_POLAR = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
_BODY_MIRROR_AXIAL = torch.linalg.det(_BODY_MIRROR_POLAR) * _BODY_MIRROR_POLAR


@dataclass(frozen=True)
class _PhasePose:
    """One real articulation pose expressed in the base-link FLU frame."""

    phase_rad: float
    flap_position_rad: float
    flap_velocity_rad_s: float
    left_joint_position_rad: float
    right_joint_position_rad: float
    left_joint_velocity_rad_s: float
    right_joint_velocity_rad_s: float
    left_origin_b: torch.Tensor
    right_origin_b: torch.Tensor
    left_probe_b: torch.Tensor
    right_probe_b: torch.Tensor
    left_chord_b: torch.Tensor
    right_chord_b: torch.Tensor
    left_span_b: torch.Tensor
    right_span_b: torch.Tensor
    left_twist_axis_b: torch.Tensor
    right_twist_axis_b: torch.Tensor
    left_angular_velocity_b: torch.Tensor
    right_angular_velocity_b: torch.Tensor


def _make_phase_robot() -> Articulation:
    return Articulation(FlappingBotCfg.replace(prim_path="/World/PhasePoseRobot"))


def _set_and_read_phase_pose(
    robot: Articulation,
    sim,
    *,
    phase_rad: float,
    wing_joint_ids: list[int],
    body_ids: list[int],
) -> _PhasePose:
    """Write mirrored joint state and read actual PhysX link transforms."""

    phase = torch.tensor([phase_rad], device=robot.device, dtype=robot.data.default_joint_pos.dtype)
    flap_position = _STROKE_AMPLITUDE_RAD * torch.cos(phase)
    flap_velocity = -_STROKE_AMPLITUDE_RAD * _PHASE_RATE_RAD_S * torch.sin(phase)
    left_position, right_position, left_velocity, right_velocity = map_symmetric_flap_coordinate_to_joint_space(
        flap_position_rad=flap_position,
        flap_velocity_rad_s=flap_velocity,
        left_joint_mid_rad=0.0,
        right_joint_mid_rad=0.0,
    )

    joint_position = robot.data.default_joint_pos.clone()
    joint_velocity = torch.zeros_like(robot.data.default_joint_vel)
    joint_position[:, wing_joint_ids[0]] = left_position
    joint_position[:, wing_joint_ids[1]] = right_position
    joint_velocity[:, wing_joint_ids[0]] = left_velocity
    joint_velocity[:, wing_joint_ids[1]] = right_velocity
    robot.write_joint_state_to_sim(joint_position, joint_velocity)
    sim.forward()
    robot.update(sim.cfg.dt)

    base_id, left_id, right_id = body_ids
    base_position_w = robot.data.body_link_pos_w[:, base_id, :]
    base_quaternion_w = robot.data.body_link_quat_w[:, base_id, :]
    left_position_w = robot.data.body_link_pos_w[:, left_id, :]
    right_position_w = robot.data.body_link_pos_w[:, right_id, :]
    left_quaternion_w = robot.data.body_link_quat_w[:, left_id, :]
    right_quaternion_w = robot.data.body_link_quat_w[:, right_id, :]

    def point_in_base(link_position_w: torch.Tensor, link_quaternion_w: torch.Tensor, point_link: torch.Tensor):
        point_w = link_position_w + quat_apply(link_quaternion_w, point_link)
        return quat_apply_inverse(base_quaternion_w, point_w - base_position_w)

    def direction_in_base(link_quaternion_w: torch.Tensor, direction_link: torch.Tensor):
        return quat_apply_inverse(base_quaternion_w, quat_apply(link_quaternion_w, direction_link))

    zero = torch.zeros((1, 3), device=robot.device)
    left_probe_link = torch.tensor([[0.0, _PROBE_SPAN_M, 0.0]], device=robot.device)
    right_probe_link = torch.tensor([[0.0, -_PROBE_SPAN_M, 0.0]], device=robot.device)
    chord_link = torch.tensor([[1.0, 0.0, 0.0]], device=robot.device)
    left_span_link = torch.tensor([[0.0, 1.0, 0.0]], device=robot.device)
    right_span_link = torch.tensor([[0.0, -1.0, 0.0]], device=robot.device)
    # Positive Wang pitching/twist vectors are axial. The established
    # Wang-to-link reflection maps local +twist to link +y on both sides.
    twist_axis_link = torch.tensor([[0.0, 1.0, 0.0]], device=robot.device)

    left_angular_velocity_b = quat_apply_inverse(
        base_quaternion_w, robot.data.body_link_ang_vel_w[:, left_id, :]
    )
    right_angular_velocity_b = quat_apply_inverse(
        base_quaternion_w, robot.data.body_link_ang_vel_w[:, right_id, :]
    )
    return _PhasePose(
        phase_rad=float(phase_rad),
        flap_position_rad=float(flap_position.item()),
        flap_velocity_rad_s=float(flap_velocity.item()),
        left_joint_position_rad=float(robot.data.joint_pos[0, wing_joint_ids[0]].item()),
        right_joint_position_rad=float(robot.data.joint_pos[0, wing_joint_ids[1]].item()),
        left_joint_velocity_rad_s=float(robot.data.joint_vel[0, wing_joint_ids[0]].item()),
        right_joint_velocity_rad_s=float(robot.data.joint_vel[0, wing_joint_ids[1]].item()),
        left_origin_b=point_in_base(left_position_w, left_quaternion_w, zero),
        right_origin_b=point_in_base(right_position_w, right_quaternion_w, zero),
        left_probe_b=point_in_base(left_position_w, left_quaternion_w, left_probe_link),
        right_probe_b=point_in_base(right_position_w, right_quaternion_w, right_probe_link),
        left_chord_b=direction_in_base(left_quaternion_w, chord_link),
        right_chord_b=direction_in_base(right_quaternion_w, chord_link),
        left_span_b=direction_in_base(left_quaternion_w, left_span_link),
        right_span_b=direction_in_base(right_quaternion_w, right_span_link),
        left_twist_axis_b=direction_in_base(left_quaternion_w, twist_axis_link),
        right_twist_axis_b=direction_in_base(right_quaternion_w, twist_axis_link),
        left_angular_velocity_b=left_angular_velocity_b,
        right_angular_velocity_b=right_angular_velocity_b,
    )


def _assert_polar_mirror(left: torch.Tensor, right: torch.Tensor, *, label: str) -> float:
    mirror = _BODY_MIRROR_POLAR.to(device=left.device, dtype=left.dtype)
    expected_right = left @ mirror.T
    torch.testing.assert_close(right, expected_right, atol=_POSE_ATOL, rtol=0.0, msg=label)
    return torch.max(torch.abs(right - expected_right)).item()


@pytest.mark.isaacsim_ci
def test_delaurier_phase_matches_real_mirrored_wing_link_pose_and_motion() -> None:
    """Validate four phase states using actual PhysX link pose and velocity."""

    with build_simulation_context(device="cpu", auto_add_lighting=False, gravity_enabled=False) as sim:
        sim._app_control_on_stop_handle = None
        robot = _make_phase_robot()
        sim.reset()
        robot.update(sim.cfg.dt)
        wing_joint_ids, _ = robot.find_joints(["left_wing", "right_wing"], preserve_order=True)
        body_ids, _ = robot.find_bodies(["base_link", "left_wing", "right_wing"], preserve_order=True)
        assert len(wing_joint_ids) == 2 and len(body_ids) == 3
        wing_joint_ids = [int(index) for index in wing_joint_ids]
        body_ids = [int(index) for index in body_ids]

        phases = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
        poses = {
            phase: _set_and_read_phase_pose(
                robot,
                sim,
                phase_rad=phase,
                wing_joint_ids=wing_joint_ids,
                body_ids=body_ids,
            )
            for phase in phases
        }

        mirror_errors: list[float] = []
        for phase, pose in poses.items():
            mirror_errors.append(
                _assert_polar_mirror(pose.left_origin_b, pose.right_origin_b, label=f"phase={phase}: link origin")
            )
            mirror_errors.append(
                _assert_polar_mirror(pose.left_probe_b, pose.right_probe_b, label=f"phase={phase}: span probe")
            )
            mirror_errors.append(
                _assert_polar_mirror(pose.left_chord_b, pose.right_chord_b, label=f"phase={phase}: chord direction")
            )
            mirror_errors.append(
                _assert_polar_mirror(pose.left_span_b, pose.right_span_b, label=f"phase={phase}: outward span")
            )
            axial_mirror = _BODY_MIRROR_AXIAL.to(
                device=pose.left_twist_axis_b.device, dtype=pose.left_twist_axis_b.dtype
            )
            torch.testing.assert_close(
                pose.right_twist_axis_b,
                pose.left_twist_axis_b @ axial_mirror.T,
                atol=_POSE_ATOL,
                rtol=0.0,
                msg=f"phase={phase}: axial pitching direction",
            )
            mirror_errors.append(
                torch.max(torch.abs(pose.right_twist_axis_b - pose.left_twist_axis_b @ axial_mirror.T)).item()
            )
            assert pose.left_joint_position_rad == pytest.approx(-pose.right_joint_position_rad, abs=_POSE_ATOL)
            assert pose.left_joint_velocity_rad_s == pytest.approx(-pose.right_joint_velocity_rad_s, abs=_POSE_ATOL)

        phase_zero = poses[0.0]
        phase_half = poses[math.pi]
        phase_quarter = poses[math.pi / 2.0]
        phase_three_quarter = poses[3.0 * math.pi / 2.0]
        assert phase_zero.left_probe_b[0, 2].item() > phase_quarter.left_probe_b[0, 2].item()
        assert phase_half.left_probe_b[0, 2].item() < phase_quarter.left_probe_b[0, 2].item()
        torch.testing.assert_close(
            phase_quarter.left_probe_b,
            phase_three_quarter.left_probe_b,
            atol=_POSE_ATOL,
            rtol=0.0,
        )

        # At pi/2 the left joint angular velocity is negative and the right is
        # positive, yet both span probes move toward body-FLU -z. At 3pi/2 the
        # signs reverse and both probes move toward +z.
        assert phase_quarter.left_joint_velocity_rad_s < 0.0 < phase_quarter.right_joint_velocity_rad_s
        assert phase_three_quarter.right_joint_velocity_rad_s < 0.0 < phase_three_quarter.left_joint_velocity_rad_s
        assert phase_quarter.left_angular_velocity_b[0, 0].item() < -_DIRECTION_EPS
        assert phase_quarter.right_angular_velocity_b[0, 0].item() > _DIRECTION_EPS
        assert phase_three_quarter.left_angular_velocity_b[0, 0].item() > _DIRECTION_EPS
        assert phase_three_quarter.right_angular_velocity_b[0, 0].item() < -_DIRECTION_EPS

        epsilon = 1.0e-3
        for phase, expected_z_direction in (
            (math.pi / 2.0, -1.0),
            (3.0 * math.pi / 2.0, 1.0),
        ):
            before = _set_and_read_phase_pose(
                robot, sim, phase_rad=phase - epsilon, wing_joint_ids=wing_joint_ids, body_ids=body_ids
            )
            after = _set_and_read_phase_pose(
                robot, sim, phase_rad=phase + epsilon, wing_joint_ids=wing_joint_ids, body_ids=body_ids
            )
            left_z_slope = (after.left_probe_b[0, 2] - before.left_probe_b[0, 2]).item() / (2.0 * epsilon)
            right_z_slope = (after.right_probe_b[0, 2] - before.right_probe_b[0, 2]).item() / (2.0 * epsilon)
            assert math.copysign(1.0, left_z_slope) == expected_z_direction
            assert math.copysign(1.0, right_z_slope) == expected_z_direction

        current_phase = torch.tensor(phases, dtype=torch.float64)
        phase_d, phase_rate_d, phase_acceleration_d = resolve_delaurier_phase(
            current_phase=current_phase,
            current_phase_rate=torch.full_like(current_phase, _PHASE_RATE_RAD_S),
            current_phase_acceleration=torch.zeros_like(current_phase),
            phase_direction=1.0,
            phase_offset_rad=0.0,
        )
        torch.testing.assert_close(phase_d, current_phase, atol=1.0e-12, rtol=0.0)
        torch.testing.assert_close(phase_rate_d, torch.full_like(current_phase, _PHASE_RATE_RAD_S), atol=1.0e-12, rtol=0.0)
        torch.testing.assert_close(phase_acceleration_d, torch.zeros_like(current_phase), atol=1.0e-12, rtol=0.0)

        for phase in phases:
            pose = poses[phase]
            interpretation = {
                0.0: "positive-body-z endpoint",
                math.pi / 2.0: "midpoint moving toward body -z (downstroke)",
                math.pi: "negative-body-z endpoint",
                3.0 * math.pi / 2.0: "midpoint moving toward body +z (upstroke)",
            }[phase]
            print(
                "[PHASE_POSE] "
                f"phase_rad={phase:.9f}; q_rad={pose.flap_position_rad:.9f}; "
                f"qd_rad_s={pose.flap_velocity_rad_s:.9f}; "
                f"left_joint=({pose.left_joint_position_rad:.9f},{pose.left_joint_velocity_rad_s:.9f}); "
                f"right_joint=({pose.right_joint_position_rad:.9f},{pose.right_joint_velocity_rad_s:.9f}); "
                f"left_probe_b={pose.left_probe_b.tolist()}; right_probe_b={pose.right_probe_b.tolist()}; "
                f"interpretation={interpretation}"
            )
        print(f"[PHASE_POSE] mirror_max_abs_error={max(mirror_errors):.3e}; tolerance={_POSE_ATOL:.3e}")

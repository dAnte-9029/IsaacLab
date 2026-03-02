"""Unit tests for PX4-like guidance utilities."""

from __future__ import annotations

import math

import torch

from flapping_bot.px4_like.guidance import AirspeedDirectionController, DirectionalGuidance


def test_directional_guidance_on_track_points_forward():
    guidance = DirectionalGuidance()
    pos = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    ground_vel = torch.tensor([[7.0, 0.0]], dtype=torch.float32)
    wind_vel = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    unit_tangent = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    closest = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    out = guidance.guide_to_path(
        curr_pos_local=pos,
        ground_vel=ground_vel,
        wind_vel=wind_vel,
        unit_path_tangent=unit_tangent,
        position_on_path=closest,
        path_curvature=0.0,
    )

    assert torch.allclose(out.signed_track_error, torch.zeros_like(out.signed_track_error), atol=1.0e-5)
    assert torch.allclose(out.course_setpoint, torch.zeros_like(out.course_setpoint), atol=1.0e-4)
    assert torch.allclose(
        out.lateral_acceleration_feedforward,
        torch.zeros_like(out.lateral_acceleration_feedforward),
        atol=1.0e-5,
    )


def test_directional_guidance_offtrack_turns_back_to_line():
    guidance = DirectionalGuidance()
    pos = torch.tensor([[0.0, 3.0]], dtype=torch.float32)
    ground_vel = torch.tensor([[7.0, 0.0]], dtype=torch.float32)
    wind_vel = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    unit_tangent = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    closest = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    out = guidance.guide_to_path(
        curr_pos_local=pos,
        ground_vel=ground_vel,
        wind_vel=wind_vel,
        unit_path_tangent=unit_tangent,
        position_on_path=closest,
        path_curvature=0.0,
    )

    assert out.signed_track_error.item() > 0.0
    assert out.course_setpoint.item() < 0.0


def test_airspeed_direction_controller_saturates_past_90deg():
    controller = AirspeedDirectionController()
    heading_sp = torch.tensor([0.0], dtype=torch.float32)
    heading = torch.tensor([math.pi], dtype=torch.float32)
    airspeed = torch.tensor([5.0], dtype=torch.float32)

    lateral_accel = controller.control_heading(heading_sp, heading, airspeed)
    expected = controller.settings.p_gain * airspeed
    assert torch.allclose(torch.abs(lateral_accel), expected, atol=1.0e-4)

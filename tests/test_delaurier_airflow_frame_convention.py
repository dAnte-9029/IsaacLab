"""Contract tests for the FLU-to-DeLaurier airflow boundary."""

from __future__ import annotations

import math

import pytest
import torch

from flapping_bot.physics.delaurier_airflow import (
    body_air_velocity_to_delaurier_section_velocity,
    compute_delaurier_axis_incidence,
)
from flapping_bot.physics.qsm_delaurier1993 import (
    DeLaurierParams,
    compute_delaurier_strip_loads,
    integrate_delaurier_strip_wrench,
    transform_wang_wrench_to_link,
)
from flapping_bot.physics.qsm_wang2016 import WingGeometry


_DTYPE = torch.float64
_ATOL = 1.0e-12
_FLU_TO_FRD = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=_DTYPE))
_WANG_TO_LEFT_LINK = torch.tensor(
    [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=_DTYPE
)
_WANG_TO_RIGHT_LINK = torch.tensor(
    [[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=_DTYPE
)


def test_horizontal_air_relative_velocity_has_zero_axis_incidence() -> None:
    velocity_flu = torch.tensor([[8.0, 0.0, 0.0]], dtype=_DTYPE)
    theta_a = compute_delaurier_axis_incidence(air_velocity_body=velocity_flu, body_frame="FLU")
    torch.testing.assert_close(theta_a, torch.zeros_like(theta_a), atol=_ATOL, rtol=0.0)


@pytest.mark.parametrize(
    "vertical_velocity_flu, expected_sign",
    ((-1.0, 1.0), (1.0, -1.0)),
)
def test_flu_vertical_component_has_documented_theta_a_sign(
    vertical_velocity_flu: float,
    expected_sign: float,
) -> None:
    # FLU -z is physically downward. DeLaurier +w_D is also downward, so it
    # produces positive theta_a; the physically upward case is negative.
    velocity_flu = torch.tensor([[8.0, 0.0, vertical_velocity_flu]], dtype=_DTYPE)
    theta_a = compute_delaurier_axis_incidence(air_velocity_body=velocity_flu, body_frame="FLU")
    assert math.copysign(1.0, theta_a.item()) == expected_sign
    assert abs(theta_a.item()) == pytest.approx(math.atan2(1.0, 8.0), abs=_ATOL)


def test_flu_and_frd_inputs_of_same_physical_velocity_are_equivalent() -> None:
    velocity_flu = torch.tensor([[7.5, 0.8, -1.2], [9.0, -0.4, 0.7]], dtype=_DTYPE)
    velocity_frd = velocity_flu @ _FLU_TO_FRD.T
    section_from_flu = body_air_velocity_to_delaurier_section_velocity(velocity_flu, body_frame="FLU")
    section_from_frd = body_air_velocity_to_delaurier_section_velocity(velocity_frd, body_frame="FRD")
    theta_from_flu = compute_delaurier_axis_incidence(air_velocity_body=velocity_flu, body_frame="FLU")
    theta_from_frd = compute_delaurier_axis_incidence(air_velocity_body=velocity_frd, body_frame="FRD")

    torch.testing.assert_close(section_from_flu, velocity_frd, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(section_from_flu, section_from_frd, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(theta_from_flu, theta_from_frd, atol=_ATOL, rtol=0.0)


def test_total_pitch_adds_axis_wing_and_dynamic_terms_once() -> None:
    velocity_flu = torch.tensor([[8.0, 0.0, -0.8]], dtype=_DTYPE)
    theta_a = compute_delaurier_axis_incidence(air_velocity_body=velocity_flu, body_frame="FLU")
    theta_w = torch.tensor([0.06], dtype=_DTYPE)
    delta_theta = torch.tensor([[-0.03, -0.08]], dtype=_DTYPE)
    theta_bar = theta_a + theta_w
    theta = theta_bar.view(1, 1) + delta_theta
    torch.testing.assert_close(theta, theta_a.view(1, 1) + theta_w.view(1, 1) + delta_theta, atol=_ATOL, rtol=0.0)


def _static_strip_wrench(theta_a: torch.Tensor):
    geometry = WingGeometry.from_input(
        {"R": 0.8, "N": 4, "c": 0.12, "dhat": 0.0, "aspect_ratio": 4.0},
        dtype=_DTYPE,
    )
    batch_size = int(theta_a.numel())
    num_strips = int(geometry.x_mid.numel())
    zeros = torch.zeros((batch_size, num_strips), dtype=_DTYPE)
    theta_bar = theta_a.view(batch_size, 1)
    loads = compute_delaurier_strip_loads(
        zeros,
        zeros,
        zeros,
        theta_bar.expand(batch_size, num_strips),
        zeros,
        zeros,
        geometry,
        rho=1.225,
        U=8.0,
        theta_a=theta_bar,
        theta_bar=theta_bar,
        omega_ref=torch.ones_like(zeros),
        params=DeLaurierParams(alpha0_rad=0.0, c_mac=0.0, cd_f=0.028),
        enable_separation=False,
    )
    return loads, integrate_delaurier_strip_wrench(loads)


def test_theta_a_sign_drives_the_expected_attached_normal_force_sign() -> None:
    velocity_flu = torch.tensor([[8.0, 0.0, -1.0], [8.0, 0.0, 1.0]], dtype=_DTYPE)
    theta_a = compute_delaurier_axis_incidence(air_velocity_body=velocity_flu, body_frame="FLU")
    loads, wrench = _static_strip_wrench(theta_a)
    assert torch.all(loads.dN_c[0] > 0.0)
    assert torch.all(loads.dN_c[1] < 0.0)
    assert wrench.force_wang[0, 1].item() > 0.0
    assert wrench.force_wang[1, 1].item() < 0.0


def test_flu_frd_equivalence_propagates_through_force_and_moment() -> None:
    velocity_flu = torch.tensor([[8.0, 0.0, -0.9]], dtype=_DTYPE)
    velocity_frd = velocity_flu @ _FLU_TO_FRD.T
    theta_flu = compute_delaurier_axis_incidence(air_velocity_body=velocity_flu, body_frame="FLU")
    theta_frd = compute_delaurier_axis_incidence(air_velocity_body=velocity_frd, body_frame="FRD")
    _loads, wrench = _static_strip_wrench(torch.cat((theta_flu, theta_frd)))
    torch.testing.assert_close(wrench.force_wang[0], wrench.force_wang[1], atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(
        wrench.moment_wang_about_wing_origin[0],
        wrench.moment_wang_about_wing_origin[1],
        atol=_ATOL,
        rtol=0.0,
    )


def test_symmetric_wings_do_not_gain_lateral_asymmetry_from_frame_conversion() -> None:
    velocity_flu = torch.tensor([[8.0, 0.0, -0.7], [8.0, 0.0, -0.7]], dtype=_DTYPE)
    theta_a = compute_delaurier_axis_incidence(air_velocity_body=velocity_flu, body_frame="FLU")
    _loads, wrench = _static_strip_wrench(theta_a)
    force_left, moment_left = transform_wang_wrench_to_link(
        wrench.force_wang[0:1], wrench.moment_wang_about_wing_origin[0:1], _WANG_TO_LEFT_LINK
    )
    force_right, moment_right = transform_wang_wrench_to_link(
        wrench.force_wang[1:2], wrench.moment_wang_about_wing_origin[1:2], _WANG_TO_RIGHT_LINK
    )
    torch.testing.assert_close(force_left[:, 1] + force_right[:, 1], torch.zeros(1, dtype=_DTYPE), atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(moment_left[:, 0] + moment_right[:, 0], torch.zeros(1, dtype=_DTYPE), atol=_ATOL, rtol=0.0)

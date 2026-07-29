"""Pure tests for DeLaurier prescribed linear-spanwise dynamic twist."""

from __future__ import annotations

import math

import pytest
import torch

from flapping_bot.physics.delaurier_twist import (
    compute_delaurier_dynamic_twist,
    resolve_delaurier_phase,
    validate_delaurier_dynamic_twist_mode,
)
from flapping_bot.physics.qsm_delaurier1993 import (
    DeLaurierParams,
    compute_delaurier_strip_loads,
    integrate_delaurier_strip_wrench,
    transform_wang_wrench_to_link,
    translate_wrench_moment,
)
from flapping_bot.physics.qsm_wang2016 import WingGeometry


_DTYPE = torch.float64
_ATOL = 1.0e-12
_WANG_TO_LEFT_LINK = torch.tensor(
    [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
    dtype=_DTYPE,
)
_WANG_TO_RIGHT_LINK = torch.tensor(
    [[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
    dtype=_DTYPE,
)


def _compute_twist(
    *,
    strip_span_m: torch.Tensor,
    strip_width_m: torch.Tensor,
    mean_pitch_rad: torch.Tensor | float = 0.0,
    tip_twist_amplitude_rad: torch.Tensor | float = 0.2,
    phase_rad: torch.Tensor | None = None,
    phase_rate_rad_s: torch.Tensor | None = None,
    phase_acceleration_rad_s2: torch.Tensor | None = None,
    enabled: bool = True,
    semi_span_m: torch.Tensor | float | None = None,
):
    phase = torch.tensor([0.7], dtype=_DTYPE) if phase_rad is None else phase_rad
    phase_rate = torch.tensor([5.0], dtype=_DTYPE) if phase_rate_rad_s is None else phase_rate_rad_s
    phase_acceleration = (
        torch.tensor([0.0], dtype=_DTYPE)
        if phase_acceleration_rad_s2 is None
        else phase_acceleration_rad_s2
    )
    return compute_delaurier_dynamic_twist(
        strip_span_m=strip_span_m,
        strip_width_m=strip_width_m,
        mean_pitch_rad=mean_pitch_rad,
        tip_twist_amplitude_rad=tip_twist_amplitude_rad,
        phase_rad=phase,
        phase_rate_rad_s=phase_rate,
        phase_acceleration_rad_s2=phase_acceleration,
        enabled=enabled,
        semi_span_m=semi_span_m,
    )


def test_disabled_and_zero_tip_modes_are_exact_no_twist_regressions() -> None:
    span = torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=_DTYPE)
    width = torch.full_like(span, 0.25)
    mean_pitch = torch.tensor([0.15, -0.05], dtype=_DTYPE)
    phase = torch.tensor([0.3, 1.1], dtype=_DTYPE)
    phase_rate = torch.tensor([4.0, 7.0], dtype=_DTYPE)
    phase_acceleration = torch.tensor([0.0, 2.0], dtype=_DTYPE)

    disabled = _compute_twist(
        strip_span_m=span,
        strip_width_m=width,
        mean_pitch_rad=mean_pitch,
        phase_rad=phase,
        phase_rate_rad_s=phase_rate,
        phase_acceleration_rad_s2=phase_acceleration,
        enabled=False,
    )
    zero_tip = _compute_twist(
        strip_span_m=span,
        strip_width_m=width,
        mean_pitch_rad=mean_pitch,
        tip_twist_amplitude_rad=0.0,
        phase_rad=phase,
        phase_rate_rad_s=phase_rate,
        phase_acceleration_rad_s2=phase_acceleration,
        enabled=True,
    )

    expected_theta = mean_pitch.view(2, 1).expand(2, 4)
    for value in (disabled.delta_theta, disabled.delta_theta_dot, disabled.delta_theta_ddot):
        torch.testing.assert_close(value, torch.zeros_like(value), atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(disabled.theta, expected_theta, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(disabled.theta_dot, torch.zeros_like(disabled.theta_dot), atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(disabled.theta_ddot, torch.zeros_like(disabled.theta_ddot), atol=_ATOL, rtol=0.0)
    for field in ("theta", "theta_dot", "theta_ddot", "delta_theta", "delta_theta_dot", "delta_theta_ddot"):
        torch.testing.assert_close(getattr(zero_tip, field), getattr(disabled, field), atol=_ATOL, rtol=0.0)


def test_spanwise_distribution_uses_outer_strip_edge_for_semi_span() -> None:
    span = torch.tensor([0.0, 0.25, 0.50, 0.75], dtype=_DTYPE)
    width = torch.full_like(span, 0.50)
    tip_amplitude = 0.4
    kinematics = _compute_twist(
        strip_span_m=span,
        strip_width_m=width,
        tip_twist_amplitude_rad=tip_amplitude,
        phase_rad=torch.tensor([math.pi / 2.0], dtype=_DTYPE),
    )

    # R=max(center+width/2)=1.0 m, not max(center)=0.75 m.
    torch.testing.assert_close(kinematics.span_fraction, span.view(1, -1), atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(
        kinematics.delta_theta,
        (-tip_amplitude * span).view(1, -1),
        atol=_ATOL,
        rtol=0.0,
    )
    assert kinematics.delta_theta[0, 0].item() == pytest.approx(0.0, abs=_ATOL)
    assert kinematics.delta_theta[0, 2].item() == pytest.approx(-0.5 * tip_amplitude, abs=_ATOL)


def test_tip_amplitude_describes_the_theoretical_geometric_tip() -> None:
    span = torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=_DTYPE)
    width = torch.full_like(span, 0.2)
    tip_amplitude = 0.32
    kinematics = _compute_twist(
        strip_span_m=span,
        strip_width_m=width,
        tip_twist_amplitude_rad=tip_amplitude,
        phase_rad=torch.tensor([math.pi / 2.0], dtype=_DTYPE),
    )

    expected_last_fraction = 0.7 / 0.8
    assert kinematics.span_fraction[0, -1].item() == pytest.approx(expected_last_fraction, abs=_ATOL)
    assert kinematics.delta_theta[0, -1].item() == pytest.approx(
        -tip_amplitude * expected_last_fraction,
        abs=_ATOL,
    )
    assert abs(kinematics.delta_theta[0, -1].item()) < tip_amplitude


@pytest.mark.parametrize(
    "phase_rad, expected_delta_scale, expected_rate_scale",
    (
        (0.0, 0.0, -1.0),
        (math.pi / 2.0, -1.0, 0.0),
        (math.pi, 0.0, 1.0),
        (3.0 * math.pi / 2.0, 1.0, 0.0),
    ),
)
def test_special_phase_values(
    phase_rad: float,
    expected_delta_scale: float,
    expected_rate_scale: float,
) -> None:
    amplitude = 0.25
    phase_rate = 6.0
    kinematics = _compute_twist(
        strip_span_m=torch.tensor([1.0], dtype=_DTYPE),
        strip_width_m=torch.tensor([0.1], dtype=_DTYPE),
        semi_span_m=1.0,
        tip_twist_amplitude_rad=amplitude,
        phase_rad=torch.tensor([phase_rad], dtype=_DTYPE),
        phase_rate_rad_s=torch.tensor([phase_rate], dtype=_DTYPE),
    )

    assert kinematics.delta_theta.item() == pytest.approx(amplitude * expected_delta_scale, abs=_ATOL)
    assert kinematics.delta_theta_dot.item() == pytest.approx(
        amplitude * phase_rate * expected_rate_scale,
        abs=_ATOL,
    )


def test_analytic_derivatives_match_independent_centered_finite_differences() -> None:
    span = torch.tensor([0.2, 0.6, 0.9], dtype=_DTYPE)
    width = torch.tensor([0.2, 0.2, 0.2], dtype=_DTYPE)
    phase = 0.73
    phase_rate = 8.2
    phase_acceleration = -1.7
    dt = 2.0e-5

    def theta_at(time_s: float) -> torch.Tensor:
        phase_at_time = phase + phase_rate * time_s + 0.5 * phase_acceleration * time_s * time_s
        return _compute_twist(
            strip_span_m=span,
            strip_width_m=width,
            semi_span_m=1.0,
            mean_pitch_rad=0.11,
            tip_twist_amplitude_rad=0.27,
            phase_rad=torch.tensor([phase_at_time], dtype=_DTYPE),
            phase_rate_rad_s=torch.tensor([phase_rate + phase_acceleration * time_s], dtype=_DTYPE),
            phase_acceleration_rad_s2=torch.tensor([phase_acceleration], dtype=_DTYPE),
        ).theta

    analytic = _compute_twist(
        strip_span_m=span,
        strip_width_m=width,
        semi_span_m=1.0,
        mean_pitch_rad=0.11,
        tip_twist_amplitude_rad=0.27,
        phase_rad=torch.tensor([phase], dtype=_DTYPE),
        phase_rate_rad_s=torch.tensor([phase_rate], dtype=_DTYPE),
        phase_acceleration_rad_s2=torch.tensor([phase_acceleration], dtype=_DTYPE),
    )
    theta_minus = theta_at(-dt)
    theta_zero = theta_at(0.0)
    theta_plus = theta_at(dt)
    theta_dot_fd = (theta_plus - theta_minus) / (2.0 * dt)
    theta_ddot_fd = (theta_plus - 2.0 * theta_zero + theta_minus) / (dt * dt)

    torch.testing.assert_close(analytic.theta_dot, theta_dot_fd, atol=2.0e-8, rtol=0.0)
    torch.testing.assert_close(analytic.theta_ddot, theta_ddot_fd, atol=5.0e-7, rtol=0.0)


def test_phase_acceleration_term_is_included_and_constant_frequency_limit_is_exact() -> None:
    span = torch.tensor([0.4, 0.8], dtype=_DTYPE)
    width = torch.tensor([0.2, 0.2], dtype=_DTYPE)
    phase = torch.tensor([0.61], dtype=_DTYPE)
    phase_rate = torch.tensor([7.3], dtype=_DTYPE)
    phase_acceleration = torch.tensor([2.4], dtype=_DTYPE)
    amplitude = 0.18

    variable = _compute_twist(
        strip_span_m=span,
        strip_width_m=width,
        semi_span_m=0.9,
        tip_twist_amplitude_rad=amplitude,
        phase_rad=phase,
        phase_rate_rad_s=phase_rate,
        phase_acceleration_rad_s2=phase_acceleration,
    )
    constant = _compute_twist(
        strip_span_m=span,
        strip_width_m=width,
        semi_span_m=0.9,
        tip_twist_amplitude_rad=amplitude,
        phase_rad=phase,
        phase_rate_rad_s=phase_rate,
        phase_acceleration_rad_s2=torch.zeros_like(phase_acceleration),
    )
    expected_acceleration_contribution = (
        -amplitude
        * variable.span_fraction
        * torch.cos(phase.view(1, 1))
        * phase_acceleration.view(1, 1)
    )
    torch.testing.assert_close(
        variable.theta_ddot - constant.theta_ddot,
        expected_acceleration_contribution,
        atol=_ATOL,
        rtol=0.0,
    )


def test_mean_pitch_is_added_without_changing_dynamic_derivatives() -> None:
    span = torch.tensor([0.25, 0.75], dtype=_DTYPE)
    width = torch.tensor([0.5, 0.5], dtype=_DTYPE)
    zero_mean = _compute_twist(strip_span_m=span, strip_width_m=width, mean_pitch_rad=0.0)
    nonzero_mean = _compute_twist(strip_span_m=span, strip_width_m=width, mean_pitch_rad=0.23)

    torch.testing.assert_close(nonzero_mean.theta, zero_mean.theta + 0.23, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(nonzero_mean.theta_dot, zero_mean.theta_dot, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(nonzero_mean.theta_ddot, zero_mean.theta_ddot, atol=_ATOL, rtol=0.0)


def test_environment_phase_mapping_preserves_delaurier_plunge_and_quadrature() -> None:
    # The environment uses q=Gamma*sin(current_phase), with phase zero neutral
    # and starting upstroke. DeLaurier uses cosine plunge, so
    # phi_D=current_phase-pi/2.
    current_phase = torch.tensor([0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0], dtype=_DTYPE)
    phase_rate = torch.full_like(current_phase, 5.0)
    phase_acceleration = torch.zeros_like(current_phase)
    phase_d, phase_rate_d, phase_acceleration_d = resolve_delaurier_phase(
        current_phase=current_phase,
        current_phase_rate=phase_rate,
        current_phase_acceleration=phase_acceleration,
        phase_direction=1.0,
        phase_offset_rad=-math.pi / 2.0,
    )
    stroke_amplitude = 0.4
    strip_span = 0.8
    q = stroke_amplitude * torch.sin(current_phase)
    h = -q * strip_span
    twist = _compute_twist(
        strip_span_m=torch.tensor([strip_span], dtype=_DTYPE),
        strip_width_m=torch.tensor([0.4], dtype=_DTYPE),
        semi_span_m=1.0,
        tip_twist_amplitude_rad=0.2,
        phase_rad=phase_d,
        phase_rate_rad_s=phase_rate_d,
        phase_acceleration_rad_s2=phase_acceleration_d,
    )

    torch.testing.assert_close(phase_d, current_phase - math.pi / 2.0, atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(
        h,
        torch.tensor([0.0, -0.32, 0.0, 0.32], dtype=_DTYPE),
        atol=_ATOL,
        rtol=0.0,
    )
    torch.testing.assert_close(
        twist.delta_theta[:, 0],
        torch.tensor([0.16, 0.0, -0.16, 0.0], dtype=_DTYPE),
        atol=_ATOL,
        rtol=0.0,
    )


def _end_to_end_case(*, tip_amplitude_rad: float):
    geometry = WingGeometry.from_input(
        {"R": 0.8, "N": 4, "c": 0.12, "dhat": 0.0, "aspect_ratio": 4.0},
        dtype=_DTYPE,
    )
    phase = torch.tensor([0.63], dtype=_DTYPE)
    phase_rate = torch.tensor([9.1], dtype=_DTYPE)
    phase_acceleration = torch.tensor([0.0], dtype=_DTYPE)
    stroke_amplitude = 0.35
    y = geometry.x_mid.view(1, -1)
    q = stroke_amplitude * torch.cos(phase)
    qd = -stroke_amplitude * phase_rate * torch.sin(phase)
    qdd = -stroke_amplitude * phase_rate.square() * torch.cos(phase)
    h = -q.view(1, 1) * y
    hdot = -qd.view(1, 1) * y
    hddot = -qdd.view(1, 1) * y
    theta_bar = torch.tensor([0.08], dtype=_DTYPE)
    twist = _compute_twist(
        strip_span_m=geometry.x_mid,
        strip_width_m=geometry.dx,
        semi_span_m=geometry.R,
        mean_pitch_rad=theta_bar,
        tip_twist_amplitude_rad=tip_amplitude_rad,
        phase_rad=phase,
        phase_rate_rad_s=phase_rate,
        phase_acceleration_rad_s2=phase_acceleration,
        enabled=True,
    )
    loads = compute_delaurier_strip_loads(
        h,
        hdot,
        hddot,
        twist.theta,
        twist.theta_dot,
        twist.theta_ddot,
        geometry,
        rho=1.225,
        U=8.0,
        theta_a=0.04,
        theta_bar=theta_bar.view(1, 1),
        omega_ref=phase_rate.view(1, 1).expand(1, geometry.x_mid.numel()),
        params=DeLaurierParams(alpha0_rad=0.01, c_mac=0.0, cd_f=0.028),
        enable_separation=False,
    )
    return geometry, twist, loads, integrate_delaurier_strip_wrench(loads)


def test_dynamic_twist_changes_force_moment_power_and_apparent_mass_couple() -> None:
    geometry, no_twist, loads_zero, wrench_zero = _end_to_end_case(tip_amplitude_rad=0.0)
    _geometry, with_twist, loads_twist, wrench_twist = _end_to_end_case(tip_amplitude_rad=0.21)

    assert torch.count_nonzero(with_twist.theta - no_twist.theta).item() > 0
    assert torch.count_nonzero(with_twist.theta_dot).item() > 0
    assert torch.count_nonzero(with_twist.theta_ddot).item() > 0
    for field in ("dN_c", "dN_a", "dT_s", "dM_a", "power_input", "alpha", "alpha_prime", "alpha_le"):
        assert not torch.allclose(
            getattr(loads_twist, field),
            getattr(loads_zero, field),
            atol=1.0e-10,
            rtol=0.0,
        ), field
    assert not torch.allclose(wrench_twist.force_wang, wrench_zero.force_wang, atol=1.0e-10, rtol=0.0)
    assert not torch.allclose(
        wrench_twist.moment_wang_about_wing_origin,
        wrench_zero.moment_wang_about_wing_origin,
        atol=1.0e-10,
        rtol=0.0,
    )
    torch.testing.assert_close(loads_zero.dM_a, torch.zeros_like(loads_zero.dM_a), atol=_ATOL, rtol=0.0)
    assert torch.any(loads_twist.dM_a.abs() > 0.0)

    chord = geometry.c.view(1, -1)
    strip_width = geometry.dx.view(1, -1)
    expected_dM_a = -(
        (1.225 * math.pi * chord**3) * (with_twist.theta_dot * 8.0) / 16.0
        + (1.225 * math.pi * chord**4) * with_twist.theta_ddot / 128.0
    ) * strip_width
    torch.testing.assert_close(loads_twist.dM_a, expected_dM_a, atol=_ATOL, rtol=0.0)


def test_left_right_dynamic_twist_and_integrated_wrench_are_mirror_consistent() -> None:
    geometry = WingGeometry.from_input(
        {"R": 0.8, "N": 4, "c": 0.12, "dhat": 0.0, "aspect_ratio": 4.0},
        dtype=_DTYPE,
    )
    phase = torch.tensor([0.71, 0.71], dtype=_DTYPE)
    phase_rate = torch.tensor([8.4, 8.4], dtype=_DTYPE)
    twist = _compute_twist(
        strip_span_m=geometry.x_mid,
        strip_width_m=geometry.dx,
        semi_span_m=geometry.R,
        mean_pitch_rad=torch.tensor([0.07, 0.07], dtype=_DTYPE),
        tip_twist_amplitude_rad=0.19,
        phase_rad=phase,
        phase_rate_rad_s=phase_rate,
        phase_acceleration_rad_s2=torch.zeros_like(phase),
    )
    torch.testing.assert_close(twist.delta_theta[0], twist.delta_theta[1], atol=_ATOL, rtol=0.0)

    y = geometry.x_mid.view(1, -1).expand(2, -1)
    q = 0.34 * torch.cos(phase)
    qd = -0.34 * phase_rate * torch.sin(phase)
    qdd = -0.34 * phase_rate.square() * torch.cos(phase)
    loads = compute_delaurier_strip_loads(
        -q.view(2, 1) * y,
        -qd.view(2, 1) * y,
        -qdd.view(2, 1) * y,
        twist.theta,
        twist.theta_dot,
        twist.theta_ddot,
        geometry,
        rho=1.225,
        U=8.0,
        theta_a=0.04,
        theta_bar=torch.tensor([[0.07], [0.07]], dtype=_DTYPE),
        omega_ref=phase_rate.view(2, 1).expand(2, geometry.x_mid.numel()),
        params=DeLaurierParams(alpha0_rad=0.01, c_mac=0.0, cd_f=0.028),
        enable_separation=False,
    )
    wrench = integrate_delaurier_strip_wrench(loads)
    torch.testing.assert_close(wrench.force_wang[0], wrench.force_wang[1], atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(
        wrench.moment_wang_about_wing_origin[0],
        wrench.moment_wang_about_wing_origin[1],
        atol=_ATOL,
        rtol=0.0,
    )

    force_left, moment_left = transform_wang_wrench_to_link(
        wrench.force_wang[0:1],
        wrench.moment_wang_about_wing_origin[0:1],
        _WANG_TO_LEFT_LINK,
    )
    force_right, moment_right = transform_wang_wrench_to_link(
        wrench.force_wang[1:2],
        wrench.moment_wang_about_wing_origin[1:2],
        _WANG_TO_RIGHT_LINK,
    )
    base_com = torch.zeros((1, 3), dtype=_DTYPE)
    total_force = force_left + force_right
    total_moment = translate_wrench_moment(
        force_left,
        moment_left,
        torch.tensor([[0.0, 0.2, 0.0]], dtype=_DTYPE),
        base_com,
    ) + translate_wrench_moment(
        force_right,
        moment_right,
        torch.tensor([[0.0, -0.2, 0.0]], dtype=_DTYPE),
        base_com,
    )
    torch.testing.assert_close(total_force[:, 1], torch.zeros(1, dtype=_DTYPE), atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(total_moment[:, 0], torch.zeros(1, dtype=_DTYPE), atol=_ATOL, rtol=0.0)
    torch.testing.assert_close(total_moment[:, 2], torch.zeros(1, dtype=_DTYPE), atol=_ATOL, rtol=0.0)
    assert torch.all(total_moment[:, 1].abs() > 0.0)

    # Free moment and local twist angular velocity are both axial vectors;
    # reflection therefore preserves their mechanical-power dot product.
    zero_force = torch.zeros((1, 3), dtype=_DTYPE)
    free_left_wang = wrench.moment_from_dM_a_wang[0:1]
    free_right_wang = wrench.moment_from_dM_a_wang[1:2]
    _unused, free_left_link = transform_wang_wrench_to_link(zero_force, free_left_wang, _WANG_TO_LEFT_LINK)
    _unused, free_right_link = transform_wang_wrench_to_link(zero_force, free_right_wang, _WANG_TO_RIGHT_LINK)
    theta_dot_scalar = twist.theta_dot.sum(dim=1, keepdim=True)
    omega_twist_link = torch.cat(
        (torch.zeros_like(theta_dot_scalar), theta_dot_scalar, torch.zeros_like(theta_dot_scalar)),
        dim=1,
    )
    torch.testing.assert_close(
        torch.sum(free_left_link * omega_twist_link[0:1], dim=1),
        torch.sum(free_right_link * omega_twist_link[1:2], dim=1),
        atol=_ATOL,
        rtol=0.0,
    )


def test_dynamic_twist_modes_are_mutually_exclusive_and_invalid_mode_is_rejected() -> None:
    assert validate_delaurier_dynamic_twist_mode("disabled") == "disabled"
    assert validate_delaurier_dynamic_twist_mode("delaurier_linear_spanwise") == "delaurier_linear_spanwise"
    assert validate_delaurier_dynamic_twist_mode("legacy_qd_scaled_proxy") == "legacy_qd_scaled_proxy"
    with pytest.raises(ValueError, match="Unsupported DeLaurier dynamic_twist_mode"):
        validate_delaurier_dynamic_twist_mode("linear_plus_legacy")

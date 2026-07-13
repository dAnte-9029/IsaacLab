"""Pure-model tests for DeLaurier strip load and wrench integration."""

from __future__ import annotations

import torch

from flapping_bot.physics.qsm_delaurier1993 import (
    DeLaurierParams,
    DeLaurierStripLoads,
    compute_aero_wrench_delaurier1993,
    compute_delaurier_strip_loads,
    integrate_delaurier_strip_wrench,
    transform_wang_wrench_to_link,
    translate_wrench_moment,
)
from flapping_bot.physics.qsm_wang2016 import WingGeometry


def _manual_strip_loads() -> DeLaurierStripLoads:
    """Return two attached-flow strips with non-zero force and free couples."""

    dtype = torch.float64
    dN_c = torch.tensor([[2.0, 3.0]], dtype=dtype)
    dN_a = torch.tensor([[5.0, 7.0]], dtype=dtype)
    dT_s = torch.tensor([[11.0, 13.0]], dtype=dtype)
    dD_camber = torch.tensor([[1.0, 2.0]], dtype=dtype)
    dD_f = torch.tensor([[0.5, 0.25]], dtype=dtype)
    dM_ac = torch.tensor([[0.4, 0.1]], dtype=dtype)
    dM_a = torch.tensor([[-0.2, 0.3]], dtype=dtype)
    span = torch.tensor([[0.10, 0.20]], dtype=dtype)
    chord = torch.tensor([[0.20, 0.10]], dtype=dtype)
    d_hat = torch.zeros_like(span)
    return DeLaurierStripLoads(
        dN_c=dN_c,
        dN_a=dN_a,
        dT_s=dT_s,
        dD_camber=dD_camber,
        dD_f=dD_f,
        dM_ac=dM_ac,
        dM_a=dM_a,
        span=span,
        chord=chord,
        strip_width=torch.full_like(span, 0.1),
        d_hat=d_hat,
        resultant_normal_force=dN_c + dN_a,
        resultant_chordwise_force=dT_s - dD_camber - dD_f,
        power_input=torch.zeros_like(span),
        separation_weight=torch.zeros_like(span),
    )


def _delaurier_inputs() -> tuple[torch.Tensor, ...]:
    dtype = torch.float64
    return (
        torch.tensor([[0.010, 0.020, 0.030, 0.040]], dtype=dtype),
        torch.tensor([[0.80, 0.90, 1.00, 1.10]], dtype=dtype),
        torch.tensor([[2.0, 2.1, 2.2, 2.3]], dtype=dtype),
        torch.full((1, 4), 0.12, dtype=dtype),
        torch.full((1, 4), 0.35, dtype=dtype),
        torch.full((1, 4), 0.40, dtype=dtype),
    )


def _delaurier_geometry() -> WingGeometry:
    return WingGeometry.from_input(
        {"R": 0.24, "N": 4, "c": 0.06, "dhat": 0.0, "aspect_ratio": 4.0},
        dtype=torch.float64,
    )


def test_strip_force_matches_legacy_aggregate_without_separation() -> None:
    h, hdot, hddot, theta, thetad, thetadd = _delaurier_inputs()
    geometry = _delaurier_geometry()
    params = DeLaurierParams(alpha0_rad=0.01, c_mac=0.02, cd_f=0.028)

    strip_loads = compute_delaurier_strip_loads(
        h,
        hdot,
        hddot,
        theta,
        thetad,
        thetadd,
        geometry,
        rho=1.225,
        U=8.0,
        theta_a=0.04,
        theta_bar=0.08,
        omega_ref=24.0,
        params=params,
        enable_separation=False,
    )
    strip_wrench = integrate_delaurier_strip_wrench(strip_loads)
    legacy_force, _legacy_moment, _power, _sep = compute_aero_wrench_delaurier1993(
        h,
        hdot,
        hddot,
        theta,
        thetad,
        thetadd,
        geometry,
        rho=1.225,
        U=8.0,
        theta_a=0.04,
        theta_bar=0.08,
        omega_ref=24.0,
        params=params,
        enable_separation=False,
    )

    assert torch.allclose(strip_wrench.force_wang, legacy_force, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(
        strip_loads.normal_force_total.sum(dim=1), legacy_force[:, 1], atol=1.0e-12, rtol=1.0e-12
    )
    assert torch.allclose(
        strip_loads.chordwise_force_total.sum(dim=1), legacy_force[:, 2], atol=1.0e-12, rtol=1.0e-12
    )


def test_strip_moment_is_sum_of_named_components() -> None:
    wrench = integrate_delaurier_strip_wrench(_manual_strip_loads())

    component_sum = sum(wrench.component_moments_wang.values())
    assert torch.allclose(component_sum, wrench.moment_wang_about_wing_origin, atol=1.0e-12, rtol=1.0e-12)


def test_chordwise_application_point_is_invariant_along_chord_line() -> None:
    strip_loads = _manual_strip_loads()
    moment_at_pitch_axis = integrate_delaurier_strip_wrench(strip_loads).moment_from_chordwise_wang
    moment_at_le = integrate_delaurier_strip_wrench(
        strip_loads,
        chordwise_application_chord_fraction_from_le=0.0,
    ).moment_from_chordwise_wang
    moment_at_quarter_chord = integrate_delaurier_strip_wrench(
        strip_loads,
        chordwise_application_chord_fraction_from_le=0.25,
    ).moment_from_chordwise_wang
    moment_at_mid_chord = integrate_delaurier_strip_wrench(
        strip_loads,
        chordwise_application_chord_fraction_from_le=0.5,
    ).moment_from_chordwise_wang

    assert torch.allclose(moment_at_pitch_axis, moment_at_le, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(moment_at_pitch_axis, moment_at_quarter_chord, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(moment_at_pitch_axis, moment_at_mid_chord, atol=1.0e-12, rtol=1.0e-12)


def test_normal_force_application_points_use_quarter_and_mid_chord() -> None:
    dtype = torch.float64
    zero = torch.zeros((1, 1), dtype=dtype)
    strip_loads = DeLaurierStripLoads(
        dN_c=torch.tensor([[2.0]], dtype=dtype),
        dN_a=torch.tensor([[3.0]], dtype=dtype),
        dT_s=zero,
        dD_camber=zero,
        dD_f=zero,
        dM_ac=zero,
        dM_a=zero,
        span=torch.tensor([[0.10]], dtype=dtype),
        chord=torch.tensor([[0.20]], dtype=dtype),
        strip_width=torch.tensor([[0.20]], dtype=dtype),
        d_hat=zero,
        resultant_normal_force=torch.tensor([[5.0]], dtype=dtype),
        resultant_chordwise_force=zero,
        power_input=zero,
        separation_weight=zero,
    )

    wrench = integrate_delaurier_strip_wrench(strip_loads)
    assert torch.allclose(wrench.moment_from_dN_c_wang, torch.tensor([[0.10, 0.0, 0.20]], dtype=dtype))
    assert torch.allclose(wrench.moment_from_dN_a_wang, torch.tensor([[0.30, 0.0, 0.30]], dtype=dtype))


def test_free_moment_switches_only_remove_the_selected_couple() -> None:
    strip_loads = _manual_strip_loads()
    both = integrate_delaurier_strip_wrench(strip_loads)
    without_ac = integrate_delaurier_strip_wrench(strip_loads, include_aerodynamic_center_moment=False)
    without_apparent_mass = integrate_delaurier_strip_wrench(strip_loads, include_apparent_mass_moment=False)

    assert torch.allclose(
        both.moment_wang_about_wing_origin - without_ac.moment_wang_about_wing_origin,
        both.moment_from_dM_ac_wang,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert torch.allclose(
        both.moment_wang_about_wing_origin - without_apparent_mass.moment_wang_about_wing_origin,
        both.moment_from_dM_a_wang,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_left_right_wang_mapping_preserves_symmetric_pitch_couples() -> None:
    wrench = integrate_delaurier_strip_wrench(_manual_strip_loads())
    wang_to_left_link = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64
    )
    wang_to_right_link = torch.tensor(
        [[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64
    )
    force_left, moment_left = transform_wang_wrench_to_link(
        wrench.force_wang, wrench.moment_wang_about_wing_origin, wang_to_left_link
    )
    force_right, moment_right = transform_wang_wrench_to_link(
        wrench.force_wang, wrench.moment_wang_about_wing_origin, wang_to_right_link
    )
    origin_left = torch.tensor([[0.0, 0.20, 0.0]], dtype=torch.float64)
    origin_right = torch.tensor([[0.0, -0.20, 0.0]], dtype=torch.float64)
    base_com = torch.zeros((1, 3), dtype=torch.float64)
    total_force = force_left + force_right
    total_moment = (
        translate_wrench_moment(force_left, moment_left, origin_left, base_com)
        + translate_wrench_moment(force_right, moment_right, origin_right, base_com)
    )

    assert torch.allclose(total_force[:, 1], torch.zeros(1, dtype=torch.float64), atol=1.0e-12)
    assert torch.allclose(total_moment[:, 0], torch.zeros(1, dtype=torch.float64), atol=1.0e-12)
    assert torch.allclose(total_moment[:, 2], torch.zeros(1, dtype=torch.float64), atol=1.0e-12)
    assert torch.all(total_moment[:, 1].abs() > 0.0)

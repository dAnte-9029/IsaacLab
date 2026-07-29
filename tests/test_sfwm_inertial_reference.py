from __future__ import annotations

import math

import pytest
import torch

from flapping_bot.physics import (
    compute_amini_sfwm_inertial_response,
    measured_flapping_bot_amini_params,
)


def test_measured_sfwm_parameter_mapping_is_self_consistent() -> None:
    params = measured_flapping_bot_amini_params()

    assert params.total_mass_kg == pytest.approx(0.90415, abs=1.0e-12)
    assert params.wing_to_weight_ratio == pytest.approx(2.0 * 0.06077 / 0.90415)
    assert params.wing_cg_from_pivot_x_m == pytest.approx(-0.06040)
    assert params.wing_cg_from_pivot_y_m == pytest.approx(0.29394)
    assert params.constant_pitch_inertia_kg_m2 > params.body_inertia_yy_kg_m2

    # Amini Eqs. (A41)--(A42) must close for the derived point O.
    body_cg_x_flu = -0.13103
    point_o_x_flu = params.point_o_in_base_flu_m[0]
    body_cg_from_o_x = body_cg_x_flu - point_o_x_flu
    expected_body_x = (
        -2.0
        * params.wing_mass_each_kg
        / params.body_mass_kg
        * (params.wing_pivot_from_o_x_m + params.wing_cg_from_pivot_x_m)
    )
    assert body_cg_from_o_x == pytest.approx(expected_body_x, abs=1.0e-12)
    expected_body_z = (
        2.0
        * params.wing_mass_each_kg
        / params.body_mass_kg
        * (
            params.wing_cg_from_pivot_y_m * math.sin(params.gamma_mean_rad)
            - params.wing_pivot_from_o_z_m
        )
    )
    assert params.body_cg_from_o_z_m == pytest.approx(expected_body_z, abs=1.0e-12)


def test_sfwm_response_matches_hand_calculation_and_preserves_dtype() -> None:
    params = measured_flapping_bot_amini_params()
    gamma = torch.tensor([0.1], dtype=torch.float64)
    gamma_dot = torch.tensor([2.0], dtype=torch.float64)
    gamma_ddot = torch.tensor([-3.0], dtype=torch.float64)

    response = compute_amini_sfwm_inertial_response(
        gamma_rad=gamma,
        gamma_dot_rad_s=gamma_dot,
        gamma_ddot_rad_s2=gamma_ddot,
        params=params,
    )

    common = -3.0 * math.cos(0.1) - 4.0 * math.sin(0.1)
    expected_vertical = params.wing_to_weight_ratio * params.wing_cg_from_pivot_y_m * common
    expected_pitch = (
        -2.0
        * params.wing_cg_from_pivot_y_m
        * (params.wing_cg_from_pivot_x_m + params.wing_pivot_from_o_x_m)
        * params.wing_mass_each_kg
        / params.constant_pitch_inertia_kg_m2
        * common
    )
    assert response.vertical_acceleration_frd_m_s2.dtype == torch.float64
    assert response.vertical_acceleration_frd_m_s2.item() == pytest.approx(expected_vertical)
    assert response.pitch_angular_acceleration_frd_rad_s2.item() == pytest.approx(expected_pitch)


def test_sfwm_inertial_amplitude_has_frequency_squared_scaling() -> None:
    params = measured_flapping_bot_amini_params()
    amplitude = math.radians(30.0)
    phase = torch.linspace(0.0, 2.0 * math.pi, 2001, dtype=torch.float64)
    peak_by_frequency: dict[float, float] = {}
    for frequency_hz in (2.0, 5.0):
        omega = 2.0 * math.pi * frequency_hz
        stroke = amplitude * torch.sin(phase)
        response = compute_amini_sfwm_inertial_response(
            gamma_rad=params.gamma_mean_rad + stroke,
            gamma_dot_rad_s=amplitude * omega * torch.cos(phase),
            gamma_ddot_rad_s2=-amplitude * omega**2 * torch.sin(phase),
            params=params,
        )
        peak_by_frequency[frequency_hz] = float(
            torch.max(torch.abs(response.vertical_acceleration_frd_m_s2)).item()
        )
    assert peak_by_frequency[5.0] / peak_by_frequency[2.0] == pytest.approx((5.0 / 2.0) ** 2)

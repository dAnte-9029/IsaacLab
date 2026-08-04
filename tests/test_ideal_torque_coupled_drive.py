from __future__ import annotations

import math

import pytest
import torch

from flapping_bot.physics import (
    IDEAL_TORQUE_EQUIVALENT_INERTIA_KG_M2,
    RIGHT_WING_LINK_MASS_PROPERTIES,
    SINGLE_WING_HINGE_INERTIA_KG_M2,
    apply_quintic_amplitude_ramp,
    compute_common_aerodynamic_hinge_torque,
    compute_ideal_torque_drive_effort,
    ideal_torque_drive_gains,
)


def test_equivalent_inertia_uses_two_measured_wings_about_the_hinge() -> None:
    wing = RIGHT_WING_LINK_MASS_PROPERTIES
    expected_single = (
        wing.inertia_kg_m2[0][0]
        + wing.mass_kg * (wing.com_m[1] ** 2 + wing.com_m[2] ** 2)
    )

    assert SINGLE_WING_HINGE_INERTIA_KG_M2 == pytest.approx(expected_single)
    assert IDEAL_TORQUE_EQUIVALENT_INERTIA_KG_M2 == pytest.approx(2.0 * expected_single)


def test_drive_gains_follow_inertia_scaled_second_order_form() -> None:
    stiffness, damping = ideal_torque_drive_gains(
        equivalent_inertia_kg_m2=0.02,
        natural_frequency_hz=10.0,
        damping_ratio=0.8,
    )
    omega_n = 2.0 * math.pi * 10.0

    assert stiffness == pytest.approx(0.02 * omega_n**2)
    assert damping == pytest.approx(2.0 * 0.8 * 0.02 * omega_n)


def test_discrete_drive_gains_place_a_repeated_critical_pole() -> None:
    dt = 1.0 / 480.0
    inertia = 0.02
    frequency_hz = 50.0
    stiffness, damping = ideal_torque_drive_gains(
        equivalent_inertia_kg_m2=inertia,
        natural_frequency_hz=frequency_hz,
        damping_ratio=1.0,
        physics_dt_s=dt,
    )
    pole = math.exp(-2.0 * math.pi * frequency_hz * dt)

    assert stiffness == pytest.approx(inertia * (1.0 - pole) ** 2 / dt**2)
    assert damping == pytest.approx(inertia * (1.0 - pole**2) / dt)


def test_drive_effort_adds_inertia_feedforward_and_pd_feedback() -> None:
    effort = compute_ideal_torque_drive_effort(
        position_rad=torch.tensor([0.1], dtype=torch.float64),
        velocity_rad_s=torch.tensor([0.2], dtype=torch.float64),
        reference_position_rad=torch.tensor([0.12], dtype=torch.float64),
        reference_velocity_rad_s=torch.tensor([0.3], dtype=torch.float64),
        reference_acceleration_rad_s2=torch.tensor([4.0], dtype=torch.float64),
        equivalent_inertia_kg_m2=0.02,
        natural_frequency_hz=10.0,
        damping_ratio=1.0,
        effort_limit_nm=100.0,
    )
    stiffness, damping = ideal_torque_drive_gains(
        equivalent_inertia_kg_m2=0.02,
        natural_frequency_hz=10.0,
        damping_ratio=1.0,
    )
    expected = 0.02 * 4.0 + stiffness * 0.02 + damping * 0.1

    torch.testing.assert_close(effort, torch.tensor([expected], dtype=torch.float64))


def test_drive_effort_is_clipped_by_the_ideal_safety_limit() -> None:
    effort = compute_ideal_torque_drive_effort(
        position_rad=torch.zeros(2),
        velocity_rad_s=torch.zeros(2),
        reference_position_rad=torch.tensor([10.0, -10.0]),
        reference_velocity_rad_s=torch.zeros(2),
        reference_acceleration_rad_s2=torch.zeros(2),
        equivalent_inertia_kg_m2=0.02,
        natural_frequency_hz=10.0,
        damping_ratio=1.0,
        effort_limit_nm=3.0,
    )

    torch.testing.assert_close(effort, torch.tensor([3.0, -3.0]))


def test_common_aerodynamic_torque_projects_opposed_joint_work() -> None:
    force = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]])
    moment_about_com = torch.tensor([[[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]]])
    com = torch.tensor([[[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]])

    common_torque = compute_common_aerodynamic_hinge_torque(
        force_link_n=force,
        moment_link_about_com_nm=moment_about_com,
        wing_com_position_link_m=com,
    )

    # Left hinge torque is +0.7 N m, right is -0.7 N m, and
    # q_right=-q gives Q=tau_left-tau_right.
    torch.testing.assert_close(common_torque, torch.tensor([1.4]))


def test_quintic_ramp_starts_at_rest_and_recovers_full_kinematics() -> None:
    position = torch.tensor([0.2, 0.2])
    velocity = torch.tensor([3.0, 3.0])
    acceleration = torch.tensor([-4.0, -4.0])
    ramped = apply_quintic_amplitude_ramp(
        position_rad=position,
        velocity_rad_s=velocity,
        acceleration_rad_s2=acceleration,
        elapsed_time_s=torch.tensor([0.0, 2.0]),
        duration_s=torch.tensor([1.0, 1.0]),
    )

    torch.testing.assert_close(ramped[0], torch.tensor([0.0, 0.2]))
    torch.testing.assert_close(ramped[1], torch.tensor([0.0, 3.0]))
    torch.testing.assert_close(ramped[2], torch.tensor([0.0, -4.0]))

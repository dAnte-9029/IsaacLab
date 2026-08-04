from __future__ import annotations

import math

import pytest
import torch

from flapping_bot.physics.sinusoidal_phase_drive import (
    SinusoidalPhaseDriveState,
    SinusoidalPhaseSpeedDriveConfig,
    compute_opposed_wing_kinematics,
    compute_phase_acceleration,
    compute_phase_inertia_terms,
    compute_phase_speed_pi_torque,
    compute_sinusoidal_constraint_effort,
    map_virtual_throttle_to_frequency_hz,
    phase_speed_pi_gains,
    project_common_joint_torque_to_phase,
    step_sinusoidal_phase_speed_drive,
)


def test_virtual_throttle_maps_zero_to_stop_and_one_to_maximum() -> None:
    throttle = torch.tensor([-0.2, 0.0, 0.4, 1.0, 1.3], dtype=torch.float64)

    frequency = map_virtual_throttle_to_frequency_hz(
        throttle_01=throttle,
        max_frequency_hz=5.0,
    )

    torch.testing.assert_close(
        frequency,
        torch.tensor([0.0, 0.0, 2.0, 5.0, 5.0], dtype=torch.float64),
    )


def test_variable_frequency_sine_kinematics_include_phase_acceleration() -> None:
    phase = torch.tensor([0.0, math.pi / 2.0], dtype=torch.float64)
    phase_rate = torch.tensor([4.0, 4.0], dtype=torch.float64)
    phase_acceleration = torch.tensor([3.0, 3.0], dtype=torch.float64)

    result = compute_opposed_wing_kinematics(
        phase_rad=phase,
        phase_rate_rad_s=phase_rate,
        phase_acceleration_rad_s2=phase_acceleration,
        amplitude_rad=0.5,
        left_joint_mid_rad=0.1,
        right_joint_mid_rad=-0.2,
    )

    torch.testing.assert_close(result.common_position_rad, torch.tensor([0.0, 0.5], dtype=torch.float64))
    torch.testing.assert_close(result.common_velocity_rad_s, torch.tensor([2.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(result.common_acceleration_rad_s2, torch.tensor([1.5, -8.0], dtype=torch.float64))
    torch.testing.assert_close(result.left_position_rad - 0.1, result.common_position_rad)
    torch.testing.assert_close(result.right_position_rad + 0.2, -result.common_position_rad)
    torch.testing.assert_close(result.left_velocity_rad_s, -result.right_velocity_rad_s)
    torch.testing.assert_close(result.left_acceleration_rad_s2, -result.right_acceleration_rad_s2)


def test_unwrapped_phase_has_continuous_periodic_wing_kinematics() -> None:
    epsilon = 1.0e-7
    phase = torch.tensor([2.0 * math.pi - epsilon, 2.0 * math.pi + epsilon], dtype=torch.float64)
    phase_rate = torch.full_like(phase, 6.0)
    phase_acceleration = torch.zeros_like(phase)

    result = compute_opposed_wing_kinematics(
        phase_rad=phase,
        phase_rate_rad_s=phase_rate,
        phase_acceleration_rad_s2=phase_acceleration,
        amplitude_rad=math.radians(30.0),
    )

    assert result.common_position_rad[1] - result.common_position_rad[0] == pytest.approx(
        2.0 * math.radians(30.0) * epsilon,
        abs=1.0e-14,
    )
    assert result.common_velocity_rad_s[1] == pytest.approx(
        result.common_velocity_rad_s[0].item(),
        abs=1.0e-12,
    )


def test_phase_load_projection_preserves_instantaneous_power() -> None:
    phase = torch.tensor([0.3, 1.2], dtype=torch.float64)
    phase_rate = torch.tensor([4.0, 7.0], dtype=torch.float64)
    common_torque = torch.tensor([3.0, -2.0], dtype=torch.float64)
    amplitude = 0.5
    common_velocity = amplitude * torch.cos(phase) * phase_rate

    phase_torque = project_common_joint_torque_to_phase(
        common_joint_torque_nm=common_torque,
        phase_rad=phase,
        amplitude_rad=amplitude,
    )

    torch.testing.assert_close(phase_torque * phase_rate, common_torque * common_velocity)


def test_configuration_inertia_term_preserves_unforced_energy_rate() -> None:
    phase = torch.tensor([0.2, 0.9], dtype=torch.float64)
    phase_rate = torch.tensor([3.0, 5.0], dtype=torch.float64)
    phase_inertia, phase_inertia_derivative = compute_phase_inertia_terms(
        phase_rad=phase,
        common_joint_inertia_kg_m2=0.02,
        amplitude_rad=0.5,
        phase_inertia_floor_kg_m2=1.0e-4,
    )
    phase_acceleration = compute_phase_acceleration(
        phase_rate_rad_s=phase_rate,
        drive_torque_nm=torch.zeros_like(phase),
        phase_external_torque_nm=torch.zeros_like(phase),
        phase_inertia_kg_m2=phase_inertia,
        phase_inertia_derivative_kg_m2=phase_inertia_derivative,
        phase_viscous_damping_nm_s=0.0,
    )
    kinetic_energy_rate = (
        0.5 * phase_inertia_derivative * phase_rate.pow(3)
        + phase_inertia * phase_rate * phase_acceleration
    )

    torch.testing.assert_close(kinetic_energy_rate, torch.zeros_like(kinetic_energy_rate), atol=1.0e-12, rtol=0.0)


def test_phase_speed_pi_gains_match_second_order_coefficients() -> None:
    proportional_gain, integral_gain = phase_speed_pi_gains(
        settling_time_s=0.2,
        damping_ratio=1.0,
    )
    natural_frequency = 4.0 / 0.2

    assert proportional_gain == pytest.approx(2.0 * natural_frequency)
    assert integral_gain == pytest.approx(natural_frequency**2)


def test_phase_speed_pi_holds_integral_when_error_drives_saturation() -> None:
    result = compute_phase_speed_pi_torque(
        phase_rate_rad_s=torch.tensor([0.0, 20.0], dtype=torch.float64),
        target_phase_rate_rad_s=torch.tensor([20.0, 0.0], dtype=torch.float64),
        speed_error_integral_rad=torch.tensor([0.4, -0.4], dtype=torch.float64),
        phase_inertia_kg_m2=torch.ones(2, dtype=torch.float64),
        phase_inertia_derivative_kg_m2=torch.zeros(2, dtype=torch.float64),
        physics_dt_s=0.1,
        proportional_gain_s_inv=2.0,
        integral_gain_s_inv2=3.0,
        phase_viscous_damping_nm_s=0.0,
        effort_limit_nm=1.0,
    )

    torch.testing.assert_close(
        result.next_speed_error_integral_rad,
        torch.tensor([0.4, -0.4], dtype=torch.float64),
    )
    torch.testing.assert_close(result.drive_torque_nm, torch.tensor([1.0, -1.0], dtype=torch.float64))
    assert bool(torch.all(result.saturated))


def test_phase_step_uses_continuous_phase_and_preserves_tensor_contract() -> None:
    config = SinusoidalPhaseSpeedDriveConfig()
    state = SinusoidalPhaseDriveState(
        phase_rad=torch.tensor([2.0 * math.pi, 0.2], dtype=torch.float64),
        phase_rate_rad_s=torch.tensor([0.0, 2.0], dtype=torch.float64),
        speed_error_integral_rad=torch.zeros(2, dtype=torch.float64),
    )

    result = step_sinusoidal_phase_speed_drive(
        state=state,
        throttle_01=torch.tensor([0.0, 0.4], dtype=torch.float64),
        common_joint_external_torque_nm=torch.zeros(2, dtype=torch.float64),
        physics_dt_s=1.0 / 1000.0,
        config=config,
    )

    assert result.next_state.phase_rad.dtype == torch.float64
    assert result.next_state.phase_rad.shape == (2,)
    assert result.next_state.phase_rad[0] == pytest.approx(2.0 * math.pi)
    assert result.target_frequency_hz.tolist() == pytest.approx([0.0, 2.0])
    assert bool(torch.all(torch.isfinite(result.phase_acceleration_rad_s2)))
    assert bool(torch.all(torch.isfinite(result.next_state.phase_rate_rad_s)))


def test_default_config_records_accepted_initial_numerical_choices() -> None:
    config = SinusoidalPhaseSpeedDriveConfig()

    assert config.amplitude_rad == pytest.approx(math.radians(30.0))
    assert config.max_frequency_hz == pytest.approx(5.0)
    assert config.frequency_settling_time_s == pytest.approx(0.15)
    assert config.phase_inertia_floor_kg_m2 == pytest.approx(
        0.01 * config.common_joint_inertia_kg_m2 * config.amplitude_rad**2
    )
    assert config.effort_limit_nm == pytest.approx(1000.0)


def test_coupled_config_can_disable_transformed_wing_inertia_without_zero_total_inertia() -> None:
    standalone = SinusoidalPhaseSpeedDriveConfig()
    coupled = SinusoidalPhaseSpeedDriveConfig(
        common_joint_inertia_kg_m2=0.0,
        constant_phase_inertia_kg_m2=standalone.phase_inertia_floor_kg_m2,
    )

    phase = torch.tensor([0.0, math.pi / 2.0], dtype=torch.float64)
    inertia, derivative = compute_phase_inertia_terms(
        phase_rad=phase,
        common_joint_inertia_kg_m2=coupled.common_joint_inertia_kg_m2,
        amplitude_rad=coupled.amplitude_rad,
        phase_inertia_floor_kg_m2=coupled.phase_inertia_floor_kg_m2,
    )

    torch.testing.assert_close(
        inertia,
        torch.full_like(phase, standalone.phase_inertia_floor_kg_m2),
    )
    torch.testing.assert_close(derivative, torch.zeros_like(phase))


def test_constraint_effort_sign_and_internal_power_are_consistent() -> None:
    phase = torch.tensor([0.3], dtype=torch.float64)
    phase_rate = torch.tensor([4.0], dtype=torch.float64)
    amplitude = 0.5
    reference_position = amplitude * torch.sin(phase)
    reference_velocity = amplitude * torch.cos(phase) * phase_rate
    actual_position = reference_position + 0.02
    actual_velocity = reference_velocity + 0.3

    result = compute_sinusoidal_constraint_effort(
        actual_common_position_rad=actual_position,
        actual_common_velocity_rad_s=actual_velocity,
        reference_common_position_rad=reference_position,
        reference_common_velocity_rad_s=reference_velocity,
        phase_rad=phase,
        amplitude_rad=amplitude,
        equivalent_common_inertia_kg_m2=0.02,
        natural_frequency_hz=10.0,
        damping_ratio=1.0,
        effort_limit_nm=100.0,
    )
    natural_frequency_rad_s = 2.0 * math.pi * 10.0
    stiffness = 0.02 * natural_frequency_rad_s**2
    damping = 2.0 * 0.02 * natural_frequency_rad_s
    expected_effort = -stiffness * 0.02 - damping * 0.3
    constraint_power = (
        result.common_wing_effort_nm * actual_velocity
        + result.phase_reaction_torque_nm * phase_rate
    )
    expected_power = (
        -stiffness * result.position_error_rad * result.velocity_error_rad_s
        - damping * result.velocity_error_rad_s.square()
    )

    assert result.common_wing_effort_nm.item() == pytest.approx(expected_effort)
    assert result.phase_reaction_torque_nm.item() == pytest.approx(
        -amplitude * math.cos(phase.item()) * expected_effort
    )
    torch.testing.assert_close(constraint_power, expected_power)


@pytest.mark.parametrize("physics_dt_s", [1.0 / 240.0, 1.0 / 480.0])
def test_default_five_hz_step_is_bounded_and_settles(physics_dt_s: float) -> None:
    config = SinusoidalPhaseSpeedDriveConfig()
    state = SinusoidalPhaseDriveState.zeros(
        (1,),
        device="cpu",
        dtype=torch.float64,
    )
    throttle = torch.ones(1, dtype=torch.float64)
    external_torque = torch.zeros(1, dtype=torch.float64)
    any_saturation = False

    for _ in range(round(1.0 / physics_dt_s)):
        result = step_sinusoidal_phase_speed_drive(
            state=state,
            throttle_01=throttle,
            common_joint_external_torque_nm=external_torque,
            physics_dt_s=physics_dt_s,
            config=config,
        )
        state = result.next_state
        any_saturation = any_saturation or bool(result.drive_saturated[0])

    assert bool(torch.all(torch.isfinite(state.phase_rad)))
    assert bool(torch.all(torch.isfinite(state.phase_rate_rad_s)))
    assert state.phase_rate_rad_s[0].item() / (2.0 * math.pi) == pytest.approx(
        5.0,
        abs=1.0e-6,
    )
    assert not any_saturation

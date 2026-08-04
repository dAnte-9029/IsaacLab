from __future__ import annotations

import math

import pytest
import torch

from flapping_bot.physics.ideal_inverse_dynamics_phase_drive import (
    IdealFrequencyPhaseState,
    IdealInverseDynamicsPhaseDriveConfig,
    compute_desired_common_acceleration,
    discrete_tracking_acceleration_gains,
    reduce_common_inverse_dynamics,
    step_ideal_frequency_phase,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frequency_phase_step_uses_exact_first_order_solution(dtype: torch.dtype) -> None:
    config = IdealInverseDynamicsPhaseDriveConfig(
        amplitude_rad=math.radians(30.0),
        max_frequency_hz=5.0,
        frequency_settling_time_s=0.2,
    )
    state = IdealFrequencyPhaseState(
        phase_rad=torch.tensor([0.25], dtype=dtype),
        frequency_hz=torch.tensor([1.0], dtype=dtype),
    )
    dt = 0.01
    result = step_ideal_frequency_phase(
        state=state,
        throttle_01=torch.tensor([0.6], dtype=dtype),
        physics_dt_s=dt,
        config=config,
    )
    time_constant = 0.2 / 4.0
    decay = math.exp(-dt / time_constant)
    expected_frequency = 3.0 + (1.0 - 3.0) * decay
    expected_frequency_integral = 3.0 * dt + (1.0 - 3.0) * time_constant * (1.0 - decay)

    assert float(result.next_state.frequency_hz[0]) == pytest.approx(expected_frequency, rel=2.0e-6)
    assert float(result.next_state.phase_rad[0]) == pytest.approx(
        0.25 + 2.0 * math.pi * expected_frequency_integral,
        rel=2.0e-6,
    )
    assert float(result.frequency_rate_hz_s[0]) == pytest.approx(40.0, rel=2.0e-6)
    assert result.next_state.phase_rad.dtype == dtype
    assert result.next_state.phase_rad.device == state.phase_rad.device


def test_frequency_phase_reference_is_sine_with_variable_frequency_derivatives() -> None:
    config = IdealInverseDynamicsPhaseDriveConfig(frequency_settling_time_s=0.2)
    result = step_ideal_frequency_phase(
        state=IdealFrequencyPhaseState(
            phase_rad=torch.tensor([0.0, math.pi / 2.0], dtype=torch.float64),
            frequency_hz=torch.tensor([2.0, 2.0], dtype=torch.float64),
        ),
        throttle_01=torch.tensor([0.6, 0.6], dtype=torch.float64),
        physics_dt_s=1.0 / 480.0,
        config=config,
    )
    amplitude = config.amplitude_rad
    phase_rate = 4.0 * math.pi
    phase_acceleration = 2.0 * math.pi * 20.0

    torch.testing.assert_close(
        result.kinematics.common_position_rad,
        torch.tensor([0.0, amplitude], dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.kinematics.common_velocity_rad_s,
        torch.tensor([amplitude * phase_rate, 0.0], dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.kinematics.common_acceleration_rad_s2,
        torch.tensor(
            [amplitude * phase_acceleration, -amplitude * phase_rate**2],
            dtype=torch.float64,
        ),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_tracking_acceleration_uses_discrete_error_feedback() -> None:
    dt = 1.0 / 480.0
    position_gain, velocity_gain = discrete_tracking_acceleration_gains(
        natural_frequency_hz=50.0,
        damping_ratio=1.0,
        physics_dt_s=dt,
    )
    acceleration = compute_desired_common_acceleration(
        actual_position_rad=torch.tensor([0.1]),
        actual_velocity_rad_s=torch.tensor([0.2]),
        reference_position_rad=torch.tensor([0.12]),
        reference_velocity_rad_s=torch.tensor([0.3]),
        reference_acceleration_rad_s2=torch.tensor([4.0]),
        natural_frequency_hz=50.0,
        damping_ratio=1.0,
        physics_dt_s=dt,
    )

    assert float(acceleration[0]) == pytest.approx(4.0 + position_gain * 0.02 + velocity_gain * 0.1)


def test_fixed_base_reduction_projects_opposed_joint_coordinate() -> None:
    result = reduce_common_inverse_dynamics(
        generalized_mass_matrix=torch.tensor([[[2.0, 0.5], [0.5, 3.0]]], dtype=torch.float64),
        generalized_bias_effort=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        external_generalized_effort=torch.tensor([[0.3, -0.4]], dtype=torch.float64),
        joint_direction=torch.tensor([1.0, -1.0], dtype=torch.float64),
        desired_common_acceleration_rad_s2=torch.tensor([2.0], dtype=torch.float64),
        effort_limit_nm=100.0,
    )

    torch.testing.assert_close(result.common_inertia_kg_m2, torch.tensor([4.0], dtype=torch.float64))
    torch.testing.assert_close(result.common_bias_effort_nm, torch.tensor([-1.0], dtype=torch.float64))
    torch.testing.assert_close(result.common_external_effort_nm, torch.tensor([0.7], dtype=torch.float64))
    torch.testing.assert_close(result.effort_nm, torch.tensor([6.3], dtype=torch.float64))


def test_floating_base_reduction_eliminates_unactuated_base_acceleration() -> None:
    mass = torch.zeros((1, 8, 8), dtype=torch.float64)
    mass[:, :6, :6] = 2.0 * torch.eye(6, dtype=torch.float64)
    mass[:, 6:, 6:] = torch.tensor([[2.0, 0.5], [0.5, 3.0]], dtype=torch.float64)
    mass[:, 0, 6] = 1.0
    mass[:, 6, 0] = 1.0
    bias = torch.zeros((1, 8), dtype=torch.float64)
    bias[:, 0] = 2.0
    bias[:, 6:] = torch.tensor([1.0, 2.0], dtype=torch.float64)
    external = torch.zeros((1, 8), dtype=torch.float64)
    external[:, 0] = 4.0
    external[:, 6:] = torch.tensor([0.3, -0.4], dtype=torch.float64)

    result = reduce_common_inverse_dynamics(
        generalized_mass_matrix=mass,
        generalized_bias_effort=bias,
        external_generalized_effort=external,
        joint_direction=torch.tensor([1.0, -1.0], dtype=torch.float64),
        desired_common_acceleration_rad_s2=torch.tensor([2.0], dtype=torch.float64),
        effort_limit_nm=100.0,
    )

    torch.testing.assert_close(result.common_inertia_kg_m2, torch.tensor([3.5], dtype=torch.float64))
    torch.testing.assert_close(result.common_bias_effort_nm, torch.tensor([-2.0], dtype=torch.float64))
    torch.testing.assert_close(result.common_external_effort_nm, torch.tensor([-1.3], dtype=torch.float64))
    torch.testing.assert_close(result.effort_nm, torch.tensor([6.3], dtype=torch.float64))


def test_reduced_inverse_dynamics_effort_limit_is_only_a_numerical_guard() -> None:
    result = reduce_common_inverse_dynamics(
        generalized_mass_matrix=torch.eye(2).unsqueeze(0),
        generalized_bias_effort=torch.zeros((1, 2)),
        external_generalized_effort=torch.zeros((1, 2)),
        joint_direction=torch.tensor([1.0, -1.0]),
        desired_common_acceleration_rad_s2=torch.tensor([10.0]),
        effort_limit_nm=3.0,
    )

    torch.testing.assert_close(result.effort_nm, torch.tensor([3.0]))
    assert bool(result.saturated[0])

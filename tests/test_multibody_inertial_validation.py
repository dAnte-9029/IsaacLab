from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from flapping_bot.analysis.multibody_inertial_validation import (
    compute_total_momentum_about_system_com_w,
    fit_fundamental_harmonic,
    wrapped_phase_difference_deg,
)


def test_harmonic_fit_recovers_fundamental_with_drift_and_harmonics() -> None:
    time = np.linspace(0.0, 2.0, 2001)
    frequency_hz = 3.0
    phase = 0.37
    values = (
        1.8 * np.sin(2.0 * math.pi * frequency_hz * time + phase)
        + 0.2 * np.sin(4.0 * math.pi * frequency_hz * time - 0.4)
        + 0.03 * time
        - 0.7
    )

    fit = fit_fundamental_harmonic(time, values, frequency_hz=frequency_hz)

    assert fit.amplitude == pytest.approx(1.8, abs=1.0e-12)
    assert fit.phase_rad == pytest.approx(phase, abs=1.0e-12)
    assert fit.rms_residual < 1.0e-12
    assert wrapped_phase_difference_deg(fit.phase_rad, phase) == pytest.approx(0.0, abs=1.0e-10)


def test_total_momentum_cancels_for_symmetric_internal_motion() -> None:
    masses = torch.tensor([1.0, 1.0], dtype=torch.float64)
    inertias = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    positions = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64)
    velocities = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64)
    angular_velocities = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

    system_com, linear_momentum, angular_momentum = compute_total_momentum_about_system_com_w(
        masses_kg=masses,
        inertia_about_com_principal_kg_m2=inertias,
        com_positions_w_m=positions,
        com_quaternions_wxyz=quaternions,
        com_linear_velocities_w_m_s=velocities,
        angular_velocities_w_rad_s=angular_velocities,
    )

    torch.testing.assert_close(system_com, torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(linear_momentum, torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(angular_momentum, torch.zeros(3, dtype=torch.float64))


def test_wrapped_phase_difference_handles_branch_cut() -> None:
    assert wrapped_phase_difference_deg(math.radians(-179.0), math.radians(179.0)) == pytest.approx(2.0)

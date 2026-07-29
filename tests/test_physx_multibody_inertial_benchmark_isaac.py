"""Headless integration check for the PhysX/Amini inertial benchmark."""

from __future__ import annotations

"""Launch Isaac Sim before importing Isaac Lab simulation modules."""

from _delaurier_isaac_app import simulation_app

"""The remaining import requires a running Isaac Sim application."""

import math

import pytest

from flapping_bot.analysis.physx_multibody_inertial_benchmark import (
    run_physx_multibody_inertial_benchmark,
)


@pytest.mark.isaacsim_ci
def test_measured_multibody_inertia_closes_against_amini_reference() -> None:
    result = run_physx_multibody_inertial_benchmark(
        frequencies_hz=(2.0,),
        time_steps_s=(1.0 / 240.0,),
        total_cycles=4,
        measurement_cycles=2,
    )
    case = result["cases"][0]

    assert case["max_sync_error_deg"] < 0.1
    assert case["max_tracking_error_deg"] < 0.5
    assert case["relative_linear_momentum_residual"] < 1.0e-4
    assert case["relative_angular_momentum_residual"] < 1.0e-2
    assert case["physx_to_amini_vertical_amplitude_ratio"] == pytest.approx(1.0, rel=0.01)
    assert abs(case["physx_minus_amini_vertical_phase_deg"]) < 1.0
    assert case["physx_to_amini_pitch_amplitude_ratio"] == pytest.approx(1.0, rel=0.05)
    assert abs(case["physx_minus_amini_pitch_phase_deg"]) < 1.0

    for harmonic in case["harmonics"].values():
        assert math.isfinite(harmonic["amplitude"])
        assert math.isfinite(harmonic["phase_rad"])

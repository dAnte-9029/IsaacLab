"""Headless smoke test for the multibody aerodynamic benchmark."""

from __future__ import annotations

from _delaurier_isaac_app import simulation_app

import pytest

from flapping_bot.analysis.physx_multibody_aero_benchmark import (
    run_free_multibody_aero_job,
)


@pytest.mark.isaacsim_ci
def test_multibody_aero_free_worker_is_finite_and_reports_momentum_closure() -> None:
    result = run_free_multibody_aero_job(
        frequency_hz=4.0,
        dt_s=1.0 / 240.0,
        duration_s=0.1,
    )
    free_case = result["free_cases"][0]

    assert free_case["stable"] is True
    assert free_case["max_sync_error_deg"] < 0.1
    assert free_case["max_single_wing_force_n"] > 0.0
    assert "relative_linear_closure_error" in free_case["momentum_closure"]
    assert "relative_angular_closure_error" in free_case["momentum_closure"]

from __future__ import annotations

import math

import numpy as np
import pytest

from flapping_bot.analysis.multibody_aero_validation import (
    compare_fixed_mode_cases,
    compute_impulse_momentum_closure,
    compute_symmetric_wrench_residuals,
    summarize_time_step_convergence,
)


def test_impulse_momentum_closure_is_exact_for_constant_wrench() -> None:
    dt_s = 0.02
    steps = 20
    force = np.tile(np.array([[2.0, -1.0, 0.5]]), (steps, 1))
    moment = np.tile(np.array([[0.1, 0.2, -0.3]]), (steps, 1))
    linear = np.vstack((np.zeros((1, 3)), np.cumsum(force * dt_s, axis=0)))
    angular = np.vstack((np.zeros((1, 3)), np.cumsum(moment * dt_s, axis=0)))

    result = compute_impulse_momentum_closure(
        linear_momentum_w_kg_m_s=linear,
        angular_momentum_about_com_w_kg_m2_s=angular,
        external_force_w_n=force,
        external_moment_about_com_w_nm=moment,
        dt_s=dt_s,
    )

    assert result["max_linear_residual_kg_m_s"] == pytest.approx(0.0, abs=1.0e-14)
    assert result["max_angular_residual_kg_m2_s"] == pytest.approx(0.0, abs=1.0e-14)


def test_time_step_summary_uses_finest_case_as_reference() -> None:
    cases = [
        {
            "frequency_hz": 4.0,
            "dt_s": 1.0 / 240.0,
            "means": {"force_z": 10.2},
            "harmonics": {"force_z": {"amplitude": 5.1, "phase_rad": math.radians(12.0)}},
        },
        {
            "frequency_hz": 4.0,
            "dt_s": 1.0 / 480.0,
            "means": {"force_z": 10.0},
            "harmonics": {"force_z": {"amplitude": 5.0, "phase_rad": math.radians(10.0)}},
        },
    ]

    rows = summarize_time_step_convergence(cases, signal_names=("force_z",))
    coarse = next(row for row in rows if math.isclose(float(row["dt_s"]), 1.0 / 240.0))
    fine = next(row for row in rows if math.isclose(float(row["dt_s"]), 1.0 / 480.0))

    assert coarse["relative_amplitude_error"] == pytest.approx(0.02)
    assert coarse["phase_error_deg"] == pytest.approx(2.0)
    assert coarse["relative_mean_error"] == pytest.approx(0.02)
    assert fine["relative_amplitude_error"] == pytest.approx(0.0)


def test_symmetric_wrench_residual_ignores_longitudinal_components() -> None:
    force = np.array([[8.0, 0.01, -12.0], [9.0, -0.01, -11.0]])
    moment = np.array([[0.001, 2.0, -0.002], [-0.001, 2.2, 0.002]])

    result = compute_symmetric_wrench_residuals(
        force_b_n=force,
        moment_b_about_base_com_nm=moment,
    )

    assert result["relative_lateral_force_rms"] < 1.0e-3
    assert result["relative_roll_yaw_moment_rms"] < 2.0e-3


def test_fixed_mode_comparison_matches_case_keys() -> None:
    def _case(mode: str, amplitude: float) -> dict[str, object]:
        return {
            "frequency_hz": 4.0,
            "airspeed_mps": 8.0,
            "angle_of_attack_deg": 0.0,
            "coupling_mode": mode,
            "stable": True,
            "means": {"force_z": 10.0},
            "harmonics": {
                "force_z": {
                    "amplitude": amplitude,
                    "phase_rad": math.radians(5.0),
                }
            },
        }

    rows = compare_fixed_mode_cases(
        [_case("actual", 5.5), _case("baseline", 5.0)],
        actual_mode="actual",
        baseline_mode="baseline",
        signal_names=("force_z",),
    )

    assert len(rows) == 1
    assert rows[0]["relative_amplitude_difference"] == pytest.approx(0.1)

"""Pure numerical helpers for multibody aerodynamic validation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .multibody_inertial_validation import wrapped_phase_difference_deg


def compute_impulse_momentum_closure(
    *,
    linear_momentum_w_kg_m_s: np.ndarray,
    angular_momentum_about_com_w_kg_m2_s: np.ndarray,
    external_force_w_n: np.ndarray,
    external_moment_about_com_w_nm: np.ndarray,
    dt_s: float,
) -> dict[str, float | list[float]]:
    """Compare momentum changes with integrated external wrench.

    Momentum arrays have shape ``(T + 1, 3)`` and wrench arrays have shape
    ``(T, 3)``. Each wrench sample is the value applied over the following
    integration interval. All quantities use world axes; angular quantities
    are about the instantaneous whole-system COM.
    """

    linear = np.asarray(linear_momentum_w_kg_m_s, dtype=np.float64)
    angular = np.asarray(angular_momentum_about_com_w_kg_m2_s, dtype=np.float64)
    force = np.asarray(external_force_w_n, dtype=np.float64)
    moment = np.asarray(external_moment_about_com_w_nm, dtype=np.float64)
    if (
        linear.ndim != 2
        or angular.shape != linear.shape
        or linear.shape[1] != 3
        or force.shape != moment.shape
        or force.ndim != 2
        or force.shape[1] != 3
        or linear.shape[0] != force.shape[0] + 1
    ):
        raise ValueError("Momentum must have shape (T+1,3) and wrench shape (T,3).")
    if float(dt_s) <= 0.0:
        raise ValueError("dt_s must be positive.")
    if not all(np.all(np.isfinite(value)) for value in (linear, angular, force, moment)):
        raise ValueError("Momentum-closure inputs must be finite.")

    linear_impulse = np.vstack(
        (np.zeros((1, 3)), np.cumsum(force * float(dt_s), axis=0))
    )
    angular_impulse = np.vstack(
        (np.zeros((1, 3)), np.cumsum(moment * float(dt_s), axis=0))
    )
    linear_change = linear - linear[0]
    angular_change = angular - angular[0]
    linear_residual = linear_change - linear_impulse
    angular_residual = angular_change - angular_impulse
    linear_scale = max(
        float(np.max(np.linalg.norm(linear_change, axis=1))),
        float(np.max(np.linalg.norm(linear_impulse, axis=1))),
        1.0e-12,
    )
    angular_scale = max(
        float(np.max(np.linalg.norm(angular_change, axis=1))),
        float(np.max(np.linalg.norm(angular_impulse, axis=1))),
        1.0e-12,
    )
    return {
        "initial_linear_momentum_w_kg_m_s": linear[0].tolist(),
        "final_linear_momentum_w_kg_m_s": linear[-1].tolist(),
        "integrated_force_impulse_w_n_s": linear_impulse[-1].tolist(),
        "initial_angular_momentum_about_com_w_kg_m2_s": angular[0].tolist(),
        "final_angular_momentum_about_com_w_kg_m2_s": angular[-1].tolist(),
        "integrated_moment_impulse_about_com_w_nm_s": angular_impulse[-1].tolist(),
        "initial_linear_momentum_norm_kg_m_s": float(np.linalg.norm(linear[0])),
        "final_linear_momentum_change_norm_kg_m_s": float(
            np.linalg.norm(linear_change[-1])
        ),
        "integrated_force_impulse_norm_n_s": float(
            np.linalg.norm(linear_impulse[-1])
        ),
        "initial_angular_momentum_norm_kg_m2_s": float(
            np.linalg.norm(angular[0])
        ),
        "final_angular_momentum_change_norm_kg_m2_s": float(
            np.linalg.norm(angular_change[-1])
        ),
        "integrated_moment_impulse_norm_nm_s": float(
            np.linalg.norm(angular_impulse[-1])
        ),
        "max_linear_residual_kg_m_s": float(
            np.max(np.linalg.norm(linear_residual, axis=1))
        ),
        "rms_linear_residual_kg_m_s": float(
            np.sqrt(np.mean(np.sum(np.square(linear_residual), axis=1)))
        ),
        "relative_linear_closure_error": float(
            np.max(np.linalg.norm(linear_residual, axis=1)) / linear_scale
        ),
        "max_angular_residual_kg_m2_s": float(
            np.max(np.linalg.norm(angular_residual, axis=1))
        ),
        "rms_angular_residual_kg_m2_s": float(
            np.sqrt(np.mean(np.sum(np.square(angular_residual), axis=1)))
        ),
        "relative_angular_closure_error": float(
            np.max(np.linalg.norm(angular_residual, axis=1)) / angular_scale
        ),
    }


def summarize_time_step_convergence(
    cases: Sequence[dict[str, object]],
    *,
    signal_names: Sequence[str],
) -> list[dict[str, float | str]]:
    """Compare harmonic metrics against the finest step for each frequency."""

    if not cases:
        return []
    reference_dt_s = min(float(case["dt_s"]) for case in cases)
    rows: list[dict[str, float | str]] = []
    frequencies = sorted({float(case["frequency_hz"]) for case in cases})
    for frequency_hz in frequencies:
        group = [
            case for case in cases if float(case["frequency_hz"]) == frequency_hz
        ]
        reference = next(
            case for case in group if math.isclose(float(case["dt_s"]), reference_dt_s)
        )
        for case in sorted(group, key=lambda item: float(item["dt_s"])):
            for signal_name in signal_names:
                harmonic = case["harmonics"][signal_name]
                reference_harmonic = reference["harmonics"][signal_name]
                amplitude_reference = max(
                    abs(float(reference_harmonic["amplitude"])),
                    1.0e-12,
                )
                mean_reference_scale = max(
                    abs(float(reference["means"][signal_name])),
                    abs(float(reference_harmonic["amplitude"])),
                    1.0e-12,
                )
                rows.append(
                    {
                        "frequency_hz": frequency_hz,
                        "dt_s": float(case["dt_s"]),
                        "reference_dt_s": reference_dt_s,
                        "signal": signal_name,
                        "relative_amplitude_error": abs(
                            float(harmonic["amplitude"])
                            - float(reference_harmonic["amplitude"])
                        )
                        / amplitude_reference,
                        "phase_error_deg": abs(
                            wrapped_phase_difference_deg(
                                float(harmonic["phase_rad"]),
                                float(reference_harmonic["phase_rad"]),
                            )
                        ),
                        "relative_mean_error": abs(
                            float(case["means"][signal_name])
                            - float(reference["means"][signal_name])
                        )
                        / mean_reference_scale,
                    }
                )
    return rows


def compute_symmetric_wrench_residuals(
    *,
    force_b_n: np.ndarray,
    moment_b_about_base_com_nm: np.ndarray,
) -> dict[str, float]:
    """Measure forbidden lateral/roll/yaw content for a symmetric test case."""

    force = np.asarray(force_b_n, dtype=np.float64)
    moment = np.asarray(moment_b_about_base_com_nm, dtype=np.float64)
    if force.ndim != 2 or force.shape != moment.shape or force.shape[1] != 3:
        raise ValueError("Force and moment must share shape (T,3).")
    force_scale = max(float(np.sqrt(np.mean(np.sum(force**2, axis=1)))), 1.0e-12)
    moment_scale = max(float(np.sqrt(np.mean(np.sum(moment**2, axis=1)))), 1.0e-12)
    forbidden_force = force[:, 1]
    forbidden_moment = moment[:, (0, 2)]
    return {
        "lateral_force_rms_n": float(np.sqrt(np.mean(forbidden_force**2))),
        "relative_lateral_force_rms": float(
            np.sqrt(np.mean(forbidden_force**2)) / force_scale
        ),
        "roll_yaw_moment_rms_nm": float(
            np.sqrt(np.mean(np.sum(forbidden_moment**2, axis=1)))
        ),
        "relative_roll_yaw_moment_rms": float(
            np.sqrt(np.mean(np.sum(forbidden_moment**2, axis=1))) / moment_scale
        ),
    }


def compare_fixed_mode_cases(
    cases: Sequence[dict[str, object]],
    *,
    actual_mode: str,
    baseline_mode: str,
    signal_names: Sequence[str],
) -> list[dict[str, float | str]]:
    """Compare matched fixed-body actual and commanded-shadow cases."""

    def _index(mode: str) -> dict[tuple[float, float, float], dict[str, object]]:
        return {
            (
                float(case["frequency_hz"]),
                float(case["airspeed_mps"]),
                float(case["angle_of_attack_deg"]),
            ): case
            for case in cases
            if case.get("coupling_mode") == mode
            and bool(case.get("stable", True))
        }

    actual_cases = _index(actual_mode)
    baseline_cases = _index(baseline_mode)
    rows: list[dict[str, float | str]] = []
    for key, actual in sorted(actual_cases.items()):
        if key not in baseline_cases:
            continue
        baseline = baseline_cases[key]
        for signal_name in signal_names:
            actual_harmonic = actual["harmonics"][signal_name]
            baseline_harmonic = baseline["harmonics"][signal_name]
            amplitude_scale = max(
                abs(float(baseline_harmonic["amplitude"])),
                1.0e-12,
            )
            mean_scale = max(
                abs(float(baseline["means"][signal_name])),
                abs(float(baseline_harmonic["amplitude"])),
                1.0e-12,
            )
            rows.append(
                {
                    "frequency_hz": key[0],
                    "airspeed_mps": key[1],
                    "angle_of_attack_deg": key[2],
                    "signal": signal_name,
                    "relative_mean_difference": (
                        float(actual["means"][signal_name])
                        - float(baseline["means"][signal_name])
                    )
                    / mean_scale,
                    "relative_amplitude_difference": (
                        float(actual_harmonic["amplitude"])
                        - float(baseline_harmonic["amplitude"])
                    )
                    / amplitude_scale,
                    "phase_difference_deg": wrapped_phase_difference_deg(
                        float(actual_harmonic["phase_rad"]),
                        float(baseline_harmonic["phase_rad"]),
                    ),
                }
            )
    return rows


__all__ = [
    "compare_fixed_mode_cases",
    "compute_impulse_momentum_closure",
    "compute_symmetric_wrench_residuals",
    "summarize_time_step_convergence",
]

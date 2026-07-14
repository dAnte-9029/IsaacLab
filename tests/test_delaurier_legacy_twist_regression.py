"""Frozen numerical regression for the historical ``qd`` twist proxy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flapping_bot.physics.delaurier_twist import (
    compute_legacy_qd_scaled_twist,
    validate_delaurier_dynamic_twist_mode,
)


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "delaurier_legacy_qd_scaled_proxy_v1.json"
_STRAIGHT_ENV_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py"
)
_FLOAT64_ATOL = 1.0e-12
_FLOAT32_ATOL = 2.0e-7


def _load_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _run_case(case: dict[str, object], *, dtype: torch.dtype):
    tensor = lambda name: torch.tensor(case[name], dtype=dtype)
    return compute_legacy_qd_scaled_twist(
        num_strips=int(case["num_strips"]),
        mean_pitch_rad=tensor("mean_pitch"),
        joint_velocity_rad_s=tensor("joint_velocity"),
        joint_acceleration_rad_s2=tensor("joint_acceleration"),
        joint_jerk_rad_s3=tensor("joint_jerk"),
        reference_velocity_rad_s=float(case["reference_velocity"]),
        maximum_twist_rad=float(case["maximum_twist"]),
        twist_limit_rad=float(case["twist_limit"]),
        twist_sign=tensor("twist_sign"),
    )


@pytest.mark.parametrize("case_index", range(4))
def test_legacy_qd_scaled_proxy_matches_frozen_float64_fixture(case_index: int) -> None:
    fixture = _load_fixture()
    assert fixture["fixture_version"] == "delaurier_legacy_qd_scaled_proxy_v1"
    assert fixture["source_commit"] == "dd4d1935f3b7c105948f445803e720942760fbd3"
    case = fixture["cases"][case_index]
    result = _run_case(case, dtype=torch.float64)

    for result_field, fixture_field in (
        ("theta", "expected_theta"),
        ("theta_dot", "expected_theta_dot"),
        ("theta_ddot", "expected_theta_ddot"),
    ):
        expected = torch.tensor(case[fixture_field], dtype=torch.float64)
        torch.testing.assert_close(
            getattr(result, result_field),
            expected,
            atol=_FLOAT64_ATOL,
            rtol=0.0,
            msg=f"legacy fixture case={case['name']} field={result_field}",
        )


def test_legacy_qd_scaled_proxy_matches_runtime_float32_precision() -> None:
    fixture = _load_fixture()
    for case in fixture["cases"]:
        result = _run_case(case, dtype=torch.float32)
        for result_field, fixture_field in (
            ("theta", "expected_theta"),
            ("theta_dot", "expected_theta_dot"),
            ("theta_ddot", "expected_theta_ddot"),
        ):
            expected = torch.tensor(case[fixture_field], dtype=torch.float32)
            torch.testing.assert_close(
                getattr(result, result_field),
                expected,
                atol=_FLOAT32_ATOL,
                rtol=0.0,
                msg=f"legacy float32 case={case['name']} field={result_field}",
            )


def test_legacy_angle_clamps_do_not_silently_clamp_historical_derivative_path() -> None:
    case = _load_fixture()["cases"][2]
    result = _run_case(case, dtype=torch.float64)
    assert torch.max(torch.abs(result.delta_theta)).item() == pytest.approx(0.15, abs=_FLOAT64_ATOL)
    assert result.theta_dot[:, 0].tolist() == pytest.approx([0.07, -0.08], abs=_FLOAT64_ATOL)
    assert result.theta_ddot[:, 0].tolist() == pytest.approx([0.11, -0.12], abs=_FLOAT64_ATOL)


def test_legacy_mode_remains_explicit_and_mutually_exclusive() -> None:
    assert validate_delaurier_dynamic_twist_mode("disabled") == "disabled"
    assert validate_delaurier_dynamic_twist_mode("delaurier_linear_spanwise") == "delaurier_linear_spanwise"
    assert validate_delaurier_dynamic_twist_mode("legacy_qd_scaled_proxy") == "legacy_qd_scaled_proxy"
    with pytest.raises(ValueError, match="Unsupported DeLaurier dynamic_twist_mode"):
        validate_delaurier_dynamic_twist_mode("delaurier_linear_spanwise+legacy_qd_scaled_proxy")


def test_default_environment_does_not_enter_legacy_proxy() -> None:
    source = _STRAIGHT_ENV_PATH.read_text(encoding="utf-8")
    assert 'dynamic_twist_mode: str = "disabled"' in source
    assert "compute_legacy_qd_scaled_twist(" in source

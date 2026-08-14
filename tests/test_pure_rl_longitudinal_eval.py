from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/flapping_rl/pure_rl_longitudinal_eval.py"
SPEC = importlib.util.spec_from_file_location("pure_rl_longitudinal_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pure_rl_longitudinal_eval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pure_rl_longitudinal_eval
SPEC.loader.exec_module(pure_rl_longitudinal_eval)


@pytest.mark.parametrize(
    ("stage_id", "expected_angles", "expected_count"),
    (
        ("c2a", {0.0, -2.0, 2.0, -4.0, 4.0}, 80),
        ("c2b", {0.0, -2.0, 2.0, -4.0, 4.0, -6.0, 6.0}, 112),
        ("c2c", {0.0, -4.0, 4.0, -8.0, 8.0, -12.0, 12.0}, 112),
    ),
)
def test_promotion_grid_is_exact_cartesian_product(
    stage_id: str,
    expected_angles: set[float],
    expected_count: int,
) -> None:
    cases = pure_rl_longitudinal_eval.build_longitudinal_evaluation_grid(stage_id)

    assert len(cases) == expected_count
    assert len({case.case_id for case in cases}) == expected_count
    assert {case.signed_slope_deg for case in cases} == expected_angles
    assert {case.heading_rad for case in cases} == {0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0}
    assert {case.flap_phase_rad for case in cases} == {0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0}
    assert all(case.entry_length_m == 17.5 for case in cases)
    assert all(case.slope_length_m == 25.0 for case in cases)
    assert all(case.episode_duration_s == 12.0 for case in cases)
    assert all(case.promotion_eligible for case in cases)


def test_stage_specific_diagnostic_grid_is_separate_and_not_promotion_eligible() -> None:
    legacy_cases = pure_rl_longitudinal_eval.build_longitudinal_diagnostic_grid("c2b")
    assert {case.signed_slope_deg for case in legacy_cases} == {-10.0, 10.0}

    cases = pure_rl_longitudinal_eval.build_longitudinal_diagnostic_grid("c2c")

    assert len(cases) == 32
    assert {case.signed_slope_deg for case in cases} == {-15.0, 15.0}
    assert all(not case.promotion_eligible for case in cases)


def test_c2c_uses_version_two_evaluation_contract() -> None:
    cases = pure_rl_longitudinal_eval.build_longitudinal_evaluation_grid("c2c")
    summary = pure_rl_longitudinal_eval.summarize_longitudinal_evaluation(
        [_episode_row(case) for case in cases],
        expected_cases=cases,
        checkpoint="model_50.pt",
        ppo_iteration=50,
    )

    assert summary["evaluation_contract"] == "pure_rl_longitudinal_c2c_v2"


def _episode_row(case, *, success: bool = True, error_m: float = 0.2) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "stage_id": case.stage_id,
        "task": case.task,
        "promotion_eligible": case.promotion_eligible,
        "terminated": not success,
        "success": success,
        "recovery_reached": success,
        "cross_track_error_m": [error_m, -error_m],
        "height_error_m": [error_m, -error_m],
        "tangent_velocity_mps": [6.0, 6.5],
        "finite_metrics": True,
        "termination_causes": {
            "ground": False,
            "tilt": False,
            "cross_track": False,
            "height_error": not success,
        },
    }


def test_summary_reports_directions_and_passes_approved_gate() -> None:
    cases = pure_rl_longitudinal_eval.build_longitudinal_evaluation_grid("c2a")
    summary = pure_rl_longitudinal_eval.summarize_longitudinal_evaluation(
        [_episode_row(case) for case in cases],
        expected_cases=cases,
        checkpoint="/checkpoints/model_300.pt",
        ppo_iteration=300,
    )

    assert summary["grid_complete"] is True
    assert summary["overall_survival_rate"] == pytest.approx(1.0)
    assert summary["climb_success_rate"] == pytest.approx(1.0)
    assert summary["descent_success_rate"] == pytest.approx(1.0)
    assert summary["recovery_reached_rate"] == pytest.approx(1.0)
    assert summary["mean_abs_cross_track_error_m"] == pytest.approx(0.2)
    assert summary["mean_abs_height_error_m"] == pytest.approx(0.2)
    assert summary["p95_abs_height_error_m"] == pytest.approx(0.2)
    assert summary["reverse_motion_fraction"] == pytest.approx(0.0)
    assert summary["finite_metrics"] is True
    assert pure_rl_longitudinal_eval.row_meets_longitudinal_promotion_gate(summary)


def test_summary_reports_signed_slope_success_and_termination_breakdown() -> None:
    cases = pure_rl_longitudinal_eval.build_longitudinal_evaluation_grid("c2c")
    rows = [_episode_row(case) for case in cases]
    climb_12 = [index for index, case in enumerate(cases) if case.signed_slope_deg == 12.0]
    for offset, row_index in enumerate(climb_12[:3]):
        rows[row_index]["terminated"] = True
        rows[row_index]["success"] = False
        rows[row_index]["recovery_reached"] = offset != 0
        rows[row_index]["termination_causes"] = {
            "ground": offset == 0,
            "tilt": offset == 1,
            "cross_track": False,
            "height_error": offset >= 1,
        }

    summary = pure_rl_longitudinal_eval.summarize_longitudinal_evaluation(
        rows,
        expected_cases=cases,
        checkpoint="model_500.pt",
        ppo_iteration=500,
    )

    by_slope = {row["signed_slope_deg"]: row for row in summary["slope_breakdown"]}
    assert list(by_slope) == [-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0]
    assert by_slope[8.0]["case_count"] == 16
    assert by_slope[8.0]["success_rate"] == pytest.approx(1.0)
    assert by_slope[12.0] == {
        "signed_slope_deg": 12.0,
        "task": "climb",
        "case_count": 16,
        "survival_rate": pytest.approx(13.0 / 16.0),
        "success_rate": pytest.approx(13.0 / 16.0),
        "recovery_reached_rate": pytest.approx(15.0 / 16.0),
        "termination_counts": {
            "ground": 1,
            "tilt": 1,
            "cross_track": 0,
            "height_error": 2,
        },
    }


def test_summary_requires_complete_termination_cause_mapping() -> None:
    cases = pure_rl_longitudinal_eval.build_longitudinal_evaluation_grid("c2a")
    rows = [_episode_row(case) for case in cases]
    del rows[0]["termination_causes"]

    with pytest.raises(ValueError, match="termination_causes"):
        pure_rl_longitudinal_eval.summarize_longitudinal_evaluation(
            rows,
            expected_cases=cases,
            checkpoint="model.pt",
            ppo_iteration=50,
        )


def test_diagnostic_summary_marks_missing_level_cases_not_applicable() -> None:
    cases = pure_rl_longitudinal_eval.build_longitudinal_diagnostic_grid("c2a")

    summary = pure_rl_longitudinal_eval.summarize_longitudinal_evaluation(
        [_episode_row(case) for case in cases],
        expected_cases=cases,
        checkpoint="/checkpoints/model_300.pt",
        ppo_iteration=300,
    )

    assert summary["level_case_count"] == 0
    assert summary["level_success_rate"] is None
    assert summary["climb_success_rate"] == pytest.approx(1.0)
    assert summary["descent_success_rate"] == pytest.approx(1.0)
    assert summary["promotion_gate_passed"] is False


def test_direction_failure_blocks_promotion_even_when_overall_rate_is_high() -> None:
    cases = pure_rl_longitudinal_eval.build_longitudinal_evaluation_grid("c2c")
    failed_descent_ids = {case.case_id for case in [item for item in cases if item.task == "descent"][:5]}
    rows = [_episode_row(case, success=case.case_id not in failed_descent_ids) for case in cases]
    summary = pure_rl_longitudinal_eval.summarize_longitudinal_evaluation(
        rows,
        expected_cases=cases,
        checkpoint="/checkpoints/model_300.pt",
        ppo_iteration=300,
    )

    assert summary["overall_survival_rate"] > 0.95
    assert summary["descent_success_rate"] < 0.90
    assert not pure_rl_longitudinal_eval.row_meets_longitudinal_promotion_gate(summary)


def test_summary_fails_closed_on_duplicate_missing_or_non_finite_cases() -> None:
    cases = pure_rl_longitudinal_eval.build_longitudinal_evaluation_grid("c2a")
    rows = [_episode_row(case) for case in cases]
    with pytest.raises(ValueError, match="duplicate"):
        pure_rl_longitudinal_eval.summarize_longitudinal_evaluation(
            rows + [dict(rows[0])],
            expected_cases=cases,
            checkpoint="model.pt",
            ppo_iteration=300,
        )
    with pytest.raises(ValueError, match="complete"):
        pure_rl_longitudinal_eval.summarize_longitudinal_evaluation(
            rows[:-1],
            expected_cases=cases,
            checkpoint="model.pt",
            ppo_iteration=300,
        )
    non_finite = [dict(row) for row in rows]
    non_finite[0]["height_error_m"] = [float("nan")]
    with pytest.raises(ValueError, match="finite"):
        pure_rl_longitudinal_eval.summarize_longitudinal_evaluation(
            non_finite,
            expected_cases=cases,
            checkpoint="model.pt",
            ppo_iteration=300,
        )

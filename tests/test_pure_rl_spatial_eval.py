from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/flapping_rl/pure_rl_spatial_eval.py"
SPEC = importlib.util.spec_from_file_location("pure_rl_spatial_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pure_rl_spatial_eval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pure_rl_spatial_eval
SPEC.loader.exec_module(pure_rl_spatial_eval)


@pytest.mark.parametrize(("stage", "count"), (("c3a", 96), ("c3b", 112), ("c3c", 96)))
def test_spatial_grids_have_exact_counts_and_unique_ids(stage: str, count: int) -> None:
    cases = pure_rl_spatial_eval.build_spatial_evaluation_grid(stage)
    assert len(cases) == count
    assert len({case.case_id for case in cases}) == count
    assert {case.heading_rad for case in cases} == {0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0}


def test_spatial_grids_cover_approved_slices() -> None:
    c3a = pure_rl_spatial_eval.build_spatial_evaluation_grid("c3a")
    assert {case.geometry_roll_deg for case in c3a} == {9.0, 11.0, 13.0}
    assert {case.turn_sign for case in c3a} == {-1, 1}
    assert {case.flap_phase_rad for case in c3a} == {0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0}

    c3b = pure_rl_spatial_eval.build_spatial_evaluation_grid("c3b")
    assert {case.template_id for case in c3b} == set(range(3, 10))
    assert all(sum(item.template_id == template for item in c3b) == 16 for template in range(3, 10))
    assert {case.turn_sign for case in c3b} == {-1, 1}

    c3c = pure_rl_spatial_eval.build_spatial_evaluation_grid("c3c")
    assert {(case.geometry_roll_deg, abs(case.slope_deg)) for case in c3c} == {
        (8.0, 2.0),
        (12.0, 3.0),
        (16.0, 3.0),
    }
    assert {(case.turn_sign, case.vertical_sign) for case in c3c} == {
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    }


def _row(case, *, success: bool = True, error_m: float = 0.2) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "stage_id": case.stage_id,
        "template_id": case.template_id,
        "turn_sign": case.turn_sign,
        "vertical_sign": case.vertical_sign,
        "severity_id": case.severity_id,
        "terminated": not success,
        "events_reached": success,
        "success": success,
        "horizontal_normal_error_m": [error_m, -error_m],
        "vertical_normal_error_m": [error_m, -error_m],
        "tangent_velocity_mps": [6.0, 6.5],
        "roll_rad": [math.radians(10.0), math.radians(15.0)],
        "roll_limit_termination": False,
        "finite_metrics": True,
    }


def test_spatial_summary_passes_all_hard_gates_and_reports_slices() -> None:
    cases = pure_rl_spatial_eval.build_spatial_evaluation_grid("c3c")
    summary = pure_rl_spatial_eval.summarize_spatial_evaluation(
        [_row(case) for case in cases],
        expected_cases=cases,
        checkpoint="model_50.pt",
        ppo_iteration=50,
    )
    assert summary["case_count"] == 96
    assert summary["overall_survival_rate"] == pytest.approx(1.0)
    assert summary["all_event_completion_rate"] == pytest.approx(1.0)
    assert summary["overall_success_rate"] == pytest.approx(1.0)
    assert summary["p95_abs_roll_deg"] == pytest.approx(15.0)
    assert summary["roll_limit_termination_count"] == 0
    assert all(rate == pytest.approx(1.0) for rate in summary["slice_success_rates"].values())
    assert pure_rl_spatial_eval.row_meets_spatial_promotion_gate(summary)


def test_spatial_summary_fails_closed_on_incomplete_or_bad_slice() -> None:
    cases = pure_rl_spatial_eval.build_spatial_evaluation_grid("c3b")
    rows = [_row(case) for case in cases]
    with pytest.raises(ValueError, match="complete"):
        pure_rl_spatial_eval.summarize_spatial_evaluation(
            rows[:-1], expected_cases=cases, checkpoint="model.pt", ppo_iteration=50
        )
    failed_template = {case.case_id for case in cases if case.template_id == 3}
    failed_template = set(sorted(failed_template)[:3])
    failed_rows = [_row(case, success=case.case_id not in failed_template) for case in cases]
    summary = pure_rl_spatial_eval.summarize_spatial_evaluation(
        failed_rows, expected_cases=cases, checkpoint="model.pt", ppo_iteration=50
    )
    assert summary["overall_success_rate"] >= 0.90
    assert summary["slice_success_rates"]["template_3"] < 0.875
    assert not pure_rl_spatial_eval.row_meets_spatial_promotion_gate(summary)


def test_spatial_gate_rejects_missing_or_unexpected_slice_names() -> None:
    cases = pure_rl_spatial_eval.build_spatial_evaluation_grid("c3c")
    summary = pure_rl_spatial_eval.summarize_spatial_evaluation(
        [_row(case) for case in cases],
        expected_cases=cases,
        checkpoint="model.pt",
        ppo_iteration=50,
    )
    incomplete = dict(summary)
    incomplete["slice_success_rates"] = {"left": 1.0}
    assert not pure_rl_spatial_eval.row_meets_spatial_promotion_gate(incomplete)

    unexpected = dict(summary)
    unexpected["slice_success_rates"] = {**summary["slice_success_rates"], "unknown": 1.0}
    assert not pure_rl_spatial_eval.row_meets_spatial_promotion_gate(unexpected)

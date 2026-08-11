from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/flapping_rl"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "pure_rl_longitudinal_promotion.py"
SPEC = importlib.util.spec_from_file_location("pure_rl_longitudinal_promotion", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
promotion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = promotion
SPEC.loader.exec_module(promotion)


def _evaluation(checkpoint: str, iteration: int, *, passed: bool = True) -> dict[str, object]:
    return {
        "checkpoint": checkpoint,
        "ppo_iteration": iteration,
        "stage_id": "c2a",
        "grid_complete": True,
        "climb_case_count": 32,
        "descent_case_count": 32,
        "overall_survival_rate": 0.98 if passed else 0.80,
        "climb_success_rate": 0.95,
        "descent_success_rate": 0.95,
        "recovery_reached_rate": 0.98,
        "mean_abs_cross_track_error_m": 0.30,
        "mean_abs_height_error_m": 0.30,
        "p95_abs_height_error_m": 1.0,
        "reverse_motion_fraction": 0.0,
        "finite_metrics": True,
    }


def _retention(checkpoint: str, *, passed: bool = True) -> dict[str, object]:
    return {
        "checkpoint": checkpoint,
        "evaluation_stage": "c1_straight",
        "success_rate": 0.97 if passed else 0.80,
        "termination_rate": 0.03,
        "score": 98.0,
        "mean_abs_cross_track_error_m": 0.20,
        "mean_abs_height_error_m": 0.20,
        "finite_metrics": True,
    }


SOURCE_BASELINE = {
    "checkpoint": "/checkpoints/c1.pt",
    "score": 100.0,
    "mean_abs_cross_track_error_m": 0.10,
    "mean_abs_height_error_m": 0.10,
}


def _c1_suite_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "checkpoint": "/checkpoints/c2a.pt",
        "evaluation_contract": "pure_rl_curriculum1_v2",
        "case": "suite",
        "timeout_rate": 0.97,
        "termination_rate": 0.03,
        "score": 98.0,
        "mean_abs_cross_track_error_m": 0.20,
        "mean_abs_height_error_m": 0.18,
    }
    row.update(overrides)
    return row


def test_build_c1_retention_row_adapts_current_suite_contract() -> None:
    result = promotion.build_c1_retention_row(_c1_suite_row())

    assert result == {
        "checkpoint": "/checkpoints/c2a.pt",
        "evaluation_stage": "c1_straight",
        "success_rate": 0.97,
        "termination_rate": 0.03,
        "score": 98.0,
        "mean_abs_cross_track_error_m": 0.20,
        "mean_abs_height_error_m": 0.18,
        "finite_metrics": True,
    }


def test_build_c1_retention_row_rejects_non_suite_or_nonfinite_evidence() -> None:
    with pytest.raises(ValueError, match="suite"):
        promotion.build_c1_retention_row(_c1_suite_row(case="fixed_grid"))
    with pytest.raises(ValueError, match="finite"):
        promotion.build_c1_retention_row(_c1_suite_row(score=float("nan")))


def test_build_c1_source_baseline_uses_same_contract_metrics() -> None:
    result = promotion.build_c1_source_baseline(_c1_suite_row(checkpoint="/checkpoints/c1.pt"))

    assert result == {
        "checkpoint": "/checkpoints/c1.pt",
        "score": 98.0,
        "mean_abs_cross_track_error_m": 0.20,
        "mean_abs_height_error_m": 0.18,
    }


@pytest.mark.parametrize(
    ("num_envs", "minimum_iteration", "evaluation_interval"),
    [(64, 200, 100), (128, 100, 50), (256, 50, 25)],
)
def test_build_sample_equivalent_promotion_schedule_preserves_reference_transitions(
    num_envs: int,
    minimum_iteration: int,
    evaluation_interval: int,
) -> None:
    schedule = promotion.build_sample_equivalent_promotion_schedule(num_envs=num_envs)

    assert schedule == {
        "minimum_ppo_iteration": minimum_iteration,
        "evaluation_interval": evaluation_interval,
    }


def test_build_sample_equivalent_promotion_schedule_fails_closed_for_invalid_or_inexact_rollout() -> None:
    with pytest.raises(ValueError, match="positive"):
        promotion.build_sample_equivalent_promotion_schedule(num_envs=0)
    with pytest.raises(ValueError, match="exact"):
        promotion.build_sample_equivalent_promotion_schedule(num_envs=96)


def test_256_environment_schedule_promotes_sample_equivalent_adjacent_passes() -> None:
    evaluations = [
        _evaluation("model_25.pt", 25),
        _evaluation("model_50.pt", 50),
        _evaluation("model_75.pt", 75),
    ]
    retention = [
        _retention("model_25.pt"),
        _retention("model_50.pt"),
        _retention("model_75.pt"),
    ]

    result = promotion.evaluate_longitudinal_promotion(
        evaluations,
        c1_retention_rows=retention,
        source_c1_baseline=SOURCE_BASELINE,
        stage_id="c2a",
        **promotion.build_sample_equivalent_promotion_schedule(num_envs=256),
    )

    assert result["promoted"] is True
    assert result["ppo_iteration"] == 75
    assert result["consecutive_checkpoints"] == ["model_50.pt", "model_75.pt"]


def test_two_consecutive_passing_checkpoints_promote_the_later_checkpoint() -> None:
    evaluations = [
        _evaluation("model_100.pt", 100),
        _evaluation("model_200.pt", 200),
        _evaluation("model_300.pt", 300),
    ]
    result = promotion.evaluate_longitudinal_promotion(
        evaluations,
        c1_retention_rows=[_retention(row["checkpoint"]) for row in evaluations],
        source_c1_baseline=SOURCE_BASELINE,
        stage_id="c2a",
    )

    assert result["promoted"] is True
    assert result["checkpoint"] == "model_300.pt"
    assert result["consecutive_checkpoints"] == ["model_200.pt", "model_300.pt"]


def test_failed_retention_or_nonconsecutive_passes_do_not_promote() -> None:
    evaluations = [
        _evaluation("model_200.pt", 200),
        _evaluation("model_300.pt", 300, passed=False),
        _evaluation("model_400.pt", 400),
    ]
    retention_rows = [_retention(row["checkpoint"]) for row in evaluations]
    retention_rows[-1] = _retention("model_400.pt", passed=False)
    result = promotion.evaluate_longitudinal_promotion(
        evaluations,
        c1_retention_rows=retention_rows,
        source_c1_baseline=SOURCE_BASELINE,
        stage_id="c2a",
    )

    assert result["promoted"] is False
    assert result["checkpoint"] is None


@pytest.mark.parametrize("fault", ("duplicate", "missing_direction", "incomplete_grid", "missing_baseline"))
def test_promotion_fails_closed_on_ambiguous_or_incomplete_evidence(fault: str) -> None:
    evaluations = [_evaluation("model_200.pt", 200), _evaluation("model_300.pt", 300)]
    retention_rows = [_retention("model_200.pt"), _retention("model_300.pt")]
    baseline = SOURCE_BASELINE
    if fault == "duplicate":
        evaluations.append(dict(evaluations[1]))
    elif fault == "missing_direction":
        evaluations[0]["descent_case_count"] = 0
    elif fault == "incomplete_grid":
        evaluations[0]["grid_complete"] = False
    elif fault == "missing_baseline":
        baseline = None

    with pytest.raises(ValueError):
        promotion.evaluate_longitudinal_promotion(
            evaluations,
            c1_retention_rows=retention_rows,
            source_c1_baseline=baseline,
            stage_id="c2a",
        )

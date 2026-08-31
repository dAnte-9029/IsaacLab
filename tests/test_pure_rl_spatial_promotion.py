from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/flapping_rl"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "pure_rl_spatial_promotion.py"
SPEC = importlib.util.spec_from_file_location("pure_rl_spatial_promotion", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
promotion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = promotion
SPEC.loader.exec_module(promotion)


REQUIRED_STAGES = {
    "c3a": ("c1_straight", "c2c"),
    "c3b": ("c1_straight", "c2c", "c3a"),
    "c3c": ("c1_straight", "c2c", "c3a", "c3b"),
}


def _evaluation(checkpoint: str, iteration: int, *, stage: str = "c3c", passed: bool = True) -> dict[str, object]:
    slices = {
        "c3a": {"left": 0.95, "right": 0.95},
        "c3b": {f"template_{template_id}": 0.95 for template_id in range(3, 10)},
        "c3c": {
            "left": 0.95,
            "right": 0.95,
            "climb": 0.95,
            "descent": 0.95,
            "sign_-1_-1": 0.95,
            "sign_-1_+1": 0.95,
            "sign_+1_-1": 0.95,
            "sign_+1_+1": 0.95,
        },
    }[stage]
    return {
        "checkpoint": checkpoint,
        "ppo_iteration": iteration,
        "stage_id": stage,
        "evaluation_contract": (
            "pure_rl_spatial_c3b_v3" if stage == "c3b" else f"pure_rl_spatial_{stage}_v2"
        ),
        "grid_complete": True,
        "overall_survival_rate": 0.98 if passed else 0.80,
        "all_event_completion_rate": 0.98,
        "overall_success_rate": 0.95,
        "mean_abs_horizontal_error_m": 0.2,
        "mean_abs_vertical_error_m": 0.2,
        "p95_abs_horizontal_error_m": 1.0,
        "p95_abs_vertical_error_m": 1.0,
        "reverse_motion_fraction": 0.0,
        "p95_abs_roll_deg": 20.0,
        "roll_limit_termination_count": 0,
        "finite_metrics": True,
        "slice_success_rates": slices,
    }


def _baseline(stage: str) -> dict[str, object]:
    row: dict[str, object] = {
        "checkpoint": f"/promoted/{stage}.pt",
        "success_rate": 0.98,
    }
    if stage == "c1_straight":
        row.update(
            score=100.0,
            mean_abs_cross_track_error_m=0.10,
            mean_abs_height_error_m=0.10,
        )
    return row


def _retention(checkpoint: str, stage: str, *, success_rate: float = 0.96, passed: bool = True) -> dict[str, object]:
    contract = {
        "c1_straight": "pure_rl_curriculum1_v2",
        "c2c": "pure_rl_longitudinal_c2c_v2",
        "c3a": "pure_rl_spatial_c3a_v2",
        "c3b": "pure_rl_spatial_c3b_v3",
    }[stage]
    row: dict[str, object] = {
        "checkpoint": checkpoint,
        "evaluation_stage": stage,
        "evaluation_contract": contract,
        "success_rate": success_rate,
        "success_gate_passed": passed,
        "finite_metrics": True,
    }
    if stage == "c1_straight":
        row.update(
            termination_rate=0.03,
            score=98.0,
            mean_abs_cross_track_error_m=0.20,
            mean_abs_height_error_m=0.18,
        )
    return row


def _evidence(stage: str, checkpoints: tuple[str, ...]) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    required = REQUIRED_STAGES[stage]
    rows = [_retention(checkpoint, retained_stage) for checkpoint in checkpoints for retained_stage in required]
    baselines = {retained_stage: _baseline(retained_stage) for retained_stage in required}
    return rows, baselines


def test_256_environment_schedule_preserves_approved_transition_counts() -> None:
    assert promotion.build_sample_equivalent_promotion_schedule(num_envs=256) == {
        "minimum_ppo_iteration": 50,
        "evaluation_interval": 25,
    }


@pytest.mark.parametrize("stage", ("c3a", "c3b", "c3c"))
def test_two_adjacent_passes_require_every_stage_specific_retention_suite(stage: str) -> None:
    checkpoints = ("model_50.pt", "model_75.pt")
    retention_rows, baselines = _evidence(stage, checkpoints)
    result = promotion.evaluate_spatial_promotion(
        [_evaluation(checkpoint, iteration, stage=stage) for checkpoint, iteration in zip(checkpoints, (50, 75))],
        retention_rows=retention_rows,
        source_baselines=baselines,
        stage_id=stage,
        minimum_ppo_iteration=50,
        evaluation_interval=25,
    )
    assert result["promoted"] is True
    assert result["checkpoint"] == "model_75.pt"
    assert result["required_retention_stages"] == list(REQUIRED_STAGES[stage])


@pytest.mark.parametrize(("fault_stage", "success_rate", "gate"), (("c2c", 0.92, True), ("c3a", 0.96, False)))
def test_prior_gate_failure_or_success_drop_over_five_points_blocks_promotion(
    fault_stage: str, success_rate: float, gate: bool
) -> None:
    checkpoints = ("model_50.pt", "model_75.pt")
    retention_rows, baselines = _evidence("c3c", checkpoints)
    faulty = next(
        row
        for row in retention_rows
        if row["checkpoint"] == "model_75.pt" and row["evaluation_stage"] == fault_stage
    )
    faulty["success_rate"] = success_rate
    faulty["success_gate_passed"] = gate
    result = promotion.evaluate_spatial_promotion(
        [_evaluation("model_50.pt", 50), _evaluation("model_75.pt", 75)],
        retention_rows=retention_rows,
        source_baselines=baselines,
        stage_id="c3c",
        minimum_ppo_iteration=50,
        evaluation_interval=25,
    )
    assert result["promoted"] is False
    assert result["checkpoint_audits"][1]["forgetting"] is True


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "duplicate",
        "wrong_checkpoint",
        "wrong_stage",
        "wrong_contract",
        "nonfinite",
        "incomplete",
        "nonadjacent",
    ),
)
def test_spatial_promotion_fails_closed_on_invalid_evidence(fault: str) -> None:
    checkpoints = ("model_50.pt", "model_75.pt")
    evaluations = [_evaluation("model_50.pt", 50, stage="c3a"), _evaluation("model_75.pt", 75, stage="c3a")]
    retention_rows, baselines = _evidence("c3a", checkpoints)
    if fault == "missing":
        retention_rows.pop()
    elif fault == "duplicate":
        retention_rows.append(dict(retention_rows[-1]))
    elif fault == "wrong_checkpoint":
        retention_rows[-1]["checkpoint"] = "model_100.pt"
    elif fault == "wrong_stage":
        evaluations[-1]["stage_id"] = "c3b"
    elif fault == "wrong_contract":
        retention_rows[-1]["evaluation_contract"] = "pure_rl_spatial_c3b_v1"
    elif fault == "nonfinite":
        retention_rows[-1]["success_rate"] = float("nan")
    elif fault == "incomplete":
        evaluations[-1]["grid_complete"] = False
    elif fault == "nonadjacent":
        evaluations[-1]["ppo_iteration"] = 100

    if fault == "nonadjacent":
        result = promotion.evaluate_spatial_promotion(
            evaluations,
            retention_rows=retention_rows,
            source_baselines=baselines,
            stage_id="c3a",
            minimum_ppo_iteration=50,
            evaluation_interval=25,
        )
        assert result["promoted"] is False
    else:
        with pytest.raises(ValueError):
            promotion.evaluate_spatial_promotion(
                evaluations,
                retention_rows=retention_rows,
                source_baselines=baselines,
                stage_id="c3a",
                minimum_ppo_iteration=50,
                evaluation_interval=25,
            )

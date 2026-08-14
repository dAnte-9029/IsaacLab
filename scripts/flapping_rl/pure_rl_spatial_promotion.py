"""Fail-closed consecutive-checkpoint promotion for PureRL C3."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from pure_rl_longitudinal_promotion import build_sample_equivalent_promotion_schedule
from pure_rl_eval_common import PURE_RL_CURRICULUM1_EVAL_CONTRACT
from pure_rl_longitudinal_eval import LONGITUDINAL_EVAL_CONTRACTS
from pure_rl_spatial_eval import SPATIAL_EVAL_CONTRACTS, row_meets_spatial_promotion_gate


SPATIAL_REQUIRED_RETENTION_STAGES: dict[str, tuple[str, ...]] = {
    "c3a": ("c1_straight", "c2c"),
    "c3b": ("c1_straight", "c2c", "c3a"),
    "c3c": ("c1_straight", "c2c", "c3a", "c3b"),
}
_RETENTION_CONTRACTS = {
    "c1_straight": PURE_RL_CURRICULUM1_EVAL_CONTRACT,
    "c2c": LONGITUDINAL_EVAL_CONTRACTS["c2c"],
    "c3a": SPATIAL_EVAL_CONTRACTS["c3a"],
    "c3b": SPATIAL_EVAL_CONTRACTS["c3b"],
}


def evaluate_spatial_promotion(
    evaluation_rows: Sequence[Mapping[str, object]],
    *,
    retention_rows: Sequence[Mapping[str, object]],
    source_baselines: Mapping[str, Mapping[str, object]],
    stage_id: str,
    minimum_ppo_iteration: int,
    evaluation_interval: int,
) -> dict[str, object]:
    """Promote the later of two adjacent C3 passes with complete retention."""

    stage = _validate_stage(stage_id)
    if minimum_ppo_iteration < 0 or evaluation_interval <= 0:
        raise ValueError("Promotion iteration settings require a non-negative minimum and positive interval.")
    required_stages = SPATIAL_REQUIRED_RETENTION_STAGES[stage]
    baselines = _validate_baselines(source_baselines, required_stages)
    evaluations = tuple(evaluation_rows)
    if not evaluations:
        raise ValueError("evaluation_rows must not be empty.")

    checkpoints: list[str] = []
    iterations: list[int] = []
    for row in evaluations:
        checkpoint = _required_text(row, "checkpoint")
        iteration = _required_iteration(row)
        if str(row.get("stage_id", "")).strip().lower() != stage:
            raise ValueError(f"Evaluation row stage does not match {stage}: {checkpoint}")
        expected_contract = SPATIAL_EVAL_CONTRACTS[stage]
        if str(row.get("evaluation_contract", "")).strip() != expected_contract:
            raise ValueError(f"Evaluation row must use {expected_contract}: {checkpoint}")
        _validate_spatial_evaluation(row)
        checkpoints.append(checkpoint)
        iterations.append(iteration)
    if len(set(checkpoints)) != len(checkpoints) or len(set(iterations)) != len(iterations):
        raise ValueError("evaluation_rows contains duplicate checkpoints or PPO iterations.")
    if any(later <= earlier for earlier, later in zip(iterations, iterations[1:])):
        raise ValueError("evaluation_rows must be ordered by strictly increasing PPO iteration.")

    retention_by_cell: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in retention_rows:
        checkpoint = _required_text(row, "checkpoint")
        evaluation_stage = _required_text(row, "evaluation_stage")
        cell = (checkpoint, evaluation_stage)
        if cell in retention_by_cell:
            raise ValueError(f"duplicate retention evidence: {cell}")
        if checkpoint not in checkpoints or evaluation_stage not in required_stages:
            raise ValueError(f"unexpected retention evidence: {cell}")
        _validate_retention_row(row, stage_id=evaluation_stage)
        retention_by_cell[cell] = row
    required_cells = {
        (checkpoint, retained_stage)
        for checkpoint in checkpoints
        for retained_stage in required_stages
    }
    missing = sorted(required_cells.difference(retention_by_cell))
    unexpected = sorted(set(retention_by_cell).difference(required_cells))
    if missing or unexpected:
        raise ValueError(f"Retention evidence must be complete; missing={missing}, unexpected={unexpected}.")

    audits: list[dict[str, object]] = []
    for row, checkpoint, iteration in zip(evaluations, checkpoints, iterations):
        retention_results = {
            retained_stage: _row_meets_retention_gate(
                retention_by_cell[(checkpoint, retained_stage)],
                baselines[retained_stage],
                stage_id=retained_stage,
            )
            for retained_stage in required_stages
        }
        forgetting = not all(retention_results.values())
        current_stage_passed = row_meets_spatial_promotion_gate(row)
        audits.append(
            {
                "checkpoint": checkpoint,
                "ppo_iteration": iteration,
                "current_stage_passed": current_stage_passed,
                "retention_passed": retention_results,
                "forgetting": forgetting,
                "eligible": (
                    iteration >= minimum_ppo_iteration
                    and current_stage_passed
                    and not forgetting
                ),
            }
        )

    for previous, current in zip(audits, audits[1:]):
        adjacent = int(current["ppo_iteration"]) - int(previous["ppo_iteration"]) == evaluation_interval
        if bool(previous["eligible"]) and bool(current["eligible"]) and adjacent:
            return _result(
                stage=stage,
                required_stages=required_stages,
                audits=audits,
                current=current,
                previous=previous,
            )
    return _result(stage=stage, required_stages=required_stages, audits=audits)


def _result(
    *,
    stage: str,
    required_stages: Sequence[str],
    audits: Sequence[Mapping[str, object]],
    current: Mapping[str, object] | None = None,
    previous: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "stage_id": stage,
        "promoted": current is not None,
        "checkpoint": None if current is None else current["checkpoint"],
        "ppo_iteration": None if current is None else current["ppo_iteration"],
        "consecutive_checkpoints": (
            [] if current is None or previous is None else [previous["checkpoint"], current["checkpoint"]]
        ),
        "required_retention_stages": list(required_stages),
        "checkpoint_audits": list(audits),
    }


def _validate_baselines(
    source_baselines: Mapping[str, Mapping[str, object]],
    required_stages: Sequence[str],
) -> dict[str, dict[str, object]]:
    if set(source_baselines) != set(required_stages):
        raise ValueError(
            "source_baselines must exactly match required retention stages: "
            f"required={sorted(required_stages)}, actual={sorted(source_baselines)}."
        )
    baselines: dict[str, dict[str, object]] = {}
    for stage in required_stages:
        row = source_baselines[stage]
        baseline = {
            "checkpoint": _required_text(row, "checkpoint"),
            "success_rate": _finite_fraction(row, "success_rate"),
        }
        if stage == "c1_straight":
            baseline.update(
                score=_finite_float(row, "score"),
                mean_abs_cross_track_error_m=_finite_nonnegative(row, "mean_abs_cross_track_error_m"),
                mean_abs_height_error_m=_finite_nonnegative(row, "mean_abs_height_error_m"),
            )
        baselines[stage] = baseline
    return baselines


def _validate_spatial_evaluation(row: Mapping[str, object]) -> None:
    if not isinstance(row.get("grid_complete"), bool):
        raise ValueError("grid_complete must be boolean.")
    if not bool(row["grid_complete"]):
        raise ValueError("Spatial promotion requires a complete evaluation grid.")
    if not isinstance(row.get("finite_metrics"), bool):
        raise ValueError("finite_metrics must be boolean.")
    for name in (
        "overall_survival_rate",
        "all_event_completion_rate",
        "overall_success_rate",
        "reverse_motion_fraction",
    ):
        _finite_fraction(row, name)
    for name in (
        "mean_abs_horizontal_error_m",
        "mean_abs_vertical_error_m",
        "p95_abs_horizontal_error_m",
        "p95_abs_vertical_error_m",
        "p95_abs_roll_deg",
    ):
        _finite_nonnegative(row, name)
    roll_limit_count = row.get("roll_limit_termination_count")
    if isinstance(roll_limit_count, bool):
        raise ValueError("roll_limit_termination_count must be a non-negative integer.")
    try:
        parsed_roll_limit_count = int(roll_limit_count)
    except (TypeError, ValueError) as error:
        raise ValueError("roll_limit_termination_count must be a non-negative integer.") from error
    if parsed_roll_limit_count < 0 or float(roll_limit_count) != parsed_roll_limit_count:
        raise ValueError("roll_limit_termination_count must be a non-negative integer.")
    slice_rates = row.get("slice_success_rates")
    if not isinstance(slice_rates, Mapping) or not slice_rates:
        raise ValueError("slice_success_rates must be a non-empty mapping.")
    for value in slice_rates.values():
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("slice_success_rates must contain numeric fractions.") from error
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ValueError("slice_success_rates must contain finite fractions.")


def _validate_retention_row(row: Mapping[str, object], *, stage_id: str) -> None:
    expected_contract = _RETENTION_CONTRACTS[stage_id]
    if str(row.get("evaluation_contract", "")).strip() != expected_contract:
        raise ValueError(f"{stage_id} retention evidence must use {expected_contract}.")
    _finite_fraction(row, "success_rate")
    if not isinstance(row.get("finite_metrics"), bool):
        raise ValueError("retention finite_metrics must be boolean.")
    gate = row.get("success_gate_passed")
    if not isinstance(gate, bool):
        raise ValueError("retention success_gate_passed must be boolean.")


def _row_meets_retention_gate(
    row: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    stage_id: str,
) -> bool:
    success_rate = _finite_fraction(row, "success_rate")
    retained = (
        bool(row["finite_metrics"])
        and bool(row["success_gate_passed"])
        and float(baseline["success_rate"]) - success_rate <= 0.05 + 1.0e-12
    )
    if stage_id != "c1_straight":
        return retained
    return retained and (
        _finite_fraction(row, "termination_rate") <= 0.05
        and float(baseline["score"]) - _finite_float(row, "score") <= 5.0
        and _finite_nonnegative(row, "mean_abs_cross_track_error_m")
        <= max(2.0 * float(baseline["mean_abs_cross_track_error_m"]), 0.25)
        and _finite_nonnegative(row, "mean_abs_height_error_m")
        <= max(2.0 * float(baseline["mean_abs_height_error_m"]), 0.25)
    )


def _validate_stage(stage_id: str) -> str:
    stage = str(stage_id).strip().lower()
    if stage not in SPATIAL_REQUIRED_RETENTION_STAGES:
        raise ValueError(f"Unknown spatial promotion stage: {stage_id!r}.")
    return stage


def _required_text(row: Mapping[str, object], name: str) -> str:
    value = str(row.get(name, "")).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _required_iteration(row: Mapping[str, object]) -> int:
    raw = row.get("ppo_iteration")
    if isinstance(raw, bool):
        raise ValueError("ppo_iteration must be a non-negative integer.")
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("ppo_iteration must be a non-negative integer.") from error
    if value < 0 or float(raw) != value:
        raise ValueError("ppo_iteration must be a non-negative integer.")
    return value


def _finite_float(row: Mapping[str, object], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _finite_fraction(row: Mapping[str, object], name: str) -> float:
    value = _finite_float(row, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite fraction.")
    return value


def _finite_nonnegative(row: Mapping[str, object], name: str) -> float:
    value = _finite_float(row, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


__all__ = [
    "SPATIAL_REQUIRED_RETENTION_STAGES",
    "build_sample_equivalent_promotion_schedule",
    "evaluate_spatial_promotion",
]

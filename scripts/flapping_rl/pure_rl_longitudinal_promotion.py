"""Fail-closed consecutive-checkpoint promotion for PureRL C2."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from pure_rl_longitudinal_eval import row_meets_longitudinal_promotion_gate


def evaluate_longitudinal_promotion(
    evaluation_rows: Sequence[Mapping[str, object]],
    *,
    c1_retention_rows: Sequence[Mapping[str, object]],
    source_c1_baseline: Mapping[str, object] | None,
    stage_id: str,
    minimum_ppo_iteration: int = 200,
    evaluation_interval: int = 100,
) -> dict[str, object]:
    """Promote the later of two consecutive C2 and C1-retention passes."""

    stage = str(stage_id).strip().lower()
    if stage not in {"c2a", "c2b", "c2c"}:
        raise ValueError(f"Unknown longitudinal promotion stage: {stage_id!r}.")
    if source_c1_baseline is None:
        raise ValueError("source_c1_baseline is required for longitudinal promotion.")
    if minimum_ppo_iteration < 0 or evaluation_interval <= 0:
        raise ValueError("Promotion iteration settings must be non-negative with a positive interval.")
    baseline = _validate_source_baseline(source_c1_baseline)
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
        if not bool(row.get("grid_complete")):
            raise ValueError(f"Incomplete longitudinal evaluation grid: {checkpoint}")
        if int(row.get("climb_case_count", 0)) <= 0 or int(row.get("descent_case_count", 0)) <= 0:
            raise ValueError(f"Evaluation row is missing climb or descent cases: {checkpoint}")
        _validate_evaluation_metrics(row)
        checkpoints.append(checkpoint)
        iterations.append(iteration)
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("evaluation_rows contains duplicate checkpoints.")
    if len(set(iterations)) != len(iterations):
        raise ValueError("evaluation_rows contains duplicate PPO iterations.")
    if any(later <= earlier for earlier, later in zip(iterations, iterations[1:])):
        raise ValueError("evaluation_rows must be ordered by strictly increasing PPO iteration.")

    retention_by_checkpoint: dict[str, Mapping[str, object]] = {}
    for row in c1_retention_rows:
        checkpoint = _required_text(row, "checkpoint")
        if checkpoint in retention_by_checkpoint:
            raise ValueError(f"duplicate C1 retention row: {checkpoint}")
        if str(row.get("evaluation_stage", "")).strip() != "c1_straight":
            raise ValueError(f"C2 promotion requires c1_straight retention: {checkpoint}")
        retention_by_checkpoint[checkpoint] = row
    if set(retention_by_checkpoint) != set(checkpoints):
        missing = sorted(set(checkpoints).difference(retention_by_checkpoint))
        unexpected = sorted(set(retention_by_checkpoint).difference(checkpoints))
        raise ValueError(f"C1 retention rows must match evaluation checkpoints; missing={missing}, unexpected={unexpected}.")

    audits: list[dict[str, object]] = []
    for row, checkpoint, iteration in zip(evaluations, checkpoints, iterations):
        c2_passed = row_meets_longitudinal_promotion_gate(row)
        retention_passed = _row_meets_c1_retention_gate(retention_by_checkpoint[checkpoint], baseline)
        audits.append(
            {
                "checkpoint": checkpoint,
                "ppo_iteration": iteration,
                "c2_passed": c2_passed,
                "c1_retention_passed": retention_passed,
                "eligible": iteration >= minimum_ppo_iteration and c2_passed and retention_passed,
            }
        )

    for previous, current in zip(audits, audits[1:]):
        consecutive = int(current["ppo_iteration"]) - int(previous["ppo_iteration"]) == evaluation_interval
        if bool(previous["eligible"]) and bool(current["eligible"]) and consecutive:
            return {
                "stage_id": stage,
                "promoted": True,
                "checkpoint": current["checkpoint"],
                "ppo_iteration": current["ppo_iteration"],
                "consecutive_checkpoints": [previous["checkpoint"], current["checkpoint"]],
                "source_c1_checkpoint": baseline["checkpoint"],
                "checkpoint_audits": audits,
            }
    return {
        "stage_id": stage,
        "promoted": False,
        "checkpoint": None,
        "ppo_iteration": None,
        "consecutive_checkpoints": [],
        "source_c1_checkpoint": baseline["checkpoint"],
        "checkpoint_audits": audits,
    }


def _row_meets_c1_retention_gate(
    row: Mapping[str, object],
    baseline: Mapping[str, object],
) -> bool:
    try:
        success_rate = _finite_float(row, "success_rate")
        termination_rate = _finite_float(row, "termination_rate")
        score = _finite_float(row, "score")
        cross_track_error = _finite_float(row, "mean_abs_cross_track_error_m")
        height_error = _finite_float(row, "mean_abs_height_error_m")
        return bool(row.get("finite_metrics")) and (
            success_rate >= 0.95
            and termination_rate <= 0.05
            and float(baseline["score"]) - score <= 5.0
            and cross_track_error <= max(2.0 * float(baseline["mean_abs_cross_track_error_m"]), 0.25)
            and height_error <= max(2.0 * float(baseline["mean_abs_height_error_m"]), 0.25)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _validate_source_baseline(row: Mapping[str, object]) -> dict[str, object]:
    checkpoint = _required_text(row, "checkpoint")
    return {
        "checkpoint": checkpoint,
        "score": _finite_float(row, "score"),
        "mean_abs_cross_track_error_m": _finite_nonnegative_float(
            row,
            "mean_abs_cross_track_error_m",
        ),
        "mean_abs_height_error_m": _finite_nonnegative_float(row, "mean_abs_height_error_m"),
    }


def _validate_evaluation_metrics(row: Mapping[str, object]) -> None:
    for name in (
        "overall_survival_rate",
        "climb_success_rate",
        "descent_success_rate",
        "recovery_reached_rate",
        "mean_abs_cross_track_error_m",
        "mean_abs_height_error_m",
        "p95_abs_height_error_m",
        "reverse_motion_fraction",
    ):
        _finite_float(row, name)
    if not isinstance(row.get("finite_metrics"), bool):
        raise ValueError("finite_metrics must be boolean.")


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


def _finite_nonnegative_float(row: Mapping[str, object], name: str) -> float:
    value = _finite_float(row, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


__all__ = ["evaluate_longitudinal_promotion"]

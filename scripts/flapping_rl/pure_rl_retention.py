"""Fail-closed curriculum retention-matrix contracts for PureRL experiments."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


RETENTION_CELL_FIELDS: tuple[str, ...] = (
    "training_stage",
    "training_stage_index",
    "checkpoint",
    "evaluation_stage",
    "evaluation_stage_index",
    "evaluation_contract",
    "success_gate_passed",
    "score",
    "diagonal_score",
    "score_drop_from_diagonal",
    "forgetting_score_drop",
)

RETENTION_ROW_FIELDS: tuple[str, ...] = (
    "training_stage",
    "training_stage_index",
    "checkpoint",
    "evaluated_stages",
    "failed_evaluation_stages",
    "maximum_forgetting_score_drop",
    "retention_passed",
)

RETENTION_OPTIONAL_DETAIL_FIELDS: tuple[str, ...] = (
    "success_rate",
    "termination_rate",
    "mean_abs_cross_track_error_m",
    "mean_abs_height_error_m",
    "finite_metrics",
)


def build_retention_matrix(
    records: Sequence[Mapping[str, object]],
    *,
    stage_order: Sequence[str],
) -> dict[str, object]:
    """Build a complete lower-triangular retention matrix.

    One promoted checkpoint is required per training stage. A checkpoint at
    stage ``k`` must be evaluated on every stage from zero through ``k``.
    Missing, duplicate, future-stage, or ambiguous checkpoint records raise
    ``ValueError`` rather than producing a partial retention claim.
    """

    stages = _validate_stage_order(stage_order)
    if not records:
        raise ValueError("retention records must not be empty.")
    stage_index = {stage: index for index, stage in enumerate(stages)}

    normalized: list[dict[str, object]] = []
    checkpoints_by_stage: dict[str, set[str]] = {stage: set() for stage in stages}
    seen_cells: set[tuple[str, str]] = set()
    for raw_record in records:
        training_stage = _required_text(raw_record, "training_stage")
        evaluation_stage = _required_text(raw_record, "evaluation_stage")
        checkpoint = _required_text(raw_record, "checkpoint")
        evaluation_contract = _required_text(raw_record, "evaluation_contract")
        if training_stage not in stage_index:
            raise ValueError(f"unknown training_stage: {training_stage}")
        if evaluation_stage not in stage_index:
            raise ValueError(f"unknown evaluation_stage: {evaluation_stage}")
        if stage_index[evaluation_stage] > stage_index[training_stage]:
            raise ValueError(
                f"future evaluation stage {evaluation_stage} is invalid for {training_stage}."
            )
        cell_key = (training_stage, evaluation_stage)
        if cell_key in seen_cells:
            raise ValueError(f"duplicate retention cell: {cell_key}")
        seen_cells.add(cell_key)
        checkpoints_by_stage[training_stage].add(checkpoint)
        normalized_record: dict[str, object] = {
                "training_stage": training_stage,
                "checkpoint": checkpoint,
                "evaluation_stage": evaluation_stage,
                "evaluation_contract": evaluation_contract,
                "success_gate_passed": int(
                    _parse_gate(raw_record.get("success_gate_passed"))
                ),
                "score": _finite_float(raw_record.get("score"), name="score"),
            }
        for name in RETENTION_OPTIONAL_DETAIL_FIELDS[:-1]:
            if raw_record.get(name) not in (None, ""):
                normalized_record[name] = _finite_float(raw_record.get(name), name=name)
        if raw_record.get("finite_metrics") not in (None, ""):
            normalized_record["finite_metrics"] = _parse_gate(raw_record.get("finite_metrics"))
        normalized.append(normalized_record)

    for stage, checkpoints in checkpoints_by_stage.items():
        if len(checkpoints) != 1:
            raise ValueError(
                f"training stage {stage} must have exactly one promoted checkpoint; "
                f"found {len(checkpoints)}."
            )

    record_by_cell = {
        (str(record["training_stage"]), str(record["evaluation_stage"])): record
        for record in normalized
    }
    required_cells = {
        (training_stage, evaluation_stage)
        for training_idx, training_stage in enumerate(stages)
        for evaluation_stage in stages[: training_idx + 1]
    }
    missing = sorted(required_cells.difference(record_by_cell))
    unexpected = sorted(set(record_by_cell).difference(required_cells))
    if missing or unexpected:
        raise ValueError(
            f"retention matrix must be a complete lower triangle; missing={missing}, "
            f"unexpected={unexpected}."
        )

    diagonal_score = {
        stage: float(record_by_cell[(stage, stage)]["score"])
        for stage in stages
    }
    cells: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    for training_idx, training_stage in enumerate(stages):
        stage_cells: list[dict[str, object]] = []
        for evaluation_stage in stages[: training_idx + 1]:
            record = record_by_cell[(training_stage, evaluation_stage)]
            score = float(record["score"])
            score_drop = diagonal_score[evaluation_stage] - score
            cell = {
                **record,
                "training_stage_index": training_idx,
                "evaluation_stage_index": stage_index[evaluation_stage],
                "diagonal_score": diagonal_score[evaluation_stage],
                "score_drop_from_diagonal": score_drop,
                "forgetting_score_drop": max(0.0, score_drop),
            }
            cells.append(cell)
            stage_cells.append(cell)

        failed_stages = [
            str(cell["evaluation_stage"])
            for cell in stage_cells
            if int(cell["success_gate_passed"]) != 1
        ]
        checkpoint_rows.append(
            {
                "training_stage": training_stage,
                "training_stage_index": training_idx,
                "checkpoint": next(iter(checkpoints_by_stage[training_stage])),
                "evaluated_stages": [str(cell["evaluation_stage"]) for cell in stage_cells],
                "failed_evaluation_stages": failed_stages,
                "maximum_forgetting_score_drop": max(
                    float(cell["forgetting_score_drop"]) for cell in stage_cells
                ),
                "retention_passed": int(not failed_stages),
            }
        )

    return {
        "stage_order": list(stages),
        "cells": cells,
        "checkpoint_rows": checkpoint_rows,
        "all_retained": all(int(row["retention_passed"]) == 1 for row in checkpoint_rows),
    }


def _validate_stage_order(stage_order: Sequence[str]) -> tuple[str, ...]:
    stages = tuple(str(stage).strip() for stage in stage_order)
    if not stages or any(not stage for stage in stages):
        raise ValueError("stage_order must contain non-empty stage names.")
    if len(set(stages)) != len(stages):
        raise ValueError("stage_order must not contain duplicates.")
    return stages


def _required_text(record: Mapping[str, object], name: str) -> str:
    value = str(record.get(name, "")).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _finite_float(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _parse_gate(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true"}:
            return True
        if normalized in {"0", "false"}:
            return False
    raise ValueError("success_gate_passed must be boolean or 0/1.")


__all__ = [
    "RETENTION_CELL_FIELDS",
    "RETENTION_OPTIONAL_DETAIL_FIELDS",
    "RETENTION_ROW_FIELDS",
    "build_retention_matrix",
]

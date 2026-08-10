"""Helpers for selecting and materializing the best evaluated checkpoint."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from path_tracking_success_gate import row_meets_path_tracking_success_gate
from pure_rl_eval_common import (
    PURE_RL_CURRICULUM1_EVAL_CONTRACT,
    row_meets_pure_rl_success_gate,
)
from pure_rl_longitudinal_eval import (
    LONGITUDINAL_EVAL_CONTRACTS,
    row_meets_longitudinal_promotion_gate,
)


def _row_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _row_float_default(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def load_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    summary_csv = Path(summary_csv).expanduser().resolve()
    if not summary_csv.is_file():
        return []
    with summary_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def select_best_checkpoint_row(
    summary_csv: Path,
    *,
    case: str = "suite",
    evaluation_contract: str | None = None,
) -> dict[str, str] | None:
    rows = [row for row in load_summary_rows(summary_csv) if row.get("case") == case and row.get("checkpoint")]
    if evaluation_contract is not None:
        rows = [row for row in rows if row.get("evaluation_contract") == evaluation_contract]
    rows = [
        row
        for row in rows
        if row.get("evaluation_contract") not in LONGITUDINAL_EVAL_CONTRACTS.values()
        or str(row.get("c1_retention_passed", "")).strip().lower() in {"1", "true"}
    ]
    if not rows:
        return None

    def _is_path_tracking_row(row: dict[str, str]) -> bool:
        return "completion_rate" in row and "mean_abs_lateral_error_m" in row

    def _is_pure_rl_row(row: dict[str, str]) -> bool:
        return row.get("evaluation_contract") == PURE_RL_CURRICULUM1_EVAL_CONTRACT

    def _is_longitudinal_row(row: dict[str, str]) -> bool:
        return row.get("evaluation_contract") in LONGITUDINAL_EVAL_CONTRACTS.values()

    def _sort_key(row: dict[str, str]) -> tuple[float, float, float, float, float, float, float]:
        if _is_longitudinal_row(row):
            return (
                float(row_meets_longitudinal_promotion_gate(row)),
                _row_float(row, "overall_survival_rate"),
                min(_row_float(row, "climb_success_rate"), _row_float(row, "descent_success_rate")),
                _row_float(row, "recovery_reached_rate"),
                -_row_float(row, "mean_abs_height_error_m"),
                -_row_float(row, "p95_abs_height_error_m"),
                -_row_float(row, "mean_abs_cross_track_error_m"),
            )
        if _is_pure_rl_row(row):
            return (
                float(row_meets_pure_rl_success_gate(row)),
                _row_float(row, "timeout_rate"),
                _row_float(row, "mean_episode_duration_s"),
                -_row_float(row, "termination_rate"),
                _row_float(row, "score"),
                _row_float(row, "mean_along_track_progress_m"),
                -_row_float(row, "mean_abs_cross_track_error_m"),
            )
        if _is_path_tracking_row(row):
            completion_rate = _row_float_default(row, "completion_rate", 0.0)
            progress_ratio = _row_float_default(row, "mean_final_progress_ratio", completion_rate)
            return (
                completion_rate,
                progress_ratio,
                -_row_float(row, "termination_rate"),
                -_row_float(row, "mean_abs_lateral_error_m"),
                -_row_float(row, "mean_abs_height_error_m"),
                -_row_float(row, "mean_abs_align_error_deg"),
                -_row_float_default(row, "score", 0.0),
            )
        return (
            _row_float(row, "score"),
            -_row_float(row, "termination_rate"),
            _row_float(row, "timeout_rate"),
            -_row_float(row, "mean_abs_vx_err"),
            -_row_float(row, "mean_abs_z_err"),
            -_row_float(row, "mean_max_abs_y"),
            -_row_float(row, "mean_max_tilt_deg"),
        )

    best = max(rows, key=_sort_key)
    tied = [row for row in rows if _sort_key(row) == _sort_key(best)]
    if len(tied) > 1:
        best = min(tied, key=lambda row: int(float(row.get("ckpt_index", -1))))

    best_row = dict(best)
    if _is_longitudinal_row(best_row):
        best_row["success_gate_passed"] = str(
            int(
                row_meets_longitudinal_promotion_gate(best_row)
                and str(best_row.get("c1_retention_passed", "")).strip().lower() in {"1", "true"}
            )
        )
    elif _is_pure_rl_row(best_row):
        best_row["success_gate_passed"] = str(int(row_meets_pure_rl_success_gate(best_row)))
    elif _is_path_tracking_row(best_row):
        best_row["success_gate_passed"] = str(int(row_meets_path_tracking_success_gate(best_row)))
    return best_row


def _write_best_checkpoint_symlink(best_model_path: Path, checkpoint_path: Path) -> None:
    if best_model_path.exists() or best_model_path.is_symlink():
        best_model_path.unlink()
    rel_target = os.path.relpath(checkpoint_path, start=best_model_path.parent)
    best_model_path.symlink_to(rel_target)


def refresh_best_checkpoint_artifacts(
    run_dir: Path,
    *,
    summary_csv: Path | None = None,
    case: str = "suite",
    evaluation_contract: str | None = None,
) -> dict[str, str] | None:
    run_dir = Path(run_dir).expanduser().resolve()
    summary_csv = Path(summary_csv).expanduser().resolve() if summary_csv is not None else (run_dir / "eval" / "summary.csv")
    best = select_best_checkpoint_row(
        summary_csv,
        case=case,
        evaluation_contract=evaluation_contract,
    )
    if best is None:
        return None

    checkpoint_path = Path(best["checkpoint"]).expanduser().resolve()
    eval_dir = summary_csv.parent
    eval_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "best_checkpoint.txt").write_text(f"{checkpoint_path}\n")
    (eval_dir / "best_checkpoint.json").write_text(json.dumps(best, indent=2))

    try:
        _write_best_checkpoint_symlink(run_dir / "best_model.pt", checkpoint_path)
    except OSError:
        pass

    return best

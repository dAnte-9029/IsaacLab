"""Helpers for selecting and materializing the best evaluated checkpoint."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def _row_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def load_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    summary_csv = Path(summary_csv).expanduser().resolve()
    if not summary_csv.is_file():
        return []
    with summary_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def select_best_checkpoint_row(summary_csv: Path, *, case: str = "suite") -> dict[str, str] | None:
    rows = [row for row in load_summary_rows(summary_csv) if row.get("case") == case and row.get("checkpoint")]
    if not rows:
        return None

    def _is_path_tracking_row(row: dict[str, str]) -> bool:
        return "completion_rate" in row and "mean_abs_lateral_error_m" in row

    def _sort_key(row: dict[str, str]) -> tuple[float, float, float, float, float, float, float]:
        if _is_path_tracking_row(row):
            return (
                _row_float(row, "score"),
                _row_float(row, "completion_rate"),
                -_row_float(row, "termination_rate"),
                -_row_float(row, "mean_abs_lateral_error_m"),
                -_row_float(row, "mean_abs_height_error_m"),
                -_row_float(row, "mean_abs_align_error_deg"),
                _row_float(row, "timeout_rate"),
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
    return best


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
) -> dict[str, str] | None:
    run_dir = Path(run_dir).expanduser().resolve()
    summary_csv = Path(summary_csv).expanduser().resolve() if summary_csv is not None else (run_dir / "eval" / "summary.csv")
    best = select_best_checkpoint_row(summary_csv, case=case)
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

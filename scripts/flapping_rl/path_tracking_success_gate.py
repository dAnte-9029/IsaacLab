"""Shared success-gate helpers for path-tracking checkpoint evaluation."""

from __future__ import annotations

SUCCESS_MIN_COMPLETION_RATE = 0.80
SUCCESS_MIN_PROGRESS_RATIO = 0.85
SUCCESS_MAX_TERMINATION_RATE = 0.20


def _row_float_default(row: dict, key: str, default: float) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def row_meets_path_tracking_success_gate(row: dict) -> bool:
    """Return whether an aggregated path-tracking row satisfies the success gate."""
    completion_rate = _row_float_default(row, "completion_rate", 0.0)
    progress_ratio = _row_float_default(row, "mean_final_progress_ratio", completion_rate)
    termination_rate = _row_float_default(row, "termination_rate", 1.0)
    return (
        completion_rate >= SUCCESS_MIN_COMPLETION_RATE
        and progress_ratio >= SUCCESS_MIN_PROGRESS_RATIO
        and termination_rate <= SUCCESS_MAX_TERMINATION_RATE
    )

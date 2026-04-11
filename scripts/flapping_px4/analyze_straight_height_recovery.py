"""Analyze long-straight height recovery from a flapping path-mission trajectory CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze straight height recovery from trajectory_env0.csv.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Directory containing trajectory_env0.csv.")
    parser.add_argument("--traj_csv", type=Path, default=None, help="Path to trajectory_env0.csv.")
    parser.add_argument("--summary_json", type=Path, default=None, help="Optional summary.json path.")
    parser.add_argument("--output_json", type=Path, default=None, help="Where to write metrics JSON.")
    parser.add_argument("--height_band_m", type=float, default=0.05)
    parser.add_argument("--sustain_band_m", type=float, default=0.10)
    parser.add_argument("--sustain_time_s", type=float, default=1.0)
    parser.add_argument("--freq_sat_hz", type=float, default=4.99)
    parser.add_argument("--action_sat_abs", type=float, default=0.98)
    return parser.parse_args()


def _load_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({key: _to_float(value) for key, value in row.items()})
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def _get(row: dict[str, float], key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    return float(value) if math.isfinite(float(value)) else default


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _distance(row: dict[str, float]) -> float:
    progress = row.get("progress_s", float("nan"))
    if math.isfinite(progress):
        return float(progress)
    x = row.get("x", float("nan"))
    y = row.get("y", float("nan"))
    if math.isfinite(x) and math.isfinite(y):
        return math.hypot(float(x), float(y))
    return float("nan")


def _find_sustained_recovery(
    *,
    rows: list[dict[str, float]],
    start_idx: int,
    height_band_m: float,
    sustain_band_m: float,
    sustain_time_s: float,
) -> int | None:
    for idx in range(start_idx, len(rows)):
        height_error = _get(rows[idx], "height_error_m")
        if not math.isfinite(height_error) or abs(height_error) > height_band_m:
            continue

        start_t = _get(rows[idx], "t")
        if not math.isfinite(start_t):
            return idx
        end_t = start_t + max(float(sustain_time_s), 0.0)
        window = [row for row in rows[idx:] if _get(row, "t") <= end_t]
        if not window:
            continue
        if _get(window[-1], "t") < end_t:
            continue
        if all(abs(_get(row, "height_error_m")) <= sustain_band_m for row in window):
            return idx
    return None


def _find_cross_zero_after_min(rows: list[dict[str, float]], start_idx: int) -> int | None:
    min_height_error = _get(rows[start_idx], "height_error_m")
    if not math.isfinite(min_height_error):
        return None
    if min_height_error <= 0.0:
        predicate = lambda value: value >= 0.0
    else:
        predicate = lambda value: value <= 0.0
    for idx in range(start_idx, len(rows)):
        height_error = _get(rows[idx], "height_error_m")
        if math.isfinite(height_error) and predicate(height_error):
            return idx
    return None


def _delta_metric(rows: list[dict[str, float]], *, start_idx: int, end_idx: int | None, key: str) -> float | None:
    if end_idx is None:
        return None
    start = _get(rows[start_idx], key)
    end = _get(rows[end_idx], key)
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    return end - start


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compute_metrics(
    *,
    rows: list[dict[str, float]],
    summary: dict[str, Any],
    height_band_m: float,
    sustain_band_m: float,
    sustain_time_s: float,
    freq_sat_hz: float,
    action_sat_abs: float,
) -> dict[str, Any]:
    height_errors = [_get(row, "height_error_m") for row in rows]
    valid_indices = [idx for idx, value in enumerate(height_errors) if math.isfinite(value)]
    if not valid_indices:
        raise SystemExit("trajectory has no finite height_error_m samples")

    min_idx = min(valid_indices, key=lambda idx: height_errors[idx])
    first3_indices = [idx for idx, row in enumerate(rows) if _get(row, "t") <= 3.0 and math.isfinite(height_errors[idx])]
    first5_indices = [idx for idx, row in enumerate(rows) if _get(row, "t") <= 5.0]
    min_first3 = min((height_errors[idx] for idx in first3_indices), default=float("nan"))

    recovery_idx = _find_sustained_recovery(
        rows=rows,
        start_idx=min_idx,
        height_band_m=float(height_band_m),
        sustain_band_m=float(sustain_band_m),
        sustain_time_s=float(sustain_time_s),
    )
    zero_cross_idx = _find_cross_zero_after_min(rows, min_idx)
    recovery_rows = rows[recovery_idx:] if recovery_idx is not None else []

    distances = [_distance(row) for row in rows]
    distance_at_min = distances[min_idx]
    distance_to_recovery = None
    if recovery_idx is not None and math.isfinite(distance_at_min) and math.isfinite(distances[recovery_idx]):
        distance_to_recovery = distances[recovery_idx] - distance_at_min
    distance_to_zero = None
    if zero_cross_idx is not None and math.isfinite(distance_at_min) and math.isfinite(distances[zero_cross_idx]):
        distance_to_zero = distances[zero_cross_idx] - distance_at_min

    freq_values = [
        max(_get(row, "freq_hz", -float("inf")), _get(row, "exec_freq_hz", -float("inf"))) for row in rows
    ]
    freq_valid = [value for value in freq_values if math.isfinite(value)]
    elevon_pitch = [
        _get(row, "exec_action_elevon_pitch", _get(row, "action_elevon_pitch")) for row in rows
    ]
    elevon_valid = [value for value in elevon_pitch if math.isfinite(value)]

    metrics = {
        "num_rows": len(rows),
        "rollout_status": summary.get("status", summary.get("failure_kind")),
        "completed_path": summary.get("completed_path"),
        "progress_ratio_final": summary.get("progress_ratio_final", summary.get("final_progress_ratio")),
        "min_height_error_first3s_m": min_first3,
        "min_height_error_m": height_errors[min_idx],
        "time_at_min_height_error_s": _get(rows[min_idx], "t"),
        "distance_at_min_height_error_m": distance_at_min,
        "time_to_abs_0p05_after_min_s": _delta_metric(rows, start_idx=min_idx, end_idx=recovery_idx, key="t"),
        "distance_to_abs_0p05_after_min_m": distance_to_recovery,
        "time_to_cross_zero_after_min_s": _delta_metric(rows, start_idx=min_idx, end_idx=zero_cross_idx, key="t"),
        "distance_to_cross_zero_after_min_m": distance_to_zero,
        "mean_abs_height_error_after_recovery_m": _mean([abs(_get(row, "height_error_m")) for row in recovery_rows]),
        "max_positive_height_error_after_recovery_m": max(
            (_get(row, "height_error_m") for row in recovery_rows), default=float("nan")
        ),
        "freq_sat_fraction": (
            sum(1 for value in freq_valid if value >= float(freq_sat_hz)) / len(freq_valid) if freq_valid else float("nan")
        ),
        "elevon_pitch_sat_fraction": (
            sum(1 for value in elevon_valid if abs(value) >= float(action_sat_abs)) / len(elevon_valid)
            if elevon_valid
            else float("nan")
        ),
        "min_speed_mps": min((_get(row, "speed") for row in rows), default=float("nan")),
        "min_airspeed_mps": min((_get(row, "airspeed") for row in rows), default=float("nan")),
        "mean_tecs_pitch_sp_deg_first5s": _mean([_get(rows[idx], "tecs_pitch_sp_deg") for idx in first5_indices]),
        "mean_pitch_sp_deg_first5s": _mean([_get(rows[idx], "pitch_sp_deg") for idx in first5_indices]),
        "height_band_m": float(height_band_m),
        "sustain_band_m": float(sustain_band_m),
        "sustain_time_s": float(sustain_time_s),
    }
    return {key: _jsonable(value) for key, value in metrics.items()}


def main() -> None:
    args = _parse_args()
    if (args.run_dir is None) == (args.traj_csv is None):
        raise SystemExit("Provide exactly one of --run_dir or --traj_csv.")

    if args.run_dir is not None:
        traj_csv = args.run_dir / "trajectory_env0.csv"
        summary_json = args.summary_json or args.run_dir / "summary.json"
    else:
        traj_csv = args.traj_csv
        summary_json = args.summary_json or traj_csv.parent / "summary.json"
    output_json = args.output_json or traj_csv.parent / "straight_height_recovery_metrics.json"

    rows = _load_rows(traj_csv)
    summary = _load_summary(summary_json)
    metrics = compute_metrics(
        rows=rows,
        summary=summary,
        height_band_m=float(args.height_band_m),
        sustain_band_m=float(args.sustain_band_m),
        sustain_time_s=float(args.sustain_time_s),
        freq_sat_hz=float(args.freq_sat_hz),
        action_sat_abs=float(args.action_sat_abs),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

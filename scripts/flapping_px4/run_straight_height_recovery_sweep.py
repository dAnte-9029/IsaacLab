"""Run a compact TECS straight-height recovery sweep for the DeLaurier path mission."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


CANDIDATES: tuple[dict[str, float | str], ...] = (
    {
        "name": "baseline",
        "hold_band": 0.25,
        "capture_tc": 1.00,
        "extra_climb": 0.70,
        "pitch_speed_weight": 0.80,
        "pitch_speed_weight_capture": 0.35,
        "pitch_damping": 0.08,
    },
    {
        "name": "mild_capture",
        "hold_band": 0.15,
        "capture_tc": 0.80,
        "extra_climb": 1.00,
        "pitch_speed_weight": 0.65,
        "pitch_speed_weight_capture": 0.35,
        "pitch_damping": 0.08,
    },
    {
        "name": "balanced_fast",
        "hold_band": 0.10,
        "capture_tc": 0.65,
        "extra_climb": 1.00,
        "pitch_speed_weight": 0.55,
        "pitch_speed_weight_capture": 0.30,
        "pitch_damping": 0.10,
    },
    {
        "name": "aggressive_capture",
        "hold_band": 0.10,
        "capture_tc": 0.50,
        "extra_climb": 1.20,
        "pitch_speed_weight": 0.50,
        "pitch_speed_weight_capture": 0.25,
        "pitch_damping": 0.12,
    },
    {
        "name": "small_band",
        "hold_band": 0.05,
        "capture_tc": 0.65,
        "extra_climb": 1.00,
        "pitch_speed_weight": 0.50,
        "pitch_speed_weight_capture": 0.25,
        "pitch_damping": 0.10,
    },
    {
        "name": "altitude_priority",
        "hold_band": 0.10,
        "capture_tc": 0.65,
        "extra_climb": 1.20,
        "pitch_speed_weight": 0.45,
        "pitch_speed_weight_capture": 0.20,
        "pitch_damping": 0.12,
    },
    {
        "name": "baseline_gain0p9",
        "hold_band": 0.25,
        "capture_tc": 1.00,
        "extra_climb": 0.70,
        "pitch_speed_weight": 0.80,
        "pitch_speed_weight_capture": 0.35,
        "pitch_damping": 0.08,
        "altitude_error_gain": 0.90,
    },
    {
        "name": "small_band_gain0p9",
        "hold_band": 0.05,
        "capture_tc": 0.65,
        "extra_climb": 1.00,
        "pitch_speed_weight": 0.50,
        "pitch_speed_weight_capture": 0.25,
        "pitch_damping": 0.10,
        "altitude_error_gain": 0.90,
    },
    {
        "name": "small_band_gain1p2",
        "hold_band": 0.05,
        "capture_tc": 0.65,
        "extra_climb": 1.00,
        "pitch_speed_weight": 0.50,
        "pitch_speed_weight_capture": 0.25,
        "pitch_damping": 0.12,
        "altitude_error_gain": 1.20,
    },
    {
        "name": "small_band_gain1p5",
        "hold_band": 0.05,
        "capture_tc": 0.65,
        "extra_climb": 1.00,
        "pitch_speed_weight": 0.50,
        "pitch_speed_weight_capture": 0.25,
        "pitch_damping": 0.14,
        "altitude_error_gain": 1.50,
    },
    {
        "name": "small_band_gain1p8",
        "hold_band": 0.05,
        "capture_tc": 0.65,
        "extra_climb": 1.00,
        "pitch_speed_weight": 0.50,
        "pitch_speed_weight_capture": 0.25,
        "pitch_damping": 0.16,
        "altitude_error_gain": 1.80,
    },
    {
        "name": "fast_gain0p9",
        "hold_band": 0.05,
        "capture_tc": 0.50,
        "extra_climb": 1.20,
        "pitch_speed_weight": 0.50,
        "pitch_speed_weight_capture": 0.25,
        "pitch_damping": 0.12,
        "altitude_error_gain": 0.90,
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact straight height recovery TECS sweep.")
    parser.add_argument("--out_root", type=Path, default=Path("/tmp/straight_height_recovery_sweep_20260411"))
    parser.add_argument("--task", default="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0")
    parser.add_argument("--straight_length_m", type=float, default=180.0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--candidate", action="append", default=None, help="Candidate name to run; repeatable.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _candidate_command(*, args: argparse.Namespace, candidate: dict[str, float | str], out_dir: Path) -> list[str]:
    return [
        "./isaaclab.sh",
        "-p",
        "scripts/flapping_px4/fly_path_mission.py",
        "--task",
        str(args.task),
        "--phase",
        "level_straight",
        "--straight_length_m",
        str(float(args.straight_length_m)),
        "--steps",
        str(int(args.steps)),
        "--no-path_warmup_enabled",
        "--teacher_tecs_altitude_hold_error_band_m",
        str(candidate["hold_band"]),
        "--teacher_tecs_altitude_capture_time_const_s",
        str(candidate["capture_tc"]),
        "--teacher_tecs_altitude_error_gain",
        str(candidate.get("altitude_error_gain", 0.55)),
        "--teacher_tecs_capture_extra_climb_rate_mps",
        str(candidate["extra_climb"]),
        "--teacher_tecs_pitch_speed_weight",
        str(candidate["pitch_speed_weight"]),
        "--teacher_tecs_pitch_speed_weight_capture",
        str(candidate["pitch_speed_weight_capture"]),
        "--teacher_tecs_pitch_damping_gain",
        str(candidate["pitch_damping"]),
        "--headless",
        "--out_dir",
        str(out_dir),
    ]


def _latest_run_dir(out_dir: Path) -> Path:
    traj_paths = sorted(out_dir.glob("level_straight/*/trajectory_env0.csv"), key=lambda p: p.stat().st_mtime)
    if not traj_paths:
        raise RuntimeError(f"No trajectory_env0.csv found under {out_dir}")
    return traj_paths[-1].parent


def _run_analyzer(run_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/flapping_px4/analyze_straight_height_recovery.py",
        "--run_dir",
        str(run_dir),
    ]
    subprocess.run(cmd, check=True)
    metrics_path = run_dir / "straight_height_recovery_metrics.json"
    with metrics_path.open() as f:
        return json.load(f)


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    invalid_penalty = 1.0 if row.get("progress_ratio_final") is None else 0.0
    distance = row.get("distance_to_abs_0p05_after_min_m")
    zero_distance = row.get("distance_to_cross_zero_after_min_m")
    time = row.get("time_to_abs_0p05_after_min_s")
    overshoot = row.get("max_positive_height_error_after_recovery_m")
    freq_sat = row.get("freq_sat_fraction")
    return (
        invalid_penalty,
        float(zero_distance) if zero_distance is not None else 1.0e9,
        float(distance) if distance is not None else 1.0e9,
        float(time) if time is not None else 1.0e9,
        max(float(overshoot) if overshoot is not None else 1.0e9, 0.0)
        + 0.1 * (float(freq_sat) if freq_sat is not None else 1.0),
    )


def main() -> None:
    args = _parse_args()
    selected = set(args.candidate or [])
    candidates = [candidate for candidate in CANDIDATES if not selected or str(candidate["name"]) in selected]
    if not candidates:
        raise SystemExit(f"No candidates selected from: {[candidate['name'] for candidate in CANDIDATES]}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        out_dir = args.out_root / name
        cmd = _candidate_command(args=args, candidate=candidate, out_dir=out_dir)
        print("+ " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            run_dir = _latest_run_dir(out_dir)
            metrics = _run_analyzer(run_dir)
        else:
            run_dir = out_dir
            metrics = {}
        row = {"candidate": name, "run_dir": str(run_dir), **candidate, **metrics}
        rows.append(row)

    rows.sort(key=_rank_key)
    summary_json = args.out_root / "sweep_summary.json"
    summary_csv = args.out_root / "sweep_summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2, sort_keys=True))
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()

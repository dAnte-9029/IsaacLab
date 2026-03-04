"""Analyze straight-line rollout logs for sudden acceleration and force consistency.

Example:
  ./isaaclab.sh -p scripts/flapping_px4/analyze_trajectory.py --run_dir logs/flapping_px4/straight_line/20260303_162030
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trajectory_env0.csv from flapping_px4 rollouts.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Directory containing trajectory_env0.csv and summary.json.")
    parser.add_argument("--traj_csv", type=Path, default=None, help="Path to trajectory_env0.csv.")
    parser.add_argument("--context_steps", type=int, default=10, help="Print +-N steps around the peak.")
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _load_traj(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def main() -> None:
    args = _parse_args()
    if (args.run_dir is None) == (args.traj_csv is None):
        raise SystemExit("Provide exactly one of --run_dir or --traj_csv.")

    if args.run_dir is not None:
        run_dir = args.run_dir
        traj_path = run_dir / "trajectory_env0.csv"
        summary_path = run_dir / "summary.json"
    else:
        traj_path = args.traj_csv
        run_dir = traj_path.parent
        summary_path = run_dir / "summary.json"

    summary = _load_summary(summary_path)
    rows = _load_traj(traj_path)
    if len(rows) < 3:
        raise SystemExit(f"Not enough rows in {traj_path}.")

    dt = float(summary.get("env_dt_s", 0.008333333333333333))
    mass = float(summary.get("mass_total_kg", 0.0))

    def _get(row: dict[str, float], key: str, default: float = 0.0) -> float:
        return float(row[key]) if key in row else default

    speeds = [_get(r, "speed", (_get(r, "vx") ** 2 + _get(r, "vy") ** 2 + _get(r, "vz") ** 2) ** 0.5) for r in rows]
    vx_b = [_get(r, "vx_b", _get(r, "vx")) for r in rows]
    steps = [int(_get(r, "step")) for r in rows]
    done = [_get(r, "done", 0.0) for r in rows]
    episode_id = [_get(r, "episode_id", 0.0) for r in rows]

    # Peak speed and peak forward acceleration.
    idx_max_speed = max(range(len(rows)), key=lambda i: speeds[i])
    dvx_b_dt: list[float] = []
    valid_dvx: list[bool] = []
    for i in range(1, len(rows)):
        crosses_reset = (done[i - 1] > 0.5) or (episode_id[i] != episode_id[i - 1])
        if crosses_reset:
            dvx_b_dt.append(float("nan"))
            valid_dvx.append(False)
        else:
            dvx_b_dt.append((vx_b[i] - vx_b[i - 1]) / dt)
            valid_dvx.append(True)

    if not any(valid_dvx):
        raise SystemExit("No valid dvx samples (check that logs include done/episode_id).")

    idx_max_ax = 1 + max(range(len(dvx_b_dt)), key=lambda i: dvx_b_dt[i] if valid_dvx[i] else -1.0e30)

    print(f"run_dir: {run_dir}")
    if summary:
        print(f"summary: {json.dumps(summary, indent=2, sort_keys=True)}")
    print(f"dt_s: {dt:.6f}")
    print(f"num_steps: {len(rows)}")
    print(f"max_speed: {speeds[idx_max_speed]:.3f} m/s at step {steps[idx_max_speed]}")
    print(f"max_dvx_b_dt: {dvx_b_dt[idx_max_ax-1]:.3f} m/s^2 at step {steps[idx_max_ax-1]}->{steps[idx_max_ax]}")

    # If force columns exist, compare predicted vs numerical acceleration.
    if "ax_pred_mps2" in rows[0]:
        ax_pred = [_get(r, "ax_pred_mps2") for r in rows]
        err: list[float] = []
        err_valid: list[bool] = []
        for i in range(1, len(rows)):
            if not valid_dvx[i - 1]:
                err.append(float("nan"))
                err_valid.append(False)
            else:
                err.append(ax_pred[i] - dvx_b_dt[i - 1])
                err_valid.append(True)

        idx_max_err = 1 + max(range(len(err)), key=lambda i: abs(err[i]) if err_valid[i] else -1.0)
        abs_err = [abs(err[i]) for i in range(len(err)) if err_valid[i]]
        mean_abs_err = sum(abs_err) / max(len(abs_err), 1)
        print(f"mean_abs(ax_pred-dvx_b_dt): {mean_abs_err:.3f} m/s^2")
        print(
            "max_abs(ax_pred-dvx_b_dt):"
            f" {err[idx_max_err-1]:+.3f} m/s^2 at step {steps[idx_max_err-1]}->{steps[idx_max_err]}"
        )
    elif mass > 0.0 and "aero_total_Fx_b" in rows[0]:
        ax_pred = [_get(r, "aero_total_Fx_b") / mass for r in rows]
        err: list[float] = []
        err_valid: list[bool] = []
        for i in range(1, len(rows)):
            if not valid_dvx[i - 1]:
                err.append(float("nan"))
                err_valid.append(False)
            else:
                err.append(ax_pred[i] - dvx_b_dt[i - 1])
                err_valid.append(True)

        idx_max_err = 1 + max(range(len(err)), key=lambda i: abs(err[i]) if err_valid[i] else -1.0)
        abs_err = [abs(err[i]) for i in range(len(err)) if err_valid[i]]
        mean_abs_err = sum(abs_err) / max(len(abs_err), 1)
        print(f"mass_kg: {mass:.4f}")
        print(f"mean_abs(ax_pred-dvx_b_dt): {mean_abs_err:.3f} m/s^2")
        print(
            "max_abs(ax_pred-dvx_b_dt):"
            f" {err[idx_max_err-1]:+.3f} m/s^2 at step {steps[idx_max_err-1]}->{steps[idx_max_err]}"
        )

    # Print context around peak acceleration.
    ctx = int(args.context_steps)
    start = max(idx_max_ax - ctx, 0)
    end = min(idx_max_ax + ctx, len(rows) - 1)
    cols = [
        "episode_id",
        "step",
        "t",
        "speed",
        "vx",
        "vx_b",
        "vel_b_mismatch",
        "vz",
        "z",
        "pitch_deg",
        "pitch_sp_deg",
        "freq_hz",
        "aero_wing_Fx_b",
        "aero_tail_Fx_b",
        "aero_total_Fx_b",
        "ax_pred_mps2",
        "aero_total_Fz_b",
        "aero_total_Ty_b",
        "action_freq",
        "action_elevon_pitch",
        "done",
    ]
    print("context:")
    for i in range(start, end + 1):
        r = rows[i]
        values = []
        for c in cols:
            if c == "step":
                values.append(f"{int(_get(r, c))}")
            else:
                values.append(f"{_get(r, c, float('nan')):.3f}")
        print("  " + " ".join(values))


if __name__ == "__main__":
    main()

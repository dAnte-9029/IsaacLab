"""Plot straight-line rollout logs.

Example:
  ./isaaclab.sh -p scripts/flapping_px4/plot_trajectory.py --run_dir logs/flapping_px4/straight_line/20260304_121438

This script reads the CSV produced by scripts/flapping_px4/fly_straight_line.py and writes a PNG summary plot.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trajectory_env0.csv from flapping_px4 straight-line rollouts.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Directory containing trajectory_env0.csv and summary.json.")
    parser.add_argument("--traj_csv", type=Path, default=None, help="Path to trajectory_env0.csv.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <run_dir>/plots.png).")
    parser.add_argument("--dpi", type=int, default=160)
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


def _get(row: dict[str, float], key: str, default: float = float("nan")) -> float:
    return float(row[key]) if key in row else default


def _const_from_series(values: list[float], *, default: float) -> float:
    finite = [v for v in values if v == v]  # NaN check
    if not finite:
        return float(default)
    finite_sorted = sorted(finite)
    return float(finite_sorted[len(finite_sorted) // 2])


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

    rows = _load_traj(traj_path)
    if len(rows) < 3:
        raise SystemExit(f"Not enough rows in {traj_path}.")

    summary = _load_summary(summary_path)
    dt = float(summary.get("env_dt_s", 0.008333333333333333))

    t = [_get(r, "t", _get(r, "step", i) * dt) for i, r in enumerate(rows)]
    z = [_get(r, "z") for r in rows]
    speed = [_get(r, "speed") for r in rows]
    freq_hz = [_get(r, "freq_hz") for r in rows]

    action_freq = [_get(r, "action_freq") for r in rows]
    action_rudder = [_get(r, "action_rudder") for r in rows]
    action_elevon_pitch = [_get(r, "action_elevon_pitch") for r in rows]
    action_elevon_roll = [_get(r, "action_elevon_roll") for r in rows]

    # Height setpoint: prefer explicit summary, else reconstruct from height_err_m + z.
    height_sp = float(summary.get("height_sp_m", float("nan")))
    if not (height_sp == height_sp):
        height_err = [_get(r, "height_err_m") for r in rows]
        height_sp = _const_from_series([z_i + e_i for z_i, e_i in zip(z, height_err)], default=z[0])

    # Control surface deflections in degrees (from summary if present; else fall back to env defaults).
    rudder_max_deg = float(summary.get("rudder_max_deg", 25.0))
    elevon_max_deg = float(summary.get("elevon_max_deg", 25.0))
    elevon_trim_deg = float(summary.get("elevon_trim_deg", 0.0))
    elevon_pitch_mix = float(summary.get("elevon_pitch_mix", 1.0))
    elevon_roll_mix = float(summary.get("elevon_roll_mix", 1.0))

    rudder_deg = [rudder_max_deg * u for u in action_rudder]
    elevon_pitch_deg = [elevon_max_deg * u for u in action_elevon_pitch]
    elevon_roll_deg = [elevon_max_deg * u for u in action_elevon_roll]
    left_elevon_deg = [elevon_trim_deg + elevon_pitch_mix * p + elevon_roll_mix * r for p, r in zip(elevon_pitch_deg, elevon_roll_deg)]
    right_elevon_deg = [elevon_trim_deg + elevon_pitch_mix * p - elevon_roll_mix * r for p, r in zip(elevon_pitch_deg, elevon_roll_deg)]

    out_path = args.out if args.out is not None else (run_dir / "plots.png")

    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    ax_z = axs[0, 0]
    ax_v = axs[0, 1]
    ax_f = axs[1, 0]
    ax_u = axs[1, 1]

    ax_z.plot(t, z, label="z (m)")
    ax_z.axhline(height_sp, color="k", linestyle="--", linewidth=1.0, alpha=0.6, label="height_sp")
    ax_z.set_title("Altitude")
    ax_z.set_ylabel("m")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc="best")

    ax_v.plot(t, speed, label="speed (m/s)")
    ax_v.set_title("Speed")
    ax_v.set_ylabel("m/s")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="best")

    ax_f.plot(t, freq_hz, label="flap freq (Hz)")
    ax_f.set_title("Flapping Frequency")
    ax_f.set_xlabel("t (s)")
    ax_f.set_ylabel("Hz")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc="best")

    ax_u.plot(t, rudder_deg, label="rudder (deg)")
    ax_u.plot(t, left_elevon_deg, label="left elevon (deg)")
    ax_u.plot(t, right_elevon_deg, label="right elevon (deg)")
    ax_u2 = ax_u.twinx()
    ax_u2.plot(t, action_freq, color="k", alpha=0.25, label="throttle (normalized)")
    ax_u.set_title("Control Surfaces / Throttle")
    ax_u.set_xlabel("t (s)")
    ax_u.set_ylabel("deg")
    ax_u2.set_ylabel("normalized")
    ax_u.grid(True, alpha=0.3)
    lines, labels = ax_u.get_legend_handles_labels()
    lines2, labels2 = ax_u2.get_legend_handles_labels()
    ax_u.legend(lines + lines2, labels + labels2, loc="best")

    fig.suptitle(str(run_dir))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)

    z_min, z_max = min(z), max(z)
    v_min, v_max = min(speed), max(speed)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "traj_csv": str(traj_path),
                "summary_json": str(summary_path) if summary_path.exists() else None,
                "plots_png": str(out_path),
                "height_sp_m": float(height_sp),
                "z_min_m": float(z_min),
                "z_max_m": float(z_max),
                "speed_min_mps": float(v_min),
                "speed_max_mps": float(v_max),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


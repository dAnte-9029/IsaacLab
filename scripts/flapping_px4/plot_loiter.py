"""Plot loiter rollout logs.

Example:
  ./isaaclab.sh -p scripts/flapping_px4/plot_loiter.py --run_dir logs/flapping_px4/loiter/20260305_120000
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
    parser = argparse.ArgumentParser(description="Plot trajectory_env0.csv from flapping_px4 loiter rollouts.")
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
    finite = [v for v in values if v == v]
    if not finite:
        return float(default)
    finite_sorted = sorted(finite)
    return float(finite_sorted[len(finite_sorted) // 2])


def _mean_abs(values: list[float]) -> float:
    finite = [abs(v) for v in values if v == v]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


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
    x = [_get(r, "x") for r in rows]
    y = [_get(r, "y") for r in rows]
    z = [_get(r, "z") for r in rows]
    speed = [_get(r, "speed") for r in rows]
    airspeed = [_get(r, "airspeed") for r in rows]
    freq_hz = [_get(r, "freq_hz") for r in rows]
    radial_error = [_get(r, "radial_error_m") for r in rows]
    height_error = [_get(r, "height_err_m") for r in rows]
    wind_x = [_get(r, "wind_x_mps") for r in rows]
    wind_y = [_get(r, "wind_y_mps") for r in rows]
    action_freq = [_get(r, "action_freq") for r in rows]
    action_rudder = [_get(r, "action_rudder") for r in rows]
    action_elevon_pitch = [_get(r, "action_elevon_pitch") for r in rows]
    action_elevon_roll = [_get(r, "action_elevon_roll") for r in rows]

    height_sp = float(summary.get("height_sp_m", float("nan")))
    if not (height_sp == height_sp):
        height_sp = _const_from_series([z_i + e_i for z_i, e_i in zip(z, height_error)], default=z[0])

    center_x = float(summary.get("loiter_center_x", 0.0))
    center_y = float(summary.get("loiter_center_y", 0.0))
    radius = float(summary.get("loiter_radius_m", 0.0))

    out_path = args.out if args.out is not None else (run_dir / "plots.png")

    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=False)
    ax_z = axs[0, 0]
    ax_v = axs[0, 1]
    ax_f = axs[1, 0]
    ax_u = axs[1, 1]
    ax_path = axs[2, 0]
    ax_wind = axs[2, 1]

    ax_z.plot(t, z, label="z (m)")
    ax_z.axhline(height_sp, color="k", linestyle="--", linewidth=1.0, alpha=0.6, label="height_sp")
    ax_z.set_title("Altitude")
    ax_z.set_ylabel("m")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc="best")

    ax_v.plot(t, speed, label="speed (m/s)")
    ax_v.plot(t, airspeed, linewidth=1.1, alpha=0.9, label="airspeed (m/s)")
    ax_v.set_title("Speed")
    ax_v.set_ylabel("m/s")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="best")

    ax_f.plot(t, freq_hz, label="flap freq (Hz)")
    ax_f2 = ax_f.twinx()
    ax_f2.plot(t, radial_error, color="tab:red", alpha=0.75, label="radial error (m)")
    ax_f.set_title("Flapping and Radial Error")
    ax_f.set_xlabel("t (s)")
    ax_f.set_ylabel("Hz")
    ax_f2.set_ylabel("m")
    ax_f.grid(True, alpha=0.3)
    lines, labels = ax_f.get_legend_handles_labels()
    lines2, labels2 = ax_f2.get_legend_handles_labels()
    ax_f.legend(lines + lines2, labels + labels2, loc="best")

    ax_u.plot(t, action_rudder, label="rudder")
    ax_u.plot(t, action_elevon_pitch, label="elevon_pitch")
    ax_u.plot(t, action_elevon_roll, label="elevon_roll")
    ax_u2 = ax_u.twinx()
    ax_u2.plot(t, action_freq, color="k", alpha=0.25, label="throttle")
    ax_u.set_title("Control Inputs")
    ax_u.set_xlabel("t (s)")
    ax_u.set_ylabel("normalized")
    ax_u2.set_ylabel("normalized")
    ax_u.grid(True, alpha=0.3)
    lines, labels = ax_u.get_legend_handles_labels()
    lines2, labels2 = ax_u2.get_legend_handles_labels()
    ax_u.legend(lines + lines2, labels + labels2, loc="best")

    ax_path.plot(x, y, label="trajectory")
    if radius > 0.0:
        import numpy as np

        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        circle_x = center_x + radius * np.cos(theta)
        circle_y = center_y + radius * np.sin(theta)
        ax_path.plot(circle_x, circle_y, "k--", linewidth=1.0, alpha=0.7, label="target circle")
        ax_path.scatter([center_x], [center_y], s=16, c="k", label="center")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_title("Ground Track")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.grid(True, alpha=0.3)
    mean_abs_radial = float(summary.get("mean_abs_radial_error_m", float("nan")))
    if not (mean_abs_radial == mean_abs_radial):
        mean_abs_radial = _mean_abs(radial_error)
    ax_path.text(
        0.02,
        0.96,
        f"mean |radial| = {mean_abs_radial:.3f} m" if mean_abs_radial == mean_abs_radial else "mean |radial| = n/a",
        transform=ax_path.transAxes,
        va="top",
    )
    ax_path.legend(loc="best")

    ax_wind.plot(t, wind_x, label="wind_x (m/s)")
    ax_wind.plot(t, wind_y, label="wind_y (m/s)")
    ax_wind2 = ax_wind.twinx()
    ax_wind2.plot(t, height_error, color="tab:purple", alpha=0.8, label="height_err (m)")
    ax_wind.set_title("Wind and Height Error")
    ax_wind.set_xlabel("t (s)")
    ax_wind.set_ylabel("m/s")
    ax_wind2.set_ylabel("m")
    ax_wind.grid(True, alpha=0.3)
    lines, labels = ax_wind.get_legend_handles_labels()
    lines2, labels2 = ax_wind2.get_legend_handles_labels()
    ax_wind.legend(lines + lines2, labels + labels2, loc="best")

    fig.suptitle(str(run_dir))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "traj_csv": str(traj_path),
                "summary_json": str(summary_path) if summary_path.exists() else None,
                "plots_png": str(out_path),
                "height_sp_m": float(height_sp),
                "mean_abs_radial_error_m": mean_abs_radial,
                "mean_abs_height_error_m": _mean_abs(height_error),
                "mean_speed_mps": _const_from_series(speed, default=float("nan")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

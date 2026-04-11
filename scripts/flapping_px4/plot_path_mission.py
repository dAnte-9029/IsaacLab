"""Plot path-mission teacher rollout logs."""

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
    parser = argparse.ArgumentParser(description="Plot trajectory_env0.csv from flapping_px4 path-mission rollouts.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Directory containing rollout artifacts.")
    parser.add_argument("--traj_csv", type=Path, default=None, help="Path to trajectory_env0.csv.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <run_dir>/plots.png).")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--ground_track_y_zoom_margin_m",
        type=float,
        default=2.0,
        help="Y-axis margin for nearly straight ground tracks; set negative to disable.",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def _get(row: dict[str, float], key: str, default: float = float("nan")) -> float:
    return float(row[key]) if key in row else default


def _has_finite(values: list[float]) -> bool:
    return any(v == v for v in values)


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
        ref_path = run_dir / "reference_path.csv"
    else:
        traj_path = args.traj_csv
        run_dir = traj_path.parent
        summary_path = run_dir / "summary.json"
        ref_path = run_dir / "reference_path.csv"

    rows = _load_rows(traj_path)
    if len(rows) < 3:
        raise SystemExit(f"Not enough rows in {traj_path}.")
    summary = _load_summary(summary_path)
    ref_rows = _load_rows(ref_path) if ref_path.exists() else []

    t = [_get(r, "t") for r in rows]
    x = [_get(r, "x") for r in rows]
    y = [_get(r, "y") for r in rows]
    z = [_get(r, "z") for r in rows]
    speed = [_get(r, "speed") for r in rows]
    airspeed = [_get(r, "airspeed") for r in rows]
    lateral_error = [_get(r, "lateral_error_m") for r in rows]
    height_error = [_get(r, "height_error_m") for r in rows]
    align_error = [_get(r, "align_error_deg") for r in rows]
    progress_ratio = [_get(r, "progress_ratio") for r in rows]
    freq_hz = [_get(r, "freq_hz") for r in rows]
    action_freq = [_get(r, "action_freq") for r in rows]
    action_rudder = [_get(r, "action_rudder") for r in rows]
    action_elevon_pitch = [_get(r, "action_elevon_pitch") for r in rows]
    action_elevon_roll = [_get(r, "action_elevon_roll") for r in rows]
    wind_x = [_get(r, "wind_x_mps") for r in rows]
    wind_y = [_get(r, "wind_y_mps") for r in rows]
    reference_x = [_get(r, "reference_x") for r in rows]
    reference_y = [_get(r, "reference_y") for r in rows]

    ref_x = [_get(r, "x_ref") for r in ref_rows]
    ref_y = [_get(r, "y_ref") for r in ref_rows]
    ref_z = [_get(r, "z_ref") for r in ref_rows]
    height_sp = float(summary.get("height_sp_m", z[0]))
    warmup_s = float(summary.get("metrics_warmup_s", float("nan")))

    out_path = args.out if args.out is not None else (run_dir / "plots.png")

    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=False)
    ax_alt = axs[0, 0]
    ax_speed = axs[0, 1]
    ax_err = axs[1, 0]
    ax_ctrl = axs[1, 1]
    ax_path = axs[2, 0]
    ax_wind = axs[2, 1]

    ax_alt.plot(t, z, label="z (m)")
    ax_alt.axhline(height_sp, color="k", linestyle="--", linewidth=1.0, alpha=0.6, label="height_sp")
    ax_alt.set_title("Altitude")
    ax_alt.set_ylabel("m")
    ax_alt.grid(True, alpha=0.3)
    ax_alt.legend(loc="best")

    ax_speed.plot(t, speed, label="speed (m/s)")
    if _has_finite(airspeed):
        ax_speed.plot(t, airspeed, linewidth=1.1, alpha=0.9, label="airspeed (m/s)")
    ax_speed2 = ax_speed.twinx()
    ax_speed2.plot(t, progress_ratio, color="tab:green", alpha=0.75, label="progress ratio")
    ax_speed.set_title("Speed / Progress")
    ax_speed.set_ylabel("m/s")
    ax_speed2.set_ylabel("ratio")
    ax_speed.grid(True, alpha=0.3)
    lines, labels = ax_speed.get_legend_handles_labels()
    lines2, labels2 = ax_speed2.get_legend_handles_labels()
    ax_speed.legend(lines + lines2, labels + labels2, loc="best")

    ax_err.plot(t, lateral_error, label="lateral err (m)")
    ax_err.plot(t, height_error, label="height err (m)")
    ax_err2 = ax_err.twinx()
    ax_err2.plot(t, align_error, color="tab:purple", alpha=0.8, label="align err (deg)")
    ax_err.set_title("Tracking Errors")
    ax_err.set_xlabel("t (s)")
    ax_err.set_ylabel("m")
    ax_err2.set_ylabel("deg")
    ax_err.grid(True, alpha=0.3)
    lines, labels = ax_err.get_legend_handles_labels()
    lines2, labels2 = ax_err2.get_legend_handles_labels()
    ax_err.legend(lines + lines2, labels + labels2, loc="best")

    ax_ctrl.plot(t, action_rudder, label="rudder")
    ax_ctrl.plot(t, action_elevon_pitch, label="elevon_pitch")
    ax_ctrl.plot(t, action_elevon_roll, label="elevon_roll")
    ax_ctrl2 = ax_ctrl.twinx()
    ax_ctrl2.plot(t, action_freq, color="k", alpha=0.25, label="throttle")
    if _has_finite(freq_hz):
        ax_ctrl2.plot(t, freq_hz, color="tab:red", alpha=0.5, label="freq_hz")
    ax_ctrl.set_title("Control Inputs")
    ax_ctrl.set_xlabel("t (s)")
    ax_ctrl.set_ylabel("normalized")
    ax_ctrl2.set_ylabel("normalized / Hz")
    ax_ctrl.grid(True, alpha=0.3)
    lines, labels = ax_ctrl.get_legend_handles_labels()
    lines2, labels2 = ax_ctrl2.get_legend_handles_labels()
    ax_ctrl.legend(lines + lines2, labels + labels2, loc="best")

    ax_path.plot(x, y, label="trajectory")
    if _has_finite(reference_x) and _has_finite(reference_y):
        ax_path.plot(reference_x, reference_y, alpha=0.5, linewidth=1.0, label="closest path points")
    if ref_rows:
        ax_path.plot(ref_x, ref_y, "k--", linewidth=1.0, alpha=0.8, label="reference path")
    ax_path.set_title("Ground Track")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.grid(True, alpha=0.3)
    ax_path.set_aspect("equal", adjustable="box")
    x_finite = [v for v in x if v == v]
    y_finite = [v for v in y if v == v]
    if x_finite and y_finite and float(args.ground_track_y_zoom_margin_m) >= 0.0:
        x_span = max(x_finite) - min(x_finite)
        y_span = max(y_finite) - min(y_finite)
        if x_span > 1.0 and y_span < 0.08 * x_span:
            y_mid = 0.5 * (min(y_finite) + max(y_finite))
            y_margin = max(float(args.ground_track_y_zoom_margin_m), 0.5 * y_span, 0.5)
            ax_path.set_aspect("auto", adjustable="box")
            ax_path.set_ylim(y_mid - y_margin, y_mid + y_margin)
            ax_path.text(0.02, 0.86, "y-axis zoomed", transform=ax_path.transAxes, va="top", fontsize=8)
    ax_path.text(
        0.02,
        0.96,
        f"mean |lat| = {_mean_abs(lateral_error):.3f} m",
        transform=ax_path.transAxes,
        va="top",
    )
    ax_path.legend(loc="best")

    if _has_finite(wind_x):
        ax_wind.plot(t, wind_x, label="wind_x (m/s)")
    if _has_finite(wind_y):
        ax_wind.plot(t, wind_y, label="wind_y (m/s)")
    ax_wind2 = ax_wind.twinx()
    ax_wind2.plot(t, progress_ratio, color="tab:green", alpha=0.6, label="progress ratio")
    ax_wind.set_title("Wind / Progress")
    ax_wind.set_xlabel("t (s)")
    ax_wind.set_ylabel("m/s")
    ax_wind2.set_ylabel("ratio")
    ax_wind.grid(True, alpha=0.3)
    lines, labels = ax_wind.get_legend_handles_labels()
    lines2, labels2 = ax_wind2.get_legend_handles_labels()
    if lines or lines2:
        ax_wind.legend(lines + lines2, labels + labels2, loc="best")

    if warmup_s == warmup_s and warmup_s > 0.0:
        for axis in (ax_alt, ax_speed, ax_err, ax_ctrl, ax_wind):
            axis.axvline(warmup_s, color="tab:gray", linestyle="--", linewidth=1.0, alpha=0.8)

    completed_path = bool(summary.get("completed_path", False))
    failure_kind = summary.get("failure_kind")
    final_progress = summary.get("final_progress_ratio", summary.get("progress_ratio_final", progress_ratio[-1]))
    status = "completed" if completed_path else str(failure_kind or "running")
    fig.suptitle(f"{run_dir} | {status} progress={float(final_progress):.3f}")
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
                "reference_csv": str(ref_path) if ref_path.exists() else None,
                "plots_png": str(out_path),
                "mean_abs_lateral_error_m": _mean_abs(lateral_error),
                "mean_abs_height_error_m": _mean_abs(height_error),
                "mean_abs_align_error_deg": _mean_abs(align_error),
                "final_progress_ratio": progress_ratio[-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

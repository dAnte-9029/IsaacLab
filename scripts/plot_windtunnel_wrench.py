#!/usr/bin/env python3
"""Plot wind-tunnel rig forces/torques from `isaac_wind_tunnel_flappingbot_v50.py` CSV.

Example:
    ./isaaclab.sh -p scripts/plot_windtunnel_wrench.py \\
      --csv outputs/wang2016_v50_windtunnel.csv \\
      --frame wang \\
      --out outputs/wang2016_v50_windtunnel_wang_wrench.png
"""

from __future__ import annotations

import argparse
import csv as csvlib
from pathlib import Path


def _read_csv(path: Path) -> dict[str, list[float]]:
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(path)
        out: dict[str, list[float]] = {}
        for c in df.columns:
            out[str(c)] = df[c].astype(float).tolist()
        return out
    except Exception:
        out: dict[str, list[float]] = {}
        with path.open("r", newline="") as f:
            reader = csvlib.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            for name in reader.fieldnames:
                out[name] = []
            for row in reader:
                for name in reader.fieldnames:
                    v = row.get(name, "")
                    try:
                        out[name].append(float(v))
                    except Exception:
                        out[name].append(float("nan"))
        return out


def _slice_indices(data_len: int, *, start: int, end: int | None, every: int) -> slice:
    if start < 0:
        start = 0
    if end is None or end > data_len:
        end = data_len
    if every <= 0:
        every = 1
    return slice(start, end, every)


def _smooth(y: list[float], window: int) -> list[float]:
    """Simple moving-average smoothing (centered)."""
    if window <= 1 or len(y) < 3:
        return y
    import numpy as np  # type: ignore

    w = int(window)
    if w % 2 == 0:
        w += 1
    k = np.ones(w, dtype=np.float64) / float(w)
    arr = np.asarray(y, dtype=np.float64)
    pad = w // 2
    arr_p = np.pad(arr, (pad, pad), mode="edge")
    out = np.convolve(arr_p, k, mode="valid")
    return out.tolist()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot left/right wing wrench curves from wind-tunnel CSV.")
    p.add_argument("--csv", type=Path, required=True, help="Input CSV from isaac_wind_tunnel_flappingbot_v50.py.")
    p.add_argument(
        "--frame",
        type=str,
        choices=("wang", "link", "world"),
        default="wang",
        help="Which frame to plot from CSV columns F_* and tau_*.",
    )
    p.add_argument(
        "--total",
        action="store_true",
        help="Plot the total (Left+Right) wrench instead of separate Left/Right curves.",
    )
    p.add_argument(
        "--about-base",
        action="store_true",
        help="When plotting world torques, plot net torque about base_link (tau_about_base) if available.",
    )
    p.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    p.add_argument("--start-step", type=int, default=0, help="Start at this step index.")
    p.add_argument("--end-step", type=int, default=None, help="Stop before this step index (default: end).")
    p.add_argument("--every", type=int, default=1, help="Plot every Nth sample (downsample).")
    p.add_argument("--smooth-window", type=int, default=1, help="Optional moving-average window (odd integer).")
    p.add_argument("--show", action="store_true", help="Show the plot window (if a display is available).")
    args = p.parse_args()

    data = _read_csv(args.csv)
    if "step" not in data or "t" not in data:
        raise ValueError("CSV must contain 'step' and 't' columns.")

    n = len(data["step"])
    sl = _slice_indices(n, start=args.start_step, end=args.end_step, every=args.every)

    frame = args.frame
    if args.total and frame == "world":
        # Prefer using precomputed total columns if present.
        Fx = data.get("F_world_x_T", [])
        Fy = data.get("F_world_y_T", [])
        Fz = data.get("F_world_z_T", [])
        if Fx and Fy and Fz:
            FxT, FyT, FzT = Fx[sl], Fy[sl], Fz[sl]
        else:
            FxT = [a + b for a, b in zip(data["F_world_x_L"], data["F_world_x_R"])][sl]
            FyT = [a + b for a, b in zip(data["F_world_y_L"], data["F_world_y_R"])][sl]
            FzT = [a + b for a, b in zip(data["F_world_z_L"], data["F_world_z_R"])][sl]

        if args.about_base and all(k in data for k in ("tau_world_x_T_about_base", "tau_world_y_T_about_base", "tau_world_z_T_about_base")):
            TxT = data["tau_world_x_T_about_base"][sl]
            TyT = data["tau_world_y_T_about_base"][sl]
            TzT = data["tau_world_z_T_about_base"][sl]
        else:
            Tx = data.get("tau_world_x_T", [])
            Ty = data.get("tau_world_y_T", [])
            Tz = data.get("tau_world_z_T", [])
            if Tx and Ty and Tz:
                TxT, TyT, TzT = Tx[sl], Ty[sl], Tz[sl]
            else:
                TxT = [a + b for a, b in zip(data["tau_world_x_L"], data["tau_world_x_R"])][sl]
                TyT = [a + b for a, b in zip(data["tau_world_y_L"], data["tau_world_y_R"])][sl]
                TzT = [a + b for a, b in zip(data["tau_world_z_L"], data["tau_world_z_R"])][sl]
    else:
        FxL = data[f"F_{frame}_x_L"][sl]
        FyL = data[f"F_{frame}_y_L"][sl]
        FzL = data[f"F_{frame}_z_L"][sl]
        TxL = data[f"tau_{frame}_x_L"][sl]
        TyL = data[f"tau_{frame}_y_L"][sl]
        TzL = data[f"tau_{frame}_z_L"][sl]

        FxR = data[f"F_{frame}_x_R"][sl]
        FyR = data[f"F_{frame}_y_R"][sl]
        FzR = data[f"F_{frame}_z_R"][sl]
        TxR = data[f"tau_{frame}_x_R"][sl]
        TyR = data[f"tau_{frame}_y_R"][sl]
        TzR = data[f"tau_{frame}_z_R"][sl]

    t = data["t"][sl]
    q = data.get("wing_q", [float("nan")] * n)[sl]
    qd = data.get("wing_qd", [float("nan")] * n)[sl]

    out_path = args.out
    if out_path is None:
        out_path = Path("outputs") / f"{args.csv.stem}_{frame}_wrench.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt  # type: ignore

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    title = f"Wing Wrench ({frame} frame): {args.csv}"
    if args.total:
        title = f"Total Wing Wrench ({frame} frame): {args.csv}"
    fig.suptitle(title)

    def _plot_pair(ax, yL, yR, title: str, ylabel: str):
        ax.plot(t, yL, label="Left")
        ax.plot(t, yR, label="Right", linestyle="--")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    def _plot_total(ax, yT, title: str, ylabel: str):
        ax.plot(t, yT, label="Total")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    if args.total and frame == "world":
        FxT = _smooth(FxT, int(args.smooth_window))
        FyT = _smooth(FyT, int(args.smooth_window))
        FzT = _smooth(FzT, int(args.smooth_window))
        TxT = _smooth(TxT, int(args.smooth_window))
        TyT = _smooth(TyT, int(args.smooth_window))
        TzT = _smooth(TzT, int(args.smooth_window))
        _plot_total(axes[0, 0], FyT, "Force Fy (total)", "N")
        _plot_total(axes[1, 0], FxT, "Force Fx (total)", "N")
        _plot_total(axes[2, 0], FzT, "Force Fz (total)", "N")

        torque_suffix = " (about base_link)" if args.about_base else " (sum of torques)"
        _plot_total(axes[0, 1], TxT, f"Torque tau_x (total){torque_suffix}", "N·m")
        _plot_total(axes[1, 1], TyT, f"Torque tau_y (total){torque_suffix}", "N·m")
        _plot_total(axes[2, 1], TzT, f"Torque tau_z (total){torque_suffix}", "N·m")
        axes[0, 0].legend(loc="best")
    else:
        FxL = _smooth(FxL, int(args.smooth_window))
        FyL = _smooth(FyL, int(args.smooth_window))
        FzL = _smooth(FzL, int(args.smooth_window))
        TxL = _smooth(TxL, int(args.smooth_window))
        TyL = _smooth(TyL, int(args.smooth_window))
        TzL = _smooth(TzL, int(args.smooth_window))
        FxR = _smooth(FxR, int(args.smooth_window))
        FyR = _smooth(FyR, int(args.smooth_window))
        FzR = _smooth(FzR, int(args.smooth_window))
        TxR = _smooth(TxR, int(args.smooth_window))
        TyR = _smooth(TyR, int(args.smooth_window))
        TzR = _smooth(TzR, int(args.smooth_window))
        _plot_pair(axes[0, 0], FyL, FyR, "Force Fy", "N")
        _plot_pair(axes[1, 0], FxL, FxR, "Force Fx", "N")
        _plot_pair(axes[2, 0], FzL, FzR, "Force Fz", "N")

        _plot_pair(axes[0, 1], TxL, TxR, "Torque tau_x", "N·m")
        _plot_pair(axes[1, 1], TyL, TyR, "Torque tau_y", "N·m")
        _plot_pair(axes[2, 1], TzL, TzR, "Torque tau_z", "N·m")
        axes[0, 0].legend(loc="best")

    axes[2, 0].set_xlabel("t (s)")
    axes[2, 1].set_xlabel("t (s)")

    # Add a small annotation with commanded motion if present.
    if any(v == v for v in q):  # not all NaN
        fig.text(0.01, 0.01, f"q range: [{min(q):.3f}, {max(q):.3f}] rad, qd range: [{min(qd):.3f}, {max(qd):.3f}] rad/s")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"[OK] wrote: {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

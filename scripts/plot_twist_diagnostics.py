"""Plot twist diagnostics from a single wind-tunnel CSV.

Outputs:
  - time series: eta_tip (L/R) and wing_qd
  - scatter: eta_tip vs wing_qd
  - optional heatmap: eta(x,t)=g(x)*eta_tip(t) if you pass wing geometry + twist shape args
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure repository `source/` is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from flapping_bot.flapping_bot.physics.qsm_wang2016 import eta_shape_piecewise


@dataclass(frozen=True)
class Series:
    t: np.ndarray
    wing_qd: np.ndarray
    eta_L: np.ndarray
    eta_R: np.ndarray


def _read_series(path: Path, *, start_step: int = 0, end_step: int | None = None) -> Series:
    cols = {"t": [], "wing_qd": [], "eta_tip_L": [], "eta_tip_R": []}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = [k for k in cols.keys() if k not in set(r.fieldnames)]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        for row in r:
            for k in cols.keys():
                cols[k].append(float(row[k]))
    n = len(cols["t"])
    if n == 0:
        raise ValueError(f"Empty CSV: {path}")
    i0 = max(0, int(start_step))
    i1 = int(end_step) if end_step is not None else n
    i1 = max(i0, min(i1, n))

    def _arr(k: str) -> np.ndarray:
        return np.asarray(cols[k][i0:i1], dtype=np.float64)

    return Series(t=_arr("t"), wing_qd=_arr("wing_qd"), eta_L=_arr("eta_tip_L"), eta_R=_arr("eta_tip_R"))


def _parse_f_hz_from_name(name: str) -> float | None:
    # Supports:
    #  - auto_sweep tag: f5p000_... (p is decimal point, m is minus)
    #  - simple tag: f3_V4_amp10.csv
    m = re.search(r"(?:^|/)f(?P<f>[0-9]+(?:p[0-9]+)?)", name)
    if m:
        s = m.group("f").replace("p", ".")
        try:
            return float(s)
        except Exception:
            return None
    m2 = re.search(r"(?:^|_)f(?P<f>[0-9]+(?:\\.[0-9]+)?)", name)
    if m2:
        try:
            return float(m2.group("f"))
        except Exception:
            return None
    return None


def _load_wing_geom_csv(path: Path) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    cs: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"x_mid_m", "c_m"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(f"Invalid wing geom CSV header in {path}. Expected columns: {sorted(expected)}")
        for row in reader:
            xs.append(float(row["x_mid_m"]))
            cs.append(float(row["c_m"]))
    if len(xs) < 2:
        raise ValueError(f"--wing-geom-csv must contain at least 2 rows: {path}")
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    cs = [cs[i] for i in order]
    return xs, cs


def _infer_span_from_x_mid(xs: list[float]) -> float:
    dxs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    dx = sorted(dxs)[len(dxs) // 2]
    if dx <= 0:
        raise ValueError("Invalid x_mid spacing in --wing-geom-csv.")
    return float(xs[-1] + 0.5 * dx)


def _interp1d_np(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot twist diagnostics from a single wind-tunnel CSV.")
    p.add_argument("--csv", type=Path, required=True, help="Input CSV from isaac_wind_tunnel_flappingbot_v50.py.")
    p.add_argument("--start-step", type=int, default=0, help="Discard first N steps.")
    p.add_argument("--end-step", type=int, default=None, help="Stop before this step.")
    p.add_argument("--f-hz", type=float, default=None, help="Flapping frequency (for phase plot). If omitted, try parse from filename.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: same as CSV).")

    p.add_argument("--heatmap", action=argparse.BooleanOptionalAction, default=False, help="Also plot eta(x,t) heatmap.")
    p.add_argument("--wing-geom-csv", type=Path, default=None, help="Wing geometry CSV (for x_mid).")
    p.add_argument("--R", type=float, default=0.65, help="Span R if no --wing-geom-csv.")
    p.add_argument("--N", type=int, default=80, help="Strip count N if no --wing-geom-csv.")
    p.add_argument("--x0", type=float, default=0.0, help="Twist region start (m).")
    p.add_argument("--x1", type=float, default=0.65, help="Twist region end (m).")
    p.add_argument("--shape", type=str, choices=("uniform", "linear", "smoothstep"), default="smoothstep", help="Twist shape g(x).")

    args = p.parse_args()

    series = _read_series(args.csv, start_step=int(args.start_step), end_step=args.end_step)
    f_hz = float(args.f_hz) if args.f_hz is not None else _parse_f_hz_from_name(str(args.csv))

    out_dir = args.out_dir if args.out_dir is not None else args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.csv.stem

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting but could not be imported: {exc}") from exc

    # Time series
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(series.t, np.degrees(series.eta_L), label="eta_tip_L (deg)", lw=2)
    ax[0].plot(series.t, np.degrees(series.eta_R), label="eta_tip_R (deg)", lw=2)
    ax[0].set_ylabel("eta_tip (deg)")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(series.t, series.wing_qd, label="wing_qd (rad/s)", lw=2)
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("wing_qd (rad/s)")
    ax[1].grid(True)
    ax[1].legend()
    fig.tight_layout()
    p_time = out_dir / f"{stem}_twist_time.png"
    fig.savefig(p_time, dpi=200)
    plt.close(fig)

    # Scatter eta vs qd
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    ax.scatter(series.wing_qd, np.degrees(series.eta_L), s=6, alpha=0.6, label="L")
    ax.scatter(series.wing_qd, np.degrees(series.eta_R), s=6, alpha=0.6, label="R")
    ax.set_xlabel("wing_qd (rad/s)")
    ax.set_ylabel("eta_tip (deg)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    p_scatter = out_dir / f"{stem}_eta_vs_qd.png"
    fig.savefig(p_scatter, dpi=200)
    plt.close(fig)

    # Phase plot (optional)
    p_phase = None
    if f_hz is not None and f_hz > 0:
        phase = (series.t * f_hz) % 1.0
        phase_deg = 360.0 * phase
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.scatter(phase_deg, np.degrees(series.eta_L), s=6, alpha=0.6, label="L")
        ax.scatter(phase_deg, np.degrees(series.eta_R), s=6, alpha=0.6, label="R")
        ax.set_xlabel("phase (deg)")
        ax.set_ylabel("eta_tip (deg)")
        ax.set_xlim(0, 360)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        p_phase = out_dir / f"{stem}_eta_vs_phase.png"
        fig.savefig(p_phase, dpi=200)
        plt.close(fig)

    # Heatmap eta(x,t) (optional)
    p_hm_L = p_hm_R = None
    if bool(args.heatmap):
        R = float(args.R)
        if args.wing_geom_csv is not None:
            xs, _cs = _load_wing_geom_csv(args.wing_geom_csv)
            R = _infer_span_from_x_mid(xs)
        N = int(args.N)
        x_mid = (np.arange(N, dtype=np.float64) + 0.5) * (R / float(N))

        if args.shape == "uniform":
            g = np.ones_like(x_mid)
        else:
            # Use the same smoothstep/linear as qsm_wang2016.eta_shape_piecewise for consistency.
            import torch

            x_t = torch.as_tensor(x_mid, dtype=torch.float32)
            g_t = eta_shape_piecewise(x_t, x0=float(args.x0), x1=float(args.x1), kind=str(args.shape))
            g = g_t.cpu().numpy().astype(np.float64)

        etaL_deg = np.degrees(series.eta_L)
        etaR_deg = np.degrees(series.eta_R)
        eta_xt_L = etaL_deg[:, None] * g[None, :]
        eta_xt_R = etaR_deg[:, None] * g[None, :]

        def _plot_hm(arr: np.ndarray, title: str, out_path: Path) -> None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))
            im = ax.imshow(
                arr,
                aspect="auto",
                origin="lower",
                extent=(float(x_mid[0]), float(x_mid[-1]), float(series.t[0]), float(series.t[-1])),
                cmap="coolwarm",
            )
            ax.set_xlabel("x (m)")
            ax.set_ylabel("t (s)")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="eta(x) (deg)")
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

        p_hm_L = out_dir / f"{stem}_eta_xt_L.png"
        p_hm_R = out_dir / f"{stem}_eta_xt_R.png"
        _plot_hm(eta_xt_L, f"eta(x,t) Left (shape={args.shape})", p_hm_L)
        _plot_hm(eta_xt_R, f"eta(x,t) Right (shape={args.shape})", p_hm_R)

    print(f"[OK] wrote: {p_time}")
    print(f"[OK] wrote: {p_scatter}")
    if p_phase is not None:
        print(f"[OK] wrote: {p_phase}")
    if p_hm_L is not None and p_hm_R is not None:
        print(f"[OK] wrote: {p_hm_L}")
        print(f"[OK] wrote: {p_hm_R}")


if __name__ == "__main__":
    main()

"""Print or export the spanwise virtual twist distribution g(x) and η(x).

This is a lightweight helper to verify how `--twist-eta-shape` distributes a tip twist
η_tip across the wing span:

  η(x,t) = g(x) * η_tip(t)

It does not launch Isaac Sim.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import torch

# Ensure repository `source/` is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from flapping_bot.flapping_bot.physics.qsm_wang2016 import WingGeometry, eta_shape_piecewise


def _load_wing_geom_csv(path: Path) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    cs: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"x_mid_m", "c_m"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(f"Invalid CSV header in {path}. Expected columns: {sorted(expected)}")
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


def _interp1d_torch(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if xp.ndim != 1 or fp.ndim != 1 or xp.numel() != fp.numel():
        raise ValueError("xp/fp must be 1D tensors with the same length.")
    if xp.numel() < 2:
        raise ValueError("xp/fp must have at least 2 points.")
    if not torch.all(xp[1:] >= xp[:-1]):
        raise ValueError("xp must be sorted (non-decreasing).")

    x_clamped = torch.clamp(x, float(xp[0]), float(xp[-1]))
    idx_hi = torch.searchsorted(xp, x_clamped, right=False).clamp(1, xp.numel() - 1)
    idx_lo = idx_hi - 1
    x0 = xp[idx_lo]
    x1 = xp[idx_hi]
    y0 = fp[idx_lo]
    y1 = fp[idx_hi]
    t = (x_clamped - x0) / torch.clamp(x1 - x0, min=1e-12)
    return y0 + t * (y1 - y0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect spanwise twist shape g(x) for the wind-tunnel twist model.")
    parser.add_argument("--wing-geom-csv", type=Path, default=None, help="Optional wing geometry CSV (x_mid_m,c_m,...).")
    parser.add_argument("--R", type=float, default=0.10, help="Span R (m) if not using --wing-geom-csv.")
    parser.add_argument("--N", type=int, default=80, help="Strip count N.")
    parser.add_argument("--x0", type=float, default=0.0, help="Twist region start (m).")
    parser.add_argument("--x1", type=float, default=0.10, help="Twist region end (m).")
    parser.add_argument(
        "--shape",
        type=str,
        choices=("uniform", "linear", "smoothstep"),
        default="linear",
        help="Twist shape g(x).",
    )
    parser.add_argument("--eta-tip-deg", type=float, default=20.0, help="Example η_tip (deg) for printing η(x)=g(x)η_tip.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    device = torch.device("cpu")
    dtype = torch.float32

    if args.wing_geom_csv is not None:
        xs, cs = _load_wing_geom_csv(args.wing_geom_csv)
        R = _infer_span_from_x_mid(xs)
        xp = torch.tensor(xs, device=device, dtype=dtype)
        fp = torch.tensor(cs, device=device, dtype=dtype)

        def chord_func(x_mid: torch.Tensor) -> torch.Tensor:
            return _interp1d_torch(xp, fp, x_mid)

        wing_geom = WingGeometry.from_input(
            {"R": R, "N": int(args.N), "chord_func": chord_func, "dhat": 0.0, "aspect_ratio": 0.0},
            device=device,
            dtype=dtype,
        )
    else:
        wing_geom = WingGeometry.from_input(
            {"R": float(args.R), "N": int(args.N), "chord_func": 0.05, "dhat": 0.0, "aspect_ratio": 0.0},
            device=device,
            dtype=dtype,
        )

    x = wing_geom.x_mid
    if args.shape == "uniform":
        g = torch.ones_like(x)
    else:
        g = eta_shape_piecewise(x, x0=float(args.x0), x1=float(args.x1), kind=str(args.shape))
    eta_tip = math.radians(float(args.eta_tip_deg))
    eta_x = g * eta_tip

    print(f"[OK] R={wing_geom.R:g} N={int(x.numel())}  twist region [{args.x0:g},{args.x1:g}] shape={args.shape}")
    print(f"[OK] g(x): min={float(g.min()):.3f} max={float(g.max()):.3f}")
    print(f"[OK] eta_tip={args.eta_tip_deg:g} deg -> eta(x): min={math.degrees(float(eta_x.min())):.3f} deg max={math.degrees(float(eta_x.max())):.3f} deg")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x_mid_m", "g", "eta_deg"])
            for xi, gi, ei in zip(x.tolist(), g.tolist(), eta_x.tolist(), strict=True):
                w.writerow([float(xi), float(gi), math.degrees(float(ei))])
        print(f"[OK] wrote: {args.out_csv}")


if __name__ == "__main__":
    main()

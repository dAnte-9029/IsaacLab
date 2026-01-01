# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit spanwise wing geometry (c(xc), d̂(xc)) from sparse keypoints.

This script is IsaacSim-independent. It produces either:
  1) discrete arrays `c[i]`, `dhat[i]` sampled at `N` spanwise midpoints, or
  2) a small Python module containing piecewise-linear `chord_func(xc)` / `dhat_func(xc)`.

The fitted outputs match Wang2016 definitions used by `WingGeometry`:
  - x_c: spanwise coordinate along the pitching axis
  - chordwise direction: along +z_c (default) in the wing link frame
  - d̂(x_c): normalized distance from LE to pitching axis: d̂ = (axis - LE)/c

Input formats
-------------
JSON is recommended. Two supported schemas:

Schema A (direct samples):
[
  {"x": 0.00, "c": 0.040, "dhat": 0.25},
  {"x": 0.03, "c": 0.038, "dhat": 0.24},
  ...
]

Schema B (geometry points per station):
[
  {"x": 0.00, "le": [..], "te": [..], "axis": [..]},
  {"x": 0.03, "le": [..], "te": [..], "axis": [..]},
  ...
]

For Schema B, vectors are in the wing link frame. `x` can be omitted if you pass
`--infer-x-from le|te|axis` to project onto `--span-axis`.

.. code-block:: bash

    python source/flapping_bot/scripts/fit_wing_geom_from_points.py \\
      --in points.json --schema samples --N 40 --out-json wing_geom.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


Schema = Literal["samples", "stations", "te_polyline"]
FitKind = Literal["pwl", "poly"]


def _as_tensor(x: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)


def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = torch.linalg.norm(v).clamp(min=eps)
    return v / n


def _piecewise_linear(xq: torch.Tensor, xk: torch.Tensor, yk: torch.Tensor) -> torch.Tensor:
    """1D piecewise-linear interpolation with clamping at endpoints (torch only)."""
    xk = xk.flatten()
    yk = yk.flatten()
    if xk.numel() < 2:
        raise ValueError("Need at least 2 keypoints for piecewise-linear interpolation.")
    # Ensure increasing xk
    order = torch.argsort(xk)
    xk = xk[order]
    yk = yk[order]

    xq_flat = xq.flatten()
    idx = torch.bucketize(xq_flat, xk) - 1
    idx = torch.clamp(idx, 0, xk.numel() - 2)
    x0 = xk[idx]
    x1 = xk[idx + 1]
    y0 = yk[idx]
    y1 = yk[idx + 1]
    t = (xq_flat - x0) / torch.clamp(x1 - x0, min=1e-12)
    yq = y0 + t * (y1 - y0)
    return yq.view_as(xq)


def _polyfit_s_normalized(x: torch.Tensor, y: torch.Tensor, *, degree: int) -> tuple[torch.Tensor, float]:
    """Least-squares polynomial fit y(s) where s = x/R in [0,1].

    Returns:
        (coef, R) where coef has shape (degree+1,) for y(s)=sum_i coef[i]*s^i.
    """
    if degree < 0:
        raise ValueError("degree must be >= 0.")
    x = x.flatten()
    y = y.flatten()
    if x.numel() != y.numel():
        raise ValueError("x and y must have same length.")
    if x.numel() < degree + 1:
        raise ValueError(f"Need at least degree+1 points ({degree+1}) to fit degree={degree}.")

    R = float(torch.max(x).item())
    if R <= 0:
        raise ValueError("Cannot normalize x by R<=0.")
    s = x / R

    # Vandermonde: [1, s, s^2, ..., s^degree]
    cols = [torch.ones_like(s)]
    for p in range(1, degree + 1):
        cols.append(cols[-1] * s)
    A = torch.stack(cols, dim=-1)  # (K, degree+1)

    # Least squares
    sol = torch.linalg.lstsq(A, y).solution  # (degree+1,)
    return sol, R


def _polyval_s(coef: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Evaluate y(s)=sum_i coef[i]*s^i with Horner's method."""
    y = torch.zeros_like(s)
    for c in reversed(coef):
        y = y * s + c
    return y

@dataclass(frozen=True)
class FittedWingGeom:
    x_mid: torch.Tensor  # (N,)
    dx: torch.Tensor  # (N,)
    c: torch.Tensor  # (N,)
    dhat: torch.Tensor  # (N,)
    R: float

    def to_winggeom_dict(self) -> dict[str, Any]:
        return {
            "R": float(self.R),
            "N": int(self.x_mid.numel()),
            "x_mid": [float(v) for v in self.x_mid.tolist()],
            "c": [float(v) for v in self.c.tolist()],
            "dhat": [float(v) for v in self.dhat.tolist()],
        }


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _extract_samples(
    data: Any, *, dtype: torch.dtype, dhat_const: float | None, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, cs, ds = [], [], []
    if not isinstance(data, list):
        raise ValueError("Schema 'samples' expects a JSON list.")
    for row in data:
        xs.append(float(row["x"]) * scale)
        cs.append(float(row["c"]) * scale)
        if "dhat" in row:
            ds.append(float(row["dhat"]))
        elif dhat_const is not None:
            ds.append(float(dhat_const))
        else:
            raise KeyError("Missing `dhat` in samples. Provide it or pass --dhat-const.")
    x = _as_tensor(xs, dtype=dtype)
    c = _as_tensor(cs, dtype=dtype)
    d = _as_tensor(ds, dtype=dtype)
    return x, c, d


def _extract_stations(
    data: Any,
    *,
    dtype: torch.dtype,
    span_axis: torch.Tensor,
    chord_axis: torch.Tensor,
    infer_x_from: Literal["none", "le", "te", "axis"],
    dhat_const: float | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(data, list):
        raise ValueError("Schema 'stations' expects a JSON list.")
    span_axis = _normalize(span_axis.to(dtype=dtype))
    chord_axis = _normalize(chord_axis.to(dtype=dtype))

    xs, cs, ds = [], [], []
    for row in data:
        le = (_as_tensor(row["le"], dtype=dtype).flatten()) * scale
        te = (_as_tensor(row["te"], dtype=dtype).flatten()) * scale
        if le.numel() != 3 or te.numel() != 3:
            raise ValueError("Each of `le` and `te` must be 3D vectors.")
        ax = None
        if "axis" in row:
            ax = (_as_tensor(row["axis"], dtype=dtype).flatten()) * scale
            if ax.numel() != 3:
                raise ValueError("`axis` must be a 3D vector.")

        if "x" in row:
            x = float(row["x"]) * scale
        else:
            if infer_x_from == "none":
                raise KeyError("Missing `x` in a station. Provide `x` or set --infer-x-from.")
            if infer_x_from == "axis" and ax is None:
                raise KeyError("Missing `axis` but --infer-x-from axis requested.")
            p = {"le": le, "te": te, "axis": ax}[infer_x_from]
            x = float(torch.dot(p, span_axis).item())

        # chord length projected on chord_axis (WingGeometry uses chord length along chordwise direction).
        chord = float(torch.abs(torch.dot(te - le, chord_axis)).item())
        if chord <= 0:
            raise ValueError("Computed chord length <= 0 for a station; check chord_axis and points.")
        if ax is None:
            if dhat_const is None:
                raise KeyError("Missing `axis` in a station. Provide it or pass --dhat-const.")
            d_hat = float(dhat_const)
        else:
            d_hat = float(torch.dot(ax - le, chord_axis).item() / chord)

        xs.append(x)
        cs.append(chord)
        ds.append(d_hat)

    x = _as_tensor(xs, dtype=dtype)
    c = _as_tensor(cs, dtype=dtype)
    d = _as_tensor(ds, dtype=dtype)
    return x, c, d


def _extract_te_polyline(
    data: Any, *, dtype: torch.dtype, dhat_const: float, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trailing-edge polyline in a 2D planform: points are (x, z_te) with LE at z=0.

    The chord is inferred as c(x) = -z_te(x) (assuming z_te <= 0). We use abs() for robustness,
    and also return the signed z_te keypoints so callers can plot the trailing edge in planform.
    """
    if dhat_const is None:
        raise ValueError("Schema 'te_polyline' requires --dhat-const (pitch-axis offset fraction).")
    xs, ztes, cs, ds = [], [], [], []
    if not isinstance(data, list):
        raise ValueError("Schema 'te_polyline' expects a JSON list.")
    for row in data:
        if isinstance(row, list) and len(row) == 2:
            x_raw, z_raw = row
        elif isinstance(row, dict):
            x_raw, z_raw = row["x"], row["z"]
        else:
            raise ValueError("Each entry must be [x,z] or {\"x\":..,\"z\":..}.")
        x = float(x_raw) * scale
        z_te = float(z_raw) * scale
        c = abs(z_te)
        xs.append(x)
        ztes.append(z_te)
        cs.append(c)
        ds.append(float(dhat_const))
    xk = _as_tensor(xs, dtype=dtype)
    z_te_k = _as_tensor(ztes, dtype=dtype)
    ck = _as_tensor(cs, dtype=dtype)
    dk = _as_tensor(ds, dtype=dtype)
    return xk, ck, dk, z_te_k


def fit_wing_geom(
    xk: torch.Tensor,
    ck: torch.Tensor,
    dk: torch.Tensor,
    *,
    N: int,
    R: float | None = None,
    dtype: torch.dtype = torch.float64,
    fit_kind: FitKind = "pwl",
    poly_degree: int = 3,
) -> FittedWingGeom:
    if xk.numel() != ck.numel() or xk.numel() != dk.numel():
        raise ValueError("x/c/dhat keypoint arrays must have the same length.")
    if xk.numel() < 2:
        raise ValueError("Need at least 2 keypoints to fit.")
    if N <= 0:
        raise ValueError("N must be positive.")

    xk = xk.to(dtype=dtype)
    ck = ck.to(dtype=dtype)
    dk = dk.to(dtype=dtype)

    # Determine R and midpoints.
    R_val = float(R) if R is not None else float(torch.max(xk).item())
    if R_val <= 0:
        raise ValueError("R must be positive (either inferred from max x or provided).")
    x_edges = torch.linspace(0.0, R_val, N + 1, dtype=dtype)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx = x_edges[1:] - x_edges[:-1]

    if fit_kind == "pwl":
        c = _piecewise_linear(x_mid, xk, ck)
        dhat = _piecewise_linear(x_mid, xk, dk)
    elif fit_kind == "poly":
        c_coef, c_R = _polyfit_s_normalized(xk, ck, degree=int(poly_degree))
        d_coef, d_R = _polyfit_s_normalized(xk, dk, degree=int(poly_degree))
        # Evaluate on normalized s in [0,1]
        s_mid_c = x_mid / float(c_R)
        s_mid_d = x_mid / float(d_R)
        c = _polyval_s(c_coef, s_mid_c)
        dhat = _polyval_s(d_coef, s_mid_d)
    else:
        raise ValueError(f"Unknown fit_kind: {fit_kind}")

    if torch.any(c <= 0):
        # Polynomials can overshoot slightly; clamp to keep geometry valid.
        c = torch.clamp(c, min=1e-9)

    return FittedWingGeom(x_mid=x_mid, dx=dx, c=c, dhat=dhat, R=R_val)


def _emit_python_module(
    xk: torch.Tensor,
    ck: torch.Tensor,
    dk: torch.Tensor,
    *,
    out_py: Path,
):
    # Write a torch-based piecewise-linear function so it can be used in WingGeometry as chord_func/dhat_func.
    x_list = [float(v) for v in xk.tolist()]
    c_list = [float(v) for v in ck.tolist()]
    d_list = [float(v) for v in dk.tolist()]
    out_py.parent.mkdir(parents=True, exist_ok=True)
    out_py.write_text(
        "\n".join(
            [
                '"""Auto-generated wing geometry fit (piecewise-linear)."""',
                "",
                "from __future__ import annotations",
                "",
                "import torch",
                "",
                f"_X = torch.tensor({x_list}, dtype=torch.float64)",
                f"_C = torch.tensor({c_list}, dtype=torch.float64)",
                f"_D = torch.tensor({d_list}, dtype=torch.float64)",
                "",
                "def _pwl(xq: torch.Tensor, xk: torch.Tensor, yk: torch.Tensor) -> torch.Tensor:",
                "    order = torch.argsort(xk)",
                "    xk = xk[order]",
                "    yk = yk[order]",
                "    idx = torch.bucketize(xq, xk) - 1",
                "    idx = torch.clamp(idx, 0, xk.numel() - 2)",
                "    x0 = xk[idx]",
                "    x1 = xk[idx + 1]",
                "    y0 = yk[idx]",
                "    y1 = yk[idx + 1]",
                "    t = (xq - x0) / torch.clamp(x1 - x0, min=1e-12)",
                "    return y0 + t * (y1 - y0)",
                "",
                "def chord_func(xc: torch.Tensor) -> torch.Tensor:",
                "    xc64 = xc.to(dtype=torch.float64)",
                "    return _pwl(xc64, _X, _C).to(dtype=xc.dtype, device=xc.device)",
                "",
                "def dhat_func(xc: torch.Tensor) -> torch.Tensor:",
                "    xc64 = xc.to(dtype=torch.float64)",
                "    return _pwl(xc64, _X, _D).to(dtype=xc.dtype, device=xc.device)",
                "",
            ]
        )
        + "\n"
    )


def _write_csv(
    fitted: FittedWingGeom,
    *,
    out_csv: Path,
) -> None:
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_mid_m", "c_m", "dhat"])
        for i in range(int(fitted.x_mid.numel())):
            w.writerow([float(fitted.x_mid[i]), float(fitted.c[i]), float(fitted.dhat[i])])


def _write_te_csv(
    fitted: FittedWingGeom,
    *,
    out_csv: Path,
) -> None:
    """Write planform trailing-edge points (x_mid, z_te=-c) to CSV.

    This is meaningful for the `te_polyline` schema where leading edge is assumed z=0.
    """
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_mid_m", "z_te_m"])
        for i in range(int(fitted.x_mid.numel())):
            w.writerow([float(fitted.x_mid[i]), float(-fitted.c[i])])


def _plot_fit(
    xk: torch.Tensor,
    ck: torch.Tensor,
    dk: torch.Tensor,
    fitted: FittedWingGeom,
    *,
    out_png: Path,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - import guard
        print("[ERR] Missing matplotlib; cannot plot.")
        print("      Install it inside env_isaaclab, e.g.:")
        print("          ./isaaclab.sh -p -m pip install matplotlib")
        print("      Error:", repr(e))
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax0 = axes[0]
    ax0.plot(fitted.x_mid.cpu().numpy(), fitted.c.cpu().numpy(), label="fitted c(x) (midpoints)")
    ax0.scatter(xk.cpu().numpy(), ck.cpu().numpy(), s=30, label="keypoints", zorder=3)
    ax0.set_ylabel("chord c [m]")
    ax0.grid(True)
    ax0.legend()

    ax1 = axes[1]
    ax1.plot(fitted.x_mid.cpu().numpy(), fitted.dhat.cpu().numpy(), label="fitted d̂(x) (midpoints)")
    ax1.scatter(xk.cpu().numpy(), dk.cpu().numpy(), s=30, label="keypoints", zorder=3)
    ax1.set_xlabel("spanwise x_c [m]")
    ax1.set_ylabel("d̂ [-]")
    ax1.grid(True)
    ax1.legend()

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    print(f"[OK] wrote: {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_planform_te(
    xk: torch.Tensor,
    z_te_k: torch.Tensor,
    fitted: FittedWingGeom,
    *,
    out_png: Path,
    show: bool,
    annotate: bool,
    fit_kind: FitKind,
    poly_degree: int,
) -> None:
    """Plot leading edge (z=0) and trailing edge z_te(x) inferred from c(x).

    This is intended for `--schema te_polyline` where:
      - leading edge is assumed to be the straight line z=0,
      - trailing edge keypoints are provided as z_te(x),
      - fitted chord is c(x) = |z_te(x)|, so the fitted trailing edge is z_te_fit(x) = -c(x).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - import guard
        print("[ERR] Missing matplotlib; cannot plot.")
        print("      Install it inside env_isaaclab, e.g.:")
        print("          ./isaaclab.sh -p -m pip install matplotlib")
        print("      Error:", repr(e))
        return

    # Use a dense evaluation for a smooth curve based on keypoints and the chosen fit kind.
    R = float(fitted.R)
    device = fitted.x_mid.device
    dtype = fitted.x_mid.dtype
    x_dense = torch.linspace(0.0, R, 400, device=device, dtype=dtype)

    # Keypoints for chord are |z_te|.
    c_k = torch.abs(z_te_k.to(device=device, dtype=dtype))
    if fit_kind == "pwl":
        c_dense = _piecewise_linear(x_dense, xk.to(device=device, dtype=dtype), c_k)
    elif fit_kind == "poly":
        c_coef, c_R = _polyfit_s_normalized(xk.to(device=device, dtype=dtype), c_k, degree=int(poly_degree))
        s_dense = x_dense / float(c_R)
        c_dense = _polyval_s(c_coef, s_dense)
    else:
        raise ValueError(f"Unknown fit_kind: {fit_kind}")
    c_dense = torch.clamp(c_dense, min=0.0)
    z_dense = (-c_dense).cpu().numpy()
    x_dense_np = x_dense.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.plot(x_dense_np, 0.0 * x_dense_np, "-", color="black", linewidth=1.0, label="LE (assumed z=0)")
    ax.plot(x_dense_np, z_dense, "-", linewidth=2.0, label=f"TE fit ({fit_kind})")
    ax.scatter(xk.cpu().numpy(), z_te_k.cpu().numpy(), s=35, zorder=3, label="TE keypoints")

    if annotate:
        for xi, zi in zip(xk.cpu().tolist(), z_te_k.cpu().tolist(), strict=False):
            ax.annotate(f"({xi:.3f}, {zi:.3f})", xy=(xi, zi), xytext=(4, -10), textcoords="offset points", fontsize=8)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z_te [m]")
    ax.set_title("Planform view: leading edge and trailing edge")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    print(f"[OK] wrote: {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Fit wing geometry (c(xc), d̂(xc)) from sparse keypoints.")
    p.add_argument("--in", dest="in_path", type=Path, required=True, help="Input JSON file.")
    p.add_argument("--schema", choices=["samples", "stations", "te_polyline"], default="samples", help="Input schema.")
    p.add_argument("--N", type=int, default=20, help="Output strip count N (midpoints over [0,R]).")
    p.add_argument("--R", type=float, default=None, help="Override span R (m). Default: max(x_key).")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64", help="Compute dtype.")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor applied to x/c coordinates (and vectors for stations). Use 0.001 for mm->m.",
    )
    p.add_argument(
        "--dhat-const",
        type=float,
        default=None,
        help="If provided, use a constant d̂ when input does not include dhat/axis.",
    )

    # For stations schema:
    p.add_argument(
        "--infer-x-from",
        choices=["none", "le", "te", "axis"],
        default="none",
        help="If station entries omit `x`, infer by projecting this point onto span-axis.",
    )
    p.add_argument("--span-axis", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="Span axis in wing frame.")
    p.add_argument("--chord-axis", type=float, nargs=3, default=[0.0, 0.0, 1.0], help="Chord axis in wing frame.")

    # Outputs:
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, write outputs into this directory using default names based on the input filename.",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Write fitted arrays to JSON.")
    p.add_argument("--out-py", type=Path, default=None, help="Write piecewise-linear chord_func/dhat_func module.")
    p.add_argument("--out-csv", type=Path, default=None, help="Write fitted midpoints (x_mid,c,dhat) to CSV.")
    p.add_argument("--out-te-csv", type=Path, default=None, help="Write planform trailing edge (x_mid,z_te=-c) to CSV.")
    p.add_argument("--print-table", action="store_true", help="Print fitted midpoints table to stdout.")
    p.add_argument("--plot", action="store_true", help="Plot c(x) and d̂(x) (requires matplotlib).")
    p.add_argument("--plot-planform", action="store_true", help="Plot planform LE/TE curve for --schema te_polyline.")
    p.add_argument("--annotate", action="store_true", help="Annotate keypoint coordinates on the planform plot.")
    p.add_argument("--fit", choices=["pwl", "poly"], default="pwl", help="Fit type for c(x) and d̂(x).")
    p.add_argument("--degree", type=int, default=3, help="Polynomial degree when --fit poly.")
    p.add_argument("--plot-out", type=Path, default=None, help="Output PNG path for --plot (default derived from --out-dir).")
    p.add_argument("--show", action="store_true", help="Show the plot window (if supported by your setup).")
    args = p.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    data = _load_json(args.in_path)

    z_te_k = None
    if args.schema == "samples":
        xk, ck, dk = _extract_samples(data, dtype=dtype, dhat_const=args.dhat_const, scale=args.scale)
    elif args.schema == "stations":
        xk, ck, dk = _extract_stations(
            data,
            dtype=dtype,
            span_axis=_as_tensor(args.span_axis, dtype=dtype),
            chord_axis=_as_tensor(args.chord_axis, dtype=dtype),
            infer_x_from=args.infer_x_from,
            dhat_const=args.dhat_const,
            scale=args.scale,
        )
    else:
        xk, ck, dk, z_te_k = _extract_te_polyline(data, dtype=dtype, dhat_const=args.dhat_const, scale=args.scale)

    fitted = fit_wing_geom(xk, ck, dk, N=args.N, R=args.R, dtype=dtype, fit_kind=args.fit, poly_degree=args.degree)

    # Print summary.
    print(f"[OK] keypoints: {xk.numel()}  -> fitted N={args.N}, R={fitted.R:.6f} m (fit={args.fit})")
    print(f"     chord range: [{float(fitted.c.min()):.6f}, {float(fitted.c.max()):.6f}] m")
    print(f"     dhat  range: [{float(fitted.dhat.min()):.6f}, {float(fitted.dhat.max()):.6f}]")
    if z_te_k is not None:
        print("     TE keypoints (x, z_te):")
        for xi, zi in zip(xk.tolist(), z_te_k.tolist(), strict=False):
            print(f"       ({float(xi):.6f}, {float(zi):.6f})")

    # Resolve outputs. If --out-dir is set, default names are derived from the input filename.
    out_json = args.out_json
    out_py = args.out_py
    out_csv = args.out_csv
    out_te_csv = args.out_te_csv
    plot_out = args.plot_out
    if args.out_dir is not None:
        def _under_out_dir(p: Path) -> Path:
            if p.is_absolute():
                return p
            out_dir_parts = args.out_dir.parts
            p_parts = p.parts
            if len(p_parts) >= len(out_dir_parts) and p_parts[: len(out_dir_parts)] == out_dir_parts:
                return p
            return args.out_dir / p

        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.in_path.stem
        if out_json is None:
            out_json = args.out_dir / f"{stem}_wing_geom.json"
        else:
            out_json = _under_out_dir(out_json)
        if out_py is None:
            out_py = args.out_dir / f"{stem}_wing_geom_fit.py"
        else:
            out_py = _under_out_dir(out_py)
        if out_csv is None:
            out_csv = args.out_dir / f"{stem}_wing_geom.csv"
        else:
            out_csv = _under_out_dir(out_csv)
        if out_te_csv is None:
            out_te_csv = args.out_dir / f"{stem}_te.csv"
        else:
            out_te_csv = _under_out_dir(out_te_csv)
        if plot_out is None:
            plot_out = args.out_dir / f"{stem}_wing_geom.png"
        else:
            plot_out = _under_out_dir(plot_out)

    wrote_any = False
    if out_json is not None:
        out = fitted.to_winggeom_dict()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[OK] wrote: {out_json}")
        wrote_any = True

    if out_py is not None:
        _emit_python_module(xk, ck, dk, out_py=out_py)
        print(f"[OK] wrote: {out_py}")
        wrote_any = True

    if out_csv is not None:
        _write_csv(fitted, out_csv=out_csv)
        print(f"[OK] wrote: {out_csv}")
        wrote_any = True
    if out_te_csv is not None:
        _write_te_csv(fitted, out_csv=out_te_csv)
        print(f"[OK] wrote: {out_te_csv}")
        wrote_any = True

    if args.print_table:
        print("\n[x_mid, c, dhat] fitted midpoints:")
        for i in range(int(fitted.x_mid.numel())):
            print(f"{i:03d}  x={float(fitted.x_mid[i]):.6f}  c={float(fitted.c[i]):.6f}  dhat={float(fitted.dhat[i]):.6f}")
        if z_te_k is not None:
            print("\n[x_mid, z_te] inferred trailing edge midpoints (z_te = -c):")
            for i in range(int(fitted.x_mid.numel())):
                print(f"{i:03d}  x={float(fitted.x_mid[i]):.6f}  z_te={float(-fitted.c[i]):.6f}")

    if args.plot:
        if plot_out is None:
            plot_out = Path(f"{args.in_path.stem}_wing_geom.png")
        _plot_fit(xk, ck, dk, fitted, out_png=plot_out, show=bool(args.show))

    if args.plot_planform:
        if args.schema != "te_polyline" or z_te_k is None:
            raise ValueError("--plot-planform requires --schema te_polyline.")
        out = plot_out
        if out is None:
            out = Path(f"{args.in_path.stem}_planform.png")
        else:
            out = out.with_name(out.stem + "_planform" + out.suffix)
        _plot_planform_te(
            xk,
            z_te_k,
            fitted,
            out_png=out,
            show=bool(args.show),
            annotate=bool(args.annotate),
            fit_kind=args.fit,
            poly_degree=int(args.degree),
        )

    if not wrote_any:
        print("[INFO] No output file requested. Use --out-dir and/or --out-json/--out-py.")


if __name__ == "__main__":
    main()

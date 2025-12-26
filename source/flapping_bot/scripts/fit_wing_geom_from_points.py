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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trailing-edge polyline in a 2D planform: points are (x, z_te) with LE at z=0.

    The chord is inferred as c(x) = -z_te(x) (assuming z_te <= 0). We use abs() for robustness.
    """
    if dhat_const is None:
        raise ValueError("Schema 'te_polyline' requires --dhat-const (pitch-axis offset fraction).")
    xs, cs, ds = [], [], []
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
        cs.append(c)
        ds.append(float(dhat_const))
    xk = _as_tensor(xs, dtype=dtype)
    ck = _as_tensor(cs, dtype=dtype)
    dk = _as_tensor(ds, dtype=dtype)
    return xk, ck, dk


def fit_wing_geom(
    xk: torch.Tensor,
    ck: torch.Tensor,
    dk: torch.Tensor,
    *,
    N: int,
    R: float | None = None,
    dtype: torch.dtype = torch.float64,
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

    c = _piecewise_linear(x_mid, xk, ck)
    dhat = _piecewise_linear(x_mid, xk, dk)

    if torch.any(c <= 0):
        raise ValueError("Fitted chord has non-positive values; check inputs.")

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
    args = p.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    data = _load_json(args.in_path)

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
        xk, ck, dk = _extract_te_polyline(data, dtype=dtype, dhat_const=args.dhat_const, scale=args.scale)

    fitted = fit_wing_geom(xk, ck, dk, N=args.N, R=args.R, dtype=dtype)

    # Print summary.
    print(f"[OK] keypoints: {xk.numel()}  -> fitted N={args.N}, R={fitted.R:.6f} m")
    print(f"     chord range: [{float(fitted.c.min()):.6f}, {float(fitted.c.max()):.6f}] m")
    print(f"     dhat  range: [{float(fitted.dhat.min()):.6f}, {float(fitted.dhat.max()):.6f}]")

    # Resolve outputs. If --out-dir is set, default names are derived from the input filename.
    out_json = args.out_json
    out_py = args.out_py
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.in_path.stem
        if out_json is None:
            out_json = args.out_dir / f"{stem}_wing_geom.json"
        elif not out_json.is_absolute():
            out_json = args.out_dir / out_json
        if out_py is None:
            out_py = args.out_dir / f"{stem}_wing_geom_fit.py"
        elif not out_py.is_absolute():
            out_py = args.out_dir / out_py

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

    if not wrote_any:
        print("[INFO] No output file requested. Use --out-dir and/or --out-json/--out-py.")


if __name__ == "__main__":
    main()

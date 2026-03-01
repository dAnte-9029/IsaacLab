"""Utilities to build WingGeometry from a chord-distribution CSV.

The DeLaurier/Wang scripts in this repository commonly use a CSV with columns:
  - x_mid_m
  - c_m
  - dhat

We use the CSV primarily to define the chord distribution c(x). The WingGeometry
class internally discretizes x uniformly on [0, R], so we interpolate c(x) from
the CSV samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import csv
import torch

from .qsm_wang2016 import WingGeometry

Tensor = torch.Tensor


def load_wing_geom_csv(path: Path) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    cs: list[float] = []
    dhats: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"x_mid_m", "c_m", "dhat"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(f"Invalid wing geometry CSV header in {path}. Expected columns: {sorted(expected)}")
        for row in reader:
            xs.append(float(row["x_mid_m"]))
            cs.append(float(row["c_m"]))
            dhats.append(float(row["dhat"]))
    if len(xs) < 2:
        raise ValueError(f"Wing geometry CSV must contain at least 2 rows: {path}")
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    cs = [cs[i] for i in order]
    dhats = [dhats[i] for i in order]
    return xs, cs, dhats


def infer_span_from_x_mid(xs: list[float]) -> float:
    if len(xs) < 2:
        raise ValueError("Need at least 2 x_mid samples to infer R.")
    dxs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    dxs_sorted = sorted(dxs)
    dx = float(dxs_sorted[len(dxs_sorted) // 2])  # median
    if dx <= 0:
        raise ValueError("x_mid must be strictly increasing.")
    return float(xs[-1] + 0.5 * dx)


def _interp1d_torch(xp: Tensor, fp: Tensor, x: Tensor) -> Tensor:
    """1D linear interpolation like numpy.interp, supports CUDA tensors."""
    if xp.ndim != 1 or fp.ndim != 1:
        raise ValueError("xp/fp must be 1D tensors.")
    if xp.numel() != fp.numel():
        raise ValueError("xp and fp must have the same length.")
    if xp.numel() < 2:
        raise ValueError("xp must have at least 2 points.")
    if not torch.all(xp[1:] > xp[:-1]):
        raise ValueError("xp must be strictly increasing.")

    x_clamped = torch.clamp(x, float(xp[0].item()), float(xp[-1].item()))
    idx_hi = torch.searchsorted(xp, x_clamped, right=True)
    idx_lo = torch.clamp(idx_hi - 1, 0, xp.numel() - 2)
    idx_hi = idx_lo + 1

    x0 = xp[idx_lo]
    x1 = xp[idx_hi]
    y0 = fp[idx_lo]
    y1 = fp[idx_hi]
    t = (x_clamped - x0) / torch.clamp(x1 - x0, min=1e-12)
    return y0 + t * (y1 - y0)


@dataclass(frozen=True)
class WingGeomCsvInfo:
    R: float
    S: float
    c_root: float
    c_tip: float
    aspect_ratio_geom: float


def build_wing_geometry_from_csv(
    path: Path,
    *,
    N: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    dhat: float = 0.0,
    aspect_ratio: float | None = None,
) -> tuple[WingGeometry, WingGeomCsvInfo]:
    """Create a WingGeometry using chord distribution from CSV samples."""
    xs, cs, _dhats = load_wing_geom_csv(path)
    R = infer_span_from_x_mid(xs)

    # Estimate planform area from the CSV assuming uniform x_mid spacing.
    dxs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    dxs_sorted = sorted(dxs)
    dx = float(dxs_sorted[len(dxs_sorted) // 2])
    S = float(sum(ci * dx for ci in cs))
    AR_geom = (R * R / S) if S > 0 else float("nan")

    xp = torch.tensor(xs, device=device, dtype=dtype)
    fp = torch.tensor(cs, device=device, dtype=dtype)

    def chord_func(x_mid: Tensor) -> Tensor:
        return _interp1d_torch(xp, fp, x_mid)

    AR_use = float(AR_geom) if aspect_ratio is None else float(aspect_ratio)

    geom = WingGeometry.from_input(
        {
            "R": float(R),
            "N": int(N),
            "chord_func": chord_func,
            "dhat": float(dhat),
            "aspect_ratio": float(AR_use),
        },
        device=device,
        dtype=dtype,
    )
    info = WingGeomCsvInfo(R=float(R), S=float(S), c_root=float(cs[0]), c_tip=float(cs[-1]), aspect_ratio_geom=float(AR_geom))
    return geom, info


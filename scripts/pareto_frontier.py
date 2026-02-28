#!/usr/bin/env python3
"""Generate a Pareto frontier CSV (maximize Fx,Fz, minimize Pin)."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _is_finite(x: float) -> bool:
    return math.isfinite(x) and not math.isnan(x)


def _minmax(vals: list[float]) -> tuple[float, float]:
    finite = [v for v in vals if _is_finite(v)]
    if not finite:
        return (0.0, 1.0)
    vmin = min(finite)
    vmax = max(finite)
    if abs(vmax - vmin) < 1e-9:
        vmin -= 1.0
        vmax += 1.0
    return (vmin, vmax)


def _format_tick(v: float) -> str:
    if not _is_finite(v):
        return ""
    av = abs(v)
    if av >= 1000.0:
        return f"{v:.0f}"
    if av >= 100.0:
        return f"{v:.1f}"
    if av >= 10.0:
        return f"{v:.2f}"
    if av >= 1.0:
        return f"{v:.3f}"
    if av >= 0.1:
        return f"{v:.4f}"
    return f"{v:.3g}"


def _write_svg_scatter(
    path: Path,
    x: list[float],
    y: list[float],
    x_front: list[float],
    y_front: list[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if len(x) < 2 or len(y) < 2:
        return
    xmin, xmax = _minmax(x)
    ymin, ymax = _minmax(y)
    width = 700.0
    height = 380.0
    margin = 50.0
    tick_size = 5.0
    n_ticks = 5

    def to_px(xv: float, yv: float) -> tuple[float, float]:
        px = margin + (xv - xmin) / (xmax - xmin) * (width - 2 * margin)
        py = margin + (ymax - yv) / (ymax - ymin) * (height - 2 * margin)
        return px, py

    pts_front = []
    for xv, yv in zip(x_front, y_front):
        if not (_is_finite(xv) and _is_finite(yv)):
            continue
        px, py = to_px(xv, yv)
        pts_front.append(f"{px:.1f},{py:.1f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}">\n')
        f.write('<rect width="100%" height="100%" fill="#ffffff"/>\n')
        f.write(f'<text x="{margin:.0f}" y="{margin - 12:.0f}" font-size="14" '
                f'font-family="sans-serif">{title}</text>\n')
        # Axes.
        f.write(f'<line x1="{margin:.1f}" y1="{height - margin:.1f}" '
                f'x2="{width - margin:.1f}" y2="{height - margin:.1f}" '
                f'stroke="#333" stroke-width="1"/>\n')
        f.write(f'<line x1="{margin:.1f}" y1="{margin:.1f}" '
                f'x2="{margin:.1f}" y2="{height - margin:.1f}" '
                f'stroke="#333" stroke-width="1"/>\n')
        # Ticks.
        for i in range(n_ticks):
            xt = xmin + (xmax - xmin) * i / (n_ticks - 1)
            px, _ = to_px(xt, ymin)
            f.write(
                f'<line x1="{px:.1f}" y1="{height - margin:.1f}" '
                f'x2="{px:.1f}" y2="{height - margin + tick_size:.1f}" '
                f'stroke="#333" stroke-width="1"/>\n'
            )
            f.write(
                f'<text x="{px:.1f}" y="{height - margin + 18:.1f}" font-size="10" '
                f'font-family="sans-serif" text-anchor="middle">{_format_tick(xt)}</text>\n'
            )
        for i in range(n_ticks):
            yt = ymin + (ymax - ymin) * i / (n_ticks - 1)
            _, py = to_px(xmin, yt)
            f.write(
                f'<line x1="{margin - tick_size:.1f}" y1="{py:.1f}" '
                f'x2="{margin:.1f}" y2="{py:.1f}" '
                f'stroke="#333" stroke-width="1"/>\n'
            )
            f.write(
                f'<text x="{margin - tick_size - 2:.1f}" y="{py + 3:.1f}" font-size="10" '
                f'font-family="sans-serif" text-anchor="end">{_format_tick(yt)}</text>\n'
            )
        # Scatter.
        for xv, yv in zip(x, y):
            if not (_is_finite(xv) and _is_finite(yv)):
                continue
            px, py = to_px(xv, yv)
            f.write(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.2" fill="#7aa6d9" opacity="0.7"/>\n')
        # Frontier.
        if pts_front:
            f.write(f'<polyline points="{" ".join(pts_front)}" fill="none" stroke="#d44" stroke-width="2"/>\n')
        # Labels.
        f.write(f'<text x="{width / 2:.1f}" y="{height - 8:.1f}" font-size="11" '
                f'font-family="sans-serif">{xlabel}</text>\n')
        f.write(f'<text x="{8:.1f}" y="{height / 2:.1f}" font-size="11" '
                f'font-family="sans-serif" transform="rotate(-90 8,{height / 2:.1f})">{ylabel}</text>\n')
        f.write("</svg>\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Compute Pareto frontier (maximize Fx,Fz, minimize Pin).")
    p.add_argument("--in-csv", type=Path, required=True, help="Input CSV (e.g., modulation_scan.csv).")
    p.add_argument("--out-csv", type=Path, default=None, help="Output CSV for Pareto points.")
    p.add_argument("--out-svg", type=Path, default=None, help="Optional SVG (Fx vs Pin) with Pareto line.")
    p.add_argument("--fx-col", type=str, default="mean_thrust_N")
    p.add_argument("--fz-col", type=str, default="mean_lift_N")
    p.add_argument("--pin-col", type=str, default="mean_pin_W")
    p.add_argument("--delta-col", type=str, default="delta")
    p.add_argument("--min-fx", type=float, default=None, help="Filter: Fx >= min.")
    p.add_argument("--min-fz", type=float, default=None, help="Filter: Fz >= min.")
    p.add_argument("--max-pin", type=float, default=None, help="Filter: Pin <= max.")
    args = p.parse_args()

    if args.out_csv is None:
        args.out_csv = args.in_csv.with_name(args.in_csv.stem + "_pareto.csv")

    with args.in_csv.open() as f:
        r = csv.DictReader(f)
        rows = list(r)
        if not rows:
            raise ValueError(f"Empty CSV: {args.in_csv}")

    pts = []
    for row in rows:
        fx = _safe_float(row.get(args.fx_col, ""))
        fz = _safe_float(row.get(args.fz_col, ""))
        pin = _safe_float(row.get(args.pin_col, ""))
        if not (_is_finite(fx) and _is_finite(fz) and _is_finite(pin)):
            continue
        if args.min_fx is not None and fx < float(args.min_fx):
            continue
        if args.min_fz is not None and fz < float(args.min_fz):
            continue
        if args.max_pin is not None and pin > float(args.max_pin):
            continue
        delta = _safe_float(row.get(args.delta_col, "nan"))
        pts.append((delta, fx, fz, pin, row))

    pareto = []
    for i, p in enumerate(pts):
        _, fx_i, fz_i, pin_i, _ = p
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            _, fx_j, fz_j, pin_j, _ = q
            if (
                fx_j >= fx_i
                and fz_j >= fz_i
                and pin_j <= pin_i
                and (fx_j > fx_i or fz_j > fz_i or pin_j < pin_i)
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    pareto.sort(key=lambda x: x[3])  # sort by pin

    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta", "Fx", "Fz", "Pin", "source_csv"])
        for delta, fx, fz, pin, row in pareto:
            w.writerow([delta, fx, fz, pin, row.get("csv", "")])

    print(f"[OK] pareto points: {len(pareto)} -> {args.out_csv}")

    if args.out_svg is not None:
        xs = [p[1] for p in pts]
        ys = [p[3] for p in pts]
        xs_f = [p[1] for p in pareto]
        ys_f = [p[3] for p in pareto]
        _write_svg_scatter(
            args.out_svg,
            xs,
            ys,
            xs_f,
            ys_f,
            title="Pareto frontier (maximize Fx,Fz, minimize Pin)",
            xlabel="Fx (N)",
            ylabel="Pin (W)",
        )
        print(f"[OK] svg: {args.out_svg}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot world-frame wing forces to SVG without matplotlib (left/right curves)."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _read_csv(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for name in r.fieldnames:
            out[name] = []
        for row in r:
            for name in r.fieldnames:
                try:
                    out[name].append(float(row.get(name, "")))
                except Exception:
                    out[name].append(float("nan"))
    return out


def _slice(data: list[float], start: int, end: int | None, every: int) -> list[float]:
    if start < 0:
        start = 0
    if end is None or end > len(data):
        end = len(data)
    if every <= 0:
        every = 1
    return data[start:end:every]


def _minmax(vals: list[float]) -> tuple[float, float]:
    finite = [v for v in vals if v == v]
    if not finite:
        return (0.0, 1.0)
    vmin = min(finite)
    vmax = max(finite)
    if abs(vmax - vmin) < 1e-9:
        vmin -= 1.0
        vmax += 1.0
    return vmin, vmax


def _panel_axes(y0: float, y1: float, x0: float, x1: float, title: str, ylabel: str) -> list[str]:
    out: list[str] = []
    out.append(f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#333" stroke-width="1"/>')
    out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y1:.1f}" stroke="#333" stroke-width="1"/>')
    out.append(f'<text x="{x0:.1f}" y="{y0 - 6:.1f}" font-size="12" font-family="sans-serif">{title}</text>')
    out.append(
        f'<text x="{x0 - 42:.1f}" y="{(y0 + y1) / 2:.1f}" font-size="10" '
        f'font-family="sans-serif" transform="rotate(-90 {x0 - 42:.1f},{(y0 + y1) / 2:.1f})">{ylabel}</text>'
    )
    return out


def _polyline(xs: list[float], ys: list[float], *, color: str) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>'


def main() -> None:
    p = argparse.ArgumentParser(description="Plot world-frame wing forces to SVG.")
    p.add_argument("--csv", type=Path, required=True, help="Input CSV from isaac_wind_tunnel_flappingbot_v50.py.")
    p.add_argument("--out", type=Path, default=None, help="Output SVG path.")
    p.add_argument("--start-step", type=int, default=0, help="Start step index.")
    p.add_argument("--end-step", type=int, default=None, help="Stop before this step index.")
    p.add_argument("--every", type=int, default=1, help="Plot every Nth sample.")
    args = p.parse_args()

    data = _read_csv(args.csv)
    if "t" not in data:
        raise ValueError("CSV must contain 't' column.")

    t = _slice(data["t"], args.start_step, args.end_step, args.every)
    FxL = _slice(data.get("F_world_x_L", []), args.start_step, args.end_step, args.every)
    FyL = _slice(data.get("F_world_y_L", []), args.start_step, args.end_step, args.every)
    FzL = _slice(data.get("F_world_z_L", []), args.start_step, args.end_step, args.every)
    FxR = _slice(data.get("F_world_x_R", []), args.start_step, args.end_step, args.every)
    FyR = _slice(data.get("F_world_y_R", []), args.start_step, args.end_step, args.every)
    FzR = _slice(data.get("F_world_z_R", []), args.start_step, args.end_step, args.every)
    q = data.get("wing_q", data.get("wing_q_R", []))
    q = _slice(q, args.start_step, args.end_step, args.every)
    q_deg = [math.degrees(v) if v == v else float("nan") for v in q]
    eta_L = _slice(data.get("eta_tip_L", []), args.start_step, args.end_step, args.every)
    eta_R = _slice(data.get("eta_tip_R", []), args.start_step, args.end_step, args.every)
    if eta_L and eta_R:
        theta_tip_L = [math.degrees(v) if v == v else float("nan") for v in eta_L]
        theta_tip_R = [math.degrees(v) if v == v else float("nan") for v in eta_R]
    else:
        theta_tip_L = _slice(data.get("del_theta_tip_deg_L", []), args.start_step, args.end_step, args.every)
        theta_tip_R = _slice(data.get("del_theta_tip_deg_R", []), args.start_step, args.end_step, args.every)
        if theta_tip_L and theta_tip_R:
            print("[WARN] eta_tip_* not found; using del_theta_tip_deg_*.")
        else:
            theta_tip_L = [float("nan")] * len(t)
            theta_tip_R = [float("nan")] * len(t)

    if not t:
        raise ValueError("No data in requested range.")

    out_path = args.out
    if out_path is None:
        out_path = Path("outputs_DeLaurier") / f"{args.csv.stem}_world_forces.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width = 900.0
    height = 1060.0
    margin_l = 70.0
    margin_r = 20.0
    margin_t = 40.0
    margin_b = 40.0
    gap = 25.0
    panel_h = (height - margin_t - margin_b - 4 * gap) / 5.0
    x0 = margin_l
    x1 = width - margin_r

    # Normalize x for all panels.
    t_min, t_max = _minmax(t)

    def x_to_px(xv: float) -> float:
        return x0 + (xv - t_min) / (t_max - t_min) * (x1 - x0)

    def panel_y_map(values: list[float], y_top: float, y_bot: float):
        vmin, vmax = _minmax(values)
        def y_to_px(yv: float) -> float:
            return y_top + (vmax - yv) / (vmax - vmin) * (y_bot - y_top)
        return y_to_px

    # Panel data order: Fy, Fx, Fz, q, theta_tip
    panels = [
        ("Fy (world)", FyL, FyR, "N"),
        ("Fx (world)", FxL, FxR, "N"),
        ("Fz (world)", FzL, FzR, "N"),
        ("Flap angle q", q_deg, None, "deg"),
        ("Tip twist eta", theta_tip_L, theta_tip_R, "deg"),
    ]

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}">')
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    lines.append(f'<text x="{margin_l:.1f}" y="{margin_t - 16:.1f}" font-size="14" font-family="sans-serif">'
                 f'World-frame wing forces: {args.csv}</text>')

    for idx, (title, yL, yR, unit) in enumerate(panels):
        y_top = margin_t + idx * (panel_h + gap)
        y_bot = y_top + panel_h
        y_all = yL if yR is None else (yL + yR)
        y_map = panel_y_map(y_all, y_top, y_bot)
        y_min, y_max = _minmax(y_all)

        # Grid (vertical only, 5 divisions)
        for k in range(6):
            xv = t_min + (t_max - t_min) * k / 5.0
            x_px = x_to_px(xv)
            lines.append(f'<line x1="{x_px:.1f}" y1="{y_top:.1f}" x2="{x_px:.1f}" y2="{y_bot:.1f}" '
                         f'stroke="#e0e0e0" stroke-width="1"/>')
            if idx == len(panels) - 1:
                lines.append(f'<text x="{x_px - 8:.1f}" y="{y_bot + 14:.1f}" font-size="10" '
                             f'font-family="sans-serif">{xv:.2f}</text>')

        # Horizontal grid + y tick labels (5 divisions).
        for k in range(6):
            yv = y_min + (y_max - y_min) * k / 5.0
            y_px = y_map(yv)
            lines.append(f'<line x1="{x0:.1f}" y1="{y_px:.1f}" x2="{x1:.1f}" y2="{y_px:.1f}" '
                         f'stroke="#e0e0e0" stroke-width="1"/>')
            lines.append(
                f'<text x="{x0 - 48:.1f}" y="{y_px + 4:.1f}" font-size="10" font-family="sans-serif">{yv:.1f}</text>'
            )

        lines.extend(_panel_axes(y_top, y_bot, x0, x1, title, unit))

        xs = [x_to_px(v) for v in t]
        ys_L = [y_map(v) for v in yL]
        ys_R = [y_map(v) for v in yR] if yR is not None else None
        lines.append(_polyline(xs, ys_L, color="#2a5caa"))
        if ys_R is not None:
            lines.append(_polyline(xs, ys_R, color="#d67f29"))

        # Legend only on top panel.
        if idx == 0:
            lx = x1 - 140
            ly = y_top + 8
            lines.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="120" height="32" fill="#ffffff" stroke="#ccc"/>')
            lines.append(f'<line x1="{lx+8:.1f}" y1="{ly+12:.1f}" x2="{lx+28:.1f}" y2="{ly+12:.1f}" '
                         f'stroke="#2a5caa" stroke-width="2"/>')
            lines.append(f'<text x="{lx+34:.1f}" y="{ly+16:.1f}" font-size="10" font-family="sans-serif">Left</text>')
            lines.append(f'<line x1="{lx+8:.1f}" y1="{ly+24:.1f}" x2="{lx+28:.1f}" y2="{ly+24:.1f}" '
                         f'stroke="#d67f29" stroke-width="2"/>')
            lines.append(f'<text x="{lx+34:.1f}" y="{ly+28:.1f}" font-size="10" font-family="sans-serif">Right</text>')

    lines.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{height - 8:.1f}" font-size="11" '
                 f'font-family="sans-serif">t (s)</text>')
    lines.append("</svg>")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

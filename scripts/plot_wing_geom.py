#!/usr/bin/env python3
"""Write a simple SVG planform view of the wing geometry CSV and annotate key points."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_geom_csv(path: Path) -> tuple[list[float], list[float], list[float]]:
    x_mid: list[float] = []
    chord: list[float] = []
    dhat: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_mid.append(float(row["x_mid_m"]))
            chord.append(float(row["c_m"]))
            dhat.append(float(row["dhat"]))
    return x_mid, chord, dhat


def _avg_diff(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    return sum(diffs) / float(len(diffs))


def main() -> int:
    parser = argparse.ArgumentParser(description="Write wing planform SVG from x_mid/c/dhat CSV.")
    parser.add_argument(
        "--wing-geom-csv",
        type=Path,
        default=Path("outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv"),
        help="Wing geometry CSV containing x_mid_m, c_m, dhat.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs_DeLaurier/wing_geom_plot.svg"),
        help="Output SVG path.",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.05,
        help="Optional gap length (m) from hinge to wing root for annotation.",
    )
    parser.add_argument("--title", type=str, default="Wing planform (local span-chord)")
    args = parser.parse_args()

    x_mid, chord, dhat = _read_geom_csv(args.wing_geom_csv)
    if len(x_mid) < 2:
        raise ValueError("Need at least two spanwise stations to plot.")
    dx = _avg_diff(x_mid)
    x_root = x_mid[0] - 0.5 * dx
    x_tip = x_mid[-1] + 0.5 * dx
    c_root = chord[0]
    c_tip = chord[-1]
    d_root = dhat[0] * c_root
    d_tip = dhat[-1] * c_tip

    max_chord = max(chord)
    scale = 1000.0  # px per meter
    margin = 40.0
    width = (x_tip - x_root) * scale + 2 * margin
    height = max_chord * scale + 2 * margin

    def to_px(x: float, y: float) -> tuple[float, float]:
        px = margin + (x - x_root) * scale
        py = margin + (max_chord - y) * scale
        return px, py

    # Build outline polygon points.
    outline = []
    for x, c in zip(x_mid, chord):
        outline.append(to_px(x, 0.0))
    for x, c in zip(reversed(x_mid), reversed(chord)):
        outline.append(to_px(x, c))

    key_points = {
        "root_LE": (x_root, 0.0),
        "root_axis": (x_root, d_root),
        "root_TE": (x_root, c_root),
        "tip_LE": (x_tip, 0.0),
        "tip_axis": (x_tip, d_tip),
        "tip_TE": (x_tip, c_tip),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}">\n')
        f.write('<rect width="100%" height="100%" fill="#ffffff"/>\n')
        f.write(f'<text x="{margin:.0f}" y="{margin - 12:.0f}" font-size="14" '
                f'font-family="sans-serif">{args.title}</text>\n')

        # Planform polygon.
        pts = " ".join([f"{px:.1f},{py:.1f}" for px, py in outline])
        f.write(f'<polygon points="{pts}" fill="#c8d5eb" stroke="#3b4a6b" stroke-width="1"/>\n')

        # Leading edge and trailing edge.
        le_pts = " ".join([f"{to_px(x, 0.0)[0]:.1f},{to_px(x, 0.0)[1]:.1f}" for x in x_mid])
        te_pts = " ".join([f"{to_px(x, c)[0]:.1f},{to_px(x, c)[1]:.1f}" for x, c in zip(x_mid, chord)])
        f.write(f'<polyline points="{le_pts}" fill="none" stroke="#3b4a6b" stroke-width="1.2"/>\n')
        f.write(f'<polyline points="{te_pts}" fill="none" stroke="#3b4a6b" stroke-width="1.2"/>\n')

        # Pitch axis.
        pa_pts = " ".join(
            [f"{to_px(x, d * c)[0]:.1f},{to_px(x, d * c)[1]:.1f}" for x, c, d in zip(x_mid, chord, dhat)]
        )
        f.write(f'<polyline points="{pa_pts}" fill="none" stroke="#d1495b" stroke-width="1.5"/>\n')

        # Gap marker.
        if args.gap > 0.0:
            gx, _ = to_px(args.gap, 0.0)
            f.write(f'<line x1="{gx:.1f}" y1="{margin:.1f}" x2="{gx:.1f}" y2="{height - margin:.1f}" '
                    f'stroke="#7d6b91" stroke-width="1" stroke-dasharray="4,4"/>\n')
            f.write(f'<text x="{gx + 4:.1f}" y="{margin + 12:.1f}" font-size="10" '
                    f'font-family="sans-serif">gap end</text>\n')

        # Key points.
        for name, (x, y) in key_points.items():
            px, py = to_px(x, y)
            f.write(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="#1f2d3a"/>\n')
            label = f"{name} ({x:.3f}, {y:.3f})"
            f.write(f'<text x="{px + 6:.1f}" y="{py - 6:.1f}" font-size="10" '
                    f'font-family="sans-serif">{label}</text>\n')

        # Axes labels.
        f.write(f'<text x="{width / 2:.1f}" y="{height - 8:.1f}" font-size="11" '
                f'font-family="sans-serif">spanwise x (m)</text>\n')
        f.write(f'<text x="{8:.1f}" y="{height / 2:.1f}" font-size="11" '
                f'font-family="sans-serif" transform="rotate(-90 8,{height / 2:.1f})">chordwise y (m)</text>\n')
        f.write("</svg>\n")

    print("Key points (span x, chord y) in meters:")
    for name, (x, y) in key_points.items():
        print(f"  {name:9s} : ({x:.4f}, {y:.4f})")
    print(f"Saved plot: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

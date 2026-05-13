#!/usr/bin/env python3
"""Plot one 2D polygon from ordered points written directly in this file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


Point2D = tuple[float, float]


# Edit these points directly. The script draws them in order and closes the polygon automatically.
POINTS: list[Point2D] = [
    (0.0, 140.0),
    (85, 140),
    (300, 0),
    (255, -80),
    (45, -80),
    (0,0)
]

TITLE = "2D Polygon"


def validate_points(points: Iterable[Point2D]) -> list[Point2D]:
    """Validate and normalize a polygon point sequence."""

    pts = [(float(x), float(y)) for x, y in points]
    base = pts[:-1] if len(pts) >= 2 and pts[0] == pts[-1] else pts
    if len(base) < 3:
        raise ValueError("A polygon needs at least three vertices.")
    return pts


def closed_polygon(points: Iterable[Point2D]) -> list[Point2D]:
    """Return polygon vertices with exactly one closing point."""

    pts = validate_points(points)
    if pts[0] == pts[-1]:
        return pts
    return [*pts, pts[0]]


def plot_polygon(
    points: Iterable[Point2D],
    *,
    title: str,
    annotate_points: bool = True,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Draw one closed polygon."""

    import matplotlib.pyplot as plt

    pts = validate_points(points)
    outline = closed_polygon(pts)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    xs = [x for x, _ in outline]
    ys = [y for _, y in outline]

    ax.plot(xs, ys, "-o", linewidth=1.5, markersize=5.0, color="tab:blue")
    ax.fill(xs, ys, alpha=0.15, color="tab:blue")

    if annotate_points:
        for idx, (x, y) in enumerate(pts):
            ax.annotate(str(idx), (x, y), xytext=(6, 6), textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot one 2D polygon from ordered points defined in this script.")
    parser.add_argument("--save", type=Path, default=None, help="Optional output image path, such as polygon.png")
    parser.add_argument("--title", type=str, default=TITLE, help="Figure title")
    parser.add_argument("--hide-labels", action="store_true", help="Do not show point indices")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    args = parser.parse_args()

    plot_polygon(
        POINTS,
        title=args.title,
        annotate_points=not args.hide_labels,
        save_path=args.save,
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

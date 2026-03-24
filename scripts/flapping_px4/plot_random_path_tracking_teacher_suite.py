"""Plot suite-level grids for random path-tracking teacher benchmarks."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SEGMENT_CODES = {"straight": "S", "turn": "T", "loiter": "L"}


@dataclass(frozen=True)
class SuiteEpisode:
    """Artifact bundle for one benchmark episode."""

    mission_seed: int
    run_dir: Path
    traj_path: Path
    ref_path: Path | None
    summary_path: Path | None
    mission_segments: tuple[str, ...]
    segment_label: str
    completed_path: bool
    has_progress_jump: bool
    final_progress_ratio: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 5x10 grid summaries for a random path-tracking teacher suite.")
    parser.add_argument("--suite_dir", type=Path, required=True, help="Suite output directory containing episodes.csv.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory (default: <suite_dir>/plots).")
    parser.add_argument("--cols", type=int, default=10, help="Number of subplot columns.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _load_rows(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def _resolve_artifact_path(path_str: str | None, suite_dir: Path) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidates = [
        (REPO_ROOT / path).resolve(),
        (suite_dir / path).resolve(),
        (suite_dir.parent / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return (REPO_ROOT / path).resolve()


def build_segment_label(segments: list[str] | tuple[str, ...]) -> str:
    """Build a compact mission-segment label such as ``S-L-T``."""
    if not segments:
        return "?"
    return "-".join(_SEGMENT_CODES.get(str(segment), str(segment)[:1].upper()) for segment in segments)


def infer_grid_shape(*, num_items: int, cols: int) -> tuple[int, int]:
    """Infer subplot grid shape from item count and requested column count."""
    safe_cols = max(1, int(cols))
    safe_items = max(0, int(num_items))
    rows = max(1, int(math.ceil(float(safe_items) / float(safe_cols))))
    return rows, safe_cols


def discover_suite_episodes(suite_dir: Path) -> list[SuiteEpisode]:
    """Discover suite episodes from ``episodes.csv`` and per-episode summaries."""
    suite_dir = Path(suite_dir).resolve()
    episodes_csv = suite_dir / "episodes.csv"
    if not episodes_csv.exists():
        raise FileNotFoundError(f"Missing suite manifest: {episodes_csv}")

    episodes: list[SuiteEpisode] = []
    with episodes_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_dir = _resolve_artifact_path(row.get("run_dir"), suite_dir)
            summary_path = _resolve_artifact_path(row.get("summary_path"), suite_dir)
            traj_path = _resolve_artifact_path(row.get("traj_path"), suite_dir)
            if run_dir is None or traj_path is None:
                continue
            ref_path = run_dir / "reference_path.csv"
            summary = _load_json(summary_path or (run_dir / "summary.json"))
            mission_segments = tuple(str(segment) for segment in summary.get("mission_segments", ()))
            mission_seed = int(summary.get("mission_seed", row.get("mission_seed", 0)))
            completed_path = _parse_bool(row.get("completed_path", summary.get("completed_path", False)))
            has_progress_jump = _parse_bool(row.get("has_progress_jump", summary.get("has_progress_jump", False)))
            final_progress_ratio = float(summary.get("final_progress_ratio", row.get("final_progress_ratio", 0.0) or 0.0))
            episodes.append(
                SuiteEpisode(
                    mission_seed=mission_seed,
                    run_dir=run_dir,
                    traj_path=traj_path,
                    ref_path=ref_path if ref_path.exists() else None,
                    summary_path=summary_path if summary_path is not None and summary_path.exists() else None,
                    mission_segments=mission_segments,
                    segment_label=build_segment_label(mission_segments),
                    completed_path=completed_path,
                    has_progress_jump=has_progress_jump,
                    final_progress_ratio=final_progress_ratio,
                )
            )
    return sorted(episodes, key=lambda episode: episode.mission_seed)


def _episode_title(episode: SuiteEpisode) -> str:
    status = "✓" if episode.completed_path and not episode.has_progress_jump else "✗"
    return f"{episode.mission_seed:02d} {episode.segment_label} {status}"


def _suite_title(suite_dir: Path, summary: dict[str, Any]) -> str:
    jump_free = int(summary.get("jump_free_completed_episodes", 0))
    total = int(summary.get("episodes", 0))
    return f"{suite_dir.name} | jump-free {jump_free}/{total}"


def _plot_xy_grid(
    *,
    episodes: list[SuiteEpisode],
    suite_dir: Path,
    summary: dict[str, Any],
    out_path: Path,
    cols: int,
    dpi: int,
) -> None:
    rows, cols = infer_grid_shape(num_items=len(episodes), cols=cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.1))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for index, axis in enumerate(axes_flat):
        if index >= len(episodes):
            axis.axis("off")
            continue
        episode = episodes[index]
        traj_rows = _load_rows(episode.traj_path)
        ref_rows = _load_rows(episode.ref_path)
        x = [row["x"] for row in traj_rows]
        y = [row["y"] for row in traj_rows]
        ref_x = [row["x_ref"] for row in ref_rows]
        ref_y = [row["y_ref"] for row in ref_rows]
        color = "tab:blue" if episode.completed_path and not episode.has_progress_jump else "tab:red"
        if ref_rows:
            axis.plot(ref_x, ref_y, linestyle="--", linewidth=0.9, color="0.7")
        if traj_rows:
            axis.plot(x, y, linewidth=1.0, color=color)
            axis.scatter([x[0]], [y[0]], s=8, color="tab:green", zorder=3)
            axis.scatter([x[-1]], [y[-1]], s=8, color=color, zorder=3)
        else:
            axis.text(0.5, 0.5, "missing", ha="center", va="center", transform=axis.transAxes, fontsize=7)
        axis.set_title(_episode_title(episode), fontsize=7, pad=2)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.2, linewidth=0.4)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(f"{_suite_title(suite_dir, summary)} | XY actual=blue ref=gray dashed", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _compute_altitude_limits(episodes: list[SuiteEpisode]) -> tuple[float, float]:
    z_min = float("inf")
    z_max = float("-inf")
    for episode in episodes:
        for row in _load_rows(episode.traj_path):
            z_min = min(z_min, row["z"], row.get("reference_z", row["z"]))
            z_max = max(z_max, row["z"], row.get("reference_z", row["z"]))
    if not math.isfinite(z_min) or not math.isfinite(z_max):
        return 0.0, 1.0
    padding = max(0.2, 0.05 * (z_max - z_min + 1.0e-6))
    return z_min - padding, z_max + padding


def _plot_altitude_grid(
    *,
    episodes: list[SuiteEpisode],
    suite_dir: Path,
    summary: dict[str, Any],
    out_path: Path,
    cols: int,
    dpi: int,
) -> None:
    rows, cols = infer_grid_shape(num_items=len(episodes), cols=cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 1.8), sharey=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    z_limits = _compute_altitude_limits(episodes)
    max_time_s = 0.0
    traj_cache: dict[int, list[dict[str, float]]] = {}
    for episode in episodes:
        traj_rows = _load_rows(episode.traj_path)
        traj_cache[episode.mission_seed] = traj_rows
        if traj_rows:
            max_time_s = max(max_time_s, traj_rows[-1]["t"])

    for index, axis in enumerate(axes_flat):
        if index >= len(episodes):
            axis.axis("off")
            continue
        episode = episodes[index]
        traj_rows = traj_cache[episode.mission_seed]
        if traj_rows:
            t = [row["t"] for row in traj_rows]
            z = [row["z"] for row in traj_rows]
            ref_z = [row.get("reference_z", row["z"]) for row in traj_rows]
            color = "tab:blue" if episode.completed_path and not episode.has_progress_jump else "tab:red"
            axis.plot(t, ref_z, linestyle="--", linewidth=0.8, color="0.7")
            axis.plot(t, z, linewidth=1.0, color=color)
        else:
            axis.text(0.5, 0.5, "missing", ha="center", va="center", transform=axis.transAxes, fontsize=7)
        axis.set_title(_episode_title(episode), fontsize=7, pad=2)
        axis.set_xlim(0.0, max_time_s if max_time_s > 0.0 else 1.0)
        axis.set_ylim(*z_limits)
        axis.grid(True, alpha=0.2, linewidth=0.4)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(f"{_suite_title(suite_dir, summary)} | altitude actual=blue ref=gray dashed", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir is not None else (suite_dir / "plots")
    episodes = discover_suite_episodes(suite_dir)
    if not episodes:
        raise SystemExit(f"No episodes discovered under {suite_dir}")
    summary = _load_json(suite_dir / "summary.json")

    xy_path = out_dir / "suite_xy_grid.png"
    altitude_path = out_dir / "suite_altitude_grid.png"
    _plot_xy_grid(episodes=episodes, suite_dir=suite_dir, summary=summary, out_path=xy_path, cols=int(args.cols), dpi=int(args.dpi))
    _plot_altitude_grid(
        episodes=episodes,
        suite_dir=suite_dir,
        summary=summary,
        out_path=altitude_path,
        cols=int(args.cols),
        dpi=int(args.dpi),
    )

    print(
        json.dumps(
            {
                "suite_dir": str(suite_dir),
                "episodes": len(episodes),
                "xy_grid_png": str(xy_path),
                "altitude_grid_png": str(altitude_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

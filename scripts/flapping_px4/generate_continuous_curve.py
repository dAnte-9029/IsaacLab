"""Generate or inspect a replayable, kinematically screened continuous path file."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import subprocess
import sys

import torch

# Resolve the project package from this checkout for standalone wrapper calls.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/flapping_bot"))

from flapping_bot.path_tracking.continuous_curve import (
    CONTINUOUS_CURVE_CONTRACT,
    build_continuous_curves,
    continuous_curve_diagnostic_grid,
    load_continuous_curves,
    sample_accepted_continuous_curve_specs,
    save_continuous_curves,
)


def main() -> None:
    """Write a new JSON recipe file, optionally exporting dense tables and a plot."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Validate and replay an existing v1 JSON file.")
    source.add_argument("--diagnostic-grid", action="store_true", help="Generate the frozen 96-case diagnostic grid.")
    parser.add_argument("--output", type=Path, help="New recipe JSON; never overwrites an existing file.")
    parser.add_argument("--count", type=int, default=6, help="Number of screened random paths (default: 6).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--table-output", type=Path, help="Optional compressed NumPy archive of all dense path fields.")
    parser.add_argument("--plot", type=Path, help="Optional 3-D preview figure (first six paths).")
    args = parser.parse_args()
    if args.input is None and args.output is None:
        parser.error("Generation requires --output.")
    for destination in (args.output, args.table_output, args.plot):
        if destination is not None and destination.exists():
            parser.error(f"Refusing to overwrite {destination}")
    if args.input is not None:
        specs = load_continuous_curves(args.input)
    elif args.diagnostic_grid:
        specs = continuous_curve_diagnostic_grid()
    else:
        specs = sample_accepted_continuous_curve_specs(
            args.count, generator=torch.Generator().manual_seed(args.seed), dtype=torch.float64)
    batch = build_continuous_curves(specs, dtype=torch.float64)
    if args.output is not None:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        save_continuous_curves(args.output, specs, provenance={
            "git_commit": commit, "git_dirty": dirty, "command": sys.argv,
            "candidate_seed": args.seed if args.input is None and not args.diagnostic_grid else None,
        })
    if args.table_output is not None:
        import numpy as np

        with args.table_output.open("xb") as stream:
            np.savez_compressed(stream, **{field.name: getattr(batch, field.name).numpy() for field in fields(batch)})
    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(10, 7))
        axes = figure.add_subplot(111, projection="3d")
        indices = torch.linspace(0, len(specs) - 1, min(6, len(specs))).long().tolist()
        for index in indices:
            spec = specs[index]
            points = batch.points_world_m[index].numpy()
            completion_index = round(
                CONTINUOUS_CURVE_CONTRACT.completion_progress_m / CONTINUOUS_CURVE_CONTRACT.sample_spacing_m
            )
            prefix, continuation = points[:completion_index + 1], points[completion_index:]
            line, = axes.plot(prefix[:, 0], prefix[:, 1], prefix[:, 2], label=spec.curve_id)
            axes.plot(continuation[:, 0], continuation[:, 1], continuation[:, 2], "--", color=line.get_color())
        axes.set(xlabel="World x [m]", ylabel="World y [m]", zlabel="Height [m]",
                 title="Continuous reference paths (not flight results)\nSolid: 0-140 m; dashed: continuation; independent axis scales")
        axes.legend(fontsize=6)
        figure.savefig(args.plot, dpi=150, bbox_inches="tight")
        plt.close(figure)
    print(f"{len(specs)} paths passed sampled kinematic screening; dynamic feasibility is unverified.")
    if args.output is not None:
        print(f"Replay file: {args.output}")


if __name__ == "__main__":
    main()

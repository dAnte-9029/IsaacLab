"""Run standalone ULLT paper-motion and convergence checks without Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New artifact directory; refuses overwrite")
    parser.add_argument("--sections", type=int, default=9)
    parser.add_argument("--steps", type=int, default=384, help="Steps over each complete three-cycle trace")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("one_state", "four_state", "six_state", "quasi_steady"),
        default=["one_state", "four_state", "six_state", "quasi_steady"],
    )
    parser.add_argument("--smoke", action="store_true", help="72 steps, 5 sections; still runs convergence gates")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--wing-csv", type=Path, help="Optional root-local wing CSV for prescribed-flapping checks")
    parser.add_argument("--wing-urdf", type=Path, help="URDF paired with --wing-csv")
    args = parser.parse_args()
    if bool(args.wing_csv) != bool(args.wing_urdf):
        parser.error("--wing-csv and --wing-urdf must be supplied together")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "source/flapping_bot"))
    import numpy as np
    import torch

    from flapping_bot.analysis.ullt_validation import plot_stage1, run_csv_flapping_checks, run_stage1_validation

    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def git(*parts):
        return subprocess.run(["git", *parts], cwd=root, text=True, capture_output=True, check=True).stdout.strip()

    files = [
        Path(__file__).resolve(),
        root / "source/flapping_bot/flapping_bot/config/ullt.py",
        root / "source/flapping_bot/flapping_bot/physics/ullt_state_space.py",
        root / "source/flapping_bot/flapping_bot/physics/ullt_geometry.py",
        root / "source/flapping_bot/flapping_bot/analysis/ullt_validation.py",
        root / "tests/test_ullt_state_space.py",
    ]
    paper_files = list((root / "docs/papers").glob("Izraelevitz*2017*.pdf"))
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status": git("status", "--porcelain"),
        "command": sys.argv,
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": None,
        "randomness": "none",
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files + paper_files
        },
    }
    if args.wing_csv:
        provenance["geometry_sha256"] = {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.wing_csv, args.wing_urdf)
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    summary, traces = run_stage1_validation(
        sections=5 if args.smoke else args.sections,
        steps=72 if args.smoke else args.steps,
        models=tuple(args.models),
        progress=lambda s: print(s, flush=True),
    )
    if args.wing_csv:
        print("CSV/URDF prescribed flapping (no twist, no free body)", flush=True)
        wing, wing_traces = run_csv_flapping_checks(args.wing_csv, args.wing_urdf, steps=48 if args.smoke else 128)
        summary["csv_urdf_numerical_checks"] = wing
        summary["all_numerical_gates_passed"] &= wing["all_finite"]
        traces.update(wing_traces)
    summary["provenance"] = provenance
    np.savez_compressed(args.output_dir / "traces.npz", **traces)
    if not args.no_plot:
        plot_stage1(traces, args.output_dir / "loads.png")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({
            "all_numerical_gates_passed": summary["all_numerical_gates_passed"],
            "output_dir": str(args.output_dir.resolve()),
        }),
        flush=True,
    )
    return 0 if summary["all_numerical_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Try six prescribed-twist candidates on three input-selected train flights."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identification-root", type=Path, required=True)
    parser.add_argument("--wing-csv", type=Path, required=True)
    parser.add_argument("--wing-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sections-per-wing", type=int, default=10)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "source/flapping_bot"))
    import torch
    from flapping_bot.analysis.ullt_flight_replay import FlightReplayConfig, load_registered_comparison, sha256
    from flapping_bot.analysis.ullt_twist_screen import run_twist_screen

    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame, provenance = load_registered_comparison(args.identification_root)
    files = [
        Path(__file__),
        args.wing_csv,
        args.wing_urdf,
        *(root / "source/flapping_bot/flapping_bot/physics").glob("ullt*.py"),
        *(root / "source/flapping_bot/flapping_bot/analysis").glob("ullt*.py"),
        root / "source/flapping_bot/flapping_bot/config/ullt.py",
    ]
    provenance.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
            "source_geometry_sha256": {str(p.resolve()): sha256(p) for p in files},
            "python": sys.executable,
            "torch": torch.__version__,
        }
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    summary = run_twist_screen(
        frame,
        args.wing_csv,
        args.wing_urdf,
        args.output_dir,
        FlightReplayConfig(sections_per_wing=args.sections_per_wing),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

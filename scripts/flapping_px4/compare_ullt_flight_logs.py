"""Compare frozen log whole-aircraft wrench, DeLaurier wings and ULLT wings."""

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
    parser.add_argument(
        "--full-runs",
        action="store_true",
        help="Replay all continuous runs; default: two input-selected windows per flight",
    )
    parser.add_argument("--smoke", action="store_true", help="First 4 seconds per run; no complete-data claim")
    parser.add_argument("--section-model", choices=("linear", "nonlinear_flat_plate"), default="linear")
    parser.add_argument("--polar-prior", choices=("early", "late"), default="early")
    parser.add_argument(
        "--fixed-twist", action="store_true", help="Use prescribed (35 deg,-0.5,0 deg), without fitting"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "source/flapping_bot"))
    import torch
    from flapping_bot.analysis.ullt_flight_replay import (
        FlightReplayConfig,
        load_registered_comparison,
        replay_registered_flights,
        sha256,
    )
    from flapping_bot.analysis.ullt_flight_report import (
        compute_flight_tables,
        summarize_tables,
        write_flight_report,
    )

    from flapping_bot.config.ullt import ULLTConfig
    from flapping_bot.config.nonlinear_section import flat_plate_prior
    from flapping_bot.physics.ullt_twist import PrescribedTwistConfig

    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame, provenance = load_registered_comparison(args.identification_root)
    print(f"Verified registered data: {len(frame)} rows, {frame.log_id.nunique()} flights; no test loaded", flush=True)
    provenance.update({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        "python": sys.executable,
        "torch": torch.__version__,
        "smoke": args.smoke,
    })
    sources = [
        Path(__file__).resolve(),
        *(root / "source/flapping_bot/flapping_bot/analysis").glob("ullt_flight_*.py"),
        root / "source/flapping_bot/flapping_bot/config/ullt.py",
        root / "source/flapping_bot/flapping_bot/config/nonlinear_section.py",
        root / "source/flapping_bot/flapping_bot/physics/flat_plate_section.py",
        root / "source/flapping_bot/flapping_bot/physics/ullt_nonlinear.py",
        root / "source/flapping_bot/flapping_bot/physics/ullt_state_space.py",
        root / "source/flapping_bot/flapping_bot/physics/ullt_geometry.py",
        args.wing_csv,
        args.wing_urdf,
    ]
    provenance["source_geometry_sha256"] = {str(p.resolve()): sha256(p) for p in sources}
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    config = FlightReplayConfig(
        sections_per_wing=args.sections_per_wing,
        backend_config=ULLTConfig(
            model="four_state",
            air_density_kg_m3=1.0,
            section_model=args.section_model,
            nonlinear_polar=flat_plate_prior(args.polar_prior),
            collect_solver_statistics=True,
        ),
        twist=PrescribedTwistConfig(35, -0.5, 0) if args.fixed_twist else PrescribedTwistConfig(),
        maximum_rows_per_run=401 if args.smoke else None,
        windowed_screen=not args.full_runs,
    )
    result, diagnostic = replay_registered_flights(
        frame, args.wing_csv, args.wing_urdf, config, lambda s: print(s, flush=True)
    )
    # Save expensive replay before statistics/plotting, including every excluded row.
    result.to_parquet(args.output_dir / "replay.parquet", index=False)
    (args.output_dir / "replay_diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    print("Replay saved; calculating cycle/phase/transition/cross-flight metrics", flush=True)
    tables = compute_flight_tables(result, config)
    summary = summarize_tables(tables, result)
    summary["replay"] = diagnostic
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_flight_report(result, tables, summary, args.output_dir)
    print(
        json.dumps({
            "report": str(args.output_dir / "report.html"),
            "scored_flights": summary["scored_flights"],
            "scored_rows": summary["scored_rows"],
            "rejected_runs": len(diagnostic["failures"]),
        }),
        flush=True,
    )
    return int(args.section_model != "linear" and bool(diagnostic["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())

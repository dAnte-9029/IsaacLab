"""Recompute diagnostics from an immutable replay, without rerunning physics."""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "source/flapping_bot"))
    import pandas as pd
    from flapping_bot.analysis.ullt_flight_replay import FlightReplayConfig, sha256
    from flapping_bot.analysis.ullt_flight_report import (
        compute_flight_tables,
        summarize_tables,
        write_flight_report,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    replay = args.replay_dir / "replay.parquet"
    diagnostic = json.loads((args.replay_dir / "replay_diagnostics.json").read_text())
    config = FlightReplayConfig(**diagnostic["configuration"])
    frame = pd.read_parquet(replay, use_threads=False)
    provenance = {
        "source_replay_dir": str(args.replay_dir.resolve()),
        "source_replay_sha256": sha256(replay),
        "source_manifest_sha256": sha256(args.replay_dir / "manifest.json"),
        "report_source_sha256": {
            str(p.resolve()): sha256(p)
            for p in (Path(__file__), root / "source/flapping_bot/flapping_bot/analysis/ullt_flight_report.py")
        },
        "change": (
            "Require complete original log/segment cycle, including rows outside selected windows; expose transition"
            " macro metrics"
        ),
        "replay_recomputed": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    tables = compute_flight_tables(frame, config)
    summary = summarize_tables(tables, frame)
    summary["replay"] = diagnostic
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_flight_report(frame, tables, summary, args.output_dir)
    print(args.output_dir / "report.html", flush=True)


if __name__ == "__main__":
    main()

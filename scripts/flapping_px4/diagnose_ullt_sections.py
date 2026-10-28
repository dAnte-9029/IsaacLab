"""Diagnose section incidence, area and wrench components on the prior three-log screen."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-screen", type=Path, required=True)
    parser.add_argument("--wing-csv", type=Path, required=True)
    parser.add_argument("--wing-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "source/flapping_bot"))
    import torch
    from flapping_bot.analysis.ullt_flight_replay import sha256
    from flapping_bot.analysis.ullt_section_attribution import (
        collect_section_case,
        summarize_sections,
        write_attribution_tables,
    )
    from flapping_bot.analysis.ullt_section_report import write_section_report

    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = [
        Path(__file__),
        args.wing_csv,
        args.wing_urdf,
        *(root / "source/flapping_bot/flapping_bot/physics").glob("ullt*.py"),
        *(root / "source/flapping_bot/flapping_bot/analysis").glob("ullt*.py"),
        root / "source/flapping_bot/flapping_bot/config/ullt.py",
    ]
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "source_sha256": {str(p.resolve()): sha256(p) for p in files},
        "source_screen": str(args.source_screen.resolve()),
        "source_screen_manifest_sha256": sha256(args.source_screen / "manifest.json"),
        "python": sys.executable,
        "torch": torch.__version__,
        "scope": "Diagnostic decomposition of wing predictions vs whole-aircraft reconstructed loads; no causal/local truth claims",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cases, verifications = [], {}
    for name in ("zero", "A35_k-0.5_p0"):
        sections, replay, verification = collect_section_case(
            args.source_screen, name, args.wing_csv, args.wing_urdf, args.output_dir
        )
        verifications[name] = verification
        cases.append(summarize_sections(sections, replay, name))
    tables = write_attribution_tables(cases, args.output_dir)
    summary = {
        "verification": verifications,
        "area_fractions": json.loads(tables["area_fractions_macro"].to_json(orient="records")),
        "threshold_metrics": json.loads(tables["threshold_metrics_macro"].to_json(orient="records")),
        "incidence_summary": json.loads(tables["incidence_summary_macro"].to_json(orient="records")),
        "artifacts_sha256": {p.name: sha256(p) for p in args.output_dir.glob("*_sections.parquet")},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_section_report(tables, args.output_dir)
    print(
        json.dumps({"report": str(args.output_dir / "report.html"), "verification": verifications}, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

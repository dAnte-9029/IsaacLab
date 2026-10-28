"""Fixed nonlinear section priors on exactly three archived training flights."""

from pathlib import Path
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--wing-csv", type=Path)
    parser.add_argument("--wing-urdf", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Regenerate reports and verify saved full replays")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/flapping_bot"))
    from flapping_bot.analysis.ullt_nonlinear_experiment import run_experiment, write_experiment_report

    if args.report_only:
        from flapping_bot.analysis.ullt_nonlinear_verification import verify_nonlinear_artifacts

        write_experiment_report(args.output_dir)
        if not any(args.output_dir.glob("*_failure.json")):
            verify_nonlinear_artifacts(args.output_dir, args.source)
    else:
        if args.wing_csv is None or args.wing_urdf is None:
            parser.error("Replay requires --wing-csv and --wing-urdf")
        run_experiment(args.source, args.wing_csv, args.wing_urdf, args.output_dir, args.smoke)
    print(f"Saved diagnostics: {args.output_dir}", flush=True)
    return int(any(args.output_dir.glob("*_failure.json")))


if __name__ == "__main__":
    raise SystemExit(main())

"""Select the best evaluated checkpoint from a run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from checkpoint_selection import refresh_best_checkpoint_artifacts, select_best_checkpoint_row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best evaluated checkpoint from eval/summary.csv.")
    parser.add_argument("--log_dir", type=str, required=True, help="Run directory that contains eval/summary.csv.")
    parser.add_argument("--case", type=str, default="suite", help="Row type to select, default: suite.")
    parser.add_argument(
        "--no_write_artifacts",
        action="store_true",
        help="Only print the selected row; do not update best_model.pt / best_checkpoint artifacts.",
    )
    parser.add_argument("--json", action="store_true", help="Print the selected row as JSON.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.log_dir).expanduser().resolve()
    summary_csv = run_dir / "eval" / "summary.csv"

    if args.no_write_artifacts:
        best_row = select_best_checkpoint_row(summary_csv, case=args.case)
    else:
        best_row = refresh_best_checkpoint_artifacts(run_dir, summary_csv=summary_csv, case=args.case)

    if best_row is None:
        raise SystemExit(f"No matching '{args.case}' rows found in {summary_csv}")

    if args.json:
        print(json.dumps(best_row, indent=2))
    else:
        print(best_row["checkpoint"])


if __name__ == "__main__":
    main()

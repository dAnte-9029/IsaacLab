"""Build CSV and JSON retention evidence from promoted-stage evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from pure_rl_retention import (
    RETENTION_CELL_FIELDS,
    RETENTION_ROW_FIELDS,
    build_retention_matrix,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Combined stage-evaluation CSV.")
    parser.add_argument(
        "--stage-order",
        required=True,
        help="Comma-separated completed curriculum stages, in training order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    with args.input.expanduser().open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    stage_order = tuple(stage.strip() for stage in args.stage_order.split(","))
    result = build_retention_matrix(records, stage_order=stage_order)

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "retention_matrix.csv", result["cells"], RETENTION_CELL_FIELDS)
    serializable_rows = [
        {
            **row,
            "evaluated_stages": ",".join(row["evaluated_stages"]),
            "failed_evaluation_stages": ",".join(row["failed_evaluation_stages"]),
        }
        for row in result["checkpoint_rows"]
    ]
    _write_csv(output_dir / "retention_checkpoints.csv", serializable_rows, RETENTION_ROW_FIELDS)
    with (output_dir / "retention_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return 0 if bool(result["all_retained"]) else 2


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

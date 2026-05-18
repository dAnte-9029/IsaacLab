#!/usr/bin/env python3
"""Export DeLaurier prior predictions for residual system-identification training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FLAPPING_BOT_SOURCE = _REPO_ROOT / "source" / "flapping_bot"
if str(_FLAPPING_BOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(_FLAPPING_BOT_SOURCE))

from flapping_bot.analysis.delaurier_gain_calibration import (
    OfflineDeLaurierConfig,
    PhysicalCalibrationParameters,
    parameter_vector_to_config,
    predict_delaurier_wrench,
)


DEFAULT_SPLIT_ROOT = Path(
    "/home/zn/flap-system-identification/dataset/"
    "canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1"
)
DEFAULT_PARAMETERS = Path("docs/analysis/effective_wrench/delaurier_physical_calibration_v1/parameters.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parameters-csv", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--prior-name", default="delaurier_physical_calibrated_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--wing-geom-csv", type=Path, default=OfflineDeLaurierConfig.wing_geom_csv)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    return parser.parse_args()


def _load_parameters(path: Path) -> PhysicalCalibrationParameters:
    table = pd.read_csv(path)
    return PhysicalCalibrationParameters.from_mapping(dict(zip(table["parameter"], table["value"])))


def _load_split(split_root: Path, split: str, max_rows: int | None) -> pd.DataFrame:
    frame = pd.read_parquet(split_root / f"{split}_samples.parquet")
    if max_rows is not None:
        frame = frame.iloc[:max_rows].copy()
    if "label_valid" in frame.columns:
        frame = frame.loc[frame["label_valid"].to_numpy(dtype=bool)].copy()
    return frame.reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    base_cfg = OfflineDeLaurierConfig(wing_geom_csv=args.wing_geom_csv)
    cfg = parameter_vector_to_config(_load_parameters(args.parameters_csv), base_cfg)

    row_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        frame = _load_split(args.split_root, split, args.max_rows_per_split)
        print(f"[INFO] Predicting {split}: {len(frame)} rows")
        predictions = predict_delaurier_wrench(
            frame,
            cfg=cfg,
            batch_size=int(args.batch_size),
            device=str(args.device),
        )
        predictions.astype("float32").to_parquet(args.output_root / f"{split}_predictions.parquet", index=False)
        row_counts[split] = int(len(predictions))

    manifest = {
        "split_root": str(args.split_root),
        "output_root": str(args.output_root),
        "parameters_csv": str(args.parameters_csv),
        "prior_name": str(args.prior_name),
        "row_counts": row_counts,
        "prediction_files": {split: f"{split}_predictions.parquet" for split in ("train", "val", "test")},
    }
    (args.output_root / "prior_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[INFO] Wrote prior predictions to {args.output_root}")


if __name__ == "__main__":
    main()

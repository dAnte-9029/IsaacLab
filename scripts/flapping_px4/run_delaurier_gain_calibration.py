#!/usr/bin/env python3
"""Run the gain-calibrated DeLaurier effective-wrench baseline."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FLAPPING_BOT_SOURCE = _REPO_ROOT / "source" / "flapping_bot"
if str(_FLAPPING_BOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(_FLAPPING_BOT_SOURCE))

from flapping_bot.analysis.delaurier_gain_calibration import (
    TARGET_COLUMNS,
    OfflineDeLaurierConfig,
    apply_channel_gains,
    fit_channel_gains,
    metrics_by_channel,
    predict_delaurier_wrench,
)


DEFAULT_SPLIT_ROOT = Path(
    "/home/zn/flap-system-identification/dataset/"
    "canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--wing-geom-csv", type=Path, default=OfflineDeLaurierConfig.wing_geom_csv)
    return parser.parse_args()


def _load_split(split_root: Path, split: str, max_rows: int | None) -> pd.DataFrame:
    path = split_root / f"{split}_samples.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if max_rows is not None:
        frame = frame.iloc[:max_rows].copy()
    invalid = frame["label_valid"].to_numpy(dtype=bool) if "label_valid" in frame.columns else None
    if invalid is not None:
        frame = frame.loc[invalid].copy()
    missing_targets = [column for column in TARGET_COLUMNS if column not in frame.columns]
    if missing_targets:
        raise ValueError(f"{path} is missing target columns: {missing_targets}")
    return frame.reset_index(drop=True)


def _write_readme(
    output_dir: Path,
    *,
    split_root: Path,
    gains: pd.Series,
    metrics: pd.DataFrame,
    rows_by_split: dict[str, int],
    save_predictions: bool,
) -> None:
    test_rows = metrics[(metrics["split"] == "test") & (metrics["model"] == "gain_calibrated")]
    uncal_rows = metrics[(metrics["split"] == "test") & (metrics["model"] == "uncalibrated")]
    summary_lines = [
        "# Gain-calibrated DeLaurier baseline",
        "",
        f"- Dataset: `{split_root}`",
        "- Calibration: one scalar least-squares gain per wrench channel, fitted on train split only.",
        "- Evaluation: the fitted train gains are frozen and applied to train/val/test predictions.",
        f"- Predictions saved: `{bool(save_predictions)}`",
        "",
        "## Rows",
        "",
    ]
    for split, count in rows_by_split.items():
        summary_lines.append(f"- {split}: {count}")
    summary_lines.extend(["", "## Fitted gains", ""])
    for target, gain in gains.items():
        summary_lines.append(f"- {target}: {gain:.9g}")
    summary_lines.extend(["", "## Test RMSE", ""])
    merged = uncal_rows[["target", "rmse"]].merge(
        test_rows[["target", "rmse"]], on="target", suffixes=("_uncalibrated", "_gain_calibrated")
    )
    for row in merged.itertuples(index=False):
        summary_lines.append(
            f"- {row.target}: {row.rmse_uncalibrated:.6g} -> {row.rmse_gain_calibrated:.6g}"
        )
    summary_lines.append("")
    (output_dir / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("docs/analysis/effective_wrench") / f"delaurier_gain_calibration_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OfflineDeLaurierConfig(wing_geom_csv=args.wing_geom_csv)
    split_frames = {
        split: _load_split(args.split_root, split, args.max_rows_per_split)
        for split in ("train", "val", "test")
    }
    rows_by_split = {split: len(frame) for split, frame in split_frames.items()}

    predictions: dict[str, pd.DataFrame] = {}
    for split, frame in split_frames.items():
        print(f"[INFO] Predicting {split}: {len(frame)} rows")
        predictions[split] = predict_delaurier_wrench(
            frame,
            cfg=cfg,
            batch_size=int(args.batch_size),
            device=str(args.device),
        )

    gains = fit_channel_gains(predictions["train"], split_frames["train"])
    calibrated = {
        split: apply_channel_gains(prediction, gains)
        for split, prediction in predictions.items()
    }

    metric_frames = []
    for split in ("train", "val", "test"):
        for model_name, prediction in (
            ("uncalibrated", predictions[split]),
            ("gain_calibrated", calibrated[split]),
        ):
            metrics = metrics_by_channel(split_frames[split], prediction)
            metrics.insert(0, "model", model_name)
            metrics.insert(0, "split", split)
            metric_frames.append(metrics)
    all_metrics = pd.concat(metric_frames, ignore_index=True)

    gains.rename("gain").to_csv(output_dir / "gains.csv", header=True)
    all_metrics.to_csv(output_dir / "metrics_by_split.csv", index=False)
    if args.save_predictions:
        for split in ("train", "val", "test"):
            predictions[split].astype("float32").to_parquet(output_dir / f"{split}_uncalibrated_predictions.parquet")
            calibrated[split].astype("float32").to_parquet(output_dir / f"{split}_gain_calibrated_predictions.parquet")

    summary = {
        "split_root": str(args.split_root),
        "output_dir": str(output_dir),
        "rows_by_split": rows_by_split,
        "gains": {key: float(value) for key, value in gains.items()},
        "save_predictions": bool(args.save_predictions),
        "target_columns": TARGET_COLUMNS,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_readme(
        output_dir,
        split_root=args.split_root,
        gains=gains,
        metrics=all_metrics,
        rows_by_split=rows_by_split,
        save_predictions=bool(args.save_predictions),
    )
    print(f"[INFO] Wrote results to {output_dir}")


if __name__ == "__main__":
    main()

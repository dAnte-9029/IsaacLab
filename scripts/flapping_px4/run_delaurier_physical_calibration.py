#!/usr/bin/env python3
"""Run the physically calibrated DeLaurier effective-wrench baseline."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FLAPPING_BOT_SOURCE = _REPO_ROOT / "source" / "flapping_bot"
if str(_FLAPPING_BOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(_FLAPPING_BOT_SOURCE))

from flapping_bot.analysis.delaurier_gain_calibration import (
    PHYSICAL_PARAMETER_BOUNDS,
    TARGET_COLUMNS,
    OfflineDeLaurierConfig,
    PhysicalCalibrationParameters,
    metrics_by_channel,
    parameter_vector_to_config,
    predict_delaurier_wrench,
    random_search_physical_calibration,
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
    parser.add_argument("--calibration-rows", type=int, default=12000)
    parser.add_argument("--n-candidates", type=int, default=48)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--regularization-weight", type=float, default=0.01)
    parser.add_argument("--wing-geom-csv", type=Path, default=OfflineDeLaurierConfig.wing_geom_csv)
    return parser.parse_args()


def _load_split(split_root: Path, split: str, max_rows: int | None) -> pd.DataFrame:
    path = split_root / f"{split}_samples.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if max_rows is not None:
        frame = frame.iloc[:max_rows].copy()
    if "label_valid" in frame.columns:
        frame = frame.loc[frame["label_valid"].to_numpy(dtype=bool)].copy()
    missing_targets = [column for column in TARGET_COLUMNS if column not in frame.columns]
    if missing_targets:
        raise ValueError(f"{path} is missing target columns: {missing_targets}")
    return frame.reset_index(drop=True)


def _select_calibration_subset(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    if rows <= 0:
        raise ValueError("--calibration-rows must be positive.")
    if rows >= len(frame):
        return frame.reset_index(drop=True)
    if "log_id" not in frame.columns:
        indices = np.linspace(0, len(frame) - 1, num=rows, dtype=int)
        return frame.iloc[np.unique(indices)].reset_index(drop=True)

    groups = [group for _log_id, group in frame.groupby("log_id", sort=False)]
    per_log = max(1, int(np.ceil(rows / len(groups))))
    selected: list[pd.DataFrame] = []
    for group in groups:
        take = min(per_log, len(group))
        indices = np.linspace(0, len(group) - 1, num=take, dtype=int)
        selected.append(group.iloc[np.unique(indices)])
    subset = pd.concat(selected, ignore_index=False).sort_index()
    if len(subset) > rows:
        trim_indices = np.linspace(0, len(subset) - 1, num=rows, dtype=int)
        subset = subset.iloc[np.unique(trim_indices)]
    return subset.reset_index(drop=True)


def _target_scale(train_frame: pd.DataFrame) -> pd.Series:
    scale = train_frame.loc[:, TARGET_COLUMNS].astype(float).std(axis=0)
    return scale.replace(0.0, 1.0).fillna(1.0)


def _parameters_table(parameters: PhysicalCalibrationParameters) -> pd.DataFrame:
    rows = []
    for name, bound in PHYSICAL_PARAMETER_BOUNDS.items():
        value = float(getattr(parameters, name))
        rows.append(
            {
                "parameter": name,
                "value": value,
                "nominal": float(bound.nominal),
                "low": float(bound.low),
                "high": float(bound.high),
                "hit_low": bool(np.isclose(value, bound.low)),
                "hit_high": bool(np.isclose(value, bound.high)),
            }
        )
    return pd.DataFrame(rows)


def _write_readme(
    output_dir: Path,
    *,
    split_root: Path,
    rows_by_split: dict[str, int],
    calibration_rows: int,
    search_result_objective: float,
    parameters: PhysicalCalibrationParameters,
    metrics: pd.DataFrame,
) -> None:
    test_uncal = metrics[(metrics["split"] == "test") & (metrics["model"] == "uncalibrated")]
    test_phys = metrics[(metrics["split"] == "test") & (metrics["model"] == "physically_calibrated")]
    merged = test_uncal[["target", "rmse"]].merge(
        test_phys[["target", "rmse"]],
        on="target",
        suffixes=("_uncalibrated", "_physically_calibrated"),
    )

    lines = [
        "# Physically calibrated DeLaurier baseline",
        "",
        f"- Dataset: `{split_root}`",
        f"- Calibration rows: `{calibration_rows}`",
        f"- Best normalized objective: `{search_result_objective:.9g}`",
        "- Calibration: bounded random search over eight interpretable physical parameters using train data only.",
        "- Evaluation: final parameters are frozen and evaluated on full train/val/test splits.",
        "",
        "## Rows",
        "",
    ]
    for split, count in rows_by_split.items():
        lines.append(f"- {split}: {count}")
    lines.extend(["", "## Parameters", ""])
    for name, value in parameters.as_dict().items():
        bound = PHYSICAL_PARAMETER_BOUNDS[name]
        hit = ""
        if np.isclose(value, bound.low):
            hit = " (hit low bound)"
        elif np.isclose(value, bound.high):
            hit = " (hit high bound)"
        lines.append(f"- {name}: {value:.9g}{hit}")
    lines.extend(["", "## Test RMSE", ""])
    for row in merged.itertuples(index=False):
        lines.append(f"- {row.target}: {row.rmse_uncalibrated:.6g} -> {row.rmse_physically_calibrated:.6g}")
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("docs/analysis/effective_wrench") / f"delaurier_physical_calibration_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OfflineDeLaurierConfig(wing_geom_csv=args.wing_geom_csv)
    split_frames = {
        split: _load_split(args.split_root, split, args.max_rows_per_split)
        for split in ("train", "val", "test")
    }
    rows_by_split = {split: len(frame) for split, frame in split_frames.items()}
    calibration_frame = _select_calibration_subset(split_frames["train"], int(args.calibration_rows))
    scale = _target_scale(split_frames["train"])

    def predictor(frame: pd.DataFrame, parameters: PhysicalCalibrationParameters) -> pd.DataFrame:
        cfg = parameter_vector_to_config(parameters, base_cfg)
        return predict_delaurier_wrench(
            frame,
            cfg=cfg,
            batch_size=int(args.batch_size),
            device=str(args.device),
        )

    print(
        f"[INFO] Searching {args.n_candidates} random candidates "
        f"+ nominal on {len(calibration_frame)} calibration rows"
    )
    search_result = random_search_physical_calibration(
        calibration_frame,
        calibration_frame,
        scale,
        n_candidates=int(args.n_candidates),
        seed=int(args.seed),
        predictor=predictor,
        regularization_weight=float(args.regularization_weight),
    )
    best_parameters = search_result.best_parameters
    best_cfg = parameter_vector_to_config(best_parameters, base_cfg)

    metric_frames = []
    for split, frame in split_frames.items():
        print(f"[INFO] Evaluating {split}: {len(frame)} rows")
        uncalibrated = predict_delaurier_wrench(
            frame,
            cfg=base_cfg,
            batch_size=int(args.batch_size),
            device=str(args.device),
        )
        physically_calibrated = predict_delaurier_wrench(
            frame,
            cfg=best_cfg,
            batch_size=int(args.batch_size),
            device=str(args.device),
        )
        for model_name, prediction in (
            ("uncalibrated", uncalibrated),
            ("physically_calibrated", physically_calibrated),
        ):
            metrics = metrics_by_channel(frame, prediction)
            metrics.insert(0, "model", model_name)
            metrics.insert(0, "split", split)
            metric_frames.append(metrics)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    parameters_table = _parameters_table(best_parameters)
    parameters_table.to_csv(output_dir / "parameters.csv", index=False)
    search_result.trace.to_csv(output_dir / "optimization_trace.csv", index=False)
    all_metrics.to_csv(output_dir / "metrics_by_split.csv", index=False)

    summary = {
        "split_root": str(args.split_root),
        "output_dir": str(output_dir),
        "rows_by_split": rows_by_split,
        "calibration_rows_requested": int(args.calibration_rows),
        "calibration_rows_used": int(len(calibration_frame)),
        "n_candidates": int(args.n_candidates),
        "seed": int(args.seed),
        "regularization_weight": float(args.regularization_weight),
        "best_objective": float(search_result.best_objective),
        "best_parameters": best_parameters.as_dict(),
        "target_columns": TARGET_COLUMNS,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_readme(
        output_dir,
        split_root=args.split_root,
        rows_by_split=rows_by_split,
        calibration_rows=len(calibration_frame),
        search_result_objective=float(search_result.best_objective),
        parameters=best_parameters,
        metrics=all_metrics,
    )
    print(f"[INFO] Wrote results to {output_dir}")


if __name__ == "__main__":
    main()

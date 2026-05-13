#!/usr/bin/env python3
"""Run teacher-forced short-window rigid-body rollouts from real flight logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FLAPPING_BOT_SOURCE = _REPO_ROOT / "source" / "flapping_bot"
_SYSID_SOURCE = Path("/home/zn/flap-system-identification/src")
for path in (_FLAPPING_BOT_SOURCE, _SYSID_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flapping_bot.analysis.delaurier_gain_calibration import (
    TARGET_COLUMNS,
    OfflineDeLaurierConfig,
    PhysicalCalibrationParameters,
    apply_channel_gains,
    parameter_vector_to_config,
    predict_delaurier_wrench,
)
from flapping_bot.analysis.short_window_rollout import (
    RigidBodyParams,
    integrate_wrench_window,
    rollout_error_summary,
)


DEFAULT_SPLIT_ROOT = Path(
    "/home/zn/flap-system-identification/dataset/"
    "canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1"
)
DEFAULT_MODEL_BUNDLE = Path(
    "/home/zn/flap-system-identification/artifacts/20260507_temporal_backbone_final/runs/"
    "final_transformer_d64_l2_h4_hist128/causal_transformer_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt"
)
DEFAULT_PHYSICAL_PARAMS = Path("docs/analysis/effective_wrench/delaurier_physical_calibration_v1/parameters.csv")
DEFAULT_GAIN_PARAMS = Path("docs/analysis/effective_wrench/delaurier_gain_calibration_v1/gains.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-bundle", type=Path, default=DEFAULT_MODEL_BUNDLE)
    parser.add_argument("--physical-params", type=Path, default=DEFAULT_PHYSICAL_PARAMS)
    parser.add_argument("--gain-params", type=Path, default=DEFAULT_GAIN_PARAMS)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/analysis/effective_wrench/short_window_rollout_level1_v1"))
    parser.add_argument("--horizons-s", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--windows-per-horizon", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wing-geom-csv", type=Path, default=OfflineDeLaurierConfig.wing_geom_csv)
    return parser.parse_args()


def _load_split(split_root: Path, split: str) -> pd.DataFrame:
    frame = pd.read_parquet(split_root / f"{split}_samples.parquet")
    if "label_valid" in frame.columns:
        frame = frame.loc[frame["label_valid"].to_numpy(dtype=bool)].copy()
    return frame.reset_index(drop=True)


def _load_physical_parameters(path: Path) -> PhysicalCalibrationParameters:
    table = pd.read_csv(path)
    values = dict(zip(table["parameter"], table["value"]))
    return PhysicalCalibrationParameters.from_mapping(values)


def _load_gain_parameters(path: Path) -> pd.Series:
    table = pd.read_csv(path, index_col=0)
    if "gain" not in table.columns:
        raise ValueError(f"Missing gain column in {path}")
    return table["gain"].astype(float)


def _select_windows(frame: pd.DataFrame, *, horizon_s: float, count: int) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for _log_id, group in frame.groupby("log_id", sort=False):
        idx = group.index.to_numpy(dtype=int)
        time = group["time_s"].to_numpy(dtype=float)
        if len(idx) < 3:
            continue
        dt = float(np.nanmedian(np.diff(time)))
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 0.01
        steps = max(2, int(round(float(horizon_s) / dt)) + 1)
        if len(idx) < steps:
            continue
        for local_start in range(0, len(idx) - steps + 1):
            windows.append((int(idx[local_start]), int(idx[local_start + steps - 1]) + 1))
    if len(windows) <= count:
        return windows
    selected = np.linspace(0, len(windows) - 1, num=count, dtype=int)
    return [windows[int(i)] for i in np.unique(selected)]


def _prediction_metadata(bundle_path: Path, frame: pd.DataFrame, *, split: str, batch_size: int, device: str) -> pd.DataFrame:
    import torch
    from system_identification.training import prediction_metadata_frame_for_bundle

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    return prediction_metadata_frame_for_bundle(
        bundle,
        frame,
        split_name=split,
        batch_size=batch_size,
        device=device,
    )


def _summary_table(window_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "pos_rmse_m",
        "vel_rmse_mps",
        "att_rmse_deg",
        "omega_rmse_radps",
        "final_pos_err_m",
        "final_vel_err_mps",
        "final_att_err_deg",
        "final_omega_err_radps",
    ]
    rows = []
    for (horizon_s, method), group in window_metrics.groupby(["horizon_s", "method"], sort=True):
        row: dict[str, float | str | int] = {
            "horizon_s": float(horizon_s),
            "method": str(method),
            "window_count": int(len(group)),
            "diverged_count": int(group["diverged"].fillna(False).astype(bool).sum()) if "diverged" in group.columns else 0,
        }
        for column in metric_columns:
            values = group[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{column}_valid_count"] = int(len(finite))
            if len(finite) == 0:
                row[f"{column}_median"] = np.nan
                row[f"{column}_iqr"] = np.nan
            else:
                row[f"{column}_median"] = float(np.median(finite))
                row[f"{column}_iqr"] = float(np.percentile(finite, 75.0) - np.percentile(finite, 25.0))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_readme(output_dir: Path, summary: pd.DataFrame, *, model_bundle: Path, split: str) -> None:
    lines = [
        "# Level 1 short-window rollout",
        "",
        "- Type: teacher-forced offline rigid-body integration.",
        "- Inputs: real log initial state and real-log-aligned predicted wrench sequence.",
        "- This is an auxiliary test; it does not close the loop through simulated-state aero inputs.",
        f"- Split: `{split}`",
        f"- NN bundle: `{model_bundle}`",
        "",
        "## Median Final Position Error",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.horizon_s:.3g}s {row.method}: "
            f"{row.final_pos_err_m_median:.6g} m "
            f"(IQR {row.final_pos_err_m_iqr:.6g})"
        )
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_frame = _load_split(args.split_root, args.split)
    print(f"[INFO] Loading NN predictions from {args.model_bundle}")
    aligned = _prediction_metadata(
        args.model_bundle,
        raw_frame,
        split=args.split,
        batch_size=int(args.batch_size),
        device=str(args.device),
    ).reset_index(drop=True)

    base_cfg = OfflineDeLaurierConfig(wing_geom_csv=args.wing_geom_csv)
    physical_cfg = parameter_vector_to_config(_load_physical_parameters(args.physical_params), base_cfg)
    gains = _load_gain_parameters(args.gain_params)

    print(f"[INFO] Predicting DeLaurier wrenches on aligned rows: {len(aligned)}")
    delaurier_uncal = predict_delaurier_wrench(aligned, cfg=base_cfg, batch_size=int(args.batch_size), device=str(args.device))
    delaurier_physical = predict_delaurier_wrench(
        aligned,
        cfg=physical_cfg,
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    delaurier_gain = apply_channel_gains(delaurier_uncal, gains)
    label_oracle = aligned.loc[:, TARGET_COLUMNS].reset_index(drop=True)
    nn_wrench = pd.DataFrame(
        {column: aligned[f"pred_{column}"].to_numpy(dtype=float) for column in TARGET_COLUMNS},
        index=aligned.index,
    )

    methods = {
        "label_oracle": label_oracle,
        "delaurier_uncalibrated": delaurier_uncal,
        "delaurier_gain_calibrated": delaurier_gain,
        "delaurier_physically_calibrated": delaurier_physical,
        "nn_effective_wrench": nn_wrench,
    }

    params = RigidBodyParams.default_flapper()
    metric_rows = []
    for horizon_s in args.horizons_s:
        windows = _select_windows(aligned, horizon_s=float(horizon_s), count=int(args.windows_per_horizon))
        print(f"[INFO] Horizon {horizon_s:g}s: {len(windows)} windows")
        for window_id, (start, stop) in enumerate(windows):
            real_window = aligned.iloc[start:stop].reset_index(drop=True)
            for method, wrench in methods.items():
                predicted = integrate_wrench_window(
                    real_window,
                    wrench.iloc[start:stop].reset_index(drop=True),
                    params=params,
                )
                row = rollout_error_summary(real_window, predicted)
                row.update(
                    {
                        "horizon_s": float(horizon_s),
                        "window_id": int(window_id),
                        "method": method,
                        "log_id": str(real_window["log_id"].iloc[0]) if "log_id" in real_window.columns else "",
                        "start_time_s": float(real_window["time_s"].iloc[0]),
                        "end_time_s": float(real_window["time_s"].iloc[-1]),
                        "sample_count": int(len(real_window)),
                    }
                )
                metric_rows.append(row)

    window_metrics = pd.DataFrame(metric_rows)
    summary = _summary_table(window_metrics)
    window_metrics.to_csv(args.output_dir / "window_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    run_summary = {
        "split_root": str(args.split_root),
        "split": str(args.split),
        "model_bundle": str(args.model_bundle),
        "aligned_rows": int(len(aligned)),
        "horizons_s": [float(v) for v in args.horizons_s],
        "windows_per_horizon": int(args.windows_per_horizon),
        "methods": list(methods),
        "rollout_type": "teacher_forced_wrench_sequence",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_readme(args.output_dir, summary, model_bundle=args.model_bundle, split=args.split)
    print(f"[INFO] Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()

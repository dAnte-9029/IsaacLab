from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "flapping_rl" / "checkpoint_selection.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_selection", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
checkpoint_selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint_selection)


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_select_best_suite_row_prefers_highest_score(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    _write_summary(
        summary_csv,
        [
            {"checkpoint": str(tmp_path / "model_0.pt"), "case": "suite", "ckpt_index": 0, "score": 25.0, "termination_rate": 0.5, "timeout_rate": 0.5, "mean_abs_vx_err": 1.0, "mean_abs_z_err": 1.0, "mean_max_abs_y": 1.0, "mean_max_tilt_deg": 10.0},
            {"checkpoint": str(tmp_path / "model_1.pt"), "case": "calm", "ckpt_index": 1, "score": 90.0, "termination_rate": 0.0, "timeout_rate": 1.0, "mean_abs_vx_err": 0.1, "mean_abs_z_err": 0.1, "mean_max_abs_y": 0.1, "mean_max_tilt_deg": 5.0},
            {"checkpoint": str(tmp_path / "model_2.pt"), "case": "suite", "ckpt_index": 2, "score": 55.0, "termination_rate": 0.0, "timeout_rate": 1.0, "mean_abs_vx_err": 0.3, "mean_abs_z_err": 0.2, "mean_max_abs_y": 0.2, "mean_max_tilt_deg": 8.0},
        ],
    )

    best = checkpoint_selection.select_best_checkpoint_row(summary_csv)

    assert best is not None
    assert best["checkpoint"].endswith("model_2.pt")
    assert float(best["score"]) == 55.0


def test_select_best_suite_row_breaks_ties_by_stability_then_earlier_checkpoint(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    _write_summary(
        summary_csv,
        [
            {"checkpoint": str(tmp_path / "model_10.pt"), "case": "suite", "ckpt_index": 10, "score": 62.0, "termination_rate": 0.0, "timeout_rate": 1.0, "mean_abs_vx_err": 0.1, "mean_abs_z_err": 0.0, "mean_max_abs_y": 0.0, "mean_max_tilt_deg": 10.0},
            {"checkpoint": str(tmp_path / "model_20.pt"), "case": "suite", "ckpt_index": 20, "score": 62.0, "termination_rate": 0.0, "timeout_rate": 1.0, "mean_abs_vx_err": 0.1, "mean_abs_z_err": 0.0, "mean_max_abs_y": 0.0, "mean_max_tilt_deg": 10.0},
            {"checkpoint": str(tmp_path / "model_30.pt"), "case": "suite", "ckpt_index": 30, "score": 62.0, "termination_rate": 0.2, "timeout_rate": 0.8, "mean_abs_vx_err": 0.05, "mean_abs_z_err": 0.0, "mean_max_abs_y": 0.0, "mean_max_tilt_deg": 9.0},
        ],
    )

    best = checkpoint_selection.select_best_checkpoint_row(summary_csv)

    assert best is not None
    assert best["checkpoint"].endswith("model_10.pt")


def test_refresh_best_checkpoint_artifacts_writes_text_json_and_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)
    model_1 = run_dir / "model_1.pt"
    model_2 = run_dir / "model_2.pt"
    model_1.write_text("one")
    model_2.write_text("two")
    summary_csv = eval_dir / "summary.csv"
    _write_summary(
        summary_csv,
        [
            {"checkpoint": str(model_1.resolve()), "case": "suite", "ckpt_index": 1, "score": 20.0, "termination_rate": 1.0, "timeout_rate": 0.0, "mean_abs_vx_err": 2.0, "mean_abs_z_err": 2.0, "mean_max_abs_y": 2.0, "mean_max_tilt_deg": 60.0},
            {"checkpoint": str(model_2.resolve()), "case": "suite", "ckpt_index": 2, "score": 50.0, "termination_rate": 0.0, "timeout_rate": 1.0, "mean_abs_vx_err": 0.3, "mean_abs_z_err": 0.1, "mean_max_abs_y": 0.2, "mean_max_tilt_deg": 12.0},
        ],
    )

    best = checkpoint_selection.refresh_best_checkpoint_artifacts(run_dir, summary_csv=summary_csv)

    assert best is not None
    assert best["checkpoint"] == str(model_2.resolve())
    assert (run_dir / "best_checkpoint.txt").read_text().strip() == str(model_2.resolve())
    payload = json.loads((eval_dir / "best_checkpoint.json").read_text())
    assert payload["checkpoint"] == str(model_2.resolve())
    best_model = run_dir / "best_model.pt"
    assert best_model.exists()
    assert best_model.resolve() == model_2.resolve()


def test_select_best_suite_row_supports_path_tracking_schema(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    _write_summary(
        summary_csv,
        [
            {
                "checkpoint": str(tmp_path / "model_10.pt"),
                "case": "suite",
                "ckpt_index": 10,
                "score": 71.0,
                "completion_rate": 0.7,
                "termination_rate": 0.3,
                "timeout_rate": 0.7,
                "mean_abs_lateral_error_m": 0.8,
                "mean_abs_height_error_m": 0.4,
                "mean_abs_align_error_deg": 12.0,
            },
            {
                "checkpoint": str(tmp_path / "model_20.pt"),
                "case": "suite",
                "ckpt_index": 20,
                "score": 71.0,
                "completion_rate": 0.9,
                "termination_rate": 0.1,
                "timeout_rate": 0.9,
                "mean_abs_lateral_error_m": 0.4,
                "mean_abs_height_error_m": 0.2,
                "mean_abs_align_error_deg": 7.0,
            },
        ],
    )

    best = checkpoint_selection.select_best_checkpoint_row(summary_csv)

    assert best is not None
    assert best["checkpoint"].endswith("model_20.pt")

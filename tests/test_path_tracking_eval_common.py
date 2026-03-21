from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "flapping_rl" / "path_tracking_eval_common.py"
SPEC = importlib.util.spec_from_file_location("path_tracking_eval_common", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
path_tracking_eval_common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(path_tracking_eval_common)


def test_resolve_eval_suite_defaults_path_tracking_task_to_truth_suite() -> None:
    resolved = path_tracking_eval_common.resolve_eval_suite(
        "Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0",
        "straight_standard",
    )

    assert resolved == "path_tracking_truth_nowind_v1"


def test_aggregate_case_row_reports_completion_and_path_errors() -> None:
    case = {
        "name": "straight_nowind",
        "wind_enabled": False,
        "wind_xy_mps": (0.0, 0.0),
        "wind_ou_enabled": False,
        "wind_ou_sigma_xy_mps": (0.0, 0.0),
    }
    episode_rows = [
        {
            "mean_abs_lateral_error_m": 0.20,
            "mean_abs_height_error_m": 0.10,
            "mean_abs_align_error_deg": 3.0,
            "p95_abs_lateral_error_m": 0.40,
            "p95_abs_height_error_m": 0.20,
            "p95_abs_align_error_deg": 5.0,
            "final_progress_ratio": 1.0,
            "completed": 1,
            "terminated": 0,
            "time_out": 1,
        },
        {
            "mean_abs_lateral_error_m": 0.30,
            "mean_abs_height_error_m": 0.20,
            "mean_abs_align_error_deg": 4.0,
            "p95_abs_lateral_error_m": 0.60,
            "p95_abs_height_error_m": 0.30,
            "p95_abs_align_error_deg": 7.0,
            "final_progress_ratio": 0.75,
            "completed": 0,
            "terminated": 1,
            "time_out": 0,
        },
    ]

    row = path_tracking_eval_common.aggregate_case_row(
        checkpoint=Path("/tmp/model_20.pt"),
        ckpt_index=20,
        case=case,
        episode_rows=episode_rows,
    )

    assert row["case"] == "straight_nowind"
    assert row["episodes"] == 2
    assert math.isclose(row["completion_rate"], 0.5)
    assert math.isclose(row["termination_rate"], 0.5)
    assert math.isclose(row["timeout_rate"], 0.5)
    assert math.isclose(row["mean_abs_lateral_error_m"], 0.25)
    assert math.isclose(row["mean_abs_height_error_m"], 0.15)
    assert math.isclose(row["mean_abs_align_error_deg"], 3.5)
    assert math.isclose(row["mean_p95_abs_lateral_error_m"], 0.5)
    assert math.isclose(row["mean_p95_abs_height_error_m"], 0.25)
    assert math.isclose(row["mean_p95_abs_align_error_deg"], 6.0)
    assert math.isclose(row["mean_final_progress_ratio"], 0.875)
    assert row["score"] > 0.0


def test_aggregate_case_row_rejects_empty_episode_list() -> None:
    case = {
        "name": "straight_nowind",
        "wind_enabled": False,
        "wind_xy_mps": (0.0, 0.0),
        "wind_ou_enabled": False,
        "wind_ou_sigma_xy_mps": (0.0, 0.0),
    }

    with pytest.raises(ValueError):
        path_tracking_eval_common.aggregate_case_row(
            checkpoint=Path("/tmp/model_0.pt"),
            ckpt_index=0,
            case=case,
            episode_rows=[],
        )

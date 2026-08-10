from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/flapping_rl/pure_rl_retention.py"
SPEC = importlib.util.spec_from_file_location("pure_rl_retention", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pure_rl_retention = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pure_rl_retention
SPEC.loader.exec_module(pure_rl_retention)


def _record(
    training_stage: str,
    evaluation_stage: str,
    *,
    score: float,
    passed: bool = True,
) -> dict[str, object]:
    return {
        "training_stage": training_stage,
        "checkpoint": f"/checkpoints/{training_stage}.pt",
        "evaluation_stage": evaluation_stage,
        "evaluation_contract": f"{evaluation_stage}_v1",
        "success_gate_passed": int(passed),
        "score": score,
    }


def test_retention_matrix_builds_complete_lower_triangle_and_score_drop() -> None:
    records = [
        _record("straight", "straight", score=10.0),
        _record("trajectory", "straight", score=8.5),
        _record("trajectory", "trajectory", score=12.0),
        _record("wind", "straight", score=7.0),
        _record("wind", "trajectory", score=11.0),
        _record("wind", "wind", score=9.0),
    ]

    result = pure_rl_retention.build_retention_matrix(
        records,
        stage_order=("straight", "trajectory", "wind"),
    )

    assert result["all_retained"] is True
    assert len(result["cells"]) == 6
    wind_straight = next(
        cell
        for cell in result["cells"]
        if cell["training_stage"] == "wind" and cell["evaluation_stage"] == "straight"
    )
    assert wind_straight["diagonal_score"] == pytest.approx(10.0)
    assert wind_straight["score_drop_from_diagonal"] == pytest.approx(3.0)
    assert wind_straight["forgetting_score_drop"] == pytest.approx(3.0)
    assert [row["retention_passed"] for row in result["checkpoint_rows"]] == [1, 1, 1]


def test_retention_matrix_exposes_old_task_failure_at_new_stage() -> None:
    records = [
        _record("straight", "straight", score=10.0),
        _record("trajectory", "straight", score=8.0, passed=False),
        _record("trajectory", "trajectory", score=12.0),
    ]

    result = pure_rl_retention.build_retention_matrix(
        records,
        stage_order=("straight", "trajectory"),
    )

    assert result["all_retained"] is False
    assert result["checkpoint_rows"][1]["retention_passed"] == 0
    assert result["checkpoint_rows"][1]["failed_evaluation_stages"] == ["straight"]


@pytest.mark.parametrize("fault", ("missing", "duplicate", "unknown", "multiple_checkpoints"))
def test_retention_matrix_fails_closed_on_incomplete_or_ambiguous_input(fault: str) -> None:
    records = [
        _record("straight", "straight", score=10.0),
        _record("trajectory", "straight", score=9.0),
        _record("trajectory", "trajectory", score=12.0),
    ]
    if fault == "missing":
        records.pop(1)
    elif fault == "duplicate":
        records.append(dict(records[1]))
    elif fault == "unknown":
        records[1]["evaluation_stage"] = "gust"
    elif fault == "multiple_checkpoints":
        records[1]["checkpoint"] = "/checkpoints/other.pt"

    with pytest.raises(ValueError):
        pure_rl_retention.build_retention_matrix(
            records,
            stage_order=("straight", "trajectory"),
        )


def test_retention_cli_writes_machine_readable_matrix(tmp_path: Path) -> None:
    input_csv = tmp_path / "evaluations.csv"
    records = [
        _record("straight", "straight", score=10.0),
        _record("trajectory", "straight", score=9.0),
        _record("trajectory", "trajectory", score=12.0),
    ]
    with input_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    output_dir = tmp_path / "retention"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH.with_name("build_pure_rl_retention_matrix.py")),
            "--input",
            str(input_csv),
            "--stage-order",
            "straight,trajectory",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "retention_matrix.csv").is_file()
    assert (output_dir / "retention_checkpoints.csv").is_file()
    summary = json.loads((output_dir / "retention_summary.json").read_text())
    assert summary["all_retained"] is True

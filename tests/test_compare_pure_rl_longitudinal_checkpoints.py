from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/flapping_rl/compare_pure_rl_longitudinal_checkpoints.py"
)
SPEC = importlib.util.spec_from_file_location("compare_pure_rl_longitudinal_checkpoints", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def _row(step: int, *, label: str, actual_vz: float, desired_vz: float) -> dict[str, float | str]:
    phase = "entry" if step < 2 else ("slope" if step < 10 else "recovery")
    row: dict[str, float | str] = {
        "label": label,
        "step": float(step),
        "time_s": 0.1 * (step + 1),
        "phase": phase,
        "height_error_m": -0.1 * step,
        "vertical_velocity_mps": actual_vz,
        "desired_vertical_velocity_mps": desired_vz,
        "pitch_deg": 2.0,
        "terminated": 0.0,
    }
    for action_name in compare._ACTION_NAMES:
        row[f"applied_{action_name}_action"] = 0.2
    return row


def test_worker_command_freezes_same_case_and_cpu(tmp_path: Path) -> None:
    args = argparse.Namespace(
        slope_deg=12.0,
        heading_deg=90.0,
        flap_phase_deg=180.0,
        entry_length_m=17.5,
        slope_length_m=25.0,
        duration_s=12.0,
        seed=3,
    )
    command = compare._worker_command(
        checkpoint=tmp_path / "model_175.pt",
        label="candidate",
        output_dir=tmp_path / "output",
        args=args,
    )

    assert "--worker" in command
    assert command[command.index("--slope-deg") + 1] == "12.0"
    assert command[command.index("--heading-deg") + 1] == "90.0"
    assert command[command.index("--flap-phase-deg") + 1] == "180.0"
    assert command[command.index("--device") + 1] == "cpu"
    assert "--headless" in command


def test_summarize_trace_reports_sustained_vertical_response_delay() -> None:
    rows = [
        _row(
            step,
            label="candidate",
            actual_vz=0.0 if step < 4 else 1.0,
            desired_vz=0.0 if step < 2 or step >= 10 else 1.0,
        )
        for step in range(12)
    ]

    summary = compare._summarize_trace(rows)

    assert summary["slope_onset_time_s"] == pytest.approx(0.3)
    assert summary["vertical_velocity_90pct_response_delay_s"] == pytest.approx(0.2)
    assert summary["slope_mean_abs_height_error_m"] == pytest.approx(0.55)


def test_align_traces_fails_closed_on_time_mismatch() -> None:
    baseline = [_row(0, label="baseline", actual_vz=0.0, desired_vz=0.0)]
    candidate = [_row(0, label="candidate", actual_vz=0.0, desired_vz=0.0)]
    candidate[0]["time_s"] = 0.2

    with pytest.raises(ValueError, match="time mismatch"):
        compare._align_traces(baseline, candidate)

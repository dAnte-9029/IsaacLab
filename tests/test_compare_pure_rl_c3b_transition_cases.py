from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/flapping_rl/compare_pure_rl_c3b_transition_cases.py"
)
SPEC = importlib.util.spec_from_file_location("compare_pure_rl_c3b_transition_cases", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def _row(step: int, *, case_id: str, template_id: int, failure: bool) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "case_id": case_id,
        "expected_result": "failure" if failure else "control",
        "template_id": float(template_id),
        "time_s": 0.1 * (step + 1),
        "progress_m": float(step),
        "active_slope_deg": 12.0 if (step >= 5 if template_id == 5 else step >= 2) else 0.0,
        "active_curvature_rad_per_m": 0.03 if (step >= 2 if template_id == 5 else step >= 5) else 0.0,
        "cross_track_error_m": 0.1 * step,
        "height_error_m": -0.05 * step,
        "lateral_normal_velocity_mps": 0.2,
        "vertical_normal_velocity_mps": 0.3,
        "abs_roll_deg": float(step) + (4.0 if failure else 0.0),
        "pitch_deg": 3.0,
        "roll_rate_rad_s": 0.4,
        "pitch_rate_rad_s": 0.1,
        "yaw_rate_rad_s": 0.2,
        "actual_flap_frequency_hz": 5.0,
        "applied_flap_frequency_action": 0.1,
        "applied_rudder_action": 0.2,
        "applied_left_elevon_action": 0.3,
        "applied_right_elevon_action": -0.3,
        "reached_all_events": float(step == 9 and not failure),
        "terminated": float(step == 9 and failure),
        "roll_limit_termination": float(step == 9 and failure),
        "ground_termination": 0.0,
        "tilt_termination": 0.0,
        "cross_track_termination": 0.0,
        "height_termination": 0.0,
        "tail_limit_active": 0.0,
        "frequency_limit_active": 0.0,
        "normalized_action_delta": 0.2,
    }
    return row


def test_worker_command_freezes_matched_case_and_cpu(tmp_path: Path) -> None:
    args = argparse.Namespace(
        slope_deg=12.0,
        geometry_roll_deg=13.5,
        turn_sign=-1,
        flap_phase_deg=0.0,
        duration_s=20.0,
        seed=0,
    )
    command = compare._worker_command(
        checkpoint=tmp_path / "model_50.pt",
        case_spec=compare._CASE_SPECS[1],
        output_dir=tmp_path / "output",
        args=args,
    )

    assert command[command.index("--template-id") + 1] == "5"
    assert command[command.index("--heading-deg") + 1] == "180.0"
    assert command[command.index("--slope-deg") + 1] == "12.0"
    assert command[command.index("--turn-sign") + 1] == "-1"
    assert command[command.index("--device") + 1] == "cpu"
    assert "--headless" in command


@pytest.mark.parametrize("template_id, second_event_time", [(5, 0.6), (7, 0.6)])
def test_summarize_trace_locates_second_event_and_roll_failure(
    template_id: int,
    second_event_time: float,
) -> None:
    rows = [
        _row(step, case_id="failure", template_id=template_id, failure=True)
        for step in range(10)
    ]

    summary = compare._summarize_trace(rows)

    assert summary["second_event_state_action"]["time_s"] == pytest.approx(second_event_time)
    assert summary["roll_limit_termination"] is True
    assert summary["reached_all_events"] is False
    assert summary["max_abs_roll_deg"] == pytest.approx(13.0)


def test_comparison_delta_uses_failure_minus_control() -> None:
    control = compare._summarize_trace(
        [_row(step, case_id="control", template_id=5, failure=False) for step in range(10)]
    )
    failure = compare._summarize_trace(
        [_row(step, case_id="failure", template_id=5, failure=True) for step in range(10)]
    )

    delta = compare._comparison_delta(control, failure)

    assert delta["max_abs_roll_delta_deg"] == pytest.approx(4.0)
    assert delta["failure_roll_limit_termination"] is True
    assert delta["control_reached_all_events"] is True

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/flapping_rl/pure_rl_backend_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("pure_rl_backend_diagnostics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostics
SPEC.loader.exec_module(diagnostics)


def _case(
    case_id: str,
    *,
    success: bool,
    task: str = "level",
    cause: str = "timeout",
    duration_s: float = 12.0,
    path_error_m: float = 0.1,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "backend_id": "gpu_implicit_candidate",
        "task": task,
        "success": success,
        "termination_cause": cause,
        "duration_s": duration_s,
        "path_error_m": path_error_m,
        "finite_metrics": True,
    }


def test_validate_case_rows_rejects_duplicate_ids_and_nonfinite_required_metrics() -> None:
    rows = [_case("a", success=True), _case("b", success=False)]
    validated = diagnostics.validate_case_rows(rows, backend_id="gpu_implicit_candidate")
    assert tuple(row["case_id"] for row in validated) == ("a", "b")

    with pytest.raises(ValueError, match="duplicate case_id"):
        diagnostics.validate_case_rows([rows[0], rows[0]], backend_id="gpu_implicit_candidate")

    bad = _case("bad", success=False)
    bad["path_error_m"] = float("nan")
    with pytest.raises(ValueError, match="path_error_m"):
        diagnostics.validate_case_rows([bad], backend_id="gpu_implicit_candidate")

    with pytest.raises(ValueError, match="backend_id"):
        diagnostics.validate_case_rows(rows, backend_id="cpu_native_authority")


def test_validate_trace_arrays_requires_aligned_time_and_four_channel_actions() -> None:
    trace = {
        "time_s": np.array([0.0, 1.0 / 60.0, 2.0 / 60.0]),
        "actions": np.zeros((3, 4), dtype=np.float32),
        "done": np.zeros(3, dtype=bool),
        "terminated": np.zeros(3, dtype=bool),
        "time_out": np.zeros(3, dtype=bool),
        "physical_state_valid": np.ones(3, dtype=bool),
        "height_error_m": np.zeros(3, dtype=np.float32),
    }
    diagnostics.validate_trace_arrays(trace)

    with pytest.raises(ValueError, match="four channels"):
        diagnostics.validate_trace_arrays({**trace, "actions": np.zeros((3, 3))})
    with pytest.raises(ValueError, match="aligned"):
        diagnostics.validate_trace_arrays({**trace, "height_error_m": np.zeros(2)})
    with pytest.raises(ValueError, match="finite"):
        diagnostics.validate_trace_arrays({**trace, "time_s": np.array([0.0, float("nan"), 0.1])})


def test_validate_trace_arrays_rejects_terminal_or_reset_mixed_physical_samples() -> None:
    trace = {
        "time_s": np.array([0.0, 1.0 / 60.0]),
        "actions": np.zeros((2, 4), dtype=np.float32),
        "done": np.array([False, True]),
        "terminated": np.array([False, False]),
        "time_out": np.array([False, True]),
        "physical_state_valid": np.array([True, False]),
    }

    with pytest.raises(ValueError, match="non-terminal physical states"):
        diagnostics.validate_trace_arrays(trace)


def _drive_trace(frequency_hz: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.arange(1, 481, dtype=np.float64) / 480.0
    target = np.deg2rad(30.0) * np.sin(2.0 * np.pi * frequency_hz * time_s)
    actual = np.column_stack((target, target))
    return time_s, target, actual


def test_drive_gate_summary_uses_frozen_full_and_steady_windows() -> None:
    time_s, target, actual = _drive_trace()
    actual[0] += np.deg2rad(1.5)

    summary = diagnostics.summarize_drive_gate_trace(
        time_s=time_s,
        target_position_rad=target,
        actual_position_rad=actual,
        frequency_hz=5.0,
    )

    assert summary["sample_count"] == 480
    assert summary["steady_sample_count"] == 385
    assert summary["full_max_abs_tracking_error_deg"] == pytest.approx(1.5)
    assert summary["steady_rms_tracking_error_deg"] == pytest.approx(0.0, abs=1.0e-10)
    assert summary["steady_p99_abs_tracking_error_deg"] == pytest.approx(0.0, abs=1.0e-10)
    assert summary["maximum_amplitude_error_percent"] == pytest.approx(0.0, abs=1.0e-10)
    assert summary["maximum_phase_time_error_ms"] == pytest.approx(0.0, abs=1.0e-10)
    assert summary["steady_wing_sync_rms_deg"] == pytest.approx(0.0, abs=1.0e-10)
    assert summary["gate_passed"] is True


def test_drive_gate_summary_fails_a_full_run_spike_over_two_degrees() -> None:
    time_s, target, actual = _drive_trace(frequency_hz=4.0)
    actual[0] += np.deg2rad(2.01)

    summary = diagnostics.summarize_drive_gate_trace(
        time_s=time_s,
        target_position_rad=target,
        actual_position_rad=actual,
        frequency_hz=4.0,
    )

    assert summary["full_max_abs_tracking_error_deg"] == pytest.approx(2.01)
    assert summary["gate_passed"] is False


def _paired_plant_trace(*, force_scale: float = 1.0) -> dict[str, np.ndarray]:
    time_s = np.arange(0, 121, dtype=np.float64) / 60.0
    sample_count = time_s.size
    quaternion = np.zeros((sample_count, 4), dtype=np.float64)
    quaternion[:, 0] = 1.0
    force = np.zeros((sample_count, 3), dtype=np.float64)
    force[:, 0] = 10.0 * force_scale
    moment = np.zeros((sample_count, 3), dtype=np.float64)
    moment[:, 1] = 0.2 * force_scale
    return {
        "time_s": time_s,
        "actions": np.zeros((sample_count, 4), dtype=np.float64),
        "root_position_m": np.zeros((sample_count, 3), dtype=np.float64),
        "root_quaternion_wxyz": quaternion,
        "root_linear_velocity_world_mps": np.zeros((sample_count, 3), dtype=np.float64),
        "root_angular_velocity_body_rad_s": np.zeros((sample_count, 3), dtype=np.float64),
        "requested_frequency_hz": np.full(sample_count, 4.0),
        "applied_frequency_hz": np.full(sample_count, 4.0),
        "actual_tail_angle_rad": np.zeros((sample_count, 3), dtype=np.float64),
        "wing_actual_position_rad": np.zeros((sample_count, 2), dtype=np.float64),
        "wing_tracking_error_rad": np.zeros((sample_count, 2), dtype=np.float64),
        "total_aero_force_body_n": force,
        "total_aero_moment_body_nm": moment,
    }


def test_paired_plant_comparison_passes_identical_action_replay() -> None:
    cpu = _paired_plant_trace()
    gpu = _paired_plant_trace()
    gpu["root_quaternion_wxyz"] *= -1.0

    comparison = diagnostics.compare_paired_plant_traces(cpu, gpu)

    assert comparison["common_sample_count"] == 121
    assert comparison["common_duration_s"] == pytest.approx(2.0)
    assert comparison["maximum_abs_action_delta"] == pytest.approx(0.0)
    assert comparison["force_integrated_vector_error_percent"] == pytest.approx(0.0)
    assert comparison["moment_integrated_vector_error_percent"] == pytest.approx(0.0)
    assert comparison["short_horizon_max_attitude_delta_deg"] == pytest.approx(0.0)
    assert comparison["gate_passed"] is True
    assert comparison["gate_failures"] == []


def test_paired_plant_comparison_fails_force_impulse_above_five_percent() -> None:
    cpu = _paired_plant_trace()
    gpu = _paired_plant_trace(force_scale=1.06)

    comparison = diagnostics.compare_paired_plant_traces(cpu, gpu)

    assert comparison["force_integrated_vector_error_percent"] == pytest.approx(6.0)
    assert comparison["force_waveform_relative_rms_percent"] == pytest.approx(6.0)
    assert "force_integrated_vector_error_percent" in comparison["gate_failures"]
    assert comparison["gate_passed"] is False


def _tail_servo_trace() -> dict[str, np.ndarray]:
    time_s = np.arange(25, dtype=np.float64) * 0.1
    levels = np.repeat(np.deg2rad([0.0, 20.0, 0.0, -20.0, 0.0]), 5)
    command = np.column_stack((levels, levels, levels))
    moment = np.ones((time_s.size, 3, 3), dtype=np.float64)
    moment[:, :, 1] += command
    return {
        "time_s": time_s,
        "tail_command_rad": command,
        "tail_actual_rad": command.copy(),
        "tail_moment_body_nm": moment,
    }


def test_paired_tail_servo_comparison_passes_identical_three_surface_traces() -> None:
    cpu = _tail_servo_trace()
    gpu = _tail_servo_trace()

    comparison = diagnostics.compare_paired_tail_servo_traces(
        cpu,
        gpu,
        aerodynamics_enabled=True,
    )

    assert comparison["surface_count"] == 3
    assert comparison["maximum_abs_command_delta_deg"] == pytest.approx(0.0)
    assert comparison["gate_passed"] is True
    assert comparison["gate_failures"] == []
    assert tuple(surface["name"] for surface in comparison["surfaces"]) == (
        "rudder",
        "left_elevon",
        "right_elevon",
    )
    assert all(surface["transition_count"] == 4 for surface in comparison["surfaces"])
    assert all(surface["actual_angle_rms_delta_deg"] == pytest.approx(0.0) for surface in comparison["surfaces"])
    assert all(surface["tail_moment_waveform_rms_percent"] == pytest.approx(0.0) for surface in comparison["surfaces"])


def test_paired_tail_servo_comparison_fails_angle_and_loaded_moment_thresholds() -> None:
    cpu = _tail_servo_trace()
    gpu = _tail_servo_trace()
    gpu["tail_actual_rad"][:, 0] += np.deg2rad(0.6)
    gpu["tail_moment_body_nm"][:, 0, :] *= 1.2

    comparison = diagnostics.compare_paired_tail_servo_traces(
        cpu,
        gpu,
        aerodynamics_enabled=True,
    )

    rudder = comparison["surfaces"][0]
    assert rudder["actual_angle_max_delta_deg"] == pytest.approx(0.6)
    assert rudder["tail_moment_waveform_rms_percent"] == pytest.approx(20.0)
    assert "rudder.actual_angle_max_delta_deg" in comparison["gate_failures"]
    assert "rudder.tail_moment_waveform_rms_percent" in comparison["gate_failures"]
    assert comparison["gate_passed"] is False


def test_paired_tail_servo_comparison_rejects_command_mismatch() -> None:
    cpu = _tail_servo_trace()
    gpu = _tail_servo_trace()
    gpu["tail_command_rad"][6, 1] += 1.0e-3

    with pytest.raises(ValueError, match="commands must match"):
        diagnostics.compare_paired_tail_servo_traces(
            cpu,
            gpu,
            aerodynamics_enabled=False,
        )


def test_diagnostic_selection_keeps_all_small_failure_sets_and_at_most_two_worst_successes() -> None:
    rows = [
        _case("fail_level", success=False, cause="tilt", duration_s=2.0),
        _case("fail_climb", success=False, task="climb", cause="height", duration_s=3.0),
        _case("ok_best", success=True, path_error_m=0.1),
        _case("ok_worst", success=True, path_error_m=0.8),
        _case("ok_second", success=True, path_error_m=0.6),
    ]

    selected = diagnostics.select_diagnostic_cases(rows, maximum_cases=4, maximum_successes=2)

    assert set(selected[:2]) == {"fail_level", "fail_climb"}
    assert selected[2:] == ("ok_worst", "ok_second")


def test_diagnostic_selection_covers_causes_and_tasks_before_filling_large_failure_set() -> None:
    rows = [
        _case("tilt_level", success=False, cause="tilt", task="level", duration_s=1.0),
        _case("height_climb", success=False, cause="height", task="climb", duration_s=2.0),
        _case("cross_descent", success=False, cause="cross_track", task="descent", duration_s=3.0),
        _case("ground_level", success=False, cause="ground", task="level", duration_s=4.0),
    ]
    rows.extend(
        _case(f"extra_{index}", success=False, cause="height", task="level", duration_s=5.0 + index)
        for index in range(8)
    )

    first = diagnostics.select_diagnostic_cases(rows, maximum_cases=8)
    second = diagnostics.select_diagnostic_cases(list(reversed(rows)), maximum_cases=8)

    assert first == second
    selected_rows = {row["case_id"]: row for row in rows if row["case_id"] in first}
    assert {row["termination_cause"] for row in selected_rows.values()} >= {
        "tilt",
        "height",
        "cross_track",
        "ground",
    }
    assert {row["task"] for row in selected_rows.values()} >= {"level", "climb", "descent"}


@pytest.mark.parametrize(
    ("case_updates", "comparison", "expected_primary"),
    [
        ({"finite_metrics": False}, {}, "runtime_or_nonfinite"),
        ({}, {"replay_outcome": "passed"}, "policy_feedback_shift"),
        (
            {},
            {"replay_outcome": "failed", "max_abs_wing_tracking_error_deg": 1.5},
            "wing_drive_tracking",
        ),
        (
            {},
            {"replay_outcome": "failed", "wrench_relative_rms_delta": 0.4},
            "aero_load_shift",
        ),
        ({"frequency_limit_fraction": 0.4}, {}, "actuator_or_governor_limit"),
        ({"termination_cause": "tilt", "max_tilt_deg": 80.0}, {}, "flight_state_instability"),
        ({"termination_cause": "height"}, {}, "path_tracking_only"),
        ({"termination_cause": "unknown"}, {}, "unattributed"),
    ],
)
def test_failure_evidence_reports_ranked_noncausal_attribution(
    case_updates: dict[str, object],
    comparison: dict[str, object],
    expected_primary: str,
) -> None:
    row = _case("failed", success=False, cause="unknown")
    row.update(case_updates)

    evidence = diagnostics.rank_failure_evidence(row, comparison=comparison)

    assert evidence[0]["label"] == expected_primary
    assert evidence[0]["causal_claim"] is False
    assert evidence[0]["closed_loop_outcome"] == "failed"
    assert "raw_evidence" in evidence[0]
    assert "first_event_time_s" in evidence[0]

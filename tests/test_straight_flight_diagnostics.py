from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "flapping_rl" / "diagnose_straight_flight_checkpoint.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("straight_flight_diagnostics", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_rollout_records_counts_termination_reasons_and_tracking_errors() -> None:
    diag = _load_module()
    step_rows = [
        {"episode": 0, "z_err_m": 0.2, "vx_err_mps": -0.4, "tilt_deg": 5.0, "cmd_a0": 0.1, "cmd_a1": 0.0},
        {"episode": 0, "z_err_m": -0.4, "vx_err_mps": 0.2, "tilt_deg": 8.0, "cmd_a0": 0.2, "cmd_a1": 0.0},
        {"episode": 1, "z_err_m": 1.0, "vx_err_mps": -2.0, "tilt_deg": 70.0, "cmd_a0": 0.99, "cmd_a1": -0.96},
    ]
    episode_rows = [
        {"episode": 0, "steps": 2, "termination_reason": "timeout"},
        {"episode": 1, "steps": 1, "termination_reason": "tilt"},
    ]

    summary = diag.summarize_rollout_records(step_rows, episode_rows, action_saturation_threshold=0.95)

    assert summary["episodes"] == 2
    assert summary["timeout_rate"] == 0.5
    assert summary["tilt_rate"] == 0.5
    assert summary["mean_episode_steps"] == 1.5
    assert summary["mean_abs_z_err_m"] == 0.5333333333333333
    assert summary["mean_abs_vx_err_mps"] == 0.8666666666666667
    assert summary["max_tilt_deg"] == 70.0
    assert summary["cmd_a0_saturation_rate"] == 1 / 3
    assert summary["cmd_a1_saturation_rate"] == 1 / 3
    assert summary["any_action_saturation_rate"] == 1 / 3


def test_infer_termination_reason_prioritizes_timeout_then_failure_modes() -> None:
    diag = _load_module()

    assert diag.infer_termination_reason(True, False, 0.0, 0.0, 0.0, 0.05, 75.0, 20.0) == "timeout"
    assert diag.infer_termination_reason(False, True, 0.01, 10.0, 0.0, 0.05, 75.0, 20.0) == "ground"
    assert diag.infer_termination_reason(False, True, 2.0, 80.0, 0.0, 0.05, 75.0, 20.0) == "tilt"
    assert diag.infer_termination_reason(False, True, 2.0, 10.0, 25.0, 0.05, 75.0, 20.0) == "lateral"
    assert diag.infer_termination_reason(False, True, 2.0, 10.0, 0.0, 0.05, 75.0, 20.0) == "terminated"


def test_refine_terminated_reason_uses_near_threshold_episode_history() -> None:
    diag = _load_module()

    rows = [
        {"height_m": 10.0, "tilt_deg": 8.0, "y_m": 0.0},
        {"height_m": 9.5, "tilt_deg": 74.2, "y_m": 0.1},
    ]

    reason = diag.refine_terminated_reason_from_episode_rows(
        "terminated",
        rows,
        ground_height_m=0.05,
        terminate_tilt_deg=75.0,
        terminate_abs_y_m=20.0,
    )

    assert reason == "tilt"

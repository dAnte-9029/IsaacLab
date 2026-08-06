from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/flapping_rl/pure_rl_eval_common.py"
SPEC = importlib.util.spec_from_file_location("pure_rl_eval_common", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pure_rl_eval_common = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pure_rl_eval_common
SPEC.loader.exec_module(pure_rl_eval_common)


def _nominal_step_metrics(count: int = 4) -> dict[str, list[float]]:
    return {
        "cross_track_error_m": [0.1] * count,
        "height_error_m": [0.2] * count,
        "along_track_progress_m": [float(index + 1) for index in range(count)],
        "along_track_velocity_mps": [1.0] * count,
        "tilt_rad": [0.1] * count,
        "angular_rate_rad_s": [0.2] * count,
        "actual_flap_frequency_hz": [2.5] * count,
        "frequency_limit_active": [0.0] * count,
        "tail_limit_active": [0.0] * count,
        "normalized_action_delta": [0.05] * count,
    }


def test_allocate_episode_quotas_is_exact_and_not_fast_failure_biased() -> None:
    assert pure_rl_eval_common.allocate_episode_quotas(16, 16) == (1,) * 16
    assert pure_rl_eval_common.allocate_episode_quotas(18, 16) == (2, 2) + (1,) * 14
    assert sum(pure_rl_eval_common.allocate_episode_quotas(5, 16)) == 5


def test_reset_schedule_contract_accepts_repeated_grid_and_rejects_drift() -> None:
    headings = torch.tensor([0.0, 1.0, 0.0, 1.0])
    phases = torch.tensor([0.5, 1.5, 0.5, 1.5])
    pure_rl_eval_common.assert_pure_rl_reset_schedule(
        actual_heading_rad=headings,
        actual_flap_phase_rad=phases,
        expected_heading_schedule_rad=(0.0, 1.0),
        expected_flap_phase_schedule_rad=(0.5, 1.5),
    )

    with pytest.raises(RuntimeError, match="registered heading/phase schedule"):
        pure_rl_eval_common.assert_pure_rl_reset_schedule(
            actual_heading_rad=headings + torch.tensor([0.0, 0.0, 0.1, 0.0]),
            actual_flap_phase_rad=phases,
            expected_heading_schedule_rad=(0.0, 1.0),
            expected_flap_phase_schedule_rad=(0.5, 1.5),
        )


def test_episode_summary_uses_route_progress_without_target_speed_error() -> None:
    row = pure_rl_eval_common.summarize_pure_rl_episode(
        step_metrics=_nominal_step_metrics(),
        step_dt_s=0.25,
        terminated=False,
        time_out=True,
        termination_causes={"ground": False, "tilt": False, "cross_track": False, "height": False},
    )

    assert row["episode_duration_s"] == pytest.approx(1.0)
    assert row["along_track_progress_m"] == pytest.approx(4.0)
    assert row["mean_along_track_velocity_mps"] == pytest.approx(1.0)
    assert row["mean_abs_cross_track_error_m"] == pytest.approx(0.1)
    assert row["mean_abs_height_error_m"] == pytest.approx(0.2)
    assert row["time_out"] == 1
    assert "mean_abs_vx_err" not in row


def test_success_gate_requires_survival_path_progress_and_unsaturated_authority() -> None:
    passing = {
        "timeout_rate": 0.8125,
        "termination_rate": 0.1875,
        "mean_abs_cross_track_error_m": 0.49,
        "mean_abs_height_error_m": 0.49,
        "mean_along_track_progress_m": 0.1,
        "frequency_limit_fraction": 0.25,
        "tail_limit_fraction": 0.10,
    }
    assert pure_rl_eval_common.row_meets_pure_rl_success_gate(passing)

    for field_name, failing_value in (
        ("timeout_rate", 0.79),
        ("termination_rate", 0.21),
        ("mean_abs_cross_track_error_m", 0.51),
        ("mean_abs_height_error_m", 0.51),
        ("mean_along_track_progress_m", 0.0),
        ("frequency_limit_fraction", 0.26),
        ("tail_limit_fraction", 0.11),
    ):
        candidate = dict(passing)
        candidate[field_name] = failing_value
        assert not pure_rl_eval_common.row_meets_pure_rl_success_gate(candidate)


def test_score_rewards_survival_progress_and_path_quality_without_vx_target() -> None:
    better = {
        "timeout_rate": 1.0,
        "mean_along_track_progress_m": 12.0,
        "mean_abs_cross_track_error_m": 0.1,
        "mean_abs_height_error_m": 0.1,
        "mean_max_tilt_deg": 10.0,
        "frequency_limit_fraction": 0.0,
        "tail_limit_fraction": 0.0,
    }
    worse = dict(better)
    worse.update(
        timeout_rate=0.0,
        mean_along_track_progress_m=-1.0,
        mean_abs_cross_track_error_m=1.0,
    )

    assert pure_rl_eval_common.compute_pure_rl_score(better) > pure_rl_eval_common.compute_pure_rl_score(worse)
    assert "mean_abs_vx_err" not in better


def test_case_aggregation_marks_contract_gate_and_termination_causes(tmp_path: Path) -> None:
    episode = pure_rl_eval_common.summarize_pure_rl_episode(
        step_metrics=_nominal_step_metrics(),
        step_dt_s=3.0,
        terminated=False,
        time_out=True,
        termination_causes={"ground": False, "tilt": False, "cross_track": False, "height": False},
    )
    case = {
        "name": "nominal_nowind",
        "wind_enabled": False,
        "wind_xy_mps": (0.0, 0.0),
        "wind_ou_enabled": False,
        "straight_line_heading_schedule_rad": (0.0,) * 16,
    }

    row = pure_rl_eval_common.aggregate_pure_rl_case_row(
        checkpoint=tmp_path / "model_0.pt",
        ckpt_index=0,
        case=case,
        episode_rows=[episode] * 16,
    )

    assert row["evaluation_contract"] == "pure_rl_curriculum1_v1"
    assert row["episodes"] == 16
    assert row["heading_phase_pairs"] == 16
    assert row["ground_termination_rate"] == pytest.approx(0.0)
    assert row["success_gate_passed"] == 1
    assert row["score"] > 0.0

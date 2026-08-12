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
        "frequency_slew_hz_per_s": [-2.0, -1.0, 0.0, 1.0][:count],
        "frequency_governor_limited": [1.0, 0.0, 0.0, 1.0][:count],
    }


def test_allocate_episode_quotas_is_exact_and_not_fast_failure_biased() -> None:
    assert pure_rl_eval_common.allocate_episode_quotas(16, 16) == (1,) * 16
    assert pure_rl_eval_common.allocate_episode_quotas(18, 16) == (2, 2) + (1,) * 14
    assert sum(pure_rl_eval_common.allocate_episode_quotas(5, 16)) == 5


def test_longitudinal_task_mapping_keeps_c1_contract_distinct() -> None:
    assert pure_rl_eval_common.longitudinal_stage_for_task(
        "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-Direct-v0"
    ) == "c2a"
    assert pure_rl_eval_common.longitudinal_stage_for_task(
        pure_rl_eval_common.MEASURED_PURE_RL_TASK_ID
    ) is None
    assert pure_rl_eval_common.is_measured_pure_rl_task(
        "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2c-Direct-v0"
    )


def test_spatial_task_mapping_and_generic_curriculum_stage_are_explicit() -> None:
    for stage in ("c3a", "c3b", "c3c"):
        task = f"Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3{stage[-1]}-Direct-v0"
        assert pure_rl_eval_common.spatial_stage_for_task(task) == stage
        assert pure_rl_eval_common.curriculum_stage_for_task(task) == stage
        assert pure_rl_eval_common.longitudinal_stage_for_task(task) is None
        assert pure_rl_eval_common.pure_rl_backend_for_task(task) == "cpu_native_authority"
    c2 = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2c-Direct-v0"
    assert pure_rl_eval_common.curriculum_stage_for_task(c2) == "c2c"
    assert pure_rl_eval_common.curriculum_stage_for_task(pure_rl_eval_common.MEASURED_PURE_RL_TASK_ID) is None


def test_gpu_implicit_task_mapping_is_explicit_and_preserves_stage_identity() -> None:
    gpu_c1 = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuImplicit-Direct-v0"
    gpu_c2a = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuImplicit-Direct-v0"

    assert pure_rl_eval_common.is_measured_pure_rl_task(gpu_c1)
    assert pure_rl_eval_common.is_measured_pure_rl_task(gpu_c2a)
    assert pure_rl_eval_common.longitudinal_stage_for_task(gpu_c1) is None
    assert pure_rl_eval_common.longitudinal_stage_for_task(gpu_c2a) == "c2a"
    assert pure_rl_eval_common.pure_rl_backend_for_task(gpu_c1) == "gpu_implicit_candidate"
    assert pure_rl_eval_common.pure_rl_backend_for_task(gpu_c2a) == "gpu_implicit_candidate"
    phase_matched_c1 = (
        "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-GpuPhaseMatched-Direct-v0"
    )
    phase_matched_c2a = (
        "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-GpuPhaseMatched-Direct-v0"
    )
    assert pure_rl_eval_common.is_measured_pure_rl_task(phase_matched_c1)
    assert pure_rl_eval_common.is_measured_pure_rl_task(phase_matched_c2a)
    assert pure_rl_eval_common.longitudinal_stage_for_task(phase_matched_c1) is None
    assert pure_rl_eval_common.longitudinal_stage_for_task(phase_matched_c2a) == "c2a"
    assert pure_rl_eval_common.pure_rl_backend_for_task(phase_matched_c1) == "gpu_implicit_candidate"
    assert pure_rl_eval_common.pure_rl_backend_for_task(phase_matched_c2a) == "gpu_implicit_candidate"
    assert (
        pure_rl_eval_common.pure_rl_backend_for_task(pure_rl_eval_common.MEASURED_PURE_RL_TASK_ID)
        == "cpu_native_authority"
    )

    with pytest.raises(ValueError, match="Unknown measured PureRL task"):
        pure_rl_eval_common.pure_rl_backend_for_task("not-a-pure-rl-task")


def test_step_metrics_include_spatial_route_and_roll_telemetry() -> None:
    class Env:
        pass

    env = Env()
    for attribute in (
        "_eval_pure_rl_cross_track_error_m",
        "_eval_pure_rl_height_error_m",
        "_eval_pure_rl_along_track_progress_m",
        "_eval_pure_rl_along_track_velocity_mps",
        "_eval_pure_rl_tilt_rad",
        "_eval_pure_rl_angular_rate_rad_s",
        "_eval_pure_rl_actual_flap_frequency_hz",
        "_eval_pure_rl_frequency_limit_active",
        "_eval_pure_rl_tail_limit_active",
        "_eval_pure_rl_normalized_action_delta",
        "_eval_pure_rl_frequency_slew_hz_per_s",
        "_eval_pure_rl_frequency_governor_limited",
        "_eval_pure_rl_ground_termination",
        "_eval_pure_rl_tilt_termination",
        "_eval_pure_rl_cross_track_termination",
        "_eval_pure_rl_height_termination",
    ):
        setattr(env, attribute, torch.zeros(2))
    env._eval_pure_rl_lateral_normal_velocity_mps = torch.zeros(2)
    env._eval_pure_rl_vertical_normal_velocity_mps = torch.zeros(2)
    env._eval_pure_rl_active_slope_rad = torch.zeros(2)
    env._eval_pure_rl_active_curvature_rad_per_m = torch.zeros(2)
    env._eval_pure_rl_turn_activity = torch.zeros(2)
    env._eval_pure_rl_reached_all_events = torch.ones(2, dtype=torch.bool)
    env._eval_pure_rl_roll_limit_termination = torch.zeros(2, dtype=torch.bool)
    env._eval_pure_rl_abs_roll_rad = torch.tensor([0.1, 0.2])
    env._pure_rl_spatial_path = type(
        "Path",
        (),
        {
            "task_family_id": torch.tensor([3, 3]),
            "template_id": torch.tensor([10, 10]),
            "turn_sign": torch.tensor([-1.0, 1.0]),
            "vertical_sign": torch.tensor([-1.0, 1.0]),
            "peak_geometry_roll_rad": torch.tensor([0.1, 0.2]),
            "peak_slope_rad": torch.tensor([-0.1, 0.1]),
        },
    )()
    metrics = pure_rl_eval_common.read_pure_rl_step_metrics(env)
    for name in (
        "active_curvature_rad_per_m",
        "active_slope_rad",
        "turn_activity",
        "reached_all_events",
        "roll_limit_termination",
        "abs_roll_rad",
        "sampled_template_id",
        "sampled_turn_sign",
        "sampled_vertical_sign",
    ):
        assert name in metrics


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


def test_spatial_reset_schedule_contract_checks_path_geometry_and_direction() -> None:
    kwargs = {
        "actual_heading_rad": torch.tensor([0.0, 1.0]),
        "actual_flap_phase_rad": torch.tensor([0.5, 1.5]),
        "actual_template_id": torch.tensor([2, 10]),
        "actual_geometry_roll_rad": torch.deg2rad(torch.tensor([9.0, 16.0])),
        "actual_slope_rad": torch.deg2rad(torch.tensor([0.0, -3.0])),
        "actual_turn_sign": torch.tensor([-1.0, 1.0]),
        "expected_heading_schedule_rad": (0.0, 1.0),
        "expected_flap_phase_schedule_rad": (0.5, 1.5),
        "expected_template_schedule": (2, 10),
        "expected_geometry_roll_deg_schedule": (9.0, 16.0),
        "expected_slope_deg_schedule": (0.0, -3.0),
        "expected_turn_sign_schedule": (-1, 1),
    }
    pure_rl_eval_common.assert_pure_rl_spatial_reset_schedule(**kwargs)

    drifted = dict(kwargs)
    drifted["actual_geometry_roll_rad"] = kwargs["actual_geometry_roll_rad"] + torch.tensor([0.0, 0.1])
    with pytest.raises(RuntimeError, match="registered spatial path schedule"):
        pure_rl_eval_common.assert_pure_rl_spatial_reset_schedule(**drifted)


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
    assert row["mean_abs_frequency_slew_hz_per_s"] == pytest.approx(1.0)
    assert row["frequency_governor_limited_fraction"] == pytest.approx(0.5)
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

    assert row["evaluation_contract"] == "pure_rl_curriculum1_v2"
    assert row["episodes"] == 16
    assert row["heading_phase_pairs"] == 16
    assert row["ground_termination_rate"] == pytest.approx(0.0)
    assert row["success_gate_passed"] == 1
    assert row["score"] > 0.0

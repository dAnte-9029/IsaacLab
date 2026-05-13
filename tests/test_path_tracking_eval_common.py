from __future__ import annotations

import importlib.util
import math
import types
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "flapping_rl" / "path_tracking_eval_common.py"
SPEC = importlib.util.spec_from_file_location("path_tracking_eval_common", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
path_tracking_eval_common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(path_tracking_eval_common)


def test_resolve_eval_suite_defaults_path_tracking_task_to_estimated_suite() -> None:
    resolved = path_tracking_eval_common.resolve_eval_suite(
        "Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0",
        "straight_standard",
    )

    assert resolved == "path_tracking_estimated_nowind_v1"


def test_resolve_eval_suite_defaults_primitive_path_tracking_task_to_estimated_primitive_suite() -> None:
    resolved = path_tracking_eval_common.resolve_eval_suite(
        "Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0",
        "straight_standard",
    )

    assert resolved == "path_tracking_estimated_primitives_nowind_v1"


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
            "stalled": 0,
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
            "stalled": 1,
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
    assert math.isclose(row["stall_rate"], 0.5)
    assert math.isclose(row["mean_abs_lateral_error_m"], 0.25)
    assert math.isclose(row["mean_abs_height_error_m"], 0.15)
    assert math.isclose(row["mean_abs_align_error_deg"], 3.5)
    assert math.isclose(row["mean_p95_abs_lateral_error_m"], 0.5)
    assert math.isclose(row["mean_p95_abs_height_error_m"], 0.25)
    assert math.isclose(row["mean_p95_abs_align_error_deg"], 6.0)
    assert math.isclose(row["mean_final_progress_ratio"], 0.875)
    assert row["score"] > 0.0


def test_aggregate_case_row_emits_fixed_loiter_schema_for_non_loiter_case() -> None:
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
            "time_out": 0,
            "stalled": 0,
        }
    ]

    row = path_tracking_eval_common.aggregate_case_row(
        checkpoint=Path("/tmp/model_20.pt"),
        ckpt_index=20,
        case=case,
        episode_rows=episode_rows,
    )

    assert "mean_loiter_radial_error_m" in row
    assert "mean_loiter_progress_ratio" in row
    assert "quarter_turn_rate" in row
    assert math.isnan(float(row["mean_loiter_radial_error_m"]))
    assert math.isnan(float(row["mean_loiter_progress_ratio"]))
    assert math.isnan(float(row["quarter_turn_rate"]))


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


def test_compute_path_tracking_score_penalizes_low_progress_when_completion_matches() -> None:
    better = path_tracking_eval_common.compute_path_tracking_score(
        completion_rate=0.0,
        mean_final_progress_ratio=0.80,
        mean_abs_lateral_error_m=0.05,
        mean_abs_height_error_m=0.05,
        mean_abs_align_error_deg=1.0,
        termination_rate=0.0,
    )
    worse = path_tracking_eval_common.compute_path_tracking_score(
        completion_rate=0.0,
        mean_final_progress_ratio=0.02,
        mean_abs_lateral_error_m=0.05,
        mean_abs_height_error_m=0.05,
        mean_abs_align_error_deg=1.0,
        termination_rate=0.0,
    )

    assert better > worse


def test_compute_path_tracking_score_rejects_near_zero_progress_timeouts() -> None:
    stalled_timeout = path_tracking_eval_common.compute_path_tracking_score(
        completion_rate=0.0,
        mean_final_progress_ratio=0.004,
        mean_abs_lateral_error_m=0.00003,
        mean_abs_height_error_m=0.001,
        mean_abs_align_error_deg=0.03,
        termination_rate=0.0,
    )
    meaningful_progress = path_tracking_eval_common.compute_path_tracking_score(
        completion_rate=0.0,
        mean_final_progress_ratio=0.31,
        mean_abs_lateral_error_m=0.36,
        mean_abs_height_error_m=0.35,
        mean_abs_align_error_deg=5.3,
        termination_rate=0.5,
    )

    assert meaningful_progress > stalled_timeout


def test_path_tracking_success_gate_rejects_near_zero_progress_timeout() -> None:
    passed = path_tracking_eval_common.row_meets_path_tracking_success_gate(
        {
            "completion_rate": 0.0,
            "mean_final_progress_ratio": 0.004,
            "termination_rate": 0.0,
            "timeout_rate": 1.0,
        }
    )

    assert passed is False


def test_aggregate_case_row_sets_success_gate_flag() -> None:
    case = {
        "name": "straight_nowind",
        "wind_enabled": False,
        "wind_xy_mps": (0.0, 0.0),
        "wind_ou_enabled": False,
        "wind_ou_sigma_xy_mps": (0.0, 0.0),
    }
    episode_rows = [
        {
            "mean_abs_lateral_error_m": 0.10,
            "mean_abs_height_error_m": 0.10,
            "mean_abs_align_error_deg": 2.0,
            "p95_abs_lateral_error_m": 0.15,
            "p95_abs_height_error_m": 0.15,
            "p95_abs_align_error_deg": 3.0,
            "final_progress_ratio": 0.95,
            "completed": 1,
            "terminated": 0,
            "time_out": 0,
        },
        {
            "mean_abs_lateral_error_m": 0.12,
            "mean_abs_height_error_m": 0.11,
            "mean_abs_align_error_deg": 2.5,
            "p95_abs_lateral_error_m": 0.16,
            "p95_abs_height_error_m": 0.16,
            "p95_abs_align_error_deg": 3.5,
            "final_progress_ratio": 0.92,
            "completed": 1,
            "terminated": 0,
            "time_out": 0,
        },
    ]

    row = path_tracking_eval_common.aggregate_case_row(
        checkpoint=Path("/tmp/model_100.pt"),
        ckpt_index=100,
        case=case,
        episode_rows=episode_rows,
    )

    assert int(row["success_gate_passed"]) == 1


def test_aggregate_suite_row_sets_success_gate_flag() -> None:
    case_rows = [
        {
            "checkpoint": "/tmp/model_100.pt",
            "case": "straight_nowind",
            "ckpt_index": 100,
            "episodes": 2,
            "mean_abs_lateral_error_m": 0.2,
            "mean_abs_height_error_m": 0.2,
            "mean_abs_align_error_deg": 3.0,
            "mean_p95_abs_lateral_error_m": 0.3,
            "mean_p95_abs_height_error_m": 0.3,
            "mean_p95_abs_align_error_deg": 5.0,
            "mean_final_progress_ratio": 0.9,
            "completion_rate": 1.0,
            "termination_rate": 0.0,
            "timeout_rate": 0.0,
            "stall_rate": 0.0,
            "wind_enabled": 0,
            "wind_x_mps": 0.0,
            "wind_y_mps": 0.0,
            "wind_ou_enabled": 0,
            "wind_ou_sigma_x_mps": 0.0,
            "wind_ou_sigma_y_mps": 0.0,
            "score": 10.0,
            "success_gate_passed": 1,
        },
        {
            "checkpoint": "/tmp/model_100.pt",
            "case": "loiter_nowind",
            "ckpt_index": 100,
            "episodes": 2,
            "mean_abs_lateral_error_m": 0.3,
            "mean_abs_height_error_m": 0.3,
            "mean_abs_align_error_deg": 4.0,
            "mean_p95_abs_lateral_error_m": 0.4,
            "mean_p95_abs_height_error_m": 0.4,
            "mean_p95_abs_align_error_deg": 6.0,
            "mean_final_progress_ratio": 0.88,
            "completion_rate": 1.0,
            "termination_rate": 0.0,
            "timeout_rate": 0.0,
            "stall_rate": 0.5,
            "wind_enabled": 0,
            "wind_x_mps": 0.0,
            "wind_y_mps": 0.0,
            "wind_ou_enabled": 0,
            "wind_ou_sigma_x_mps": 0.0,
            "wind_ou_sigma_y_mps": 0.0,
            "score": 9.0,
            "success_gate_passed": 1,
        },
    ]

    suite_row = path_tracking_eval_common.aggregate_suite_row(case_rows)

    assert int(suite_row["success_gate_passed"]) == 1


def test_compute_eval_episode_length_extends_single_loiter_case() -> None:
    cfg = types.SimpleNamespace(
        episode_length_s=18.0,
        vx_cmd=7.0,
        freeze_steps_after_reset=240,
        loiter_radius_m=20.0,
        loiter_turns=1.0,
        sim=types.SimpleNamespace(dt=1.0 / 240.0),
    )
    case = {
        "name": "loiter_primitive_nowind",
        "mission_allow_straight": False,
        "mission_allow_turn": False,
        "mission_allow_loiter": True,
    }

    episode_length_s = path_tracking_eval_common.compute_path_tracking_eval_episode_length_s(
        case,
        cfg,
        vx_cmd=None,
    )

    assert episode_length_s > 20.0


def test_apply_eval_case_to_cfg_extends_loiter_eval_horizon() -> None:
    cfg = types.SimpleNamespace(
        randomize_commands=True,
        vx_cmd=7.0,
        height_cmd=10.0,
        teacher_guidance_enabled=True,
        wind_curriculum_enabled=True,
        mission_curriculum_enabled=True,
        wind_enabled=False,
        randomize_wind=True,
        wind_xy_mps=(0.0, 0.0),
        wind_x_range_mps=(0.0, 0.0),
        wind_y_range_mps=(0.0, 0.0),
        wind_ou_enabled=False,
        wind_ou_tau_s=2.0,
        wind_ou_sigma_xy_mps=(0.0, 0.0),
        wind_ou_clip_to_range=True,
        episode_length_s=18.0,
        freeze_steps_after_reset=240,
        loiter_radius_m=20.0,
        loiter_turns=1.0,
        sim=types.SimpleNamespace(dt=1.0 / 240.0),
    )
    case = {
        "name": "loiter_primitive_nowind",
        "wind_enabled": False,
        "wind_xy_mps": (0.0, 0.0),
        "wind_ou_enabled": False,
        "wind_ou_tau_s": 2.0,
        "wind_ou_sigma_xy_mps": (0.0, 0.0),
        "mission_allow_straight": False,
        "mission_allow_turn": False,
        "mission_allow_loiter": True,
    }

    path_tracking_eval_common.apply_eval_case_to_cfg(case, cfg, vx_cmd=None, height_cmd=None)

    assert cfg.randomize_commands is False
    assert cfg.teacher_guidance_enabled is False
    assert cfg.wind_curriculum_enabled is False
    assert cfg.mission_curriculum_enabled is False
    assert cfg.episode_length_s > 20.0


def test_read_path_tracking_step_metrics_prefers_snapshot_buffers() -> None:
    env = types.SimpleNamespace(
        num_envs=2,
        _path_lateral_error_m=torch.tensor([99.0, 2.0], dtype=torch.float32),
        _path_height_error_m=torch.tensor([88.0, 3.0], dtype=torch.float32),
        _path_align_error_rad=torch.deg2rad(torch.tensor([77.0, 4.0], dtype=torch.float32)),
        _path_progress_s=torch.tensor([0.0, 2.0], dtype=torch.float32),
        _path_managers=[
            types.SimpleNamespace(total_length_m=10.0),
            types.SimpleNamespace(total_length_m=10.0),
        ],
        _eval_abs_lateral_error_m=torch.tensor([0.5, 0.25], dtype=torch.float32),
        _eval_abs_height_error_m=torch.tensor([0.4, 0.2], dtype=torch.float32),
        _eval_abs_align_error_deg=torch.tensor([6.0, 3.0], dtype=torch.float32),
        _eval_progress_ratio=torch.tensor([0.8, 0.3], dtype=torch.float32),
        _eval_stalled=torch.tensor([True, False]),
    )

    metrics = path_tracking_eval_common.read_path_tracking_step_metrics(env)

    assert torch.allclose(metrics["abs_lateral_error_m"], torch.tensor([0.5, 0.25]))
    assert torch.allclose(metrics["abs_height_error_m"], torch.tensor([0.4, 0.2]))
    assert torch.allclose(metrics["abs_align_error_deg"], torch.tensor([6.0, 3.0]))
    assert torch.allclose(metrics["progress_ratio"], torch.tensor([0.8, 0.3]))
    assert torch.equal(metrics["stalled"], torch.tensor([True, False]))


def test_read_path_tracking_step_metrics_falls_back_to_live_state() -> None:
    env = types.SimpleNamespace(
        num_envs=1,
        _path_lateral_error_m=torch.tensor([-2.0], dtype=torch.float32),
        _path_height_error_m=torch.tensor([1.5], dtype=torch.float32),
        _path_align_error_rad=torch.deg2rad(torch.tensor([30.0], dtype=torch.float32)),
        _path_progress_s=torch.tensor([4.0], dtype=torch.float32),
        _path_managers=[types.SimpleNamespace(total_length_m=10.0)],
        _stalled=torch.tensor([True]),
    )

    metrics = path_tracking_eval_common.read_path_tracking_step_metrics(env)

    assert torch.allclose(metrics["abs_lateral_error_m"], torch.tensor([2.0]))
    assert torch.allclose(metrics["abs_height_error_m"], torch.tensor([1.5]))
    assert torch.allclose(metrics["abs_align_error_deg"], torch.tensor([30.0]))
    assert torch.allclose(metrics["progress_ratio"], torch.tensor([0.4]))
    assert torch.equal(metrics["stalled"], torch.tensor([True]))


def test_compute_path_tracking_finish_masks_marks_completion_before_done() -> None:
    finish_mask, early_success_mask, completed_mask = path_tracking_eval_common.compute_path_tracking_finish_masks(
        progress_ratio=torch.tensor([1.0, 0.25, 0.4], dtype=torch.float32),
        dones=torch.tensor([0, 1, 0], dtype=torch.long),
        completion_ratio=0.98,
    )

    assert torch.equal(finish_mask, torch.tensor([True, True, False]))
    assert torch.equal(early_success_mask, torch.tensor([True, False, False]))
    assert torch.equal(completed_mask, torch.tensor([True, False, False]))


def test_summarize_path_tracking_episode_treats_completion_as_success() -> None:
    row = path_tracking_eval_common.summarize_path_tracking_episode(
        step_abs_lateral=[0.3, 0.1],
        step_abs_height=[0.2, 0.1],
        step_abs_align_deg=[6.0, 2.0],
        final_progress_ratio=1.0,
        completion_ratio=0.98,
        terminated=True,
        time_out=True,
        stalled=True,
    )

    assert row["completed"] == 1
    assert row["terminated"] == 0
    assert row["time_out"] == 0
    assert row["stalled"] == 0


def test_summarize_path_tracking_episode_reports_loiter_metrics() -> None:
    row = path_tracking_eval_common.summarize_path_tracking_episode(
        step_abs_lateral=[0.3, 0.1],
        step_abs_height=[0.2, 0.1],
        step_abs_align_deg=[6.0, 2.0],
        step_loiter_radial_error=[0.4, 0.2],
        step_loiter_progress_ratio=[0.1, 0.3],
        loiter_quarter_turn_complete=True,
        final_progress_ratio=0.35,
        completion_ratio=0.98,
        terminated=False,
        time_out=True,
        stalled=False,
    )

    assert math.isclose(float(row["mean_loiter_radial_error_m"]), 0.3)
    assert math.isclose(float(row["mean_loiter_progress_ratio"]), 0.2)
    assert int(row["quarter_turn_completed"]) == 1

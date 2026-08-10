from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path


def _load_watch_and_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "flapping_rl" / "watch_and_eval.py"
    spec = importlib.util.spec_from_file_location("watch_and_eval_test_module", module_path)
    assert spec is not None and spec.loader is not None

    isaaclab_module = types.ModuleType("isaaclab")
    app_module = types.ModuleType("isaaclab.app")

    class _FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", type=str, default="cpu")
            parser.add_argument("--headless", action="store_true")

    app_module.AppLauncher = _FakeAppLauncher
    isaaclab_module.app = app_module

    old_isaaclab = sys.modules.get("isaaclab")
    old_app = sys.modules.get("isaaclab.app")
    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.app"] = app_module

    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if old_isaaclab is None:
            sys.modules.pop("isaaclab", None)
        else:
            sys.modules["isaaclab"] = old_isaaclab
        if old_app is None:
            sys.modules.pop("isaaclab.app", None)
        else:
            sys.modules["isaaclab.app"] = old_app

    return module


def test_watch_and_eval_parser_accepts_truth_nowind_suite(monkeypatch) -> None:
    watch_and_eval = _load_watch_and_eval_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch_and_eval.py",
            "--task",
            "Isaac-FlappingBot-GenericPathTracking-Direct-v0",
            "--log_dir",
            "logs/dummy_run",
            "--eval_suite",
            "path_tracking_truth_nowind_v1",
        ],
    )

    args = watch_and_eval._parse_args()

    assert args.eval_suite == "path_tracking_truth_nowind_v1"


def test_watch_and_eval_parser_accepts_truth_primitives_nowind_suite(monkeypatch) -> None:
    watch_and_eval = _load_watch_and_eval_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch_and_eval.py",
            "--task",
            "Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0",
            "--log_dir",
            "logs/dummy_run",
            "--eval_suite",
            "path_tracking_truth_primitives_nowind_v1",
        ],
    )

    args = watch_and_eval._parse_args()

    assert args.eval_suite == "path_tracking_truth_primitives_nowind_v1"


def test_watch_and_eval_resolves_path_tracking_task_to_estimated_suite() -> None:
    watch_and_eval = _load_watch_and_eval_module()

    resolved = watch_and_eval._resolve_eval_suite(
        "Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0",
        "straight_standard",
    )

    assert resolved == "path_tracking_estimated_nowind_v1"


def test_watch_and_eval_resolves_primitive_path_tracking_task_to_estimated_primitive_suite() -> None:
    watch_and_eval = _load_watch_and_eval_module()

    resolved = watch_and_eval._resolve_eval_suite(
        "Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0",
        "straight_standard",
    )

    assert resolved == "path_tracking_estimated_primitives_nowind_v1"


def test_watch_and_eval_resolves_measured_pure_rl_to_fixed_grid_suite() -> None:
    watch_and_eval = _load_watch_and_eval_module()

    resolved = watch_and_eval._resolve_eval_suite(
        "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0",
        "straight_standard",
    )
    shape = watch_and_eval._resolve_eval_shape(
        "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0",
        resolved,
        num_envs=None,
        episodes=None,
    )

    assert resolved == "pure_rl_curriculum1_nowind_v2"
    assert shape == (16, 16)


def test_watch_and_eval_resolves_longitudinal_task_to_stage_grid_and_shape() -> None:
    watch_and_eval = _load_watch_and_eval_module()
    task = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2a-Direct-v0"
    suite = watch_and_eval._resolve_eval_suite(task, "straight_standard")

    assert suite == "pure_rl_longitudinal_c2a_v1"
    assert watch_and_eval._resolve_eval_shape(task, suite, num_envs=None, episodes=None) == (80, 80)


def test_watch_and_eval_applies_longitudinal_fixed_schedules() -> None:
    watch_and_eval = _load_watch_and_eval_module()
    cfg = types.SimpleNamespace(
        wind_enabled=True,
        wind_xy_mps=(1.0, 1.0),
        wind_ou_enabled=True,
        wind_ou_tau_s=1.0,
        wind_ou_sigma_xy_mps=(1.0, 1.0),
        randomize_wind=True,
        randomize_straight_line_heading=True,
        randomize_flap_phase_at_reset=True,
        pure_rl_eval_heading_schedule_rad=None,
        pure_rl_eval_flap_phase_schedule_rad=None,
        pure_rl_eval_longitudinal_task_schedule=None,
        pure_rl_eval_longitudinal_slope_deg_schedule=None,
        pure_rl_eval_entry_length_m_schedule=None,
        pure_rl_eval_slope_length_m_schedule=None,
    )
    case = {
        "name": "c2a_promotion_grid",
        "wind_enabled": False,
        "wind_xy_mps": (0.0, 0.0),
        "wind_ou_enabled": False,
        "wind_ou_tau_s": 2.0,
        "wind_ou_sigma_xy_mps": (0.0, 0.0),
        "straight_line_heading_schedule_rad": (0.0, 1.0),
        "flap_phase_schedule_rad": (0.5, 1.5),
        "longitudinal_task_schedule": (1, 2),
        "longitudinal_slope_deg_schedule": (4.0, -4.0),
        "longitudinal_entry_length_m_schedule": (17.5, 17.5),
        "longitudinal_slope_length_m_schedule": (25.0, 25.0),
    }

    watch_and_eval._apply_eval_case_to_cfg(case, cfg, vx_cmd=None, height_cmd=None)

    assert cfg.pure_rl_eval_longitudinal_task_schedule == (1, 2)
    assert cfg.pure_rl_eval_longitudinal_slope_deg_schedule == (4.0, -4.0)
    assert cfg.pure_rl_eval_entry_length_m_schedule == (17.5, 17.5)
    assert cfg.pure_rl_eval_slope_length_m_schedule == (25.0, 25.0)


def test_watch_and_eval_applies_pure_rl_fixed_heading_phase_schedule() -> None:
    watch_and_eval = _load_watch_and_eval_module()
    cfg = types.SimpleNamespace(
        randomize_commands=True,
        teacher_guidance_enabled=True,
        wind_curriculum_enabled=True,
        wind_enabled=True,
        randomize_wind=True,
        wind_xy_mps=(1.0, 1.0),
        wind_x_range_mps=(1.0, 1.0),
        wind_y_range_mps=(1.0, 1.0),
        wind_ou_enabled=True,
        wind_ou_tau_s=1.0,
        wind_ou_sigma_xy_mps=(1.0, 1.0),
        wind_ou_clip_to_range=True,
        randomize_straight_line_heading=True,
        randomize_flap_phase_at_reset=True,
        pure_rl_eval_heading_schedule_rad=None,
        pure_rl_eval_flap_phase_schedule_rad=None,
    )
    case = {
        "name": "fixed_grid",
        "wind_enabled": False,
        "wind_xy_mps": (0.0, 0.0),
        "wind_ou_enabled": False,
        "wind_ou_tau_s": 2.0,
        "wind_ou_sigma_xy_mps": (0.0, 0.0),
        "straight_line_heading_schedule_rad": (0.0, 1.0),
        "flap_phase_schedule_rad": (0.5, 1.5),
    }

    watch_and_eval._apply_eval_case_to_cfg(case, cfg, vx_cmd=None, height_cmd=None)

    assert cfg.randomize_straight_line_heading is False
    assert cfg.randomize_flap_phase_at_reset is False
    assert cfg.pure_rl_eval_heading_schedule_rad == (0.0, 1.0)
    assert cfg.pure_rl_eval_flap_phase_schedule_rad == (0.5, 1.5)


def test_watch_and_eval_applies_path_tracking_mission_overrides() -> None:
    watch_and_eval = _load_watch_and_eval_module()

    cfg = types.SimpleNamespace(
        randomize_commands=True,
        vx_cmd=0.0,
        height_cmd=0.0,
        teacher_guidance_enabled=True,
        wind_curriculum_enabled=True,
        wind_enabled=True,
        randomize_wind=True,
        wind_xy_mps=(1.0, 1.0),
        wind_x_range_mps=(1.0, 1.0),
        wind_y_range_mps=(1.0, 1.0),
        wind_ou_enabled=True,
        wind_ou_tau_s=1.0,
        wind_ou_sigma_xy_mps=(1.0, 1.0),
        wind_ou_clip_to_range=True,
        mission_seed=0,
        mission_increment_seed_per_reset=True,
        mission_num_segments_min=2,
        mission_num_segments_max=4,
        mission_allow_straight=True,
        mission_allow_turn=True,
        mission_allow_loiter=True,
        mission_allow_climb_on_straight=True,
    )
    case = {
        "name": "straight_nowind",
        "wind_enabled": False,
        "wind_xy_mps": (0.0, 0.0),
        "wind_ou_enabled": False,
        "wind_ou_tau_s": 2.0,
        "wind_ou_sigma_xy_mps": (0.0, 0.0),
        "mission_seed": 101,
        "mission_increment_seed_per_reset": False,
        "mission_num_segments_min": 1,
        "mission_num_segments_max": 1,
        "mission_allow_straight": True,
        "mission_allow_turn": False,
        "mission_allow_loiter": False,
        "mission_allow_climb_on_straight": False,
    }

    watch_and_eval._apply_eval_case_to_cfg(case, cfg, vx_cmd=None, height_cmd=None)

    assert cfg.wind_enabled is False
    assert cfg.wind_xy_mps == (0.0, 0.0)
    assert cfg.mission_seed == 101
    assert cfg.mission_increment_seed_per_reset is False
    assert cfg.mission_num_segments_min == 1
    assert cfg.mission_num_segments_max == 1
    assert cfg.mission_allow_straight is True
    assert cfg.mission_allow_turn is False
    assert cfg.mission_allow_loiter is False


def test_watch_and_eval_scores_path_tracking_rows_with_completion_priority() -> None:
    watch_and_eval = _load_watch_and_eval_module()

    better = watch_and_eval._score_row(
        {
            "completion_rate": 0.9,
            "mean_final_progress_ratio": 0.95,
            "mean_abs_lateral_error_m": 0.2,
            "mean_abs_height_error_m": 0.1,
            "mean_abs_align_error_deg": 4.0,
            "termination_rate": 0.0,
        }
    )
    worse = watch_and_eval._score_row(
        {
            "completion_rate": 0.4,
            "mean_final_progress_ratio": 0.45,
            "mean_abs_lateral_error_m": 0.8,
            "mean_abs_height_error_m": 0.5,
            "mean_abs_align_error_deg": 12.0,
            "termination_rate": 0.4,
        }
    )

    assert better > worse


def test_watch_and_eval_scores_pure_rl_without_legacy_vx_error() -> None:
    watch_and_eval = _load_watch_and_eval_module()
    row = {
        "evaluation_contract": "pure_rl_curriculum1_v2",
        "timeout_rate": 1.0,
        "mean_along_track_progress_m": 12.0,
        "mean_abs_cross_track_error_m": 0.1,
        "mean_abs_height_error_m": 0.1,
        "mean_max_tilt_deg": 10.0,
        "frequency_limit_fraction": 0.0,
        "tail_limit_fraction": 0.0,
    }

    score = watch_and_eval._score_row(row)

    assert score > 0.0
    assert "mean_abs_vx_err" not in row


def test_watch_and_eval_scores_path_tracking_rows_with_progress_priority() -> None:
    watch_and_eval = _load_watch_and_eval_module()

    better = watch_and_eval._score_row(
        {
            "completion_rate": 0.0,
            "mean_final_progress_ratio": 0.8,
            "mean_abs_lateral_error_m": 0.05,
            "mean_abs_height_error_m": 0.05,
            "mean_abs_align_error_deg": 1.0,
            "termination_rate": 0.0,
        }
    )
    worse = watch_and_eval._score_row(
        {
            "completion_rate": 0.0,
            "mean_final_progress_ratio": 0.02,
            "mean_abs_lateral_error_m": 0.05,
            "mean_abs_height_error_m": 0.05,
            "mean_abs_align_error_deg": 1.0,
            "termination_rate": 0.0,
        }
    )

    assert better > worse


def test_watch_and_eval_exposes_path_tracking_success_gate() -> None:
    watch_and_eval = _load_watch_and_eval_module()

    passed = watch_and_eval.row_meets_path_tracking_success_gate(
        {
            "completion_rate": 0.0,
            "mean_final_progress_ratio": 0.004,
            "termination_rate": 0.0,
            "timeout_rate": 1.0,
        }
    )

    assert passed is False


def test_path_tracking_episode_rows_include_stalled_flag_for_eval_aggregation() -> None:
    watch_source = (Path(__file__).resolve().parents[1] / "scripts" / "flapping_rl" / "watch_and_eval.py").read_text()
    eval_source = (
        Path(__file__).resolve().parents[1] / "scripts" / "flapping_rl" / "eval_path_tracking_checkpoint.py"
    ).read_text()

    assert "summarize_path_tracking_episode(" in watch_source
    assert "stalled=bool(" in watch_source
    assert "summarize_path_tracking_episode(" in eval_source
    assert "stalled=bool(" in eval_source


def test_watch_and_eval_rewrites_summary_csv_when_new_columns_are_added(tmp_path) -> None:
    watch_and_eval = _load_watch_and_eval_module()

    summary_csv = tmp_path / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "case", "score"])
        writer.writeheader()
        writer.writerow({"checkpoint": "model_1.pt", "case": "suite", "score": "10.0"})

    watch_and_eval._append_summary_row(
        summary_csv,
        {
            "checkpoint": "model_2.pt",
            "case": "suite",
            "score": 11.0,
            "stall_rate": 0.25,
        },
    )

    with summary_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert "stall_rate" in rows[0]
    assert rows[0]["checkpoint"] == "model_1.pt"
    assert rows[0]["stall_rate"] == ""
    assert rows[1]["checkpoint"] == "model_2.pt"
    assert rows[1]["stall_rate"] == "0.25"

from __future__ import annotations

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


def test_watch_and_eval_resolves_path_tracking_task_to_truth_suite() -> None:
    watch_and_eval = _load_watch_and_eval_module()

    resolved = watch_and_eval._resolve_eval_suite(
        "Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0",
        "straight_standard",
    )

    assert resolved == "path_tracking_truth_nowind_v1"


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
            "mean_abs_lateral_error_m": 0.2,
            "mean_abs_height_error_m": 0.1,
            "mean_abs_align_error_deg": 4.0,
            "termination_rate": 0.0,
        }
    )
    worse = watch_and_eval._score_row(
        {
            "completion_rate": 0.4,
            "mean_abs_lateral_error_m": 0.8,
            "mean_abs_height_error_m": 0.5,
            "mean_abs_align_error_deg": 12.0,
            "termination_rate": 0.4,
        }
    )

    assert better > worse

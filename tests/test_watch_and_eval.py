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

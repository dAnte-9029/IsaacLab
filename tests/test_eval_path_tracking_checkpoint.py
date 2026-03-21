from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_eval_path_tracking_checkpoint_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "flapping_rl" / "eval_path_tracking_checkpoint.py"
    spec = importlib.util.spec_from_file_location("eval_path_tracking_checkpoint_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_path_tracking_parser_accepts_eval_suite_and_num_envs(monkeypatch) -> None:
    eval_path_tracking_checkpoint = _load_eval_path_tracking_checkpoint_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_path_tracking_checkpoint.py",
            "--task",
            "Isaac-FlappingBot-PathTracking-DeLaurier-TeacherRL-Direct-v0",
            "--checkpoint",
            "logs/dummy_run/model_20.pt",
            "--eval_suite",
            "path_tracking_truth_nowind_v1",
            "--num_envs",
            "2",
        ],
    )

    args = eval_path_tracking_checkpoint._parse_args()

    assert args.eval_suite == "path_tracking_truth_nowind_v1"
    assert args.num_envs == 2

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "flapping_bot"
    / "flapping_bot"
    / "direct"
    / "flapping_bot"
    / "path_tracking_env.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("path_tracking_env_reward_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_progress_is_rewarded_but_low_speed_is_penalized():
    module = _load_module()
    reward_fast = module._compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.4]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
    )
    reward_slow = module._compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.1]),
        airspeed=torch.tensor([2.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
    )
    assert reward_fast.item() > reward_slow.item()

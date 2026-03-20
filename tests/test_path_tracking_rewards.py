from __future__ import annotations

import torch

from flapping_bot.direct.flapping_bot.path_tracking_env import _compute_tracking_reward


def test_progress_is_rewarded_but_low_speed_is_penalized():
    reward_fast = _compute_tracking_reward(
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
    reward_slow = _compute_tracking_reward(
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

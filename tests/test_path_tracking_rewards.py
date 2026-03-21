from __future__ import annotations

import torch

from flapping_bot.direct.flapping_bot.path_tracking_env import _compute_tracking_reward


def test_tracking_reward_prefers_progress_with_small_errors():
    reward_better = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.4]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    reward_worse = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.1]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    assert reward_better.item() > reward_worse.item()


def test_tracking_reward_penalizes_lateral_and_height_error() -> None:
    reward_small_error = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    reward_large_error = _compute_tracking_reward(
        lateral_error=torch.tensor([2.0]),
        height_error=torch.tensor([1.5]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    assert reward_small_error.item() > reward_large_error.item()


def test_tracking_reward_penalizes_action_rate() -> None:
    reward_smooth = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    reward_aggressive = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.full((1, 4), 0.8),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    assert reward_smooth.item() > reward_aggressive.item()


def test_tracking_reward_applies_termination_penalty() -> None:
    reward_alive = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    reward_terminated = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([True]),
    )
    assert reward_alive.item() > reward_terminated.item()

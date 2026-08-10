"""Pure contracts for the bounded random-action reward runtime gate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flapping_bot.analysis.pure_rl_fixed_action_reward_validation import REWARD_TERM_NAMES
from flapping_bot.analysis.pure_rl_random_action_reward_validation import (
    BoundedRandomActionProcess,
    summarize_random_action_trace,
)


def test_random_action_process_is_deterministic_bounded_and_resettable() -> None:
    base = torch.zeros((8, 4), dtype=torch.float32)
    first = BoundedRandomActionProcess(base, seed=17)
    second = BoundedRandomActionProcess(base, seed=17)

    first_action = first.sample()
    second_action = second.sample()

    torch.testing.assert_close(first_action, second_action)
    assert float(torch.max(torch.abs(first_action))) <= 1.0
    assert float(torch.max(torch.abs(first_action[1::2, 1:4]))) <= 0.2
    reset_mask = torch.tensor([False, True, False, True, False, False, False, False])
    reset_action = torch.full_like(base, 0.37)
    first.reset(reset_mask, reset_action)
    torch.testing.assert_close(first._current[reset_mask], reset_action[reset_mask])


def _accepted_synthetic_trace() -> dict[str, np.ndarray]:
    steps = 20
    num_envs = 8
    environment = np.tile(np.arange(num_envs, dtype=np.int64), steps)
    sample_step = np.repeat(np.arange(steps, dtype=np.int64), num_envs)
    family = environment % 2
    requested = np.zeros((steps, num_envs, 4), dtype=np.float64)
    sweep = np.linspace(-1.0, 1.0, steps)
    for step_index, value in enumerate(sweep):
        requested[step_index, 0::2, :] = value
        requested[step_index, 1::2, 0] = 0.5 * value
        requested[step_index, 1::2, 1:4] = 0.4 * value
    applied = requested.copy()
    total = np.full(steps * num_envs, 0.5, dtype=np.float64)
    terminated = np.zeros((steps, num_envs), dtype=bool)
    terminated[4, 0] = True
    terminated[9, 0] = True
    terminated[7, 1] = True
    episode_index = np.zeros((steps, num_envs), dtype=np.int64)
    episode_index[5:, 0] = 1
    episode_index[10:, 0] = 2
    episode_index[8:, 1] = 1
    traces: dict[str, np.ndarray] = {
        "sample_step": sample_step,
        "environment_id": environment,
        "episode_index": episode_index.reshape(-1),
        "family_index": family,
        "returned_reward": total.copy(),
        "step_requested_action": requested,
        "step_applied_action": applied,
        "step_action_comparison_valid": np.ones((steps, num_envs), dtype=bool),
        "step_returned_reward": np.full((steps, num_envs), 0.5),
        "step_observation_max_abs": np.full((steps, num_envs), 2.0),
        "step_terminated": terminated,
        "step_truncated": np.zeros((steps, num_envs), dtype=bool),
        "step_telemetry_reward_total_mean": np.full(steps, 0.5),
        "step_telemetry_terminated_fraction": terminated.mean(axis=1),
        "reset_count_by_environment": np.array([2, 1, 0, 0, 0, 0, 0, 0]),
    }
    for name in REWARD_TERM_NAMES:
        traces[name] = total.copy() if name == "total" else np.zeros_like(total)
    traces["frequency_slew_penalty"] = np.full_like(total, 0.2)
    traces["tail_action_delta_penalty"] = np.full_like(total, 0.1)
    return traces


def test_random_trace_summary_accepts_finite_reset_and_coverage_contract() -> None:
    summary = summarize_random_action_trace(_accepted_synthetic_trace(), num_envs=8)

    assert summary["all_cases_accepted"] is True
    assert all(summary["gates"].values())
    assert summary["metrics"]["reset_event_count"] == 3
    assert summary["metrics"]["environments_reset_repeatedly"] == 1
    assert summary["metrics"]["post_reset_sample_count"] > 0


@pytest.mark.parametrize(
    ("mutation", "failed_gate"),
    (
        ("action_transfer", "applied_tail_actions_match_requested_actions"),
        ("reward", "reward_reconstruction_matches_environment"),
        ("telemetry", "reward_total_telemetry_matches_environment_mean"),
        ("observation", "observation_safety_clip_contract_holds"),
        ("timeout", "no_episode_timeout"),
    ),
)
def test_random_trace_summary_rejects_runtime_regressions(
    mutation: str,
    failed_gate: str,
) -> None:
    traces = _accepted_synthetic_trace()
    if mutation == "action_transfer":
        traces["step_applied_action"][0, 0, 1] += 0.1
    elif mutation == "reward":
        traces["returned_reward"][0] += 0.1
    elif mutation == "telemetry":
        traces["step_telemetry_reward_total_mean"][0] += 0.1
    elif mutation == "observation":
        traces["step_observation_max_abs"][0, 0] = 5.1
    elif mutation == "timeout":
        traces["step_truncated"][0, 0] = True

    summary = summarize_random_action_trace(traces, num_envs=8)

    assert summary["all_cases_accepted"] is False
    assert summary["gates"][failed_gate] is False


def test_random_action_process_fails_closed_on_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        BoundedRandomActionProcess(torch.zeros((8, 3)), seed=1)
    with pytest.raises(ValueError, match="correlated_alpha"):
        BoundedRandomActionProcess(torch.zeros((8, 4)), seed=1, correlated_alpha=0.0)
    process = BoundedRandomActionProcess(torch.zeros((8, 4)), seed=1)
    with pytest.raises(ValueError, match="reset_mask"):
        process.reset(torch.zeros(8), torch.zeros((8, 4)))

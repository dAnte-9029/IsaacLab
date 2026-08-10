"""Pure contracts for the fixed-action reward runtime gate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flapping_bot.analysis.pure_rl_fixed_action_reward_validation import (
    ACTION_FAMILY_NAMES,
    build_fixed_action_matrix,
    summarize_fixed_action_trace,
)


def test_fixed_action_matrix_repeats_eight_interpretable_families() -> None:
    base = torch.zeros((16, 4), dtype=torch.float64)
    base[:, 0] = 0.6
    base[:, 2:4] = -0.3

    action, family = build_fixed_action_matrix(
        base,
        minimum_frequency_hz=0.0,
        maximum_frequency_hz=5.0,
    )

    assert family.tolist() == list(range(8)) * 2
    assert len(ACTION_FAMILY_NAMES) == 8
    torch.testing.assert_close(action[0], base[0])
    assert action[1, 0].item() == pytest.approx(-1.0)
    assert action[2, 0].item() == pytest.approx(0.0)
    assert action[3, 0].item() == pytest.approx(1.0)
    assert action[4, 1].item() == pytest.approx(0.5)
    assert action[5, 2].item() == pytest.approx(0.5)
    assert action[6, 3].item() == pytest.approx(0.5)
    torch.testing.assert_close(action[7, 1:4], torch.full((3,), 0.9, dtype=torch.float64))


def _accepted_synthetic_trace() -> tuple[dict[str, np.ndarray], np.ndarray]:
    total_steps = 10
    environment_count = 8
    step = np.repeat(np.arange(total_steps, dtype=np.int64), environment_count)
    environment = np.tile(np.arange(environment_count, dtype=np.int64), total_steps)
    family = environment.copy()
    expected_action = np.zeros((environment_count, 4), dtype=np.float64)
    expected_action[1, 0] = -1.0
    expected_action[3, 0] = 1.0
    expected_action[4, 1] = 0.5
    expected_action[5, 2] = 0.5
    expected_action[6, 3] = 0.5
    expected_action[7, 1:4] = 0.9
    applied = expected_action[environment]
    actual_frequency = np.choose(family, [4.0, 0.0, 2.5, 5.0, 4.0, 4.0, 4.0, 4.0])
    flap_penalty = np.power(actual_frequency / 5.0, 3.0)
    frequency_delta = np.where((step == 0) & np.isin(family, (1, 3)), 0.25, 0.0)
    tail_delta = np.where((step == 0) & (family >= 4), 0.1, 0.0)
    tail_limit = np.where(family == 7, 0.25, 0.0)
    total = 0.8 - 0.04 * flap_penalty - 0.01 * frequency_delta - 0.01 * tail_delta - 0.02 * tail_limit
    traces: dict[str, np.ndarray] = {
        "sample_step": step,
        "environment_id": environment,
        "family_index": family,
        "applied_action": applied,
        "actual_frequency_hz": actual_frequency,
        "returned_reward": total.copy(),
        "path": np.ones_like(total),
        "progress": np.zeros_like(total),
        "velocity": np.ones_like(total),
        "roll": np.ones_like(total),
        "angular_rate": np.ones_like(total),
        "pitch_envelope_penalty": np.zeros_like(total),
        "flap_penalty": flap_penalty,
        "frequency_slew_penalty": frequency_delta,
        "tail_action_delta_penalty": tail_delta,
        "tail_action_limit_penalty": tail_limit,
        "total": total,
        "step_returned_reward_mean": np.full(total_steps, 0.5),
        "step_telemetry_reward_total_mean": np.full(total_steps, 0.5),
        "step_terminated_fraction": np.zeros(total_steps),
        "step_telemetry_terminated_fraction": np.zeros(total_steps),
    }
    return traces, expected_action


def test_trace_summary_accepts_consistent_fixed_action_contract() -> None:
    traces, expected_action = _accepted_synthetic_trace()

    summary = summarize_fixed_action_trace(
        traces,
        expected_action_by_environment=expected_action,
        total_policy_steps=10,
    )

    assert summary["all_cases_accepted"] is True
    assert all(summary["gates"].values())
    assert summary["metrics"]["later_step_combined_delta_penalty_max"] == pytest.approx(0.0)
    assert summary["action_families"][7]["tail_action_limit_penalty_mean"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("mutation", "failed_gate"),
    (
        ("reward", "reward_reconstruction_matches_environment"),
        ("later_delta", "constant_tail_actions_have_zero_later_delta_penalty"),
        ("action", "fixed_tail_actions_preserved"),
        ("termination", "termination_telemetry_matches_returned_done"),
    ),
)
def test_trace_summary_rejects_wiring_regressions(mutation: str, failed_gate: str) -> None:
    traces, expected_action = _accepted_synthetic_trace()
    if mutation == "reward":
        traces["returned_reward"][0] += 0.1
    elif mutation == "later_delta":
        traces["tail_action_delta_penalty"][8] = 0.1
    elif mutation == "action":
        traces["applied_action"][0, 1] = 0.2
    elif mutation == "termination":
        traces["step_telemetry_terminated_fraction"][0] = 0.25

    summary = summarize_fixed_action_trace(
        traces,
        expected_action_by_environment=expected_action,
        total_policy_steps=10,
    )

    assert summary["all_cases_accepted"] is False
    assert summary["gates"][failed_gate] is False


def test_action_builder_fails_closed_on_invalid_shape_or_range() -> None:
    with pytest.raises(ValueError, match="shape"):
        build_fixed_action_matrix(torch.zeros((8, 3)), minimum_frequency_hz=0.0, maximum_frequency_hz=5.0)
    with pytest.raises(ValueError, match="exceed"):
        build_fixed_action_matrix(torch.zeros((8, 4)), minimum_frequency_hz=5.0, maximum_frequency_hz=5.0)

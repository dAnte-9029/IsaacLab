"""Pure contracts for direct-action step-response validation."""

from __future__ import annotations

import numpy as np
import pytest

from flapping_bot.analysis.pure_rl_action_step_validation import (
    sequence_level,
    summarize_step_response,
)


def test_sequence_level_holds_each_value_and_clamps_after_end() -> None:
    levels = (0.0, 2.0, 5.0)

    assert sequence_level(0.0, dwell_s=0.5, levels=levels) == 0.0
    assert sequence_level(0.499, dwell_s=0.5, levels=levels) == 0.0
    assert sequence_level(0.5, dwell_s=0.5, levels=levels) == 2.0
    assert sequence_level(100.0, dwell_s=0.5, levels=levels) == 5.0
    with pytest.raises(ValueError, match="nonnegative"):
        sequence_level(-1.0e-3, dwell_s=0.5, levels=levels)


def test_step_summary_measures_up_and_down_transitions() -> None:
    time_s = np.arange(0.0, 2.0, 0.1)
    target = np.where(time_s < 0.5, 0.0, np.where(time_s < 1.5, 1.0, 0.0))
    actual = np.zeros_like(time_s)
    actual[5:15] = np.array([0.0, 0.1, 0.4, 0.75, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0])
    actual[15:] = np.array([1.0, 0.8, 0.4, 0.1, 0.0])

    summary = summarize_step_response(time_s, target, actual)

    assert summary["transition_count"] == 2
    up, down = summary["transitions"]
    assert up["from"] == 0.0
    assert up["to"] == 1.0
    assert up["rise_time_10_90_s"] == pytest.approx(0.3)
    assert down["from"] == 1.0
    assert down["to"] == 0.0
    assert down["rise_time_10_90_s"] == pytest.approx(0.2)
    assert summary["max_abs_rate_per_s"] == pytest.approx(4.0)


def test_step_summary_rejects_misaligned_or_nonfinite_inputs() -> None:
    time_s = np.array([0.0, 0.1])
    with pytest.raises(ValueError, match="aligned"):
        summarize_step_response(time_s, np.array([0.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        summarize_step_response(time_s, np.array([0.0, 1.0]), np.array([0.0, np.nan]))

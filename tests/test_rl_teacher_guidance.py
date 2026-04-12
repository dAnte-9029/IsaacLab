from __future__ import annotations

import math
import pytest

import torch

try:
    from flapping_bot.px4_like.rl_training_utils import apply_teacher_action_envelope, compute_teacher_guidance_delta, linear_anneal
except ImportError:
    from flapping_bot.flapping_bot.px4_like.rl_training_utils import (
        apply_teacher_action_envelope,
        compute_teacher_guidance_delta,
        linear_anneal,
    )

try:
    from flapping_bot.direct.flapping_bot.state_source_contract import (
        resolve_imu_source,
        resolve_runtime_state_source_selection,
        resolve_teacher_state_inputs,
        resolve_teacher_state_source,
        teacher_should_use_truth_wind,
    )
except ImportError:
    from flapping_bot.flapping_bot.direct.flapping_bot.state_source_contract import (
        resolve_imu_source,
        resolve_runtime_state_source_selection,
        resolve_teacher_state_inputs,
        resolve_teacher_state_source,
        teacher_should_use_truth_wind,
    )


def test_linear_anneal_interpolates_and_clamps() -> None:
    assert math.isclose(linear_anneal(0, start=0.2, end=2.0, duration_steps=100), 0.2)
    assert math.isclose(linear_anneal(50, start=0.2, end=2.0, duration_steps=100), 1.1)
    assert math.isclose(linear_anneal(100, start=0.2, end=2.0, duration_steps=100), 2.0)
    assert math.isclose(linear_anneal(250, start=0.2, end=2.0, duration_steps=100), 2.0)


def test_linear_anneal_zero_duration_returns_end() -> None:
    assert math.isclose(linear_anneal(0, start=0.2, end=2.0, duration_steps=0), 2.0)


def test_apply_teacher_action_envelope_limits_deviation() -> None:
    teacher = torch.tensor([[0.5, -0.5, 0.0, 0.8]], dtype=torch.float32)
    rl = torch.tensor([[-1.0, 1.0, 0.4, -1.0]], dtype=torch.float32)

    out = apply_teacher_action_envelope(teacher, rl, delta=0.25)

    expected = torch.tensor([[0.25, -0.25, 0.25, 0.55]], dtype=torch.float32)
    assert torch.allclose(out, expected)


def test_apply_teacher_action_envelope_with_full_delta_reaches_rl_action() -> None:
    teacher = torch.tensor([[0.8, -0.8]], dtype=torch.float32)
    rl = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)

    delta = compute_teacher_guidance_delta(
        200_000,
        enabled=True,
        delta_init=0.15,
        delta_final=2.0,
        anneal_steps=160_000,
        schedule_steps=(0, 20_000, 80_000, 160_000),
        schedule_deltas=(0.15, 0.25, 0.75, 2.0),
        disable_after_steps=-1,
    )
    out = apply_teacher_action_envelope(teacher, rl, delta=delta)

    assert torch.allclose(out, rl)


def test_resolve_teacher_state_source_rejects_invalid_values() -> None:
    assert resolve_teacher_state_source("truth") == "truth"
    assert resolve_teacher_state_source("ESTIMATED") == "estimated"
    with pytest.raises(ValueError):
        resolve_teacher_state_source("bogus")


def test_teacher_estimated_mode_never_uses_truth_wind() -> None:
    assert teacher_should_use_truth_wind("truth", True)
    assert not teacher_should_use_truth_wind("estimated", True)
    assert not teacher_should_use_truth_wind("truth", False)


def test_resolve_teacher_state_inputs_normalizes_and_reports_wind_flag() -> None:
    estimated_inputs = resolve_teacher_state_inputs("ESTIMATED", "truth", True)
    assert estimated_inputs.teacher_state_source == "estimated"
    assert estimated_inputs.policy_state_source == "truth"
    assert not estimated_inputs.teacher_uses_truth_wind

    truth_inputs = resolve_teacher_state_inputs("truth", "estimated", True)
    assert truth_inputs.teacher_state_source == "truth"
    assert truth_inputs.policy_state_source == "estimated"
    assert truth_inputs.teacher_uses_truth_wind

    assert not resolve_teacher_state_inputs("truth", "truth", False).teacher_uses_truth_wind


def test_runtime_state_source_selection_defaults_teacher_and_policy_from_controller_source() -> None:
    truth_selection = resolve_runtime_state_source_selection(state_source="truth")
    assert truth_selection.controller_state_source == "truth"
    assert truth_selection.teacher_state_source == "truth"
    assert truth_selection.policy_state_source == "truth"
    assert truth_selection.imu_source == "synthetic"

    compare_selection = resolve_runtime_state_source_selection(state_source="compare", imu_source="isaacsim")
    assert compare_selection.controller_state_source == "compare"
    assert compare_selection.teacher_state_source == "estimated"
    assert compare_selection.policy_state_source == "estimated"
    assert compare_selection.imu_source == "isaacsim"


def test_resolve_imu_source_rejects_invalid_values() -> None:
    assert resolve_imu_source("synthetic") == "synthetic"
    assert resolve_imu_source("ISAACSIM") == "isaacsim"
    with pytest.raises(ValueError):
        resolve_imu_source("bogus")

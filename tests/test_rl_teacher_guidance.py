from __future__ import annotations

import math

import torch

try:
    from flapping_bot.px4_like.rl_training_utils import apply_teacher_action_envelope, linear_anneal
except ImportError:
    from flapping_bot.flapping_bot.px4_like.rl_training_utils import apply_teacher_action_envelope, linear_anneal


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

    out = apply_teacher_action_envelope(teacher, rl, delta=2.0)

    assert torch.allclose(out, rl)

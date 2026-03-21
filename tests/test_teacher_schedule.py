from __future__ import annotations

import math

import pytest

try:
    from flapping_bot.px4_like.rl_training_utils import (
        compute_teacher_guidance_delta,
        piecewise_linear_anneal,
        teacher_guidance_is_active,
    )
except ImportError:
    from flapping_bot.flapping_bot.px4_like.rl_training_utils import (
        compute_teacher_guidance_delta,
        piecewise_linear_anneal,
        teacher_guidance_is_active,
    )


def test_piecewise_linear_anneal_interpolates_between_knots() -> None:
    steps = (0, 100, 300)
    values = (0.2, 0.5, 2.0)

    assert math.isclose(piecewise_linear_anneal(0, steps=steps, values=values), 0.2)
    assert math.isclose(piecewise_linear_anneal(50, steps=steps, values=values), 0.35)
    assert math.isclose(piecewise_linear_anneal(100, steps=steps, values=values), 0.5)
    assert math.isclose(piecewise_linear_anneal(200, steps=steps, values=values), 1.25)
    assert math.isclose(piecewise_linear_anneal(600, steps=steps, values=values), 2.0)


def test_piecewise_linear_anneal_rejects_bad_schedule_shapes() -> None:
    with pytest.raises(ValueError):
        piecewise_linear_anneal(10, steps=(0, 100), values=(0.2,))

    with pytest.raises(ValueError):
        piecewise_linear_anneal(10, steps=(100, 0), values=(0.2, 0.5))


def test_teacher_guidance_is_active_respects_disable_after_steps() -> None:
    assert teacher_guidance_is_active(0, enabled=True, disable_after_steps=-1) is True
    assert teacher_guidance_is_active(99, enabled=True, disable_after_steps=100) is True
    assert teacher_guidance_is_active(100, enabled=True, disable_after_steps=100) is False
    assert teacher_guidance_is_active(0, enabled=False, disable_after_steps=-1) is False


def test_compute_teacher_guidance_delta_uses_path_tracking_schedule_knots() -> None:
    steps = (0, 20_000, 80_000, 160_000)
    values = (0.15, 0.25, 0.75, 2.0)

    assert math.isclose(
        compute_teacher_guidance_delta(
            0,
            enabled=True,
            delta_init=0.15,
            delta_final=2.0,
            anneal_steps=160_000,
            schedule_steps=steps,
            schedule_deltas=values,
            disable_after_steps=-1,
        ),
        0.15,
    )
    assert math.isclose(
        compute_teacher_guidance_delta(
            50_000,
            enabled=True,
            delta_init=0.15,
            delta_final=2.0,
            anneal_steps=160_000,
            schedule_steps=steps,
            schedule_deltas=values,
            disable_after_steps=-1,
        ),
        0.5,
    )
    assert math.isclose(
        compute_teacher_guidance_delta(
            200_000,
            enabled=True,
            delta_init=0.15,
            delta_final=2.0,
            anneal_steps=160_000,
            schedule_steps=steps,
            schedule_deltas=values,
            disable_after_steps=-1,
        ),
        2.0,
    )


def test_compute_teacher_guidance_delta_returns_final_after_disable() -> None:
    assert math.isclose(
        compute_teacher_guidance_delta(
            160_000,
            enabled=True,
            delta_init=0.15,
            delta_final=2.0,
            anneal_steps=160_000,
            schedule_steps=(0, 20_000, 80_000, 160_000),
            schedule_deltas=(0.15, 0.25, 0.75, 2.0),
            disable_after_steps=120_000,
        ),
        2.0,
    )

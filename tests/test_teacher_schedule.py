from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from flapping_bot.direct.flapping_bot.path_tracking_env import (
    _apply_recovery_teacher_delta,
    _compute_segment_aware_teacher_delta,
)

try:
    from flapping_bot.px4_like.rl_training_utils import (
        apply_teacher_guided_actions,
        apply_teacher_residual_action,
        compute_teacher_guidance_delta,
        load_runner_checkpoint_for_warm_start,
        maybe_bootstrap_teacher_guided_policy,
        piecewise_linear_anneal,
        should_randomize_initial_episode_length,
        should_bootstrap_teacher_guided_policy,
        teacher_guidance_is_active,
    )
except ImportError:
    from flapping_bot.flapping_bot.px4_like.rl_training_utils import (
        apply_teacher_guided_actions,
        apply_teacher_residual_action,
        compute_teacher_guidance_delta,
        load_runner_checkpoint_for_warm_start,
        maybe_bootstrap_teacher_guided_policy,
        piecewise_linear_anneal,
        should_randomize_initial_episode_length,
        should_bootstrap_teacher_guided_policy,
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


def test_loiter_teacher_delta_is_tighter_than_turn() -> None:
    delta = _compute_segment_aware_teacher_delta(
        base_delta=torch.tensor([1.0, 1.0], dtype=torch.float32),
        is_loiter=torch.tensor([False, True]),
        loiter_min_scale=0.6,
    )

    assert torch.allclose(delta, torch.tensor([[1.0], [0.6]], dtype=torch.float32))


def test_recovery_teacher_delta_tightens_only_masked_envs() -> None:
    delta = _apply_recovery_teacher_delta(
        base_delta=torch.tensor([[0.4], [0.4]], dtype=torch.float32),
        recovery_mask=torch.tensor([False, True]),
        recovery_scale=0.1,
    )

    assert torch.allclose(delta, torch.tensor([[0.4], [0.04]], dtype=torch.float32))


def test_apply_teacher_residual_action_scales_policy_around_teacher() -> None:
    teacher_actions = torch.tensor([[0.1, -0.2], [0.8, -0.8]], dtype=torch.float32)
    residual_actions = torch.tensor([[1.0, -0.5], [1.0, -1.0]], dtype=torch.float32)
    delta = torch.tensor([[0.2], [0.5]], dtype=torch.float32)

    executed = apply_teacher_residual_action(teacher_actions, residual_actions, delta)

    expected = torch.tensor([[0.3, -0.3], [1.0, -1.0]], dtype=torch.float32)
    assert torch.allclose(executed, expected)


def test_apply_teacher_guided_actions_rejects_unknown_mode() -> None:
    teacher_actions = torch.zeros((1, 4), dtype=torch.float32)
    rl_actions = torch.zeros((1, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="unsupported teacher guidance mode"):
        apply_teacher_guided_actions(teacher_actions, rl_actions, delta=0.2, mode="not-a-mode")


def test_should_bootstrap_teacher_guided_policy_only_for_fresh_residual_teacher_runs() -> None:
    assert (
        should_bootstrap_teacher_guided_policy(
            teacher_guidance_enabled=True,
            teacher_guidance_mode="residual",
            zero_actor_init=True,
            is_resume=False,
        )
        is True
    )
    assert (
        should_bootstrap_teacher_guided_policy(
            teacher_guidance_enabled=True,
            teacher_guidance_mode="envelope",
            zero_actor_init=True,
            is_resume=False,
        )
        is False
    )
    assert (
        should_bootstrap_teacher_guided_policy(
            teacher_guidance_enabled=True,
            teacher_guidance_mode="residual",
            zero_actor_init=False,
            is_resume=False,
        )
        is False
    )
    assert (
        should_bootstrap_teacher_guided_policy(
            teacher_guidance_enabled=True,
            teacher_guidance_mode="residual",
            zero_actor_init=True,
            is_resume=True,
        )
        is False
    )


def test_maybe_bootstrap_teacher_guided_policy_zeroes_actor_output_layer() -> None:
    class DummyPolicy:
        def __init__(self) -> None:
            self.actor = nn.Sequential(nn.Linear(3, 5), nn.ELU(), nn.Linear(5, 2))

    policy = DummyPolicy()
    with torch.no_grad():
        policy.actor[-1].weight.fill_(1.0)
        policy.actor[-1].bias.fill_(2.0)

    applied = maybe_bootstrap_teacher_guided_policy(
        policy=policy,
        teacher_guidance_enabled=True,
        teacher_guidance_mode="residual",
        zero_actor_init=True,
        is_resume=False,
    )

    assert applied is True
    assert torch.count_nonzero(policy.actor[-1].weight) == 0
    assert torch.count_nonzero(policy.actor[-1].bias) == 0


def test_load_runner_checkpoint_for_warm_start_resets_iteration_and_skips_optimizer() -> None:
    class DummyRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool, str | None]] = []
            self.current_learning_iteration = -1

        def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
            self.calls.append((path, load_optimizer, map_location))
            self.current_learning_iteration = 40
            return {"bc": True}

    runner = DummyRunner()

    infos = load_runner_checkpoint_for_warm_start(runner=runner, checkpoint_path="model_bc.pt")

    assert infos == {"bc": True}
    assert runner.calls == [("model_bc.pt", False, None)]
    assert runner.current_learning_iteration == 0


def test_should_randomize_initial_episode_length_disables_path_tracking_weights_only_warm_start() -> None:
    assert (
        should_randomize_initial_episode_length(
            task="Isaac-FlappingBot-PathTracking-DeLaurier-PrimitivePureRL-Direct-v0",
            load_weights_only=True,
        )
        is False
    )
    assert (
        should_randomize_initial_episode_length(
            task="Isaac-FlappingBot-PathTracking-DeLaurier-PrimitivePureRL-Direct-v0",
            load_weights_only=False,
        )
        is True
    )
    assert (
        should_randomize_initial_episode_length(
            task="Isaac-FlappingBot-StraightFlight-DeLaurier-PureRL-Direct-v0",
            load_weights_only=True,
        )
        is True
    )

from __future__ import annotations

import math
import importlib.util
from pathlib import Path
import sys

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from flapping_bot.px4_like.pure_rl_split_actor import (
    PURE_RL_OBSERVATION_DIM,
    PureRLSplitActor,
    PureRLSplitActorCritic,
    build_cycle_averaged_frequency_observation,
    register_pure_rl_split_actor_critic,
)
from flapping_bot.px4_like.rl_training_utils import split_actor_state_dict_from_shared_actor


OBSERVATION_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_observation.py"
)
OBSERVATION_SPEC = importlib.util.spec_from_file_location(
    "pure_rl_observation_split_actor_test",
    OBSERVATION_MODULE_PATH,
)
assert OBSERVATION_SPEC is not None and OBSERVATION_SPEC.loader is not None
pure_rl_observation = importlib.util.module_from_spec(OBSERVATION_SPEC)
sys.modules[OBSERVATION_SPEC.name] = pure_rl_observation
OBSERVATION_SPEC.loader.exec_module(pure_rl_observation)


def _valid_observation(batch_size: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    observation = torch.zeros((batch_size, PURE_RL_OBSERVATION_DIM), dtype=dtype)
    sensor = observation[:, :420].reshape(batch_size, 30, 14)
    sensor[:, :, 0] = 1.0
    sensor[:, :, 11] = 0.2  # 3 Hz under the accepted 0--5 Hz affine normalization.
    sensor[:, :, 13] = 1.0
    return observation


def test_split_actor_contract_matches_authoritative_555_layout() -> None:
    assert pure_rl_observation.PURE_RL_RAW_OBSERVATION_LAYOUT.observation_dim == PURE_RL_OBSERVATION_DIM
    actor = PureRLSplitActor(PURE_RL_OBSERVATION_DIM, (16, 8), "elu")

    actions = actor(_valid_observation(3))

    assert actions.shape == (3, 4)


def test_frequency_observation_is_exactly_invariant_to_all_phase_pairs() -> None:
    observation = _valid_observation(2, dtype=torch.float64)
    phase_changed = observation.clone()
    changed_sensor = phase_changed[:, :420].reshape(2, 30, 14)
    changed_sensor[:, :, 12] = 1.0
    changed_sensor[:, :, 13] = 0.0

    original_slow = build_cycle_averaged_frequency_observation(observation)
    changed_slow = build_cycle_averaged_frequency_observation(phase_changed)

    torch.testing.assert_close(changed_slow, original_slow, rtol=0.0, atol=0.0)
    assert changed_slow.dtype == observation.dtype
    assert changed_slow.shape == observation.shape


def test_tail_branch_can_still_respond_to_phase() -> None:
    actor = PureRLSplitActor(PURE_RL_OBSERVATION_DIM, (1, 1), "elu")
    for parameter in actor.parameters():
        torch.nn.init.zeros_(parameter)
    newest_phase_sin_index = 29 * 14 + 12
    actor.tail_actor[0].weight.data[0, newest_phase_sin_index] = 1.0
    actor.tail_actor[2].weight.data[0, 0] = 1.0
    actor.tail_actor[4].weight.data[0, 0] = 1.0

    neutral = _valid_observation(1)
    phase_changed = neutral.clone()
    phase_changed[:, newest_phase_sin_index] = 1.0
    neutral_actions = actor(neutral)
    changed_actions = actor(phase_changed)

    torch.testing.assert_close(changed_actions[:, 0], neutral_actions[:, 0], rtol=0.0, atol=0.0)
    assert changed_actions[0, 1] > neutral_actions[0, 1]


def test_one_cycle_sensor_perturbation_averages_to_zero() -> None:
    observation = _valid_observation(1, dtype=torch.float64)
    sensor = observation[:, :420].reshape(1, 30, 14)
    phase = torch.arange(20, dtype=observation.dtype) * (2.0 * math.pi / 20.0)
    sensor[0, -20:, 4] = torch.sin(phase)

    slow = build_cycle_averaged_frequency_observation(observation)
    slow_sensor = slow[:, :420].reshape(1, 30, 14)

    torch.testing.assert_close(
        slow_sensor[0, :, 4],
        torch.zeros(30, dtype=observation.dtype),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_shared_actor_warm_start_preserves_tail_and_transformed_frequency() -> None:
    observations = TensorDict(
        {"policy": _valid_observation(5)},
        batch_size=[5],
    )
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    source = ActorCritic(
        observations,
        obs_groups,
        4,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        activation="elu",
        init_noise_std=0.6,
        noise_std_type="log",
    )
    target = PureRLSplitActorCritic(
        observations,
        obs_groups,
        4,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        activation="elu",
        init_noise_std=0.6,
        noise_std_type="log",
    )
    converted = split_actor_state_dict_from_shared_actor(
        source_state=source.state_dict(),
        target_state=target.state_dict(),
    )
    target.load_state_dict(converted)

    full_observation = observations["policy"]
    source_full_actions = source.actor(full_observation)
    source_slow_actions = source.actor(
        build_cycle_averaged_frequency_observation(full_observation)
    )
    target_actions = target.actor(full_observation)

    torch.testing.assert_close(target_actions[:, 1:4], source_full_actions[:, 1:4])
    torch.testing.assert_close(target_actions[:, 0], source_slow_actions[:, 0])
    torch.testing.assert_close(target.critic(full_observation), source.critic(full_observation))
    torch.testing.assert_close(target.log_std, source.log_std)


def test_split_actor_critic_registration_is_idempotent() -> None:
    import rsl_rl.runners.on_policy_runner as runner_module

    register_pure_rl_split_actor_critic()
    register_pure_rl_split_actor_critic()

    assert runner_module.PureRLSplitActorCritic is PureRLSplitActorCritic

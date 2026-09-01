from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/flapping_bot/flapping_bot/px4_like/rl_training_utils.py"
)
SPEC = importlib.util.spec_from_file_location("rl_training_utils_yaw_consistency_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
training_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training_utils)


class _DummyPolicy:
    is_recurrent = False
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}


def _identity_attitude_observation(batch_size: int) -> torch.Tensor:
    observation = torch.randn(batch_size, 555)
    sensor_history = observation[:, : 30 * 14].reshape(batch_size, 30, 14)
    sensor_history[:, :, 0:4] = 0.0
    sensor_history[:, :, 0] = 1.0
    return observation


def test_rotate_pure_rl_observation_world_yaw_changes_only_attitude_history() -> None:
    observation = _identity_attitude_observation(2)
    original = observation.clone()
    rotated = training_utils.rotate_pure_rl_observation_world_yaw(
        observation,
        torch.tensor([math.pi, -0.5 * math.pi]),
    )

    original_sensor = observation[:, : 30 * 14].reshape(2, 30, 14)
    rotated_sensor = rotated[:, : 30 * 14].reshape(2, 30, 14)
    assert torch.allclose(rotated_sensor[0, :, 0], torch.zeros(30), atol=1.0e-6)
    assert torch.allclose(rotated_sensor[0, :, 3], torch.ones(30), atol=1.0e-6)
    assert torch.allclose(
        torch.linalg.vector_norm(rotated_sensor[:, :, 0:4], dim=-1),
        torch.ones(2, 30),
        atol=1.0e-6,
    )
    assert torch.equal(rotated_sensor[:, :, 4:], original_sensor[:, :, 4:])
    assert torch.equal(rotated[:, 30 * 14 :], observation[:, 30 * 14 :])
    assert torch.equal(observation, original)


def test_actor_yaw_consistency_augmentor_pairs_observations_and_invariant_actions() -> None:
    observation = _identity_attitude_observation(3)
    auxiliary = torch.arange(3, dtype=torch.float32).unsqueeze(1)
    obs = TensorDict(
        {"policy": observation, "auxiliary": auxiliary},
        batch_size=[3],
    )
    augmentor = training_utils.ActorYawConsistencyAugmentor(_DummyPolicy())

    torch.manual_seed(11)
    augmented_obs, _ = augmentor(obs=obs, actions=None, env=object())
    actions = torch.randn(3, 4, requires_grad=True)
    _, target_actions = augmentor(obs=None, actions=actions, env=object())

    assert augmented_obs.batch_size == torch.Size([6])
    assert torch.equal(augmented_obs["policy"][:3], observation)
    assert torch.equal(augmented_obs["auxiliary"], torch.cat((auxiliary, auxiliary), dim=0))
    assert torch.equal(target_actions[:3], actions)
    assert torch.equal(target_actions[3:], actions.detach())


def test_maybe_enable_actor_yaw_consistency_uses_auxiliary_mirror_loss_slot() -> None:
    runner = type(
        "Runner",
        (),
        {
            "env": type(
                "Env",
                (),
                {"cfg": type("Cfg", (), {"pure_rl_actor_yaw_consistency_coefficient": 0.05})()},
            )(),
            "alg": type("Alg", (), {"policy": _DummyPolicy(), "symmetry": None})(),
        },
    )()

    assert training_utils.maybe_enable_actor_yaw_consistency(runner=runner) is True
    assert runner.alg.symmetry["use_data_augmentation"] is False
    assert runner.alg.symmetry["use_mirror_loss"] is True
    assert runner.alg.symmetry["mirror_loss_coeff"] == pytest.approx(0.05)


def test_actor_yaw_consistency_is_disabled_by_default() -> None:
    runner = type(
        "Runner",
        (),
        {
            "env": type(
                "Env",
                (),
                {"cfg": type("Cfg", (), {"pure_rl_actor_yaw_consistency_coefficient": 0.0})()},
            )(),
            "alg": None,
        },
    )()

    assert training_utils.maybe_enable_actor_yaw_consistency(runner=runner) is False

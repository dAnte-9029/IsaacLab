from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/flapping_bot/flapping_bot/px4_like/rl_training_utils.py"
)
SPEC = importlib.util.spec_from_file_location("rl_training_utils_distillation_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
training_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training_utils)


class _DummyPolicy(nn.Module):
    is_recurrent = False
    state_dependent_std = False

    def __init__(self, *, bias: bool = True) -> None:
        super().__init__()
        self.actor = nn.Linear(2, 1, bias=bias)
        self.actor_obs_normalizer = nn.Identity()
        self.obs_groups = {"policy": ["policy"], "critic": ["policy"]}


def test_actor_policy_distillation_masks_current_task_and_freezes_teacher() -> None:
    policy = _DummyPolicy(bias=False)
    nn.init.constant_(policy.actor.weight, 1.0)
    augmentor = training_utils.ActorPolicyDistillationAugmentor(policy)
    nn.init.constant_(policy.actor.weight, 2.0)
    obs = TensorDict(
        {
            "policy": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            training_utils.ACTOR_DISTILLATION_MASK_KEY: torch.tensor([[1.0], [0.0]]),
        },
        batch_size=[2],
    )

    augmented_obs, _ = augmentor(obs=obs, actions=None, env=object())
    student_actions = policy.actor(obs["policy"])
    _, target_actions = augmentor(obs=None, actions=student_actions, env=object())

    assert augmented_obs.batch_size == torch.Size([4])
    assert torch.allclose(target_actions[2:], torch.tensor([[3.0], [14.0]]))
    assert all(parameter.requires_grad is False for parameter in augmentor.teacher_actor.parameters())


def test_warm_start_enables_actor_distillation_after_checkpoint_load() -> None:
    class DummyRunner:
        def __init__(self) -> None:
            self.current_learning_iteration = -1
            self.env = type(
                "Env",
                (),
                {"cfg": type("Cfg", (), {"pure_rl_actor_distillation_coefficient": 0.05})()},
            )()
            storage = type(
                "Storage",
                (),
                {
                    "observations": TensorDict(
                        {
                            "policy": torch.zeros(1, 2),
                            training_utils.ACTOR_DISTILLATION_MASK_KEY: torch.zeros(1, 1),
                        },
                        batch_size=[1],
                    )
                },
            )()
            self.alg = type("Alg", (), {"policy": _DummyPolicy(), "storage": storage, "symmetry": None})()

        def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> None:
            del path, load_optimizer, map_location
            nn.init.constant_(self.alg.policy.actor.weight, 3.0)

    runner = DummyRunner()

    training_utils.load_runner_checkpoint_for_warm_start(runner=runner, checkpoint_path="teacher.pt")

    assert runner.current_learning_iteration == 0
    assert runner.alg.symmetry["mirror_loss_coeff"] == pytest.approx(0.05)
    teacher_actor = runner.alg.symmetry["data_augmentation_func"].teacher_actor
    assert torch.allclose(teacher_actor.weight, torch.full_like(teacher_actor.weight, 3.0))


def test_actor_policy_distillation_is_disabled_by_default() -> None:
    runner = type(
        "Runner",
        (),
        {
            "env": type(
                "Env", (), {"cfg": type("Cfg", (), {"pure_rl_actor_distillation_coefficient": 0.0})()}
            )(),
            "alg": None,
        },
    )()

    assert training_utils.maybe_enable_actor_policy_distillation(runner=runner) is False

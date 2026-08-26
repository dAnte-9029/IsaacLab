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


class _CapacityPolicy(nn.Module):
    is_recurrent = False
    state_dependent_std = False

    def __init__(self, actor_hidden_dims: tuple[int, int]) -> None:
        super().__init__()
        first, second = actor_hidden_dims
        self.log_std = nn.Parameter(torch.zeros(2))
        self.actor = nn.Sequential(
            nn.Linear(5, first),
            nn.ELU(),
            nn.Linear(first, second),
            nn.ELU(),
            nn.Linear(second, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(5, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.actor_obs_normalizer = nn.Identity()
        self.obs_groups = {"policy": ["policy"], "critic": ["policy"]}


def test_net2wider_actor_state_preserves_source_action_mean_and_critic() -> None:
    torch.manual_seed(7)
    source = _CapacityPolicy((256, 128))
    target = _CapacityPolicy((512, 256))
    observations = torch.randn(13, 5)
    source_actions = source.actor(observations).detach()
    source_values = source.critic(observations).detach()

    expanded = training_utils.expand_actor_state_dict_net2wider(
        source_state=source.state_dict(),
        target_state=target.state_dict(),
    )
    target.load_state_dict(expanded)

    assert torch.allclose(target.actor(observations), source_actions, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(target.critic(observations), source_values, atol=0.0, rtol=0.0)
    assert torch.allclose(target.log_std, source.log_std, atol=0.0, rtol=0.0)
    assert not torch.allclose(target.actor[2].weight[:, :256], target.actor[2].weight[:, 256:])
    assert not torch.allclose(target.actor[4].weight[:, :128], target.actor[4].weight[:, 128:])


def test_large_actor_warm_start_uses_net2wider_and_resets_iteration(tmp_path: Path) -> None:
    torch.manual_seed(11)
    source = _CapacityPolicy((256, 128))
    target = _CapacityPolicy((512, 256))
    checkpoint = tmp_path / "model_550.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "optimizer_state_dict": {},
            "iter": 550,
            "infos": {"source": "c2c"},
        },
        checkpoint,
    )

    class DummyRunner:
        def __init__(self) -> None:
            self.current_learning_iteration = -1
            self.alg = type("Alg", (), {"policy": target, "symmetry": None})()
            self.env = type("Env", (), {"cfg": type("Cfg", (), {})()})()

        def load(self, *_args, **_kwargs):
            raise AssertionError("large actor warm start must not call strict runner.load()")

    observations = torch.randn(9, 5)
    expected_actions = source.actor(observations).detach()
    runner = DummyRunner()

    infos = training_utils.load_runner_checkpoint_for_warm_start(
        runner=runner,
        checkpoint_path=str(checkpoint),
        map_location="cpu",
    )

    assert infos == {"source": "c2c"}
    assert runner.current_learning_iteration == 0
    assert torch.allclose(target.actor(observations), expected_actions, atol=1.0e-6, rtol=1.0e-6)


def test_net2wider_rejects_non_doubled_actor_width() -> None:
    source = _CapacityPolicy((256, 128))
    unsupported_target = _CapacityPolicy((512, 128))

    with pytest.raises(ValueError, match="exact twofold hidden widths"):
        training_utils.expand_actor_state_dict_net2wider(
            source_state=source.state_dict(),
            target_state=unsupported_target.state_dict(),
        )

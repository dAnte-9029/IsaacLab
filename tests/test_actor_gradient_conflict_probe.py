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
SPEC = importlib.util.spec_from_file_location("rl_training_utils_gradient_probe_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
training_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training_utils)


class _DummyPolicy(nn.Module):
    is_recurrent = False
    state_dependent_std = False
    noise_std_type = "log"

    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(self.actor.weight)
        self.actor_obs_normalizer = nn.Identity()
        self.obs_groups = {"policy": ["policy"], "critic": ["policy"]}
        self.log_std = nn.Parameter(torch.zeros(1))

    def get_actor_obs(self, observations: TensorDict) -> torch.Tensor:
        return observations["policy"]


class _DummyStorage:
    def __init__(self) -> None:
        policy_observation = torch.tensor(
            [
                [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            ]
        )
        group = torch.tensor(
            [
                [
                    [training_utils.ACTOR_GRADIENT_PROBE_C1_GROUP],
                    [training_utils.ACTOR_GRADIENT_PROBE_C2C_GROUP],
                    [training_utils.ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP],
                    [training_utils.ACTOR_GRADIENT_PROBE_C3A_GROUP],
                ]
            ]
        )
        self.observations = TensorDict(
            {
                "policy": policy_observation,
                training_utils.ACTOR_GRADIENT_PROBE_GROUP_KEY: group,
            },
            batch_size=[1, 4],
        )
        self.actions = torch.ones(1, 4, 1)
        self.advantages = torch.tensor([[[1.0], [1.0], [1.0], [-1.0]]])
        self.actions_log_prob = torch.full((1, 4, 1), -1.4189385)


class _DummyAlgorithm:
    clip_param = 0.2

    def __init__(self) -> None:
        self.policy = _DummyPolicy()
        self.storage = _DummyStorage()

    def update(self) -> dict[str, float]:
        with torch.no_grad():
            self.policy.actor.weight[0, 0] += 0.1
        return {"surrogate": 0.0}


def test_gradient_probe_reports_conflict_and_actual_update_alignment(tmp_path: Path) -> None:
    algorithm = _DummyAlgorithm()
    output_path = tmp_path / "actor_gradient_conflict.csv"
    probe = training_utils.ActorGradientConflictProbe(
        algorithm,
        output_path=output_path,
        interval=1,
        minimum_samples=1,
    )

    metrics = probe.update()

    assert metrics["gradient_probe/cos_c2c_strong_vs_c3a"] == pytest.approx(-1.0)
    assert metrics["gradient_probe/update_alignment_c2c_strong"] == pytest.approx(1.0)
    assert metrics["gradient_probe/count_c2c_strong"] == 1.0
    assert output_path.is_file()
    assert len(output_path.read_text().splitlines()) == 2


def test_gradient_probe_is_disabled_by_default() -> None:
    runner = type(
        "Runner",
        (),
        {
            "env": type("Env", (), {"cfg": type("Cfg", (), {})()})(),
            "alg": None,
        },
    )()

    assert training_utils.maybe_enable_actor_gradient_conflict_probe(runner=runner) is False


class _TaskAwareStorage:
    def __init__(self) -> None:
        group_values = torch.tensor(
            [
                training_utils.ACTOR_GRADIENT_PROBE_C1_GROUP,
                training_utils.ACTOR_GRADIENT_PROBE_C1_GROUP,
                training_utils.ACTOR_GRADIENT_PROBE_C2C_GROUP,
                training_utils.ACTOR_GRADIENT_PROBE_C2C_GROUP,
                training_utils.ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP,
                training_utils.ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP,
                training_utils.ACTOR_GRADIENT_PROBE_C3A_GROUP,
                training_utils.ACTOR_GRADIENT_PROBE_C3A_GROUP,
                training_utils.ACTOR_GRADIENT_PROBE_ACTIVE_C3A_GROUP,
                training_utils.ACTOR_GRADIENT_PROBE_ACTIVE_C3A_GROUP,
            ]
        )
        self.observations = TensorDict(
            {
                "policy": torch.arange(20, dtype=torch.float32).reshape(1, 10, 2),
                training_utils.ACTOR_GRADIENT_PROBE_GROUP_KEY: group_values.reshape(1, 10, 1),
            },
            batch_size=[1, 10],
        )
        self.values = torch.zeros(1, 10, 1)
        self.returns = torch.tensor(
            [[[-1.0], [1.0], [8.0], [12.0], [9.0], [11.0], [98.0], [102.0], [99.0], [101.0]]]
        )
        self.advantages = torch.zeros_like(self.returns)
        self.actions = torch.zeros(1, 10, 1)
        self.actions_log_prob = torch.zeros(1, 10, 1)
        self.mu = torch.zeros(1, 10, 1)
        self.sigma = torch.ones(1, 10, 1)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        del num_mini_batches, num_epochs
        raise AssertionError("baseline generator should be replaced by task-aware PPO")


class _TaskAwareAlgorithm:
    def __init__(self) -> None:
        self.policy = _DummyPolicy()
        self.storage = _TaskAwareStorage()

    def update(self) -> dict[str, float]:
        batches = list(self.storage.mini_batch_generator(2, 1))
        assert len(batches) == 2
        for batch in batches:
            labels = set(
                batch[0][training_utils.ACTOR_GRADIENT_PROBE_GROUP_KEY].squeeze(-1).tolist()
            )
            assert labels == {0, 1, 2, 3, 4}
        return {"surrogate": 0.0}


def test_task_aware_ppo_normalizes_per_task_and_stratifies_phases(tmp_path: Path) -> None:
    algorithm = _TaskAwareAlgorithm()
    adapter = training_utils.TaskAwarePpoAdapter(
        algorithm,
        output_path=tmp_path / "task_aware_ppo.csv",
        task_weights=(0.2, 0.4, 0.4),
        minimum_task_samples=2,
        minimum_phase_samples=2,
    )

    metrics = adapter.update()

    group = algorithm.storage.observations[training_utils.ACTOR_GRADIENT_PROBE_GROUP_KEY].squeeze(-1)
    task_masks = adapter._task_masks(group)
    for mask in task_masks.values():
        task_advantages = algorithm.storage.advantages.squeeze(-1)[mask]
        assert task_advantages.mean().item() == pytest.approx(0.0, abs=1.0e-6)
        assert task_advantages.std(unbiased=False).item() == pytest.approx(1.0, abs=1.0e-6)
    assert metrics["task_aware/count_c2c_strong_phase"] == 2.0
    assert metrics["task_aware/count_c3a_active_phase"] == 2.0
    assert (tmp_path / "task_aware_ppo.csv").is_file()


def test_task_aware_ppo_supports_four_task_c3b_weights(tmp_path: Path) -> None:
    algorithm = _TaskAwareAlgorithm()
    c3b_group = torch.full(
        (1, 2, 1),
        training_utils.TASK_AWARE_C3B_GROUP,
        dtype=torch.long,
    )
    algorithm.storage.observations = torch.cat(
        (
            algorithm.storage.observations,
            TensorDict(
                {
                    "policy": torch.tensor([[[20.0, 21.0], [22.0, 23.0]]]),
                    training_utils.ACTOR_GRADIENT_PROBE_GROUP_KEY: c3b_group,
                },
                batch_size=[1, 2],
            ),
        ),
        dim=1,
    )
    for name, values in (
        ("values", torch.zeros(1, 2, 1)),
        ("returns", torch.tensor([[[-2.0], [2.0]]])),
        ("advantages", torch.zeros(1, 2, 1)),
        ("actions", torch.zeros(1, 2, 1)),
        ("actions_log_prob", torch.zeros(1, 2, 1)),
        ("mu", torch.zeros(1, 2, 1)),
        ("sigma", torch.ones(1, 2, 1)),
    ):
        setattr(algorithm.storage, name, torch.cat((getattr(algorithm.storage, name), values), dim=1))

    adapter = training_utils.TaskAwarePpoAdapter(
        algorithm,
        output_path=tmp_path / "task_aware_c3b.csv",
        task_weights=(0.15, 0.20, 0.15, 0.50),
        minimum_task_samples=2,
        minimum_phase_samples=2,
    )
    metrics = adapter._prepare_advantages()

    assert metrics["task_aware/count_c3b"] == 2.0
    c3b_mask = adapter._task_masks(
        algorithm.storage.observations[training_utils.ACTOR_GRADIENT_PROBE_GROUP_KEY].squeeze(-1)
    )["c3b"]
    c3b_advantages = algorithm.storage.advantages.squeeze(-1)[c3b_mask]
    assert c3b_advantages.mean().item() == pytest.approx(0.0, abs=1.0e-6)


def test_task_aware_ppo_supports_five_task_c3_joint_weights(tmp_path: Path) -> None:
    algorithm = _TaskAwareAlgorithm()
    for group_id, returns in (
        (training_utils.TASK_AWARE_C3B_GROUP, [[[-2.0], [2.0]]]),
        (training_utils.TASK_AWARE_C3C_GROUP, [[[-3.0], [3.0]]]),
    ):
        group = torch.full((1, 2, 1), group_id, dtype=torch.long)
        algorithm.storage.observations = torch.cat(
            (
                algorithm.storage.observations,
                TensorDict(
                    {
                        "policy": torch.zeros(1, 2, 2),
                        training_utils.ACTOR_GRADIENT_PROBE_GROUP_KEY: group,
                    },
                    batch_size=[1, 2],
                ),
            ),
            dim=1,
        )
        for name, values in (
            ("values", torch.zeros(1, 2, 1)),
            ("returns", torch.tensor(returns)),
            ("advantages", torch.zeros(1, 2, 1)),
            ("actions", torch.zeros(1, 2, 1)),
            ("actions_log_prob", torch.zeros(1, 2, 1)),
            ("mu", torch.zeros(1, 2, 1)),
            ("sigma", torch.ones(1, 2, 1)),
        ):
            setattr(algorithm.storage, name, torch.cat((getattr(algorithm.storage, name), values), dim=1))

    adapter = training_utils.TaskAwarePpoAdapter(
        algorithm,
        output_path=tmp_path / "task_aware_c3_joint.csv",
        task_weights=(0.15, 0.20, 0.15, 0.25, 0.25),
        minimum_task_samples=2,
        minimum_phase_samples=0,
    )
    metrics = adapter._prepare_advantages()

    assert metrics["task_aware/count_c3b"] == 2.0
    assert metrics["task_aware/count_c3c"] == 2.0


def test_task_aware_ppo_can_disable_per_rollout_phase_abort(tmp_path: Path) -> None:
    adapter = training_utils.TaskAwarePpoAdapter(
        _TaskAwareAlgorithm(),
        output_path=tmp_path / "task_aware_no_phase_abort.csv",
        task_weights=(0.2, 0.4, 0.4),
        minimum_task_samples=2,
        minimum_phase_samples=0,
    )

    adapter._validate_phase_coverage(
        {
            "task_aware/count_c2c_strong_phase": 0.0,
            "task_aware/count_c3a_active_phase": 0.0,
        }
    )


def test_task_aware_ppo_is_disabled_by_default() -> None:
    runner = type(
        "Runner",
        (),
        {
            "env": type("Env", (), {"cfg": type("Cfg", (), {})()})(),
            "alg": None,
        },
    )()

    assert training_utils.maybe_enable_task_aware_ppo(runner=runner) is False


class _WarmStartStorage:
    def __init__(self) -> None:
        self.clear_count = 0

    def clear(self) -> None:
        self.clear_count += 1


class _WarmStartPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.actor.weight)


class _WarmStartAlgorithm:
    def __init__(self, *, actor_delta: float = 0.01) -> None:
        self.policy = _WarmStartPolicy()
        self.storage = _WarmStartStorage()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3.0e-4)
        self.schedule = "adaptive"
        self.learning_rate = 3.0e-4
        self.num_learning_epochs = 4
        self.symmetry = None
        self.rnd = None
        self.actor_delta = float(actor_delta)
        self.update_calls: list[tuple[float, int, str]] = []

    def update(self) -> dict[str, float]:
        self.update_calls.append(
            (
                float(self.optimizer.param_groups[0]["lr"]),
                int(self.num_learning_epochs),
                str(self.schedule),
            )
        )
        with torch.no_grad():
            self.policy.actor.weight.add_(self.actor_delta)
        self.storage.clear()
        return {"value_function": 1.0, "surrogate": 2.0, "entropy": 3.0}


def test_ppo_warm_start_guard_skips_then_ramps_fixed_learning_rate() -> None:
    algorithm = _WarmStartAlgorithm()
    guard = training_utils.PpoWarmStartGuard(
        algorithm,
        burn_in_iterations=2,
        warmup_update_iterations=3,
        initial_learning_rate=1.0e-5,
        target_learning_rate=5.0e-5,
        warmup_num_learning_epochs=1,
        actor_update_norm_limit=0.10,
    )

    results = [guard.update() for _ in range(6)]

    assert algorithm.storage.clear_count == 6
    assert len(algorithm.update_calls) == 4
    assert [call[0] for call in algorithm.update_calls] == pytest.approx(
        [1.0e-5, 3.0e-5, 5.0e-5, 5.0e-5]
    )
    assert [call[1] for call in algorithm.update_calls] == [1, 1, 1, 4]
    assert [call[2] for call in algorithm.update_calls] == ["fixed"] * 4
    assert results[0]["warm_start/skipped_update"] == 1.0
    assert results[1]["warm_start/actor_update_norm"] == 0.0
    assert results[2]["warm_start/update_index"] == 0.0
    assert results[2]["warm_start/actor_update_norm"] == pytest.approx(0.01)
    assert algorithm.num_learning_epochs == 4
    assert algorithm.schedule == "fixed"
    assert algorithm.learning_rate == pytest.approx(5.0e-5)


def test_ppo_warm_start_guard_fails_closed_on_large_actor_update() -> None:
    algorithm = _WarmStartAlgorithm(actor_delta=0.2)
    guard = training_utils.PpoWarmStartGuard(
        algorithm,
        burn_in_iterations=0,
        warmup_update_iterations=1,
        initial_learning_rate=1.0e-5,
        target_learning_rate=1.0e-5,
        warmup_num_learning_epochs=1,
        actor_update_norm_limit=0.10,
    )

    with pytest.raises(RuntimeError, match="0.200000 > 0.100000"):
        guard.update()


def test_ppo_warm_start_guard_only_enforces_update_limit_during_warmup() -> None:
    algorithm = _WarmStartAlgorithm(actor_delta=0.01)
    guard = training_utils.PpoWarmStartGuard(
        algorithm,
        burn_in_iterations=0,
        warmup_update_iterations=1,
        initial_learning_rate=1.0e-5,
        target_learning_rate=1.0e-5,
        warmup_num_learning_epochs=1,
        actor_update_norm_limit=0.10,
    )

    guard.update()
    algorithm.actor_delta = 0.2

    result = guard.update()

    assert result["warm_start/update_index"] == 1.0
    assert result["warm_start/actor_update_norm"] == pytest.approx(0.2)


def test_maybe_enable_ppo_warm_start_guard_is_default_disabled_and_attaches_when_enabled() -> None:
    algorithm = _WarmStartAlgorithm()
    disabled_runner = type(
        "Runner",
        (),
        {"env": type("Env", (), {"cfg": type("Cfg", (), {})()})(), "alg": algorithm},
    )()

    assert training_utils.maybe_enable_ppo_warm_start_guard(runner=disabled_runner) is False

    cfg = type(
        "Cfg",
        (),
        {
            "pure_rl_warm_start_guard_enabled": True,
            "pure_rl_warm_start_burn_in_iterations": 3,
            "pure_rl_warm_start_update_iterations": 10,
            "pure_rl_warm_start_initial_learning_rate": 1.0e-5,
            "pure_rl_warm_start_target_learning_rate": 5.0e-5,
            "pure_rl_warm_start_num_learning_epochs": 1,
            "pure_rl_warm_start_actor_update_norm_limit": 0.10,
        },
    )()
    enabled_runner = type("Runner", (), {"env": type("Env", (), {"cfg": cfg})(), "alg": algorithm})()

    assert training_utils.maybe_enable_ppo_warm_start_guard(runner=enabled_runner) is True
    assert algorithm.ppo_warm_start_guard.burn_in_iterations == 3
    assert algorithm.update == algorithm.ppo_warm_start_guard.update
    assert algorithm.schedule == "fixed"
    assert algorithm.learning_rate == pytest.approx(1.0e-5)

from __future__ import annotations

import torch

from scripts.flapping_rl.pretrain_path_tracking_bc import (
    build_bc_parser,
    compute_bc_loss,
    compute_bc_sample_mask,
    resolve_bc_rollout_actions_and_targets,
    save_bc_checkpoint,
)


def test_bc_parser_accepts_task_and_dataset() -> None:
    parser = build_bc_parser()
    args = parser.parse_args(
        [
            "--task",
            "Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0",
            "--dataset",
            "tmp.pt",
        ]
    )
    assert args.task.endswith("PathTracking-DeLaurier-Direct-v0")
    assert args.dataset == "tmp.pt"


def test_bc_parser_accepts_online_warmstart_args() -> None:
    parser = build_bc_parser()
    args = parser.parse_args(
        [
            "--task",
            "Isaac-FlappingBot-PathTracking-DeLaurier-PrimitiveWeakTeacherRL-Direct-v0",
            "--run-name",
            "bc_smoke",
            "--num-envs",
            "64",
            "--bc-updates",
            "10",
            "--steps-per-collect",
            "8",
            "--batch-size",
            "128",
            "--learning-rate",
            "0.001",
        ]
    )
    assert args.run_name == "bc_smoke"
    assert args.num_envs == 64
    assert args.bc_updates == 10
    assert args.steps_per_collect == 8
    assert args.batch_size == 128
    assert args.learning_rate == 0.001


def test_compute_bc_loss_is_zero_when_actor_matches_teacher() -> None:
    actions = torch.tensor([[0.1, -0.2, 0.3, -0.4], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    loss = compute_bc_loss(actions, actions)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_compute_bc_loss_increases_with_action_gap() -> None:
    teacher = torch.zeros((2, 4), dtype=torch.float32)
    near = torch.full((2, 4), 0.1, dtype=torch.float32)
    far = torch.full((2, 4), 0.5, dtype=torch.float32)
    assert compute_bc_loss(far, teacher) > compute_bc_loss(near, teacher)


def test_save_bc_checkpoint_works_without_runner_logger_fields(tmp_path) -> None:
    class _DummyPolicy:
        def state_dict(self) -> dict[str, torch.Tensor]:
            return {"weight": torch.tensor([1.0], dtype=torch.float32)}

    class _DummyOptimizer:
        def state_dict(self) -> dict[str, float]:
            return {"lr": 1.0e-3}

    class _DummyAlg:
        def __init__(self) -> None:
            self.policy = _DummyPolicy()
            self.optimizer = _DummyOptimizer()

    class _DummyRunner:
        def __init__(self) -> None:
            self.alg = _DummyAlg()
            self.current_learning_iteration = 7

    checkpoint_path = tmp_path / "model_bc.pt"
    save_bc_checkpoint(_DummyRunner(), checkpoint_path, infos={"bc": {"final_loss": 0.25}})

    loaded = torch.load(checkpoint_path, map_location="cpu")
    assert loaded["iter"] == 7
    assert loaded["infos"]["bc"]["final_loss"] == 0.25
    assert loaded["model_state_dict"]["weight"].item() == 1.0


def test_compute_bc_sample_mask_skips_frozen_envs() -> None:
    freeze_steps = torch.tensor([3, 0, -1], dtype=torch.int32)
    mask = compute_bc_sample_mask(freeze_steps=freeze_steps, episode_length_buf=None, min_episode_steps=0)
    assert torch.equal(mask, torch.tensor([False, True, True]))


def test_compute_bc_sample_mask_respects_min_episode_steps() -> None:
    freeze_steps = torch.tensor([0, 0, 0], dtype=torch.int32)
    episode_length_buf = torch.tensor([0, 4, 8], dtype=torch.int64)
    mask = compute_bc_sample_mask(freeze_steps=freeze_steps, episode_length_buf=episode_length_buf, min_episode_steps=5)
    assert torch.equal(mask, torch.tensor([False, False, True]))


def test_resolve_bc_rollout_actions_and_targets_keeps_absolute_actions_for_envelope_mode() -> None:
    teacher_actions = torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=torch.float32)

    rollout_actions, target_actions = resolve_bc_rollout_actions_and_targets(
        teacher_actions=teacher_actions,
        teacher_guidance_enabled=True,
        teacher_guidance_mode="envelope",
    )

    assert torch.allclose(rollout_actions, teacher_actions)
    assert torch.allclose(target_actions, teacher_actions)


def test_resolve_bc_rollout_actions_and_targets_uses_zero_residual_for_residual_mode() -> None:
    teacher_actions = torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=torch.float32)

    rollout_actions, target_actions = resolve_bc_rollout_actions_and_targets(
        teacher_actions=teacher_actions,
        teacher_guidance_enabled=True,
        teacher_guidance_mode="residual",
    )

    assert torch.allclose(rollout_actions, torch.zeros_like(teacher_actions))
    assert torch.allclose(target_actions, torch.zeros_like(teacher_actions))

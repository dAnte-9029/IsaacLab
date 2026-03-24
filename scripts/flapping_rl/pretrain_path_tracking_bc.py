"""Behavior-cloning warm start entrypoint for generic path tracking."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F


def _add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--headless", action="store_true")
    else:
        AppLauncher.add_app_launcher_args(parser)


def build_bc_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for path-tracking BC pretraining."""
    parser = argparse.ArgumentParser(description="Pretrain a path-tracking policy from teacher demonstrations.")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--run-name", type=str, default="path_tracking_bc")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-updates", type=int, default=200)
    parser.add_argument("--steps-per-collect", type=int, default=24)
    parser.add_argument("--bc-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--checkpoint-name", type=str, default="model_bc.pt")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-max-samples", type=int, default=50_000)
    parser.add_argument("--min-episode-steps-before-collect", type=int, default=0)
    _add_app_launcher_args(parser)
    return parser


def compute_bc_loss(pred_actions: torch.Tensor, teacher_actions: torch.Tensor) -> torch.Tensor:
    """Return the smooth-L1 imitation loss on actions."""
    return F.smooth_l1_loss(pred_actions, teacher_actions)


def compute_bc_sample_mask(
    *,
    freeze_steps: torch.Tensor | None,
    episode_length_buf: torch.Tensor | None,
    min_episode_steps: int,
) -> torch.Tensor:
    """Return the env mask eligible for BC sampling."""
    if freeze_steps is None:
        if episode_length_buf is None:
            raise ValueError("freeze_steps and episode_length_buf cannot both be None.")
        mask = torch.ones_like(episode_length_buf, dtype=torch.bool)
    else:
        mask = freeze_steps <= 0

    if episode_length_buf is not None and int(min_episode_steps) > 0:
        mask = mask & (episode_length_buf >= int(min_episode_steps))
    return mask


def _resolve_teacher_guidance_mode(mode: str) -> str:
    """Return the normalized teacher-guidance action semantics."""
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"envelope", "residual"}:
        raise ValueError(f"unsupported teacher guidance mode: {mode!r}")
    return resolved_mode


def resolve_bc_rollout_actions_and_targets(
    *,
    teacher_actions: torch.Tensor,
    teacher_guidance_enabled: bool,
    teacher_guidance_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rollout actions and BC targets consistent with env action semantics.

    For teacher-envelope tasks, the policy action is the executed action, so both
    rollout actions and BC targets match the teacher action directly.

    For residual teacher-guided tasks, the policy action is interpreted as a
    residual around the teacher action. The correct teacher-following rollout and
    target are therefore both zero residual.
    """
    if not bool(teacher_guidance_enabled):
        return teacher_actions, teacher_actions

    resolved_mode = _resolve_teacher_guidance_mode(teacher_guidance_mode)
    if resolved_mode == "envelope":
        return teacher_actions, teacher_actions

    zero_residual = torch.zeros_like(teacher_actions)
    return zero_residual, zero_residual


def save_bc_checkpoint(runner, path: str | Path, infos: dict | None = None) -> None:
    """Save a checkpoint without requiring post-`learn()` logger state on the runner."""
    checkpoint_path = str(Path(path))
    saved_dict = {
        "model_state_dict": runner.alg.policy.state_dict(),
        "optimizer_state_dict": runner.alg.optimizer.state_dict(),
        "iter": runner.current_learning_iteration,
        "infos": infos,
    }
    if hasattr(runner.alg, "rnd") and getattr(runner.alg, "rnd"):
        saved_dict["rnd_state_dict"] = runner.alg.rnd.state_dict()
        saved_dict["rnd_optimizer_state_dict"] = runner.alg.rnd_optimizer.state_dict()
    torch.save(saved_dict, checkpoint_path)

    logger_type = getattr(runner, "logger_type", None)
    disable_logs = bool(getattr(runner, "disable_logs", True))
    writer = getattr(runner, "writer", None)
    if logger_type in ["neptune", "wandb"] and writer is not None and not disable_logs:
        writer.save_model(checkpoint_path, runner.current_learning_iteration)


def _append_dataset_chunk(
    *,
    obs_policy: torch.Tensor,
    teacher_actions: torch.Tensor,
    obs_chunks: list[torch.Tensor],
    action_chunks: list[torch.Tensor],
    max_samples: int,
    current_samples: int,
) -> int:
    remaining = max(int(max_samples) - int(current_samples), 0)
    if remaining <= 0:
        return current_samples
    keep = min(int(obs_policy.shape[0]), remaining)
    obs_chunks.append(obs_policy[:keep].detach().cpu())
    action_chunks.append(teacher_actions[:keep].detach().cpu())
    return current_samples + keep


def _save_dataset(
    *,
    dataset_path: Path,
    obs_chunks: list[torch.Tensor],
    action_chunks: list[torch.Tensor],
    task: str,
    run_dir: Path,
) -> None:
    obs_policy = torch.cat(obs_chunks, dim=0) if obs_chunks else torch.empty((0, 0), dtype=torch.float32)
    teacher_actions = (
        torch.cat(action_chunks, dim=0) if action_chunks else torch.empty((0, 0), dtype=torch.float32)
    )
    torch.save(
        {
            "task": str(task),
            "run_dir": str(run_dir),
            "obs_policy": obs_policy,
            "teacher_actions": teacher_actions,
        },
        dataset_path,
    )


def main() -> None:
    args, hydra_args = build_bc_parser().parse_known_args()
    if not args.task:
        raise ValueError("--task is required.")

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:
        raise RuntimeError("IsaacLab runtime is required to run path-tracking BC pretraining.") from exc

    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.scene.num_envs = int(args.num_envs)
        env_cfg.seed = int(args.seed)
        env_cfg.sim.device = args.device

        agent_cfg = load_cfg_from_registry(args.task, args.agent)
        agent_cfg.device = args.device if args.device is not None else agent_cfg.device
        agent_cfg.seed = int(args.seed)

        log_root_path = Path("logs") / "rsl_rl" / str(agent_cfg.experiment_name)
        log_root_path = log_root_path.resolve()
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{run_stamp}_{args.run_name}"
        run_dir = log_root_path / run_name
        (run_dir / "params").mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Logging BC warm start in directory: {log_root_path}")
        print(f"[INFO] BC run directory: {run_dir}")

        env_cfg.log_dir = str(run_dir)
        env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if not hasattr(env.unwrapped, "_compute_teacher_actions"):
            raise RuntimeError(f"Task '{args.task}' does not expose teacher actions for BC warm start.")
        if getattr(env.unwrapped, "_teacher_controller", None) is None:
            raise RuntimeError(f"Task '{args.task}' does not have teacher guidance enabled.")

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)
        policy_nn = runner.alg.policy
        optimizer = torch.optim.Adam(
            policy_nn.actor.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )

        dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)

        dataset_obs_chunks: list[torch.Tensor] = []
        dataset_action_chunks: list[torch.Tensor] = []
        dataset_samples = 0

        obs, _ = env.reset()
        policy_nn.reset(torch.ones(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device))

        final_loss = float("nan")
        teacher_guidance_enabled = bool(getattr(env.unwrapped.cfg, "teacher_guidance_enabled", False))
        teacher_guidance_mode = str(getattr(env.unwrapped.cfg, "teacher_guidance_mode", "envelope"))
        for update_idx in range(1, int(args.bc_updates) + 1):
            obs_policy_chunks: list[torch.Tensor] = []
            teacher_action_chunks: list[torch.Tensor] = []
            collected_steps = 0
            while collected_steps < int(args.steps_per_collect):
                teacher_actions, _ = env.unwrapped._compute_teacher_actions()
                rollout_actions, bc_targets = resolve_bc_rollout_actions_and_targets(
                    teacher_actions=teacher_actions,
                    teacher_guidance_enabled=teacher_guidance_enabled,
                    teacher_guidance_mode=teacher_guidance_mode,
                )
                sample_mask = compute_bc_sample_mask(
                    freeze_steps=getattr(env.unwrapped, "_freeze_steps", None),
                    episode_length_buf=getattr(env.unwrapped, "episode_length_buf", None),
                    min_episode_steps=int(args.min_episode_steps_before_collect),
                )
                if torch.any(sample_mask):
                    obs_policy_chunks.append(obs["policy"][sample_mask].detach().clone())
                    teacher_action_chunks.append(bc_targets[sample_mask].detach().clone())
                    collected_steps += 1
                obs, _rew, dones, _info = env.step(rollout_actions)
                policy_nn.reset(dones)

            obs_policy = torch.cat(obs_policy_chunks, dim=0)
            teacher_actions = torch.cat(teacher_action_chunks, dim=0)
            dataset_samples = _append_dataset_chunk(
                obs_policy=obs_policy,
                teacher_actions=teacher_actions,
                obs_chunks=dataset_obs_chunks,
                action_chunks=dataset_action_chunks,
                max_samples=int(args.dataset_max_samples),
                current_samples=dataset_samples,
            )

            num_samples = int(obs_policy.shape[0])
            batch_size = max(1, min(int(args.batch_size), num_samples))
            weighted_loss_sum = 0.0
            sample_count = 0

            for _ in range(int(args.bc_epochs)):
                perm = torch.randperm(num_samples, device=obs_policy.device)
                for start in range(0, num_samples, batch_size):
                    batch_ids = perm[start : start + batch_size]
                    obs_batch = {"policy": obs_policy[batch_ids]}
                    teacher_batch = teacher_actions[batch_ids]
                    pred_actions = policy_nn.act_inference(obs_batch)
                    loss = compute_bc_loss(pred_actions, teacher_batch)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_nn.actor.parameters(), max_norm=float(args.grad_clip_norm))
                    optimizer.step()

                    batch_items = int(batch_ids.numel())
                    weighted_loss_sum += float(loss.item()) * batch_items
                    sample_count += batch_items

            final_loss = weighted_loss_sum / max(sample_count, 1)
            runner.current_learning_iteration = update_idx
            info = {
                "bc": {
                    "task": str(args.task),
                    "bc_updates": int(args.bc_updates),
                    "steps_per_collect": int(args.steps_per_collect),
                    "bc_epochs": int(args.bc_epochs),
                    "batch_size": int(args.batch_size),
                    "learning_rate": float(args.learning_rate),
                    "final_loss": float(final_loss),
                    "dataset_samples": int(dataset_samples),
                }
            }

            if update_idx % max(int(args.save_interval), 1) == 0:
                save_bc_checkpoint(runner, run_dir / f"model_bc_update_{update_idx}.pt", infos=info)

            print(
                json.dumps(
                    {
                        "update": update_idx,
                        "bc_updates": int(args.bc_updates),
                        "num_samples": num_samples,
                        "mean_bc_loss": float(final_loss),
                    }
                ),
                flush=True,
            )

        save_bc_checkpoint(runner, run_dir / str(args.checkpoint_name), infos={"bc": {"final_loss": float(final_loss)}})

        if args.dataset:
            dataset_path = Path(args.dataset).expanduser().resolve()
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            _save_dataset(
                dataset_path=dataset_path,
                obs_chunks=dataset_obs_chunks,
                action_chunks=dataset_action_chunks,
                task=str(args.task),
                run_dir=run_dir,
            )
            print(f"[INFO] Saved BC dataset: {dataset_path}")

        print(f"[OK] Saved BC checkpoint: {run_dir / str(args.checkpoint_name)}")
        env.close()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

"""Smoke-test the FlappingBot straight-flight DirectRLEnv.

Usage examples:
  ./isaaclab.sh -p scripts/flapping_rl/smoke_straight_flight_env.py --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 --headless
  ./isaaclab.sh -p scripts/flapping_rl/smoke_straight_flight_env.py --task Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0 --headless --num_envs 32
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test FlappingBot straight-flight env.")
    parser.add_argument("--task", type=str, required=True, help="Gym task id.")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=240)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = _parse_args()

    # launch Isaac Sim
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    obs, _ = env.reset()
    assert "policy" in obs

    device = env.unwrapped.device
    act_dim = env.unwrapped.cfg.action_space
    actions = torch.zeros(args.num_envs, act_dim, device=device)

    for i in range(int(args.steps)):
        actions.uniform_(-1.0, 1.0)
        obs, rew, term, trunc, info = env.step(actions)
        if not torch.isfinite(obs["policy"]).all():
            raise RuntimeError(f"Non-finite observation at step {i}.")
        if not torch.isfinite(rew).all():
            raise RuntimeError(f"Non-finite reward at step {i}.")
        if (term | trunc).any():
            env.reset()

    env.close()
    simulation_app.close()
    print("[OK] Smoke test completed.")


if __name__ == "__main__":
    main()


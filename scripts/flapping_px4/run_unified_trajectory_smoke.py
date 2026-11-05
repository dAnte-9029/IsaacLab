"""One short CPU-native environment check, fixed actions, no learner/checkpoint."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if not 2 <= args.num_envs <= 16 or not 2 <= args.steps <= 120:
        parser.error('Smoke requires 2..16 environments and 2..120 steps')
    args.output.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher
    root = Path(__file__).resolve().parents[2]
    app = AppLauncher(headless=True, device='cpu', kit_args=(
        f'--ext-folder {root / "source/flapping_bot/native_extensions"} '
        '--enable omni.flapping_bot.holonomic_constraint')).app
    import gymnasium as gym
    import numpy as np
    import omni.physx
    import torch
    from flapping_bot.direct.flapping_bot.unified_trajectory_env import UnifiedTrajectoryEnvCfg
    from flapping_bot.path_tracking.unified_trajectory import reversal_locations
    cfg = UnifiedTrajectoryEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 10.
    cfg.seed = args.seed
    cfg.robot = cfg.robot.replace(spawn=cfg.robot.spawn.replace(
        asset_path=str(root / 'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf'),
        usd_dir=str(args.output.resolve() / 'generated_assets')))
    try:
        env = gym.make('Isaac-FlappingBot-UnifiedTrajectory-Direct-v0', cfg=cfg).unwrapped
    except BaseException:
        import traceback
        traceback.print_exc()
        app.close()
        raise
    try:
        obs, _ = env.reset()
        assert obs['policy'].shape == (args.num_envs, 39)
        assert torch.isfinite(obs['policy']).all()
        assert env._pure_rl_spatial_stage is None and env._pure_rl_longitudinal_stage is None
        initial_records = list(env.task_records)
        for path in env.paths:
            assert all(reversal_locations(path, cfg.sampling)[k] for k in cfg.sampling.required_reversals)
        for i, path in enumerate(env.paths):
            np.testing.assert_allclose(env.reference['preview'][i].cpu().numpy(),
                                       path.query(np.asarray(cfg.preview_distances_m))['position'], atol=3e-6)
        actions = env._actions.clone()
        rewards = []
        for step in range(args.steps):
            obs, reward, terminated, truncated, extras = env.step(actions)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            assert torch.isfinite(env._robot.data.root_state_w).all()
            assert not terminated.any() and not truncated.any()
            rewards.append(reward.tolist())
            if step == args.steps//2:
                protected = env.task_records[1]
                protected_progress = env.progress[1].clone()
                env._reset_idx(torch.tensor([0], device=env.device))
                assert env.task_records[0] != initial_records[0]
                assert env.task_records[1] == protected
                torch.testing.assert_close(env.progress[1], protected_progress)
                assert env.progress[0] == 0
        physical_progress = env.progress.tolist()
        physical_error = torch.linalg.vector_norm(env.reference['error'], dim=-1).tolist()
        # Controlled state injection checks task wiring only, not a flight completion claim.
        state = env._robot.data.root_state_w.clone()
        end = env.paths[0].query(cfg.sampling.length_m)['position']
        state[0, :3] = torch.as_tensor(end, dtype=state.dtype)
        state[0, 3:7] = torch.tensor([1., 0., 0., 0.])
        state[0, 7:] = 0
        state[1, 2] = -1.
        env._robot.write_root_state_to_sim(state)
        env.progress[0] = cfg.sampling.length_m-.1
        env.previous_position.copy_(state[:, :3])
        env._reference_valid = False
        done, timeout = env._get_dones()
        assert done[0] and env.completed[0] and not env.failed[0]
        assert done[1] and env.failed[1] and not env.completed[1]
        assert torch.isfinite(env._get_rewards()).all()
        env._reset_idx(torch.arange(args.num_envs, device=env.device))
        env.episode_length_buf[:] = env.max_episode_length-1
        done, timeout = env._get_dones()
        assert timeout.all() and not done.any() and not env.completed.any()
        obs = env._get_observations()
        assert torch.isfinite(obs['policy']).all()
        report = dict(passed=True, num_envs=args.num_envs, policy_steps=args.steps,
                      physical_duration_s=args.steps*env.step_dt, seed=args.seed,
                      physics_dt_s=cfg.sim.dt, policy_dt_s=env.step_dt,
                      plant='CPU-native measured-wing PhysX / unchanged DeLaurier',
                      sampling=asdict(cfg.sampling), limits=asdict(cfg.geometry_limits),
                      initial_tasks=initial_records, final_tasks=env.task_records,
                      progress_m=physical_progress, path_error_m=physical_error, rewards=rewards,
                      checks=['finite state/observation/reward', 'distance preview', 'partial/full reset',
                              'interior reversals', 'injected completion/safety/timeout'])
        (args.output/'summary.json').write_text(json.dumps(report, indent=2))
        print('UNIFIED_SMOKE_PASSED', args.output/'summary.json', flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()
        app.close()


if __name__ == '__main__':
    main()

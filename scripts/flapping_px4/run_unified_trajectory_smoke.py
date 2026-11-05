"""One short CPU-native environment check, fixed actions, no learner/checkpoint."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter


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
        reset_start = perf_counter()
        obs, _ = env.reset()
        reset_seconds = perf_counter()-reset_start
        assert obs['policy'].shape == (args.num_envs, 39)
        assert torch.isfinite(obs['policy']).all()
        assert env._pure_rl_spatial_stage is None and env._pure_rl_longitudinal_stage is None
        initial_records = list(env.task_records)
        velocity = env._robot.data.root_lin_vel_w
        velocity_direction = velocity/torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
        torch.testing.assert_close(env.reference['tangent'], velocity_direction, atol=1e-6, rtol=0)
        initial_direction_error = float((env.reference['tangent']-velocity_direction).abs().max())
        for path in env.paths:
            entry = path.query(np.linspace(0, cfg.sampling.initial_straight_m, 11))
            for name in ('kappa', 'gamma', 'kappa_prime', 'gamma_prime'):
                np.testing.assert_allclose(entry[name], 0., atol=1e-14)
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
                protected_cache = env.path_cache.values[1].clone()
                protected_progress = env.progress[1].clone()
                env._reset_idx(torch.tensor([0], device=env.device))
                assert env.task_records[0] != initial_records[0]
                assert env.task_records[1] == protected
                torch.testing.assert_close(env.path_cache.values[1], protected_cache, atol=0, rtol=0)
                torch.testing.assert_close(env.progress[1], protected_progress)
                assert env.progress[0] == 0
        physical_progress = env.progress.tolist()
        physical_error = torch.linalg.vector_norm(env.reference['error'], dim=-1).tolist()
        # Explicit complex path on one row; other rows retain their default tasks.
        default_sampling = cfg.sampling
        cfg.sampling = replace(default_sampling, required_reversals=('kappa', 'gamma'))
        env._reset_idx(torch.tensor([0], device=env.device))
        complex_path = env.paths[0]
        complex_record = complex_path.spec.to_dict()
        events = reversal_locations(complex_path, cfg.sampling)
        assert all(events.values())
        cfg.sampling = default_sampling
        reversal_s = events['kappa'][0]
        transition_checks = []
        # Controlled state injection checks task wiring only, not flight success.
        for delta_s in (-1., 0., 1.):
            s = reversal_s+delta_s
            q = complex_path.query(s)
            state = env._robot.data.root_state_w.clone()
            state[0, :3] = torch.as_tensor(q['position']+.15*q['normal'], dtype=state.dtype)
            state[0, 3:7] = torch.tensor([1., 0., 0., 0.])
            state[0, 7:10] = torch.as_tensor(8*q['tangent'], dtype=state.dtype)
            state[0, 10:] = 0
            env._robot.write_root_state_to_sim(state)
            env.progress[0] = s-.3
            env.high_water[0] = s-.3
            env.previous_high_water[0] = s-.3
            env.previous_position.copy_(state[:, :3])
            env.previous_action.copy_(env._actions)
            env._reference_valid = False
            done, timeout = env._get_dones()
            reward = env._get_rewards()
            assert not done[0] and not timeout[0] and not env.completed[0]
            assert abs(float(env.progress[0])-s) < .002
            actual_s = float(env.progress[0])
            np.testing.assert_allclose(env.reference['preview'][0].numpy(), complex_path.query(
                actual_s+np.asarray(cfg.preview_distances_m))['position'], atol=2e-5)
            error2 = torch.sum(env.reference['error'][0]**2)
            tracking = torch.exp(-error2/cfg.tracking_scale_m**2)
            expected_reward = (cfg.progress_weight*(env.high_water[0]-env.previous_high_water[0])*tracking
                               + cfg.tracking_weight*(tracking-1)*env.step_dt)
            torch.testing.assert_close(reward[0], expected_reward, atol=1e-6, rtol=0)
            assert float(reward[0]) > 0
            assert torch.isfinite(env._get_observations()['policy']).all()
            transition_checks.append(dict(s=actual_s, kappa=float(env.reference['kappa'][0]),
                                          reward=float(reward[0]), completed=bool(env.completed[0])))
        assert transition_checks[0]['kappa']*transition_checks[-1]['kappa'] < 0
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
                      initial_reset_seconds=reset_seconds, initial_direction_max_abs_error=initial_direction_error,
                      complex_task=complex_record, reversal_events=events, injected_transition_checks=transition_checks,
                      physics_dt_s=cfg.sim.dt, policy_dt_s=env.step_dt,
                      plant='CPU-native measured-wing PhysX / unchanged DeLaurier',
                      sampling=asdict(cfg.sampling), limits=asdict(cfg.geometry_limits),
                      initial_tasks=initial_records, final_tasks=env.task_records,
                      progress_m=physical_progress, path_error_m=physical_error, rewards=rewards,
                      checks=['finite state/observation/reward', 'distance preview', 'partial/full reset',
                              'horizontal entry/velocity agreement', 'mid-route reversal projection/preview/reward',
                              'injected completion/safety/timeout'])
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

"""Fresh CPU-native integration gate; no checkpoint evaluation or PPO training."""

from dataclasses import fields, replace
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
simulation_app = AppLauncher(
    headless=True, device="cpu",
    kit_args=f"--ext-folder {ROOT / 'source/flapping_bot/native_extensions'} --enable omni.flapping_bot.holonomic_constraint",
).app

import gymnasium as gym
import omni.physx
import pytest
import torch

from flapping_bot.path_tracking.continuous_curve import continuous_curve_diagnostic_grid, save_continuous_curves
from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightDeLaurierMeasuredPureRLC4EnvCfg,
    FlappingBotStraightFlightEnv,
)


@pytest.mark.isaacsim_ci
def test_c4_native_cpu_replay_observation_telemetry_and_partial_reset(tmp_path):
    task = gym.spec("Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C4-Direct-v0")
    assert "rsl_rl_cfg_entry_point" not in task.kwargs
    grid = continuous_curve_diagnostic_grid()
    specs = tuple(replace(grid[i], initial_altitude_m=10.0 + j) for j, i in enumerate((0, 17, 32, 49, 64, 81)))
    curve_file = tmp_path / "curves.json"
    save_continuous_curves(curve_file, specs)
    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLC4EnvCfg()
    cfg.scene.num_envs = 6
    cfg.scene.env_spacing = 5.0
    cfg.seed = 0
    cfg.pure_rl_continuous_curve_file = str(curve_file)
    cfg.robot = cfg.robot.replace(spawn=cfg.robot.spawn.replace(
        asset_path=str(ROOT / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"),
        usd_dir=str(tmp_path / "generated_assets"),
    ))
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        obs, _ = env.reset()
        assert env.device == "cpu" and not cfg.scene.replicate_physics
        assert cfg.wing_drive_variant == "native_holonomic_drive"
        assert env._native_holonomic is not None
        assert env._native_holonomic.get_active_joint_count() == len(env._native_holonomic_joint_paths) == 6
        assert cfg.sim.dt == 1.0 / 480.0 and cfg.decimation == 8
        assert obs["policy"].shape == (6, 555)
        assert cfg.pure_rl_spatial_preview_times_s == (.24, .48, .72, .96, 1.20)
        assert not cfg.pure_rl_task_aware_ppo_enabled
        path = env._pure_rl_spatial_path
        assert (path.task_family_id == 5).all() and not path.event_count.any()
        torch.testing.assert_close(env._height_cmd, torch.tensor([s.initial_altitude_m for s in specs]))
        torch.testing.assert_close(
            (env._robot.data.root_pos_w - env.scene.env_origins)[:, 2], env._height_cmd,
            atol=1e-5, rtol=0,
        )
        torch.testing.assert_close(env._phase, torch.tensor([s.initial_flap_phase_rad for s in specs]))
        first_query = env._query_pure_rl_spatial_path()
        assert env._query_pure_rl_spatial_path() is first_query
        actions = torch.zeros((6, 4), device=env.device)
        actions[:, 0] = 0.2
        for _ in range(8):
            obs, rewards, terminated, truncated, extras = env.step(actions)
            assert obs["policy"].shape == (6, 555)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
            assert torch.isfinite(env._robot.data.root_state_w).all()
            assert not terminated.any() and not truncated.any()
            for key in ("PureRLCurve/completion_reached_fraction", "PureRLCurve/mean_progress_m",
                        "PureRLCurve/design_speed_violation_fraction", "PureRLPath/mean_turn_activity"):
                assert torch.isfinite(torch.as_tensor(extras["log"][key])).all()
        protected = torch.tensor([0, 2, 3, 5])
        reset_ids = torch.tensor([1, 4])
        before = {f.name: getattr(path, f.name).clone() for f in fields(path) if isinstance(getattr(path, f.name), torch.Tensor)}
        env._pure_rl_spatial_progress_m[:] = 20.0
        for _ in range(2):
            env._reset_idx(reset_ids)
            assert env._pure_rl_spatial_query_cache is None
            torch.testing.assert_close(env._pure_rl_spatial_progress_m[reset_ids], torch.zeros(2))
            torch.testing.assert_close(env._pure_rl_spatial_progress_m[protected], torch.full((4,), 20.0))
            for name, value in before.items():
                # Selected rows replay exactly; unselected rows are untouched.
                torch.testing.assert_close(getattr(path, name), value, atol=0, rtol=0)
        obs, rewards, *_ = env.step(actions)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
        # Switch from fixed replay to the reset-time random sampler in this same
        # native scene; avoid unsupported same-process scene reconstruction.
        env._continuous_curve_specs = None
        before_random = path.points_world_m.clone()
        env._reset_idx(reset_ids)
        torch.testing.assert_close(path.points_world_m[protected], before_random[protected], atol=0, rtol=0)
        assert not torch.equal(path.points_world_m[reset_ids], before_random[reset_ids])
        first_random = path.points_world_m.clone()
        env._reset_idx(reset_ids)
        torch.testing.assert_close(path.points_world_m[protected], first_random[protected], atol=0, rtol=0)
        assert not torch.equal(path.points_world_m[reset_ids], first_random[reset_ids])
        fixed_episode_path = path.points_world_m.clone()
        for _ in range(8):
            obs, rewards, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
            assert not terminated.any() and not truncated.any()
        torch.testing.assert_close(path.points_world_m, fixed_episode_path, atol=0, rtol=0)
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()

"""Mixed task IDs, per-row preview and partial-reset isolation on native CPU."""
from pathlib import Path
from isaaclab.app import AppLauncher
ROOT = Path(__file__).resolve().parents[1]
simulation_app = AppLauncher(headless=True,device='cpu',kit_args=(
    f'--ext-folder {ROOT / "source/flapping_bot/native_extensions"} '
    '--enable omni.flapping_bot.holonomic_constraint')).app
import torch
import pytest
import omni.physx
from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv, FlappingBotStraightFlightDeLaurierMeasuredPureRLC4JointEnvCfg,
)
from flapping_bot.direct.flapping_bot.pure_rl_spatial_path import query_spatial_path
from flapping_bot.px4_like.rl_training_utils import ACTOR_GRADIENT_PROBE_GROUP_KEY


@pytest.mark.isaacsim_ci
def test_joint_preview_group_and_partial_reset(tmp_path):
    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLC4JointEnvCfg()
    cfg.scene.num_envs = 96
    cfg.seed = 0
    cfg.robot = cfg.robot.replace(spawn=cfg.robot.spawn.replace(
        asset_path=str(ROOT/'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf'),
        usd_dir=str(tmp_path/'assets')))
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        obs, _ = env.reset()
        path = env._pure_rl_spatial_path
        family = path.task_family_id
        assert set(family.tolist()) == set(range(6))
        assert obs['policy'].shape == (96,555)
        assert (obs[ACTOR_GRADIENT_PROBE_GROUP_KEY][family==5] == 7).all()
        assert (obs[ACTOR_GRADIENT_PROBE_GROUP_KEY][family==4] == 6).all()
        query = env._query_pure_rl_spatial_path()
        kwargs = dict(path=path,previous_progress_m=query.progress_m,
                      position_world_m=env._robot.data.root_pos_w-env.scene.env_origins,
                      ground_velocity_world_mps=env._robot.data.root_lin_vel_w)
        short = query_spatial_path(**kwargs)
        long = query_spatial_path(**kwargs,preview_times_s=cfg.pure_rl_c4_preview_times_s)
        # C1/C2 use their existing overrides; compare old spatial groups directly.
        legacy_spatial = (family>=2)&(family<5)
        torch.testing.assert_close(query.preview_points_world_m[legacy_spatial],short.preview_points_world_m[legacy_spatial])
        torch.testing.assert_close(query.preview_points_world_m[family==5],long.preview_points_world_m[family==5])
        before = path.points_world_m.clone()
        env._reset_idx(torch.arange(12))
        torch.testing.assert_close(path.points_world_m[12:],before[12:],atol=0,rtol=0)
        for _ in range(8):
            obs,reward,*_ = env.step(torch.zeros(96,4))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()

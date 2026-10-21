"""Curriculum labels and partial-reset isolation with the actual native plant."""

from pathlib import Path
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
simulation_app = AppLauncher(headless=True, device='cpu', kit_args=(
    f'--ext-folder {ROOT / "source/flapping_bot/native_extensions"} '
    '--enable omni.flapping_bot.holonomic_constraint')).app

import omni.physx
import pytest
import torch
from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightEnv, FlappingBotStraightFlightDeLaurierMeasuredPureRLC4JointEnvCfg,
)
from flapping_bot.path_tracking.c4_curriculum import C4Curriculum


@pytest.mark.isaacsim_ci
def test_curriculum_reset_ids_and_unchanged_nonreset_rows(tmp_path):
    teacher = C4Curriculum(seed=31)

    class TrainingEnv(FlappingBotStraightFlightEnv):
        def __init__(self, cfg):
            self._continuous_curve_sampler = teacher.sample
            super().__init__(cfg)

    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLC4JointEnvCfg()
    cfg.scene.num_envs = 96
    cfg.seed = 0
    cfg.robot = cfg.robot.replace(spawn=cfg.robot.spawn.replace(
        asset_path=str(ROOT / 'source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf'),
        usd_dir=str(tmp_path / 'assets')))
    env = TrainingEnv(cfg)
    try:
        obs, _ = env.reset()
        assert obs['policy'].shape == (96, 555)
        for _ in range(3):
            path = env._pure_rl_spatial_path
            curves = path.task_family_id == 5
            assert curves.any()
            assert ((path.hard_slice_id[curves] >= 100) & (path.hard_slice_id[curves] <= 103)).all()
            before, labels = path.points_world_m.clone(), path.hard_slice_id.clone()
            env._reset_idx(torch.arange(12))
            torch.testing.assert_close(path.points_world_m[12:], before[12:], atol=0, rtol=0)
            torch.testing.assert_close(path.hard_slice_id[12:], labels[12:], atol=0, rtol=0)
            obs, rewards, *_ = env.step(torch.zeros(96, 4))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(rewards).all()
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()

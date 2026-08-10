"""End-to-end CPU runtime gate for PureRL longitudinal curriculum stage C2a."""

from __future__ import annotations

import math
from pathlib import Path

from isaaclab.app import AppLauncher


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTENSION_PARENT = _REPO_ROOT / "source/flapping_bot/native_extensions"
simulation_app = AppLauncher(
    headless=True,
    device="cpu",
    kit_args=(
        f"--ext-folder {_EXTENSION_PARENT} "
        "--enable omni.flapping_bot.holonomic_constraint"
    ),
).app

import omni.physx
import pytest
import torch

from flapping_bot.direct.flapping_bot.pure_rl_longitudinal_path import (
    CLIMB_TASK_ID,
    DESCENT_TASK_ID,
    LEVEL_TASK_ID,
    query_longitudinal_path,
)
from flapping_bot.direct.flapping_bot.pure_rl_reward import (
    compute_pure_rl_path_reward_terms,
    compute_pure_rl_reward_terms,
)
from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aEnvCfg,
    FlappingBotStraightFlightEnv,
)


def _assert_finite(name: str, value: torch.Tensor) -> None:
    assert bool(torch.all(torch.isfinite(value))), f"non-finite tensor: {name}"


@pytest.mark.isaacsim_ci
def test_native_cpu_pure_rl_c2a_geometry_reward_and_reset_runtime_gate(tmp_path: Path) -> None:
    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aEnvCfg()
    cfg.scene.num_envs = 6
    cfg.scene.env_spacing = 5.0
    cfg.sim.device = "cpu"
    cfg.seed = 0
    cfg.randomize_straight_line_heading = False
    cfg.randomize_flap_phase_at_reset = False
    cfg.pure_rl_eval_longitudinal_task_schedule = (
        LEVEL_TASK_ID,
        CLIMB_TASK_ID,
        DESCENT_TASK_ID,
        LEVEL_TASK_ID,
        CLIMB_TASK_ID,
        DESCENT_TASK_ID,
    )
    cfg.pure_rl_eval_longitudinal_slope_deg_schedule = (0.0, 4.0, -4.0, 0.0, 2.0, -2.0)
    cfg.pure_rl_eval_heading_schedule_rad = (0.0, 0.0, 0.0, math.pi / 2.0, math.pi / 2.0, math.pi / 2.0)
    cfg.pure_rl_eval_flap_phase_schedule_rad = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0, 0.0, math.pi / 2.0)
    cfg.pure_rl_eval_entry_length_m_schedule = (15.0,) * cfg.scene.num_envs
    cfg.pure_rl_eval_slope_length_m_schedule = (20.0,) * cfg.scene.num_envs
    cfg.freeze_steps_after_reset = 0
    cfg.episode_length_s = 100.0
    cfg.terminate_ground_height = -1.0e6
    cfg.terminate_tilt_deg = 89.9
    cfg.terminate_abs_y = 1.0e6
    cfg.pure_rl_terminate_abs_height_error_m = 1.0e6
    cfg.robot = cfg.robot.replace(
        spawn=cfg.robot.spawn.replace(
            asset_path=str(
                _REPO_ROOT
                / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
            ),
            usd_dir=str(tmp_path / "generated_assets/flap_robot_552"),
        )
    )

    assert cfg.sim.device == "cpu"
    assert not cfg.scene.replicate_physics
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        observations, _ = env.reset()
        assert env._native_holonomic is not None
        assert env._native_holonomic.get_active_joint_count() == len(env._native_holonomic_joint_paths)
        assert observations["policy"].shape == (cfg.scene.num_envs, 555)
        _assert_finite("reset_observation", observations["policy"])

        path = env._pure_rl_longitudinal_path
        assert path is not None
        torch.testing.assert_close(
            path.task_id,
            torch.tensor(cfg.pure_rl_eval_longitudinal_task_schedule, device=env.device),
        )
        torch.testing.assert_close(
            path.heading_rad,
            torch.tensor(cfg.pure_rl_eval_heading_schedule_rad, device=env.device),
        )
        torch.testing.assert_close(
            torch.rad2deg(path.signed_slope_rad),
            torch.tensor(cfg.pure_rl_eval_longitudinal_slope_deg_schedule, device=env.device),
            atol=1.0e-5,
            rtol=0.0,
        )

        horizontal_tangent = torch.stack(
            (
                torch.cos(path.heading_rad),
                torch.sin(path.heading_rad),
                torch.zeros_like(path.heading_rad),
            ),
            dim=1,
        )
        synthetic_position = (path.entry_length_m - 0.25).unsqueeze(1) * horizontal_tangent
        synthetic_position[:, 2] = path.initial_altitude_m
        synthetic_velocity = 7.0 * horizontal_tangent
        synthetic_query = query_longitudinal_path(
            path=path,
            position_world_m=synthetic_position,
            ground_velocity_world_mps=synthetic_velocity,
            minimum_preview_speed_mps=cfg.pure_rl_preview_minimum_speed_mps,
            maximum_preview_speed_mps=cfg.pure_rl_preview_maximum_speed_mps,
        )
        preview_delta_z = (
            synthetic_query.preview_points_world_m[:, -1, 2]
            - synthetic_query.preview_points_world_m[:, 0, 2]
        )
        assert bool(torch.allclose(preview_delta_z[[0, 3]], torch.zeros(2, device=env.device)))
        assert bool(torch.all(preview_delta_z[[1, 4]] > 0.0))
        assert bool(torch.all(preview_delta_z[[2, 5]] < 0.0))
        basis = torch.stack(
            (
                synthetic_query.tangent_world,
                synthetic_query.lateral_normal_world,
                synthetic_query.vertical_normal_world,
            ),
            dim=1,
        )
        identity = torch.eye(3, device=env.device).expand(cfg.scene.num_envs, -1, -1)
        torch.testing.assert_close(basis @ basis.transpose(1, 2), identity, atol=1.0e-6, rtol=0.0)

        actions = torch.zeros((cfg.scene.num_envs, cfg.action_space), device=env.device)
        for _ in range(8):
            observations, rewards, terminated, truncated, extras = env.step(actions)
            for name, value in {
                "observation": observations["policy"],
                "reward": rewards,
                "root_state": env._robot.data.root_state_w,
                "path_cross_track": env._eval_pure_rl_cross_track_error_m,
                "path_height": env._eval_pure_rl_height_error_m,
                "path_tangent_velocity": env._eval_pure_rl_along_track_velocity_mps,
                "path_lateral_velocity": env._eval_pure_rl_lateral_normal_velocity_mps,
                "path_vertical_velocity": env._eval_pure_rl_vertical_normal_velocity_mps,
                "path_active_slope": env._eval_pure_rl_active_slope_rad,
            }.items():
                assert value is not None
                _assert_finite(name, value)
            assert not bool(torch.any(terminated))
            assert not bool(torch.any(truncated))

        required_telemetry = {
            "PureRLReward/total",
            "PureRLReward/path",
            "PureRLReward/progress",
            "PureRLReward/velocity",
            "PureRLPath/mean_sampled_signed_slope_rad",
            "PureRLPath/mean_active_slope_rad",
            "PureRLPath/recovery_reached_fraction",
            "PureRLPath/mean_tangent_velocity_mps",
            "PureRLPath/mean_abs_lateral_normal_velocity_mps",
            "PureRLPath/mean_abs_vertical_normal_velocity_mps",
        }
        assert required_telemetry.issubset(extras["log"])
        for name in required_telemetry:
            _assert_finite(name, torch.as_tensor(extras["log"][name]))

        actual_query = env._query_pure_rl_longitudinal_path()
        level_ids = torch.tensor([0, 3], device=env.device)
        ground_velocity = env._robot.data.root_lin_vel_w[level_ids]
        common_kwargs = {
            "cross_track_error_m": actual_query.cross_track_error_m[level_ids],
            "height_error_m": actual_query.height_error_m[level_ids],
            "roll_rad": torch.zeros(2, device=env.device),
            "pitch_rad": torch.zeros(2, device=env.device),
            "angular_velocity_body_rad_s": env._robot.data.root_ang_vel_b[level_ids],
            "actual_flap_frequency_hz": env._freq[level_ids],
            "frequency_slew_hz_per_s": env._frequency_slew_hz_per_s[level_ids],
            "applied_action": env._act_cmd[level_ids],
            "previous_applied_action": env._pure_rl_previous_reward_action[level_ids],
            "config": cfg.pure_rl_reward_cfg,
        }
        c2_terms = compute_pure_rl_path_reward_terms(
            tangent_velocity_mps=torch.sum(
                ground_velocity * actual_query.tangent_world[level_ids], dim=1
            ),
            lateral_normal_velocity_mps=torch.sum(
                ground_velocity * actual_query.lateral_normal_world[level_ids], dim=1
            ),
            vertical_normal_velocity_mps=torch.sum(
                ground_velocity * actual_query.vertical_normal_world[level_ids], dim=1
            ),
            **common_kwargs,
        )
        c1_terms = compute_pure_rl_reward_terms(
            along_track_velocity_mps=torch.sum(
                ground_velocity * env._straight_line_tangent_w[level_ids], dim=1
            ),
            cross_track_velocity_mps=torch.sum(
                ground_velocity * env._straight_line_normal_w[level_ids], dim=1
            ),
            vertical_velocity_mps=ground_velocity[:, 2],
            **common_kwargs,
        )
        for name, c2_value in c2_terms.as_dict().items():
            torch.testing.assert_close(c2_value, c1_terms.as_dict()[name], atol=1.0e-6, rtol=0.0)

        reset_ids = torch.tensor([1, 4], device=env.device)
        protected_ids = torch.tensor([0, 2, 3, 5], device=env.device)
        episode_lengths_before = env.episode_length_buf.clone()
        path_before = {
            name: getattr(path, name).clone()
            for name in (
                "task_id",
                "heading_rad",
                "signed_slope_rad",
                "entry_length_m",
                "slope_length_m",
                "initial_altitude_m",
            )
        }
        env._reset_idx(reset_ids)
        assert bool(torch.all(env.episode_length_buf[reset_ids] == 0))
        torch.testing.assert_close(
            env.episode_length_buf[protected_ids], episode_lengths_before[protected_ids]
        )
        for name, before in path_before.items():
            torch.testing.assert_close(getattr(path, name)[protected_ids], before[protected_ids])

        for _ in range(3):
            env._reset_idx(reset_ids)
        observations, rewards, _terminated, _truncated, _extras = env.step(actions)
        _assert_finite("post_reset_observation", observations["policy"])
        _assert_finite("post_reset_reward", rewards)
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()

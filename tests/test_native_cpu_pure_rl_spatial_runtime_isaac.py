"""End-to-end CPU runtime gate for PureRL spatial curriculum C3."""

from __future__ import annotations

from dataclasses import fields
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
import gymnasium as gym
import pytest
import torch

from flapping_bot.direct.flapping_bot.pure_rl_reward import (
    compute_pure_rl_spatial_path_reward_terms,
    compute_pure_rl_spatial_termination_terms,
)
from flapping_bot.direct.flapping_bot.pure_rl_longitudinal_path import query_longitudinal_path
from flapping_bot.direct.flapping_bot.pure_rl_spatial_path import (
    C3B_TURN_THEN_CLIMB_TEMPLATE_ID,
    C3C_COUPLED_TEMPLATE_ID,
    ISOLATED_TURN_TEMPLATE_ID,
    REHEARSAL_C2C_TASK_FAMILY_ID,
    query_spatial_path,
    sample_spatial_path_batch,
)
from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightDeLaurierMeasuredPureRLC3aEnvCfg,
    FlappingBotStraightFlightDeLaurierMeasuredPureRLC3cEnvCfg,
    FlappingBotStraightFlightEnv,
)


def _assert_finite(name: str, value: torch.Tensor) -> None:
    assert bool(torch.all(torch.isfinite(value))), f"non-finite tensor: {name}"


def _tensor_fields(path) -> tuple[str, ...]:
    return tuple(field.name for field in fields(path) if isinstance(getattr(path, field.name), torch.Tensor))


@pytest.mark.isaacsim_ci
def test_native_cpu_pure_rl_c3_geometry_reward_and_reset_runtime_gate(tmp_path: Path) -> None:
    for stage in ("C3a", "C3b", "C3c"):
        task_id = f"Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-{stage}-Direct-v0"
        assert gym.spec(task_id).id == task_id

    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLC3cEnvCfg()
    cfg.scene.num_envs = 6
    cfg.scene.env_spacing = 5.0
    cfg.sim.device = "cpu"
    cfg.seed = 0
    cfg.randomize_straight_line_heading = False
    cfg.randomize_flap_phase_at_reset = False
    cfg.pure_rl_eval_spatial_template_schedule = (C3C_COUPLED_TEMPLATE_ID,) * 6
    cfg.pure_rl_eval_spatial_geometry_roll_deg_schedule = (8.0, 12.0, 6.0, 8.0, 12.0, 6.0)
    cfg.pure_rl_eval_spatial_slope_deg_schedule = (4.0, -7.0, 9.0, -4.0, 7.0, -9.0)
    cfg.pure_rl_eval_spatial_turn_sign_schedule = (-1, -1, -1, 1, 1, 1)
    cfg.pure_rl_eval_heading_schedule_rad = (0.0, 0.0, math.pi / 2.0, math.pi, math.pi, -math.pi / 2.0)
    cfg.pure_rl_eval_flap_phase_schedule_rad = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0, 0.0, math.pi)
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
    assert not cfg.wind_enabled
    assert not cfg.randomize_wind
    assert not cfg.wind_ou_enabled
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        observations, _ = env.reset()
        assert env._native_holonomic is not None
        assert env._native_holonomic.get_active_joint_count() == len(env._native_holonomic_joint_paths)
        assert observations["policy"].shape == (6, 555)
        _assert_finite("reset_observation", observations["policy"])

        path = env._pure_rl_spatial_path
        progress = env._pure_rl_spatial_progress_m
        assert path is not None and progress is not None
        torch.testing.assert_close(path.template_id, torch.full((6,), C3C_COUPLED_TEMPLATE_ID, device=env.device))
        torch.testing.assert_close(
            torch.rad2deg(path.peak_geometry_roll_rad),
            torch.tensor(cfg.pure_rl_eval_spatial_geometry_roll_deg_schedule, device=env.device),
            atol=1.0e-5,
            rtol=0.0,
        )
        torch.testing.assert_close(
            torch.rad2deg(path.peak_slope_rad),
            torch.tensor(cfg.pure_rl_eval_spatial_slope_deg_schedule, device=env.device),
            atol=1.0e-5,
            rtol=0.0,
        )
        torch.testing.assert_close(
            path.turn_sign,
            torch.tensor(
                cfg.pure_rl_eval_spatial_turn_sign_schedule,
                device=env.device,
                dtype=path.turn_sign.dtype,
            ),
        )
        assert bool(torch.all(path.vertical_sign[:3] == torch.tensor([1.0, -1.0, 1.0], device=env.device)))
        coupled = (path.curvature_rad_per_m.abs() > 1.0e-6) & (path.slope_rad.abs() > 1.0e-6)
        assert bool(torch.all(torch.any(coupled, dim=1)))
        for env_id in range(6):
            assert bool(torch.all(path.curvature_rad_per_m[env_id, coupled[env_id]] * path.turn_sign[env_id] > 0.0))
            assert bool(torch.all(path.slope_rad[env_id, coupled[env_id]] * path.vertical_sign[env_id] > 0.0))

        synthetic_progress = torch.full((6,), 55.0, device=env.device)
        point_ids = torch.round(synthetic_progress / path.sample_spacing_m).to(torch.int64)
        rows = torch.arange(6, device=env.device)
        synthetic_position = path.points_world_m[rows, point_ids]
        synthetic_velocity = 7.0 * path.tangent_world[rows, point_ids]
        query = query_spatial_path(
            path=path,
            previous_progress_m=synthetic_progress,
            position_world_m=synthetic_position,
            ground_velocity_world_mps=synthetic_velocity,
            minimum_preview_speed_mps=cfg.pure_rl_preview_minimum_speed_mps,
            maximum_preview_speed_mps=cfg.pure_rl_preview_maximum_speed_mps,
        )
        basis = torch.stack(
            (query.tangent_world, query.lateral_normal_world, query.vertical_normal_world),
            dim=1,
        )
        identity = torch.eye(3, device=env.device).expand(6, -1, -1)
        torch.testing.assert_close(basis @ basis.transpose(1, 2), identity, atol=1.0e-5, rtol=0.0)
        for field_name in _tensor_fields(path):
            _assert_finite(f"path_{field_name}", getattr(path, field_name).to(torch.float32))
        for field_name in fields(query):
            value = getattr(query, field_name.name)
            if isinstance(value, torch.Tensor):
                _assert_finite(f"query_{field_name.name}", value.to(torch.float32))

        rehearsal_path = env._pure_rl_c2c_rehearsal_path
        assert rehearsal_path is not None
        path.task_family_id[0] = REHEARSAL_C2C_TASK_FAMILY_ID
        expected_c2c = query_longitudinal_path(
            path=rehearsal_path,
            position_world_m=env._robot.data.root_pos_w - env.scene.env_origins,
            ground_velocity_world_mps=env._robot.data.root_lin_vel_w,
            minimum_preview_speed_mps=cfg.pure_rl_preview_minimum_speed_mps,
            maximum_preview_speed_mps=cfg.pure_rl_preview_maximum_speed_mps,
        )
        env._pure_rl_spatial_query_cache = None
        env._pure_rl_spatial_query_step = -1
        mixed_query = env._query_pure_rl_spatial_path()
        torch.testing.assert_close(
            mixed_query.preview_points_world_m[0], expected_c2c.preview_points_world_m[0]
        )
        torch.testing.assert_close(
            mixed_query.horizontal_normal_error_m[0], expected_c2c.cross_track_error_m[0]
        )
        torch.testing.assert_close(
            mixed_query.vertical_normal_error_m[0], expected_c2c.height_error_m[0]
        )
        torch.testing.assert_close(mixed_query.tangent_world[0], expected_c2c.tangent_world[0])
        assert bool(mixed_query.reached_all_events[0]) == bool(expected_c2c.reached_recovery[0])
        path.task_family_id[0] = 3
        env._pure_rl_spatial_query_cache = None
        env._pure_rl_spatial_query_step = -1

        for stage, template, roll, slope in (
            ("c3a", ISOLATED_TURN_TEMPLATE_ID, 11.0, 0.0),
            ("c3b", C3B_TURN_THEN_CLIMB_TEMPLATE_ID, 13.5, 12.0),
        ):
            sampled = sample_spatial_path_batch(
                num_paths=2,
                stage=stage,
                device=env.device,
                dtype=torch.float32,
                initial_altitude_m=10.0,
                evaluation_template_id=torch.full((2,), template, device=env.device, dtype=torch.int64),
                evaluation_geometry_roll_deg=torch.full((2,), roll, device=env.device),
                evaluation_slope_deg=torch.tensor([slope, -slope], device=env.device),
                evaluation_turn_sign=torch.tensor([-1.0, 1.0], device=env.device),
                evaluation_heading_rad=torch.zeros(2, device=env.device),
            )
            assert bool(torch.all(sampled.template_id == template))
            _assert_finite(f"{stage}_points", sampled.points_world_m)
            assert bool(torch.any(sampled.curvature_rad_per_m[0] < 0.0))
            assert bool(torch.any(sampled.curvature_rad_per_m[1] > 0.0))
            if stage == "c3b":
                assert bool(torch.any(sampled.slope_rad[0] > 0.0))
                assert bool(torch.any(sampled.slope_rad[1] < 0.0))

        actions = torch.zeros((6, cfg.action_space), device=env.device)
        for _ in range(8):
            observations, rewards, terminated, truncated, extras = env.step(actions)
            for name, value in {
                "observation": observations["policy"],
                "reward": rewards,
                "root_state": env._robot.data.root_state_w,
                "horizontal_error": env._eval_pure_rl_cross_track_error_m,
                "vertical_error": env._eval_pure_rl_height_error_m,
                "curvature": env._eval_pure_rl_active_curvature_rad_per_m,
                "slope": env._eval_pure_rl_active_slope_rad,
                "turn_activity": env._eval_pure_rl_turn_activity,
                "abs_roll": env._eval_pure_rl_abs_roll_rad,
            }.items():
                assert value is not None
                _assert_finite(name, value)
            assert not bool(torch.any(terminated))
            assert not bool(torch.any(truncated))

        required_telemetry = {
            "PureRLReward/total",
            "PureRLReward/path",
            "PureRLReward/progress",
            "PureRLReward/roll",
            "PureRLPath/mean_sampled_template_id",
            "PureRLPath/mean_active_curvature_rad_per_m",
            "PureRLPath/mean_active_slope_rad",
            "PureRLPath/mean_turn_activity",
            "PureRLPath/all_events_reached_fraction",
            "PureRLPath/mean_abs_roll_rad",
            "PureRLTermination/roll_limit_fraction",
        }
        assert required_telemetry.issubset(extras["log"])
        for name in required_telemetry:
            _assert_finite(name, torch.as_tensor(extras["log"][name]))

        common_reward = {
            "cross_track_error_m": torch.zeros(3, device=env.device),
            "height_error_m": torch.zeros(3, device=env.device),
            "tangent_velocity_mps": torch.full((3,), 7.0, device=env.device),
            "lateral_normal_velocity_mps": torch.zeros(3, device=env.device),
            "vertical_normal_velocity_mps": torch.zeros(3, device=env.device),
            "roll_rad": torch.deg2rad(torch.tensor([25.0, 30.0, 35.0], device=env.device)),
            "pitch_rad": torch.zeros(3, device=env.device),
            "angular_velocity_body_rad_s": torch.zeros((3, 3), device=env.device),
            "actual_flap_frequency_hz": torch.full((3,), 2.0, device=env.device),
            "frequency_slew_hz_per_s": torch.zeros(3, device=env.device),
            "applied_action": torch.zeros((3, 4), device=env.device),
            "previous_applied_action": torch.zeros((3, 4), device=env.device),
            "turn_activity": torch.ones(3, device=env.device),
            "config": cfg.pure_rl_reward_cfg,
        }
        reward_terms = compute_pure_rl_spatial_path_reward_terms(**common_reward)
        torch.testing.assert_close(
            reward_terms.roll_reward,
            torch.tensor([1.0, 0.75, 0.0], device=env.device),
            atol=1.0e-6,
            rtol=0.0,
        )
        termination = compute_pure_rl_spatial_termination_terms(
            height_m=torch.full((3,), 10.0, device=env.device),
            cross_track_error_m=torch.zeros(3, device=env.device),
            height_error_m=torch.zeros(3, device=env.device),
            projected_gravity_body=torch.tensor([[0.0, 0.0, -1.0]] * 3, device=env.device),
            roll_rad=common_reward["roll_rad"],
            config=env._pure_rl_termination_cfg,
        )
        assert termination.roll_limit.tolist() == [False, False, True]

        reset_ids = torch.tensor([1, 4], device=env.device)
        protected_ids = torch.tensor([0, 2, 3, 5], device=env.device)
        path_before = {name: getattr(path, name).clone() for name in _tensor_fields(path)}
        progress.fill_(20.0)
        cfg.pure_rl_eval_spatial_template_schedule = None
        cfg.pure_rl_eval_spatial_geometry_roll_deg_schedule = None
        cfg.pure_rl_eval_spatial_slope_deg_schedule = None
        cfg.pure_rl_eval_spatial_turn_sign_schedule = None
        env._reset_idx(reset_ids)
        torch.testing.assert_close(progress[reset_ids], torch.zeros(2, device=env.device))
        torch.testing.assert_close(progress[protected_ids], torch.full((4,), 20.0, device=env.device))
        for name, before in path_before.items():
            torch.testing.assert_close(getattr(path, name)[protected_ids], before[protected_ids])
        assert not torch.equal(path.points_world_m[reset_ids], path_before["points_world_m"][reset_ids])

        observations, rewards, _terminated, _truncated, _extras = env.step(actions)
        _assert_finite("post_reset_observation", observations["policy"])
        _assert_finite("post_reset_reward", rewards)
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()


@pytest.mark.isaacsim_ci
def test_native_cpu_c3a_adaptive_sampling_runtime_update(tmp_path: Path) -> None:
    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLC3aEnvCfg()
    cfg.scene.num_envs = 8
    cfg.scene.env_spacing = 5.0
    cfg.sim.device = "cpu"
    cfg.seed = 0
    cfg.freeze_steps_after_reset = 0
    cfg.episode_length_s = 100.0
    cfg.terminate_ground_height = -1.0e6
    cfg.terminate_tilt_deg = 89.9
    cfg.terminate_abs_y = 1.0e6
    cfg.pure_rl_terminate_abs_height_error_m = 1.0e6
    cfg.pure_rl_c2c_strong_climb_probability = 0.5
    cfg.pure_rl_adaptive_task_sampling_enabled = True
    cfg.pure_rl_adaptive_sampling_interval_steps = 1
    cfg.pure_rl_adaptive_sampling_minimum_episodes = 1
    cfg.robot = cfg.robot.replace(
        spawn=cfg.robot.spawn.replace(
            asset_path=str(
                _REPO_ROOT
                / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
            ),
            usd_dir=str(tmp_path / "generated_assets/flap_robot_552_adaptive"),
        )
    )

    env = FlappingBotStraightFlightEnv(cfg)
    try:
        env.reset()
        assert env._pure_rl_adaptive_completed_counts is not None
        assert env._pure_rl_adaptive_success_counts is not None
        env._pure_rl_adaptive_completed_counts.fill_(1.0)
        env._pure_rl_adaptive_success_counts.copy_(
            torch.tensor([1.0, 0.0, 0.0], device=env.device)
        )
        initial_probabilities = env._pure_rl_adaptive_task_probabilities

        actions = torch.zeros((8, cfg.action_space), device=env.device)
        observations, rewards, _terminated, _truncated, extras = env.step(actions)

        _assert_finite("adaptive_observation", observations["policy"])
        _assert_finite("adaptive_reward", rewards)
        assert env._pure_rl_adaptive_update_count == 1
        assert env._pure_rl_adaptive_task_probabilities[1] > initial_probabilities[1]
        assert env._pure_rl_adaptive_strong_climb_probability == pytest.approx(0.6)
        assert extras["log"]["AdaptiveSampling/update_count"] == 1
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()

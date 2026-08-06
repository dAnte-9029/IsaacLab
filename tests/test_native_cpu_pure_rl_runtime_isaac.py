"""End-to-end CPU runtime gate for the canonical measured PureRL task."""

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

from flapping_bot.direct.flapping_bot.action_contract import frequency_hz_to_normalized_action
from flapping_bot.direct.flapping_bot.straight_flight_env import (
    FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
    FlappingBotStraightFlightEnv,
)


def _assert_finite(name: str, value: torch.Tensor) -> None:
    assert bool(torch.all(torch.isfinite(value))), f"non-finite tensor: {name}"


@pytest.mark.isaacsim_ci
def test_native_cpu_pure_rl_frequency_and_repeated_reset_runtime_gate(tmp_path: Path) -> None:
    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg()
    cfg.scene.num_envs = 64
    cfg.scene.env_spacing = 5.0
    cfg.sim.device = "cpu"
    cfg.seed = 0
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

    assert cfg.min_flap_hz == 0.0
    assert cfg.max_flap_hz == 5.0
    env = FlappingBotStraightFlightEnv(cfg)
    try:
        observations, _ = env.reset()
        _assert_finite("reset_observation", observations["policy"])
        assert observations["policy"].shape == (cfg.scene.num_envs, 555)
        assert float(torch.max(torch.abs(observations["policy"])).item()) <= 5.0
        assert float(torch.std(env._straight_line_heading_rad).item()) > 0.5
        assert bool(torch.all(env._phase >= 0.0))
        assert bool(torch.all(env._phase < 2.0 * math.pi))
        assert float(torch.std(env._phase).item()) > 0.5
        torch.testing.assert_close(
            env._pure_rl_sensor_history,
            env._pure_rl_sensor_history[:, 0:1, :].expand_as(env._pure_rl_sensor_history),
        )
        torch.testing.assert_close(
            env._pure_rl_action_history,
            env._pure_rl_action_history[:, 0:1, :].expand_as(env._pure_rl_action_history),
        )

        requested_frequency_hz = torch.tensor(
            [0.0, 2.5, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            device=env.device,
        ).repeat(cfg.scene.num_envs // 8)
        actions = torch.zeros((cfg.scene.num_envs, cfg.action_space), device=env.device)
        actions[:, 0] = frequency_hz_to_normalized_action(
            requested_frequency_hz,
            minimum_frequency_hz=cfg.min_flap_hz,
            maximum_frequency_hz=cfg.max_flap_hz,
        )

        left_joint_id = int(env._joint_ids[env._IDX_LEFT_WING])
        right_joint_id = int(env._joint_ids[env._IDX_RIGHT_WING])
        max_tracking_error_rad = 0.0
        max_sync_error_rad = 0.0
        tracking_peak: dict[str, float | int] = {}
        forced_reset_count = 0
        duration_s = 1.5
        total_policy_steps = int(round(duration_s / float(env.step_dt)))
        forced_reset_steps = {
            int(round(total_policy_steps * fraction))
            for fraction in (0.25, 0.5, 0.75)
        }

        for policy_step in range(total_policy_steps):
            observations, rewards, terminated, truncated, extras = env.step(actions)
            finite_tensors = {
                "observation": observations["policy"],
                "reward": rewards,
                "root_state": env._robot.data.root_state_w,
                "joint_position": env._robot.data.joint_pos,
                "joint_velocity": env._robot.data.joint_vel,
                "frequency": env._freq,
                "target_frequency": env._phase_target_frequency_hz,
                "wing_force": env._debug_last_wing_force_link_n,
                "wing_moment": env._debug_last_wing_moment_link_about_com_nm,
                "eval_cross_track_error": env._eval_pure_rl_cross_track_error_m,
                "eval_height_error": env._eval_pure_rl_height_error_m,
                "eval_along_track_progress": env._eval_pure_rl_along_track_progress_m,
                "eval_along_track_velocity": env._eval_pure_rl_along_track_velocity_mps,
                "eval_tilt": env._eval_pure_rl_tilt_rad,
                "eval_angular_rate": env._eval_pure_rl_angular_rate_rad_s,
                "eval_actual_frequency": env._eval_pure_rl_actual_flap_frequency_hz,
                "eval_action_delta": env._eval_pure_rl_normalized_action_delta,
            }
            for name, value in finite_tensors.items():
                assert value is not None
                _assert_finite(name, value)
            assert observations["policy"].shape == (cfg.scene.num_envs, 555)
            assert float(torch.max(torch.abs(observations["policy"])).item()) <= 5.0
            telemetry = extras["log"]
            required_telemetry = {
                "PureRLReward/total",
                "PureRLReward/path",
                "PureRLReward/progress",
                "PureRLPenalty/flap",
                "PureRLPenalty/frequency_action_delta",
                "PureRLPenalty/tail_action_delta",
                "PureRLState/mean_abs_cross_track_error_m",
                "PureRLState/mean_actual_flap_frequency_hz",
                "PureRLTermination/ground_fraction",
                "PureRLTermination/tilt_fraction",
                "PureRLTermination/cross_track_fraction",
                "PureRLTermination/height_error_fraction",
            }
            assert required_telemetry.issubset(telemetry)
            for name in required_telemetry:
                value = torch.as_tensor(telemetry[name])
                _assert_finite(name, value)
            assert not bool(torch.any(terminated))
            assert not bool(torch.any(truncated))

            left_position = env._robot.data.joint_pos[:, left_joint_id] - float(env._wing_mid_L)
            right_position = -(
                env._robot.data.joint_pos[:, right_joint_id] - float(env._wing_mid_R)
            )
            current_reference = float(env._wing_amp) * torch.sin(
                env._ideal_inverse_phase_state.phase_rad
            )
            tracking_error = torch.abs(left_position - current_reference)
            step_max_tracking_error = float(torch.max(tracking_error).item())
            if step_max_tracking_error > max_tracking_error_rad:
                peak_env = int(torch.argmax(tracking_error).item())
                max_tracking_error_rad = step_max_tracking_error
                tracking_peak = {
                    "policy_step": policy_step,
                    "env": peak_env,
                    "position_rad": float(left_position[peak_env].item()),
                    "reference_rad": float(current_reference[peak_env].item()),
                    "frequency_hz": float(env._freq[peak_env].item()),
                    "target_frequency_hz": float(env._phase_target_frequency_hz[peak_env].item()),
                }
            max_sync_error_rad = max(
                max_sync_error_rad,
                float(torch.max(torch.abs(left_position - right_position)).item()),
            )

            if policy_step in forced_reset_steps:
                reset_ids = torch.arange(3, cfg.scene.num_envs, 2, device=env.device)
                env._reset_idx(reset_ids)
                forced_reset_count += int(reset_ids.numel())

        protected_ids = torch.arange(3, device=env.device)
        assert torch.allclose(
            env._phase_target_frequency_hz[protected_ids],
            requested_frequency_hz[protected_ids],
            atol=1.0e-3,
            rtol=0.0,
        )
        assert torch.allclose(
            env._freq[protected_ids],
            requested_frequency_hz[protected_ids],
            atol=2.0e-2,
            rtol=0.0,
        )
        print(
            "native CPU PureRL runtime gate:"
            f" forced_resets={forced_reset_count},"
            f" final_target_hz={env._phase_target_frequency_hz[protected_ids].tolist()},"
            f" final_actual_hz={env._freq[protected_ids].tolist()},"
            f" max_tracking_deg={math.degrees(max_tracking_error_rad):.9f},"
            f" max_sync_deg={math.degrees(max_sync_error_rad):.9f},"
            f" tracking_peak={tracking_peak}"
        )
        expected_forced_resets = len(forced_reset_steps) * len(range(3, cfg.scene.num_envs, 2))
        assert forced_reset_count == expected_forced_resets
        assert max_tracking_error_rad < math.radians(0.1)
        assert max_sync_error_rad < math.radians(0.1)
    finally:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()

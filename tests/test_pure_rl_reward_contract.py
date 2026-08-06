from __future__ import annotations

from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

from isaaclab.utils.dict import update_class_from_dict


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_reward.py"
)
SPEC = importlib.util.spec_from_file_location("pure_rl_reward", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pure_rl_reward = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pure_rl_reward
SPEC.loader.exec_module(pure_rl_reward)


def _reward_inputs(count: int, *, dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    return {
        "cross_track_error_m": torch.zeros(count, dtype=dtype),
        "height_error_m": torch.zeros(count, dtype=dtype),
        "along_track_velocity_mps": torch.zeros(count, dtype=dtype),
        "cross_track_velocity_mps": torch.zeros(count, dtype=dtype),
        "vertical_velocity_mps": torch.zeros(count, dtype=dtype),
        "roll_rad": torch.zeros(count, dtype=dtype),
        "pitch_rad": torch.zeros(count, dtype=dtype),
        "angular_velocity_body_rad_s": torch.zeros((count, 3), dtype=dtype),
        "actual_flap_frequency_hz": torch.zeros(count, dtype=dtype),
        "applied_action": torch.zeros((count, 4), dtype=dtype),
        "previous_applied_action": torch.zeros((count, 4), dtype=dtype),
    }


def test_reward_config_supports_hydra_style_restore_without_changing_defaults_or_reward() -> None:
    default_config = pure_rl_reward.PureRLRewardConfig()
    restored_config = pure_rl_reward.PureRLRewardConfig(
        cross_track_scale_m=99.0,
        path_reward_weight=99.0,
    )

    update_class_from_dict(restored_config, asdict(default_config))

    assert asdict(restored_config) == asdict(default_config)
    default_terms = pure_rl_reward.compute_pure_rl_reward_terms(
        **_reward_inputs(3),
        config=default_config,
    )
    restored_terms = pure_rl_reward.compute_pure_rl_reward_terms(
        **_reward_inputs(3),
        config=restored_config,
    )
    torch.testing.assert_close(restored_terms.total_reward, default_terms.total_reward, rtol=0.0, atol=0.0)

    restored_config.cross_track_scale_m = 2.0
    assert default_config.cross_track_scale_m == pytest.approx(1.5)
    assert pure_rl_reward.PURE_RL_CURRICULUM1_REWARD_CONFIG.cross_track_scale_m == pytest.approx(1.5)


def test_nominal_terms_have_expected_unit_values_and_weighted_total() -> None:
    terms = pure_rl_reward.compute_pure_rl_reward_terms(**_reward_inputs(2))

    torch.testing.assert_close(terms.path_reward, torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(terms.progress_reward, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(terms.velocity_reward, torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(terms.roll_reward, torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(terms.angular_rate_reward, torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(terms.pitch_envelope_penalty, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(terms.flap_penalty, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(terms.total_reward, torch.full((2,), 0.8, dtype=torch.float64))
    assert set(terms.as_dict()) == {
        "path",
        "progress",
        "velocity",
        "roll",
        "angular_rate",
        "pitch_envelope_penalty",
        "flap_penalty",
        "frequency_action_delta_penalty",
        "tail_action_delta_penalty",
        "tail_action_limit_penalty",
        "total",
    }


def test_path_reward_decreases_monotonically_with_route_relative_errors() -> None:
    inputs = _reward_inputs(4)
    inputs["cross_track_error_m"][:] = torch.tensor([0.0, 0.5, 1.5, 3.0])
    cross_track_terms = pure_rl_reward.compute_pure_rl_reward_terms(**inputs)
    assert bool(torch.all(cross_track_terms.path_reward[:-1] > cross_track_terms.path_reward[1:]))

    inputs = _reward_inputs(4)
    inputs["height_error_m"][:] = torch.tensor([0.0, 0.5, 1.0, 3.0])
    height_terms = pure_rl_reward.compute_pure_rl_reward_terms(**inputs)
    assert bool(torch.all(height_terms.path_reward[:-1] > height_terms.path_reward[1:]))


def test_progress_rewards_forward_motion_without_a_target_speed() -> None:
    inputs = _reward_inputs(5)
    inputs["along_track_velocity_mps"][:] = torch.tensor([-6.0, -3.0, 0.0, 3.0, 12.0])

    progress = pure_rl_reward.compute_pure_rl_reward_terms(**inputs).progress_reward

    assert bool(torch.all(progress[:-1] < progress[1:]))
    assert progress[0].item() < 0.0
    assert progress[2].item() == pytest.approx(0.0)
    assert 0.99 < progress[-1].item() < 1.0


def test_pitch_is_free_inside_envelope_and_penalized_smoothly_outside() -> None:
    inputs = _reward_inputs(5)
    inputs["pitch_rad"][:] = torch.deg2rad(torch.tensor([0.0, 20.0, 30.0, 35.0, 45.0]))

    penalty = pure_rl_reward.compute_pure_rl_reward_terms(**inputs).pitch_envelope_penalty

    torch.testing.assert_close(penalty[:3], torch.zeros(3, dtype=torch.float64))
    assert 0.0 < penalty[3].item() < penalty[4].item() < 1.0


def test_frequency_penalty_uses_actual_frequency_and_cubic_proxy() -> None:
    inputs = _reward_inputs(3)
    inputs["actual_flap_frequency_hz"][:] = torch.tensor([0.0, 2.5, 5.0])
    inputs["applied_action"][:, 0] = torch.tensor([1.0, -1.0, 0.5])
    inputs["previous_applied_action"].copy_(inputs["applied_action"])

    terms = pure_rl_reward.compute_pure_rl_reward_terms(**inputs)

    torch.testing.assert_close(
        terms.flap_penalty,
        torch.tensor([0.0, 0.125, 1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(terms.frequency_action_delta_penalty, torch.zeros(3, dtype=torch.float64))


def test_constant_nonzero_trim_has_no_delta_penalty_and_limit_penalty_is_soft() -> None:
    inputs = _reward_inputs(3)
    actions = torch.tensor(
        [[0.0, 0.5, -0.5, 0.25], [0.0, 0.8, -0.8, 0.0], [0.0, 0.9, -0.9, 0.0]],
        dtype=torch.float64,
    )
    inputs["applied_action"].copy_(actions)
    inputs["previous_applied_action"].copy_(actions)

    terms = pure_rl_reward.compute_pure_rl_reward_terms(**inputs)

    torch.testing.assert_close(terms.frequency_action_delta_penalty, torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(terms.tail_action_delta_penalty, torch.zeros(3, dtype=torch.float64))
    assert terms.tail_action_limit_penalty[0].item() == pytest.approx(0.0)
    assert terms.tail_action_limit_penalty[1].item() == pytest.approx(0.0)
    assert terms.tail_action_limit_penalty[2].item() == pytest.approx(1.0 / 6.0)


def test_full_range_action_jump_normalizes_delta_penalties_to_one() -> None:
    inputs = _reward_inputs(1)
    inputs["previous_applied_action"][:] = -1.0
    inputs["applied_action"][:] = 1.0

    terms = pure_rl_reward.compute_pure_rl_reward_terms(**inputs)

    assert terms.frequency_action_delta_penalty.item() == pytest.approx(1.0)
    assert terms.tail_action_delta_penalty.item() == pytest.approx(1.0)
    assert terms.tail_action_limit_penalty.item() == pytest.approx(1.0)


def test_termination_reports_each_cause_and_detects_inverted_attitude() -> None:
    height = torch.tensor([10.0, 0.0, 10.0, 10.0, 10.0, 10.0], dtype=torch.float64)
    cross_track = torch.tensor([0.0, 0.0, 0.0, 3.1, 0.0, 0.0], dtype=torch.float64)
    height_error = torch.tensor([0.0, -10.0, 0.0, 0.0, 3.1, 0.0], dtype=torch.float64)
    gravity = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    terms = pure_rl_reward.compute_pure_rl_termination_terms(
        height_m=height,
        cross_track_error_m=cross_track,
        height_error_m=height_error,
        projected_gravity_body=gravity,
    )

    assert terms.ground.tolist() == [False, True, False, False, False, False]
    assert terms.tilt.tolist() == [False, False, True, False, False, True]
    assert terms.cross_track.tolist() == [False, False, False, True, False, False]
    assert terms.height_error.tolist() == [False, True, False, False, True, False]
    assert terms.terminated.tolist() == [False, True, True, True, True, True]
    assert terms.tilt_rad[0].item() == pytest.approx(0.0)
    assert terms.tilt_rad[5].item() == pytest.approx(math.pi)


def test_builders_preserve_batch_one_float32_and_fail_closed_on_invalid_input() -> None:
    inputs = _reward_inputs(1, dtype=torch.float32)
    terms = pure_rl_reward.compute_pure_rl_reward_terms(**inputs)
    assert terms.total_reward.shape == (1,)
    assert terms.total_reward.dtype == torch.float32

    inputs["actual_flap_frequency_hz"][:] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        pure_rl_reward.compute_pure_rl_reward_terms(**inputs)

    with pytest.raises(ValueError, match="shape"):
        pure_rl_reward.compute_pure_rl_termination_terms(
            height_m=torch.zeros(1),
            cross_track_error_m=torch.zeros(1),
            height_error_m=torch.zeros(1),
            projected_gravity_body=torch.zeros((1, 2)),
        )

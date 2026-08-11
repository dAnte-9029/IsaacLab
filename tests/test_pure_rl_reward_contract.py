from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, fields
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


def _reward_inputs(
    count: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    return {
        "cross_track_error_m": torch.zeros(count, dtype=dtype, device=device),
        "height_error_m": torch.zeros(count, dtype=dtype, device=device),
        "along_track_velocity_mps": torch.zeros(count, dtype=dtype, device=device),
        "cross_track_velocity_mps": torch.zeros(count, dtype=dtype, device=device),
        "vertical_velocity_mps": torch.zeros(count, dtype=dtype, device=device),
        "roll_rad": torch.zeros(count, dtype=dtype, device=device),
        "pitch_rad": torch.zeros(count, dtype=dtype, device=device),
        "angular_velocity_body_rad_s": torch.zeros((count, 3), dtype=dtype, device=device),
        "actual_flap_frequency_hz": torch.zeros(count, dtype=dtype, device=device),
        "frequency_slew_hz_per_s": torch.zeros(count, dtype=dtype, device=device),
        "applied_action": torch.zeros((count, 4), dtype=dtype, device=device),
        "previous_applied_action": torch.zeros((count, 4), dtype=dtype, device=device),
    }


def _path_reward_inputs(
    count: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    inputs = _reward_inputs(count, dtype=dtype, device=device)
    inputs["tangent_velocity_mps"] = inputs.pop("along_track_velocity_mps")
    inputs["lateral_normal_velocity_mps"] = inputs.pop("cross_track_velocity_mps")
    inputs["vertical_normal_velocity_mps"] = inputs.pop("vertical_velocity_mps")
    return inputs


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
        "frequency_slew_penalty",
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


def test_path_reward_uses_tangent_progress_and_both_normal_velocity_components() -> None:
    dtype = torch.float64
    slope_rad = math.radians(6.0)
    climb_tangent = torch.tensor([math.cos(slope_rad), 0.0, math.sin(slope_rad)], dtype=dtype)
    world_velocity = 7.0 * climb_tangent
    climb_inputs = _reward_inputs(1, dtype=dtype)
    climb_inputs.pop("along_track_velocity_mps")
    climb_inputs.pop("cross_track_velocity_mps")
    climb_inputs.pop("vertical_velocity_mps")
    climb_terms = pure_rl_reward.compute_pure_rl_path_reward_terms(
        **climb_inputs,
        tangent_velocity_mps=torch.tensor([7.0], dtype=dtype),
        lateral_normal_velocity_mps=torch.tensor([0.0], dtype=dtype),
        vertical_normal_velocity_mps=torch.tensor([0.0], dtype=dtype),
    )

    assert climb_terms.progress_reward.item() > 0.0
    assert climb_terms.velocity_reward.item() == pytest.approx(1.0)

    level_inputs = _reward_inputs(1, dtype=dtype)
    level_inputs.pop("along_track_velocity_mps")
    level_inputs.pop("cross_track_velocity_mps")
    level_inputs.pop("vertical_velocity_mps")
    level_terms = pure_rl_reward.compute_pure_rl_path_reward_terms(
        **level_inputs,
        tangent_velocity_mps=world_velocity[0:1],
        lateral_normal_velocity_mps=world_velocity[1:2],
        vertical_normal_velocity_mps=world_velocity[2:3],
    )

    assert level_terms.velocity_reward.item() < climb_terms.velocity_reward.item()


def test_c1_reward_wrapper_is_exactly_equal_to_path_reward_api() -> None:
    old_inputs = _reward_inputs(4)
    old_inputs["cross_track_error_m"][:] = torch.tensor([-1.2, -0.3, 0.4, 1.7])
    old_inputs["height_error_m"][:] = torch.tensor([0.8, -0.2, 0.0, 1.1])
    old_inputs["along_track_velocity_mps"][:] = torch.tensor([-1.0, 0.0, 4.0, 8.0])
    old_inputs["cross_track_velocity_mps"][:] = torch.tensor([0.5, -1.5, 0.0, 2.0])
    old_inputs["vertical_velocity_mps"][:] = torch.tensor([-0.8, 0.2, 1.2, 0.0])
    old_inputs["roll_rad"][:] = torch.tensor([0.0, 0.1, -0.4, 0.7])
    old_inputs["pitch_rad"][:] = torch.tensor([0.0, 0.3, -0.6, 0.9])
    old_inputs["angular_velocity_body_rad_s"][:] = torch.arange(12, dtype=torch.float64).reshape(4, 3) / 10.0
    old_inputs["actual_flap_frequency_hz"][:] = torch.tensor([0.0, 2.0, 3.5, 5.0])
    old_inputs["frequency_slew_hz_per_s"][:] = torch.tensor([0.0, -0.5, 1.0, 2.0])
    old_inputs["applied_action"][:] = torch.linspace(-0.9, 0.9, 16, dtype=torch.float64).reshape(4, 4)
    old_inputs["previous_applied_action"][:] = old_inputs["applied_action"] - 0.1

    old_terms = pure_rl_reward.compute_pure_rl_reward_terms(**old_inputs)
    path_inputs = dict(old_inputs)
    path_inputs["tangent_velocity_mps"] = path_inputs.pop("along_track_velocity_mps")
    path_inputs["lateral_normal_velocity_mps"] = path_inputs.pop("cross_track_velocity_mps")
    path_inputs["vertical_normal_velocity_mps"] = path_inputs.pop("vertical_velocity_mps")
    path_terms = pure_rl_reward.compute_pure_rl_path_reward_terms(**path_inputs)

    for field in fields(pure_rl_reward.PureRLRewardTerms):
        torch.testing.assert_close(
            getattr(old_terms, field.name),
            getattr(path_terms, field.name),
            rtol=0.0,
            atol=0.0,
        )


def test_pitch_is_free_inside_envelope_and_penalized_smoothly_outside() -> None:
    inputs = _reward_inputs(5)
    inputs["pitch_rad"][:] = torch.deg2rad(torch.tensor([0.0, 20.0, 30.0, 35.0, 45.0]))

    penalty = pure_rl_reward.compute_pure_rl_reward_terms(**inputs).pitch_envelope_penalty

    torch.testing.assert_close(penalty[:3], torch.zeros(3, dtype=torch.float64))
    assert 0.0 < penalty[3].item() < penalty[4].item() < 1.0


def test_frequency_penalty_uses_actual_frequency_and_cubic_proxy() -> None:
    inputs = _reward_inputs(3)
    inputs["actual_flap_frequency_hz"][:] = torch.tensor([0.0, 2.5, 5.0])
    inputs["frequency_slew_hz_per_s"][:] = torch.tensor([0.0, 1.0, -2.0])

    terms = pure_rl_reward.compute_pure_rl_reward_terms(**inputs)

    torch.testing.assert_close(
        terms.flap_penalty,
        torch.tensor([0.0, 0.125, 1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        terms.frequency_slew_penalty,
        torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64),
    )


def test_constant_nonzero_trim_has_no_delta_penalty_and_limit_penalty_is_soft() -> None:
    inputs = _reward_inputs(3)
    actions = torch.tensor(
        [[0.0, 0.5, -0.5, 0.25], [0.0, 0.8, -0.8, 0.0], [0.0, 0.9, -0.9, 0.0]],
        dtype=torch.float64,
    )
    inputs["applied_action"].copy_(actions)
    inputs["previous_applied_action"].copy_(actions)

    terms = pure_rl_reward.compute_pure_rl_reward_terms(**inputs)

    torch.testing.assert_close(terms.frequency_slew_penalty, torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(terms.tail_action_delta_penalty, torch.zeros(3, dtype=torch.float64))
    assert terms.tail_action_limit_penalty[0].item() == pytest.approx(0.0)
    assert terms.tail_action_limit_penalty[1].item() == pytest.approx(0.0)
    assert terms.tail_action_limit_penalty[2].item() == pytest.approx(1.0 / 6.0)


def test_physical_frequency_slew_and_full_range_tail_jump_normalize_penalties_to_one() -> None:
    inputs = _reward_inputs(1)
    inputs["previous_applied_action"][:] = -1.0
    inputs["applied_action"][:] = 1.0
    inputs["frequency_slew_hz_per_s"][:] = -2.0

    terms = pure_rl_reward.compute_pure_rl_reward_terms(**inputs)

    assert terms.frequency_slew_penalty.item() == pytest.approx(1.0)
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


def test_spatial_reward_at_zero_turn_activity_is_exactly_the_base_path_reward() -> None:
    inputs = _path_reward_inputs(4)
    inputs["cross_track_error_m"][:] = torch.tensor([-1.2, -0.3, 0.4, 1.7])
    inputs["height_error_m"][:] = torch.tensor([0.8, -0.2, 0.0, 1.1])
    inputs["tangent_velocity_mps"][:] = torch.tensor([-1.0, 0.0, 4.0, 8.0])
    inputs["lateral_normal_velocity_mps"][:] = torch.tensor([0.5, -1.5, 0.0, 2.0])
    inputs["vertical_normal_velocity_mps"][:] = torch.tensor([-0.8, 0.2, 1.2, 0.0])
    inputs["roll_rad"][:] = torch.tensor([0.0, 0.1, -0.4, 0.7])
    inputs["pitch_rad"][:] = torch.tensor([0.0, 0.3, -0.6, 0.9])
    inputs["angular_velocity_body_rad_s"][:] = torch.arange(12, dtype=torch.float64).reshape(4, 3) / 10.0
    inputs["actual_flap_frequency_hz"][:] = torch.tensor([0.0, 2.0, 3.5, 5.0])
    inputs["frequency_slew_hz_per_s"][:] = torch.tensor([0.0, -0.5, 1.0, 2.0])
    inputs["applied_action"][:] = torch.linspace(-0.9, 0.9, 16, dtype=torch.float64).reshape(4, 4)
    inputs["previous_applied_action"][:] = inputs["applied_action"] - 0.1

    baseline = pure_rl_reward.compute_pure_rl_path_reward_terms(**inputs)
    spatial = pure_rl_reward.compute_pure_rl_spatial_path_reward_terms(
        **inputs,
        turn_activity=torch.zeros(4, dtype=torch.float64),
    )

    for field in fields(pure_rl_reward.PureRLRewardTerms):
        torch.testing.assert_close(
            getattr(spatial, field.name),
            getattr(baseline, field.name),
            rtol=0.0,
            atol=0.0,
        )


def test_spatial_roll_reward_has_approved_boundaries_and_smooth_activity_blend() -> None:
    rolls_deg = torch.tensor([0.0, 25.0, 30.0, 35.0], dtype=torch.float64)
    inputs = _path_reward_inputs(12)
    inputs["roll_rad"][:] = torch.deg2rad(rolls_deg.repeat(3))
    turn_activity = torch.tensor([0.0] * 4 + [0.5] * 4 + [1.0] * 4, dtype=torch.float64)

    baseline = pure_rl_reward.compute_pure_rl_path_reward_terms(**inputs)
    spatial = pure_rl_reward.compute_pure_rl_spatial_path_reward_terms(
        **inputs,
        turn_activity=turn_activity,
    )

    active_turn_roll_reward = torch.tensor([1.0, 1.0, 0.75, 0.0], dtype=torch.float64)
    torch.testing.assert_close(spatial.roll_reward[:4], baseline.roll_reward[:4], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        spatial.roll_reward[4:8],
        0.5 * (baseline.roll_reward[4:8] + active_turn_roll_reward),
    )
    torch.testing.assert_close(spatial.roll_reward[8:], active_turn_roll_reward)

    expected_total = baseline.total_reward + (
        pure_rl_reward.PURE_RL_CURRICULUM1_REWARD_CONFIG.roll_reward_weight
        * (spatial.roll_reward - baseline.roll_reward)
    )
    torch.testing.assert_close(spatial.total_reward, expected_total, rtol=0.0, atol=0.0)
    for field in fields(pure_rl_reward.PureRLRewardTerms):
        if field.name not in {"roll_reward", "total_reward"}:
            torch.testing.assert_close(
                getattr(spatial, field.name),
                getattr(baseline, field.name),
                rtol=0.0,
                atol=0.0,
            )


def test_spatial_roll_reward_is_sign_symmetric() -> None:
    inputs = _path_reward_inputs(8)
    positive_roll = torch.deg2rad(torch.tensor([0.0, 25.0, 30.0, 35.0], dtype=torch.float64))
    inputs["roll_rad"][:] = torch.cat((positive_roll, -positive_roll))

    terms = pure_rl_reward.compute_pure_rl_spatial_path_reward_terms(
        **inputs,
        turn_activity=torch.ones(8, dtype=torch.float64),
    )

    torch.testing.assert_close(terms.roll_reward[:4], terms.roll_reward[4:], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("turn_activity", "message"),
    [
        (torch.zeros((2, 1)), "shape"),
        (torch.tensor([0.0, float("nan")]), "finite"),
        (torch.tensor([-0.1, 0.0]), r"\[0, 1\]"),
        (torch.tensor([0.0, 1.1]), r"\[0, 1\]"),
    ],
)
def test_spatial_reward_rejects_invalid_turn_activity(turn_activity: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pure_rl_reward.compute_pure_rl_spatial_path_reward_terms(
            **_path_reward_inputs(2, dtype=turn_activity.dtype),
            turn_activity=turn_activity,
        )


def test_spatial_reward_rejects_misaligned_turn_activity() -> None:
    with pytest.raises(ValueError, match="batch dimension"):
        pure_rl_reward.compute_pure_rl_spatial_path_reward_terms(
            **_path_reward_inputs(2),
            turn_activity=torch.zeros(1, dtype=torch.float64),
        )


def test_spatial_termination_preserves_base_causes_and_unions_roll_limit() -> None:
    height = torch.tensor([0.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=torch.float64)
    cross_track = torch.tensor([0.0, 0.0, 3.1, 0.0, 0.0, 0.0], dtype=torch.float64)
    height_error = torch.tensor([0.0, 0.0, 0.0, 3.1, 0.0, 0.0], dtype=torch.float64)
    gravity = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float64,
    )
    roll_rad = torch.deg2rad(torch.tensor([0.0, 0.0, 0.0, 0.0, 25.0, 35.0], dtype=torch.float64))

    base = pure_rl_reward.compute_pure_rl_termination_terms(
        height_m=height,
        cross_track_error_m=cross_track,
        height_error_m=height_error,
        projected_gravity_body=gravity,
    )
    spatial = pure_rl_reward.compute_pure_rl_spatial_termination_terms(
        height_m=height,
        cross_track_error_m=cross_track,
        height_error_m=height_error,
        projected_gravity_body=gravity,
        roll_rad=roll_rad,
    )

    for name in ("ground", "tilt", "cross_track", "height_error", "tilt_rad"):
        torch.testing.assert_close(getattr(spatial, name), getattr(base, name), rtol=0.0, atol=0.0)
    assert spatial.roll_limit.tolist() == [False, False, False, False, False, True]
    assert spatial.terminated.tolist() == [True, True, True, True, False, True]
    assert set(spatial.as_dict()) == {
        "ground",
        "tilt",
        "cross_track",
        "height_error",
        "roll_limit",
        "terminated",
    }
    with pytest.raises(FrozenInstanceError):
        spatial.roll_limit = torch.zeros_like(spatial.roll_limit)


@pytest.mark.parametrize("maximum_abs_roll_rad", [0.0, -1.0, float("inf"), float("nan")])
def test_spatial_termination_rejects_invalid_roll_threshold(maximum_abs_roll_rad: float) -> None:
    with pytest.raises(ValueError, match="maximum_abs_roll_rad"):
        pure_rl_reward.compute_pure_rl_spatial_termination_terms(
            height_m=torch.ones(1),
            cross_track_error_m=torch.zeros(1),
            height_error_m=torch.zeros(1),
            projected_gravity_body=torch.tensor([[0.0, 0.0, -1.0]]),
            roll_rad=torch.zeros(1),
            maximum_abs_roll_rad=maximum_abs_roll_rad,
        )


@pytest.mark.parametrize(
    ("roll_rad", "message"),
    [
        (torch.zeros((1, 1)), "shape"),
        (torch.tensor([float("nan")]), "finite"),
        (torch.zeros(2), "batch dimension"),
    ],
)
def test_spatial_termination_rejects_invalid_roll_vector(roll_rad: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pure_rl_reward.compute_pure_rl_spatial_termination_terms(
            height_m=torch.ones(1),
            cross_track_error_m=torch.zeros(1),
            height_error_m=torch.zeros(1),
            projected_gravity_body=torch.tensor([[0.0, 0.0, -1.0]]),
            roll_rad=roll_rad,
        )


def test_spatial_builders_preserve_batch_one_dtype_and_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = _path_reward_inputs(1, dtype=torch.float32, device=device)
    reward_terms = pure_rl_reward.compute_pure_rl_spatial_path_reward_terms(
        **inputs,
        turn_activity=torch.tensor([0.5], dtype=torch.float32, device=device),
    )
    termination_terms = pure_rl_reward.compute_pure_rl_spatial_termination_terms(
        height_m=torch.ones(1, dtype=torch.float32, device=device),
        cross_track_error_m=torch.zeros(1, dtype=torch.float32, device=device),
        height_error_m=torch.zeros(1, dtype=torch.float32, device=device),
        projected_gravity_body=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device),
        roll_rad=torch.zeros(1, dtype=torch.float32, device=device),
    )

    assert reward_terms.total_reward.shape == (1,)
    assert reward_terms.total_reward.dtype == torch.float32
    assert reward_terms.total_reward.device == inputs["roll_rad"].device
    assert termination_terms.terminated.shape == (1,)
    assert termination_terms.roll_limit.dtype == torch.bool
    assert termination_terms.roll_limit.device == inputs["roll_rad"].device

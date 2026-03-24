from __future__ import annotations

import math

import torch

from flapping_bot.direct.flapping_bot.path_tracking_env import (
    _compute_curve_aware_teacher_delta,
    _compute_loiter_angular_progress,
    _compute_loiter_milestone_bonus,
    _compute_loiter_radial_error,
    _compute_loiter_tracking_quality,
    _compute_path_completion_mask,
    _compute_path_episode_length_s,
    _compute_stall_mask,
    _compute_tracking_reward,
    _resolve_path_tracking_curriculum,
)


def test_tracking_reward_prefers_progress_with_small_errors():
    reward_better = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.4]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    reward_worse = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.1]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    assert reward_better.item() > reward_worse.item()


def test_compute_loiter_radial_error_is_zero_on_target_circle() -> None:
    radial_error = _compute_loiter_radial_error(
        position_xy=torch.tensor([[20.0, 0.0]], dtype=torch.float32),
        center_xy=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        radius_m=torch.tensor([20.0], dtype=torch.float32),
    )

    assert torch.allclose(radial_error, torch.tensor([0.0], dtype=torch.float32))


def test_compute_loiter_angular_progress_reaches_quarter_turn() -> None:
    progress = _compute_loiter_angular_progress(
        start_angle_rad=torch.tensor([0.0], dtype=torch.float32),
        current_angle_rad=torch.tensor([0.5 * math.pi], dtype=torch.float32),
        turn_direction=torch.tensor([1], dtype=torch.long),
        total_turns=torch.tensor([1.0], dtype=torch.float32),
    )

    assert torch.allclose(progress, torch.tensor([0.25], dtype=torch.float32), atol=1.0e-4)


def test_compute_loiter_angular_progress_handles_wraparound() -> None:
    progress = _compute_loiter_angular_progress(
        start_angle_rad=torch.tensor([1.5 * math.pi], dtype=torch.float32),
        current_angle_rad=torch.tensor([0.0], dtype=torch.float32),
        turn_direction=torch.tensor([1], dtype=torch.long),
        total_turns=torch.tensor([1.0], dtype=torch.float32),
    )

    assert torch.allclose(progress, torch.tensor([0.25], dtype=torch.float32), atol=1.0e-4)


def test_compute_loiter_angular_progress_handles_clockwise_turns() -> None:
    progress = _compute_loiter_angular_progress(
        start_angle_rad=torch.tensor([0.0], dtype=torch.float32),
        current_angle_rad=torch.tensor([1.5 * math.pi], dtype=torch.float32),
        turn_direction=torch.tensor([-1], dtype=torch.long),
        total_turns=torch.tensor([1.0], dtype=torch.float32),
    )

    assert torch.allclose(progress, torch.tensor([0.25], dtype=torch.float32), atol=1.0e-4)


def test_compute_loiter_angular_progress_uses_reference_to_resolve_full_turn() -> None:
    progress = _compute_loiter_angular_progress(
        start_angle_rad=torch.tensor([0.0], dtype=torch.float32),
        current_angle_rad=torch.tensor([0.0], dtype=torch.float32),
        turn_direction=torch.tensor([1], dtype=torch.long),
        total_turns=torch.tensor([1.0], dtype=torch.float32),
        reference_progress_ratio=torch.tensor([0.98], dtype=torch.float32),
    )

    assert torch.allclose(progress, torch.tensor([1.0], dtype=torch.float32), atol=1.0e-4)


def test_compute_loiter_milestone_bonus_rewards_new_thresholds_only() -> None:
    bonus = _compute_loiter_milestone_bonus(
        previous_progress_ratio=torch.tensor([0.30], dtype=torch.float32),
        progress_ratio=torch.tensor([0.55], dtype=torch.float32),
        quarter_turn_bonus=1.0,
        half_turn_bonus=1.5,
        three_quarter_turn_bonus=2.0,
    )

    assert torch.allclose(bonus, torch.tensor([1.5], dtype=torch.float32), atol=1.0e-6)


def test_compute_loiter_milestone_bonus_accumulates_multiple_crossings() -> None:
    bonus = _compute_loiter_milestone_bonus(
        previous_progress_ratio=torch.tensor([0.10], dtype=torch.float32),
        progress_ratio=torch.tensor([0.80], dtype=torch.float32),
        quarter_turn_bonus=1.0,
        half_turn_bonus=1.5,
        three_quarter_turn_bonus=2.0,
    )

    assert torch.allclose(bonus, torch.tensor([4.5], dtype=torch.float32), atol=1.0e-6)


def test_compute_loiter_tracking_quality_drops_when_orbit_tracking_degrades() -> None:
    quality_tight = _compute_loiter_tracking_quality(
        loiter_radial_error=torch.tensor([0.2], dtype=torch.float32),
        align_error=torch.tensor([0.05], dtype=torch.float32),
        loiter_tracking_radial_scale_m=1.0,
        loiter_tracking_align_scale_rad=0.25,
    )
    quality_wide = _compute_loiter_tracking_quality(
        loiter_radial_error=torch.tensor([4.0], dtype=torch.float32),
        align_error=torch.tensor([0.8], dtype=torch.float32),
        loiter_tracking_radial_scale_m=1.0,
        loiter_tracking_align_scale_rad=0.25,
    )

    assert quality_tight.item() > quality_wide.item()


def test_tracking_reward_penalizes_lateral_and_height_error() -> None:
    reward_small_error = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    reward_large_error = _compute_tracking_reward(
        lateral_error=torch.tensor([2.0]),
        height_error=torch.tensor([1.5]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    assert reward_small_error.item() > reward_large_error.item()


def test_tracking_reward_penalizes_action_rate() -> None:
    reward_smooth = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    reward_aggressive = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.full((1, 4), 0.8),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    assert reward_smooth.item() > reward_aggressive.item()


def test_tracking_reward_applies_termination_penalty() -> None:
    reward_alive = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
    )
    reward_terminated = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([True]),
    )
    assert reward_alive.item() > reward_terminated.item()


def test_tracking_reward_penalizes_stalled_episodes() -> None:
    reward_moving = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        stalled=torch.tensor([False]),
        no_progress_penalty=0.5,
    )
    reward_stalled = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        stalled=torch.tensor([True]),
        no_progress_penalty=0.5,
    )

    assert reward_moving.item() > reward_stalled.item()


def test_tracking_reward_penalizes_teacher_action_gap() -> None:
    reward_small_gap = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        teacher_action_gap_abs=torch.full((1, 4), 0.05),
        teacher_action_gap_penalty=0.5,
    )
    reward_large_gap = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        teacher_action_gap_abs=torch.full((1, 4), 0.8),
        teacher_action_gap_penalty=0.5,
    )

    assert reward_small_gap.item() > reward_large_gap.item()


def test_loiter_reward_scales_teacher_action_gap_penalty_more_aggressively() -> None:
    reward_non_loiter = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([False]),
        teacher_action_gap_abs=torch.full((1, 4), 0.3),
        teacher_action_gap_penalty=0.5,
        loiter_teacher_action_gap_penalty_scale=3.0,
    )
    reward_loiter = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([True]),
        teacher_action_gap_abs=torch.full((1, 4), 0.3),
        teacher_action_gap_penalty=0.5,
        loiter_teacher_action_gap_penalty_scale=3.0,
    )

    assert reward_non_loiter.item() > reward_loiter.item()


def test_tracking_reward_rewards_curve_tracking_when_progress_matches() -> None:
    reward_on_curve = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        curvature_m_inv=torch.tensor([0.05]),
        completed=torch.tensor([False]),
        curve_tracking_bonus=0.6,
        curve_progress_bonus=0.25,
    )
    reward_off_curve = _compute_tracking_reward(
        lateral_error=torch.tensor([2.5]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.7]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        curvature_m_inv=torch.tensor([0.05]),
        completed=torch.tensor([False]),
        curve_tracking_bonus=0.6,
        curve_progress_bonus=0.25,
    )

    assert reward_on_curve.item() > reward_off_curve.item()


def test_curve_tracking_bonus_requires_progress_for_positive_shaping() -> None:
    reward_no_progress = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.0]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        curvature_m_inv=torch.tensor([0.05]),
        completed=torch.tensor([False]),
        curve_tracking_bonus=0.6,
        curve_progress_bonus=0.25,
    )
    reward_with_progress = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.05]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        curvature_m_inv=torch.tensor([0.05]),
        completed=torch.tensor([False]),
        curve_tracking_bonus=0.6,
        curve_progress_bonus=0.25,
    )

    assert reward_no_progress.item() <= 0.0
    assert reward_with_progress.item() > reward_no_progress.item()


def test_loiter_reward_prefers_small_radial_error() -> None:
    reward_near = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([True]),
        loiter_radial_error=torch.tensor([0.2]),
        loiter_progress_ratio=torch.tensor([0.3]),
        loiter_tracking_bonus=0.8,
        loiter_progress_bonus=0.2,
    )
    reward_far = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([True]),
        loiter_radial_error=torch.tensor([3.0]),
        loiter_progress_ratio=torch.tensor([0.3]),
        loiter_tracking_bonus=0.8,
        loiter_progress_bonus=0.2,
    )

    assert reward_near.item() > reward_far.item()


def test_loiter_tracking_bonus_requires_progress_for_positive_shaping() -> None:
    reward_no_progress = _compute_tracking_reward(
        lateral_error=torch.tensor([0.64]),
        height_error=torch.tensor([0.34]),
        align_error=torch.tensor([0.17]),
        delta_s=torch.tensor([0.0]),
        airspeed=torch.tensor([6.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.15]),
        ang_rate=torch.tensor([0.2]),
        terminated=torch.tensor([False]),
        curvature_m_inv=torch.tensor([0.05]),
        is_loiter=torch.tensor([True]),
        loiter_radial_error=torch.tensor([0.64]),
        loiter_progress_ratio=torch.tensor([0.02]),
        loiter_tracking_bonus=0.8,
        loiter_progress_bonus=0.2,
        curve_tracking_bonus=0.6,
        curve_progress_bonus=0.25,
    )
    reward_with_progress = _compute_tracking_reward(
        lateral_error=torch.tensor([0.64]),
        height_error=torch.tensor([0.34]),
        align_error=torch.tensor([0.17]),
        delta_s=torch.tensor([0.05]),
        airspeed=torch.tensor([6.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.15]),
        ang_rate=torch.tensor([0.2]),
        terminated=torch.tensor([False]),
        curvature_m_inv=torch.tensor([0.05]),
        is_loiter=torch.tensor([True]),
        loiter_radial_error=torch.tensor([0.64]),
        loiter_progress_ratio=torch.tensor([0.02]),
        loiter_tracking_bonus=0.8,
        loiter_progress_bonus=0.2,
        curve_tracking_bonus=0.6,
        curve_progress_bonus=0.25,
    )

    assert reward_no_progress.item() <= 0.0
    assert reward_with_progress.item() > reward_no_progress.item()


def test_loiter_progress_bonus_does_not_reward_absolute_orbit_progress() -> None:
    reward_early = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([True]),
        loiter_radial_error=torch.tensor([0.2]),
        loiter_progress_ratio=torch.tensor([0.1]),
        loiter_tracking_bonus=0.8,
        loiter_progress_bonus=0.2,
    )
    reward_late = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([True]),
        loiter_radial_error=torch.tensor([0.2]),
        loiter_progress_ratio=torch.tensor([0.9]),
        loiter_tracking_bonus=0.8,
        loiter_progress_bonus=0.2,
    )

    assert torch.allclose(reward_early, reward_late)


def test_loiter_reward_penalizes_outward_radial_drift() -> None:
    reward_stable_radius = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([True]),
        loiter_radial_error=torch.tensor([0.2]),
        loiter_radial_error_delta=torch.tensor([0.0]),
        loiter_tracking_bonus=0.8,
        loiter_progress_bonus=0.2,
        loiter_radial_drift_penalty=5.0,
    )
    reward_outward_drift = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([True]),
        loiter_radial_error=torch.tensor([0.2]),
        loiter_radial_error_delta=torch.tensor([0.3]),
        loiter_tracking_bonus=0.8,
        loiter_progress_bonus=0.2,
        loiter_radial_drift_penalty=5.0,
    )

    assert reward_stable_radius.item() > reward_outward_drift.item()


def test_loiter_reward_applies_quarter_turn_bonus_once() -> None:
    reward_without_bonus = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([True]),
        loiter_radial_error=torch.tensor([0.2]),
        loiter_progress_ratio=torch.tensor([0.3]),
        loiter_milestone_bonus_reward=torch.tensor([0.0]),
        loiter_tracking_bonus=0.8,
    )
    reward_with_bonus = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([True]),
        loiter_radial_error=torch.tensor([0.2]),
        loiter_progress_ratio=torch.tensor([0.3]),
        loiter_milestone_bonus_reward=torch.tensor([1.0]),
        loiter_tracking_bonus=0.8,
    )

    assert reward_with_bonus.item() > reward_without_bonus.item()


def test_non_loiter_reward_ignores_loiter_specific_terms() -> None:
    reward_nominal = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([False]),
        loiter_radial_error=torch.tensor([0.2]),
        loiter_progress_ratio=torch.tensor([0.3]),
        loiter_milestone_bonus_reward=torch.tensor([0.0]),
        loiter_tracking_bonus=0.8,
        loiter_progress_bonus=0.2,
    )
    reward_changed_loiter_terms = _compute_tracking_reward(
        lateral_error=torch.tensor([0.2]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        is_loiter=torch.tensor([False]),
        loiter_radial_error=torch.tensor([3.0]),
        loiter_progress_ratio=torch.tensor([0.9]),
        loiter_milestone_bonus_reward=torch.tensor([1.0]),
        loiter_tracking_bonus=0.8,
        loiter_progress_bonus=0.2,
    )

    assert torch.allclose(reward_nominal, reward_changed_loiter_terms)


def test_tracking_reward_applies_completion_bonus() -> None:
    reward_incomplete = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        completed=torch.tensor([False]),
        completion_bonus=4.0,
    )
    reward_complete = _compute_tracking_reward(
        lateral_error=torch.tensor([0.1]),
        height_error=torch.tensor([0.1]),
        align_error=torch.tensor([0.05]),
        delta_s=torch.tensor([0.2]),
        airspeed=torch.tensor([7.0]),
        action=torch.zeros((1, 4)),
        action_delta=torch.zeros((1, 4)),
        tilt=torch.tensor([0.1]),
        ang_rate=torch.tensor([0.1]),
        terminated=torch.tensor([False]),
        completed=torch.tensor([True]),
        completion_bonus=4.0,
    )

    assert reward_complete.item() > reward_incomplete.item()


def test_compute_stall_mask_exempts_completed_missions() -> None:
    stalled = _compute_stall_mask(
        window_ready=torch.tensor([True, True, False]),
        progress_delta=torch.tensor([0.0, 0.0, 0.0]),
        progress_s=torch.tensor([9.8, 9.2, 5.0]),
        path_total_length_m=torch.tensor([10.0, 10.0, 10.0]),
        min_delta_s_m=0.5,
    )

    assert torch.equal(stalled, torch.tensor([False, True, False]))


def test_compute_path_completion_mask_marks_completed_progress() -> None:
    completed = _compute_path_completion_mask(
        progress_s=torch.tensor([9.8, 9.7, 5.0]),
        path_total_length_m=torch.tensor([10.0, 10.0, 10.0]),
        completion_ratio=0.98,
    )

    assert torch.equal(completed, torch.tensor([True, False, False]))


def test_compute_path_episode_length_s_extends_loiter_training_horizon() -> None:
    episode_length_s = _compute_path_episode_length_s(
        current_episode_length_s=18.0,
        path_total_length_m=2.0 * math.pi * 20.0,
        speed_ref_mps=6.0,
        freeze_steps=240,
        sim_dt=1.0 / 240.0,
        completion_margin_s=3.0,
        max_episode_length_s=36.0,
    )

    assert episode_length_s > 24.0


def test_resolve_path_tracking_curriculum_stages_delay_loiter() -> None:
    early = _resolve_path_tracking_curriculum(
        step=0,
        enabled=True,
        stage_steps=(0, 12_000, 36_000),
        stage_modes=("turn_only", "straight_turn", "all"),
        allow_straight=True,
        allow_turn=True,
        allow_loiter=True,
    )
    middle = _resolve_path_tracking_curriculum(
        step=20_000,
        enabled=True,
        stage_steps=(0, 12_000, 36_000),
        stage_modes=("turn_only", "straight_turn", "all"),
        allow_straight=True,
        allow_turn=True,
        allow_loiter=True,
    )
    late = _resolve_path_tracking_curriculum(
        step=50_000,
        enabled=True,
        stage_steps=(0, 12_000, 36_000),
        stage_modes=("turn_only", "straight_turn", "all"),
        allow_straight=True,
        allow_turn=True,
        allow_loiter=True,
    )

    assert early == (False, True, False)
    assert middle == (True, True, False)
    assert late == (True, True, True)


def test_resolve_path_tracking_curriculum_preserves_explicit_eval_primitive() -> None:
    straight_only = _resolve_path_tracking_curriculum(
        step=0,
        enabled=True,
        stage_steps=(0, 12_000, 36_000),
        stage_modes=("turn_only", "straight_turn", "all"),
        allow_straight=True,
        allow_turn=False,
        allow_loiter=False,
    )
    loiter_only = _resolve_path_tracking_curriculum(
        step=0,
        enabled=True,
        stage_steps=(0, 12_000, 36_000),
        stage_modes=("turn_only", "straight_turn", "all"),
        allow_straight=False,
        allow_turn=False,
        allow_loiter=True,
    )

    assert straight_only == (True, False, False)
    assert loiter_only == (False, False, True)


def test_curve_aware_teacher_delta_tightens_on_curved_segments() -> None:
    delta = _compute_curve_aware_teacher_delta(
        base_delta=1.0,
        curvature_m_inv=torch.tensor([0.0, 0.05]),
        curvature_ref_m_inv=0.05,
        min_scale=0.5,
    )

    assert torch.allclose(delta, torch.tensor([[1.0], [0.5]]))

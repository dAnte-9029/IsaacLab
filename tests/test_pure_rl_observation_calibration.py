"""Pure contracts for actor-observation calibration gates."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from flapping_bot.analysis.pure_rl_observation_calibration import (
    PARTITION_CALIBRATION,
    PARTITION_VERIFICATION,
    build_level_straight_preview,
    build_scripted_actions,
    quaternion_wxyz_from_roll_pitch_yaw,
    rotate_body_vectors_to_world,
    rotate_world_vectors_to_body,
    run_synthetic_envelope_gate,
    summarize_scalar_samples,
)


def test_quaternion_vector_rotation_round_trip_preserves_vectors() -> None:
    roll = torch.tensor([0.2, -0.4], dtype=torch.float64)
    pitch = torch.tensor([-0.3, 0.1], dtype=torch.float64)
    yaw = torch.tensor([1.1, -2.0], dtype=torch.float64)
    quaternion = quaternion_wxyz_from_roll_pitch_yaw(roll, pitch, yaw)
    body_vector = torch.tensor([[3.0, -2.0, 0.5], [-1.0, 4.0, 2.0]], dtype=torch.float64)

    world_vector = rotate_body_vectors_to_world(quaternion, body_vector)
    recovered = rotate_world_vectors_to_body(quaternion, world_vector)

    torch.testing.assert_close(recovered, body_vector, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(quaternion, dim=1),
        torch.ones(2, dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_level_preview_uses_current_along_track_speed_and_body_geometry() -> None:
    position = torch.tensor([[2.0, 1.0, 9.0]], dtype=torch.float64)
    orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    velocity = torch.tensor([[5.0, 10.0, 0.0]], dtype=torch.float64)
    route_origin = torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float64)
    route_tangent = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

    preview, preview_speed, along_track = build_level_straight_preview(
        vehicle_position_world_m=position,
        orientation_world_wxyz=orientation,
        ground_velocity_world_mps=velocity,
        route_origin_world_m=route_origin,
        route_tangent_world=route_tangent,
        minimum_preview_speed_mps=1.0,
        maximum_preview_speed_mps=12.0,
    )

    torch.testing.assert_close(preview_speed, torch.tensor([5.0], dtype=torch.float64))
    torch.testing.assert_close(along_track, torch.tensor([5.0], dtype=torch.float64))
    expected_x = torch.tensor([0.6, 1.2, 1.8, 2.4, 3.0], dtype=torch.float64)
    torch.testing.assert_close(preview[0, :, 0], expected_x)
    torch.testing.assert_close(preview[0, :, 1], torch.full((5,), -1.0, dtype=torch.float64))
    torch.testing.assert_close(preview[0, :, 2], torch.full((5,), 1.0, dtype=torch.float64))


def test_scripted_actions_cover_frequency_extremes_and_remain_bounded() -> None:
    base = torch.zeros((16, 4), dtype=torch.float64)

    action = build_scripted_actions(
        base_action=base,
        policy_step=6,
        policy_dt_s=1.0 / 60.0,
        minimum_frequency_hz=0.0,
        maximum_frequency_hz=5.0,
    )

    assert action[1, 0].item() == pytest.approx(-1.0)
    assert action[2, 0].item() == pytest.approx(0.0)
    assert action[3, 0].item() == pytest.approx(1.0)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    assert action[4, 1].item() != 0.0
    assert action[5, 2].item() != 0.0
    assert action[6, 3].item() != 0.0


def test_scalar_summary_reports_robust_and_tail_statistics() -> None:
    summary = summarize_scalar_samples(np.array([-10.0, -1.0, 0.0, 1.0, 10.0]))

    assert summary["count"] == 5
    assert summary["minimum"] == -10.0
    assert summary["maximum"] == 10.0
    assert summary["median"] == 0.0
    assert summary["median_absolute_deviation"] == 1.0
    assert summary["p99_absolute"] == pytest.approx(10.0)


def test_synthetic_envelope_gate_keeps_raw_contract_and_disjoint_partitions() -> None:
    result = run_synthetic_envelope_gate(sample_count=256, seed=1234)

    assert result.summary["all_cases_accepted"] is True
    assert result.summary["schema_version"] == "pure_rl_observation_calibration_v2"
    assert result.summary["raw_samples_retained"] is True
    assert result.summary["normalization_contract_evaluated"] is True
    assert result.summary["safety_clipping_applied_to_saved_samples"] is False
    assert result.summary["normalized_preclip_all_finite"] is True
    assert result.summary["would_clip_value_count"] == 0
    assert result.summary["would_clip_sample_count"] == 0
    assert float(result.summary["normalized_preclip_maximum_absolute_value"]) <= 5.0
    assert result.summary["raw_observation_dimension"] == 555
    partition = result.samples["partition"]
    assert np.count_nonzero(partition == PARTITION_CALIBRATION) == 192
    assert np.count_nonzero(partition == PARTITION_VERIFICATION) == 64
    assert set(np.unique(partition)) == {PARTITION_CALIBRATION, PARTITION_VERIFICATION}
    sensor = result.samples["sensor_frame"]
    assert sensor.shape == (256, 14)
    assert np.all(np.isfinite(sensor))
    assert np.max(sensor[:, 11]) <= 5.0
    assert np.min(sensor[:, 11]) >= 0.0
    assert result.summary["preview_speed_bounds_are_provisional"] is False
    assert result.summary["preview_speed_bounds_are_frozen"] is True


def test_yaw_rotated_route_and_body_produce_equivalent_preview() -> None:
    yaw = torch.tensor([0.0, math.pi / 2.0], dtype=torch.float64)
    zeros = torch.zeros_like(yaw)
    quaternion = quaternion_wxyz_from_roll_pitch_yaw(zeros, zeros, yaw)
    tangent = torch.stack((torch.cos(yaw), torch.sin(yaw), zeros), dim=1)
    position = torch.stack((2.0 * torch.cos(yaw), 2.0 * torch.sin(yaw), torch.full_like(yaw, 10.0)), dim=1)
    velocity = 5.0 * tangent
    route_origin = torch.zeros((2, 3), dtype=torch.float64)
    route_origin[:, 2] = 10.0

    preview, _preview_speed, _along_track = build_level_straight_preview(
        vehicle_position_world_m=position,
        orientation_world_wxyz=quaternion,
        ground_velocity_world_mps=velocity,
        route_origin_world_m=route_origin,
        route_tangent_world=tangent,
    )

    torch.testing.assert_close(preview[0], preview[1], atol=1.0e-12, rtol=0.0)

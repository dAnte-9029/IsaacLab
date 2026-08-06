from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_observation.py"
)
SPEC = importlib.util.spec_from_file_location("pure_rl_observation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pure_rl_observation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pure_rl_observation
SPEC.loader.exec_module(pure_rl_observation)


def test_raw_observation_layout_is_fixed_at_555_values() -> None:
    layout = pure_rl_observation.PURE_RL_RAW_OBSERVATION_LAYOUT

    assert layout.sensor_history_steps == 30
    assert layout.sensor_frame_dim == 14
    assert layout.sensor_history_dim == 420
    assert layout.action_history_steps == 30
    assert layout.action_dim == 4
    assert layout.action_history_dim == 120
    assert layout.preview_point_count == 5
    assert layout.preview_point_dim == 3
    assert layout.preview_dim == 15
    assert layout.observation_dim == 555


def test_raw_sensor_frame_preserves_units_and_contract_order() -> None:
    orientation = torch.tensor([[-2.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    previous_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    ground_velocity = torch.tensor([[12.0, -3.0, 0.5]], dtype=torch.float64)
    angular_velocity = torch.tensor([[0.2, -0.4, 1.5]], dtype=torch.float64)
    forward_air_velocity = torch.tensor([9.25], dtype=torch.float64)
    actual_frequency = torch.tensor([3.75], dtype=torch.float64)
    phase = torch.tensor([math.pi / 2.0], dtype=torch.float64)

    frame = pure_rl_observation.build_raw_sensor_frame(
        orientation_world_wxyz=orientation,
        ground_velocity_body_mps=ground_velocity,
        angular_velocity_body_rad_s=angular_velocity,
        forward_air_velocity_body_mps=forward_air_velocity,
        actual_flap_frequency_hz=actual_frequency,
        flap_phase_rad=phase,
        previous_orientation_world_wxyz=previous_orientation,
    )

    expected = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 12.0, -3.0, 0.5, 0.2, -0.4, 1.5, 9.25, 3.75, 1.0, 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(frame, expected, atol=1.0e-12, rtol=0.0)
    assert frame.dtype == orientation.dtype
    assert frame.device == orientation.device


def test_quaternion_sign_alignment_keeps_temporal_representation_continuous() -> None:
    previous = torch.tensor([[0.01, 0.0, 0.0, 0.99995]], dtype=torch.float64)
    same_rotation_opposite_sign = -previous * 3.0

    aligned = pure_rl_observation.align_quaternion_sign_to_previous(
        same_rotation_opposite_sign,
        previous,
    )
    normalized_previous = pure_rl_observation.normalize_quaternion_wxyz(previous)

    torch.testing.assert_close(aligned, normalized_previous)
    assert float(torch.sum(aligned * normalized_previous, dim=1).item()) > 0.0


def test_reset_history_repeats_real_sample_and_append_is_oldest_to_newest() -> None:
    reset_sample = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    history = pure_rl_observation.initialize_raw_history(reset_sample, history_steps=3)

    expected_reset = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]])
    torch.testing.assert_close(history, expected_reset)

    updated = pure_rl_observation.append_raw_history(
        history,
        torch.tensor([[3.0, 4.0]], dtype=torch.float32),
    )
    expected_updated = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]])
    torch.testing.assert_close(updated, expected_updated)
    torch.testing.assert_close(history, expected_reset)


def test_preview_progress_uses_current_along_track_ground_speed_only() -> None:
    progress = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
    velocity = torch.tensor(
        [
            [7.0, 100.0, 0.0],
            [-2.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
        ],
        dtype=torch.float64,
    )
    tangent = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=torch.float64,
    )

    queries, preview_speed = pure_rl_observation.compute_preview_query_progress_m(
        closest_path_progress_m=progress,
        ground_velocity_world_mps=velocity,
        path_tangent_world=tangent,
        minimum_preview_speed_mps=2.0,
        maximum_preview_speed_mps=10.0,
    )

    torch.testing.assert_close(preview_speed, torch.tensor([7.0, 2.0, 10.0, 6.0], dtype=torch.float64))
    expected_times = torch.tensor([0.12, 0.24, 0.36, 0.48, 0.60], dtype=torch.float64)
    torch.testing.assert_close(queries, progress.unsqueeze(1) + preview_speed.unsqueeze(1) * expected_times)


def test_world_route_rotation_produces_the_same_body_preview_geometry() -> None:
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    yaw_ninety = torch.tensor(
        [[math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]],
        dtype=torch.float64,
    )
    east_points = torch.tensor(
        [[[1.0, 0.0, 0.2], [2.0, 0.0, 0.4], [3.0, 0.0, 0.6], [4.0, 0.0, 0.8], [5.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    north_points = torch.stack((-east_points[..., 1], east_points[..., 0], east_points[..., 2]), dim=2)
    position = torch.zeros((1, 3), dtype=torch.float64)

    east_body = pure_rl_observation.transform_world_preview_points_to_body(
        preview_points_world_m=east_points,
        vehicle_position_world_m=position,
        orientation_world_wxyz=identity,
    )
    north_body = pure_rl_observation.transform_world_preview_points_to_body(
        preview_points_world_m=north_points,
        vehicle_position_world_m=position,
        orientation_world_wxyz=yaw_ninety,
    )

    torch.testing.assert_close(north_body, east_body, atol=1.0e-12, rtol=0.0)


def test_raw_actor_builder_concatenates_history_then_actions_then_preview_without_clipping() -> None:
    layout = pure_rl_observation.PURE_RL_RAW_OBSERVATION_LAYOUT
    batch_size = 2
    sensor = torch.arange(batch_size * layout.sensor_history_dim, dtype=torch.float64).reshape(
        batch_size,
        layout.sensor_history_steps,
        layout.sensor_frame_dim,
    )
    action = (1000.0 + torch.arange(batch_size * layout.action_history_dim, dtype=torch.float64)).reshape(
        batch_size,
        layout.action_history_steps,
        layout.action_dim,
    )
    preview = (2000.0 + torch.arange(batch_size * layout.preview_dim, dtype=torch.float64)).reshape(
        batch_size,
        layout.preview_point_count,
        layout.preview_point_dim,
    )

    observation = pure_rl_observation.build_raw_actor_observation(
        sensor_history=sensor,
        action_history=action,
        preview_points_body_m=preview,
    )
    expected = torch.cat(
        (
            sensor.reshape(batch_size, -1),
            action.reshape(batch_size, -1),
            preview.reshape(batch_size, -1),
        ),
        dim=1,
    )

    assert observation.shape == (batch_size, 555)
    torch.testing.assert_close(observation, expected)
    assert float(observation.max().item()) > 1.0


def test_fixed_scale_normalization_preserves_layout_dtype_and_exact_mapping() -> None:
    layout = pure_rl_observation.PURE_RL_RAW_OBSERVATION_LAYOUT
    sensor_frame = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 12.0, -5.0, 2.5, 5.0, -2.5, 1.0, 6.0, 2.5, 1.0, 0.0]],
        dtype=torch.float64,
    )
    sensor = sensor_frame.unsqueeze(1).expand(-1, layout.sensor_history_steps, -1).clone()
    action = torch.full(
        (1, layout.action_history_steps, layout.action_dim),
        0.25,
        dtype=torch.float64,
    )
    preview = torch.tensor(
        [[[5.0, -5.0, 2.5], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, -5.0]]],
        dtype=torch.float64,
    )

    observation = pure_rl_observation.normalize_actor_observation(
        sensor_history=sensor,
        action_history=action,
        preview_points_body_m=preview,
        apply_safety_clip=False,
    )

    expected_frame = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.5, 1.0, -0.5, 0.2, 0.5, 0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(observation[:, :14], expected_frame)
    torch.testing.assert_close(observation[:, layout.sensor_history_dim : layout.sensor_history_dim + 4], action[:, 0])
    preview_start = layout.sensor_history_dim + layout.action_history_dim
    torch.testing.assert_close(observation[:, preview_start:], (preview / 5.0).reshape(1, -1))
    assert observation.shape == (1, layout.observation_dim)
    assert observation.dtype == torch.float64


def test_normalization_has_no_clipping_inside_contract_envelope_and_clips_only_as_safety_gate() -> None:
    layout = pure_rl_observation.PURE_RL_RAW_OBSERVATION_LAYOUT
    sensor = torch.zeros((2, layout.sensor_history_steps, layout.sensor_frame_dim), dtype=torch.float32)
    sensor[..., 0] = 1.0
    sensor[..., 4] = 12.0
    sensor[..., 5:7] = 5.0
    sensor[..., 7:10] = 5.0
    sensor[..., 10] = 12.0
    sensor[..., 11] = 5.0
    sensor[..., 13] = 1.0
    action = torch.ones((2, layout.action_history_steps, layout.action_dim), dtype=torch.float32)
    preview = torch.full((2, layout.preview_point_count, layout.preview_point_dim), 5.0)

    inside = pure_rl_observation.normalize_actor_observation(
        sensor_history=sensor,
        action_history=action,
        preview_points_body_m=preview,
        apply_safety_clip=False,
    )
    assert float(torch.max(torch.abs(inside)).item()) == pytest.approx(1.0)

    sensor[..., 4] = 120.0
    unclipped = pure_rl_observation.normalize_actor_observation(
        sensor_history=sensor,
        action_history=action,
        preview_points_body_m=preview,
        apply_safety_clip=False,
    )
    clipped = pure_rl_observation.normalize_actor_observation(
        sensor_history=sensor,
        action_history=action,
        preview_points_body_m=preview,
    )
    assert float(unclipped.max().item()) == pytest.approx(10.0)
    assert float(clipped.max().item()) == pytest.approx(5.0)
    assert bool(torch.all(torch.isfinite(clipped)))


def test_raw_builder_fails_closed_on_invalid_values_and_shapes() -> None:
    with pytest.raises(ValueError, match="non-zero norm"):
        pure_rl_observation.normalize_quaternion_wxyz(torch.zeros((1, 4)))

    with pytest.raises(ValueError, match="non-negative"):
        pure_rl_observation.build_raw_sensor_frame(
            orientation_world_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            ground_velocity_body_mps=torch.zeros((1, 3)),
            angular_velocity_body_rad_s=torch.zeros((1, 3)),
            forward_air_velocity_body_mps=torch.zeros((1,)),
            actual_flap_frequency_hz=torch.tensor([-1.0]),
            flap_phase_rad=torch.zeros((1,)),
        )

    layout = pure_rl_observation.PURE_RL_RAW_OBSERVATION_LAYOUT
    with pytest.raises(ValueError, match="sensor_history"):
        pure_rl_observation.build_raw_actor_observation(
            sensor_history=torch.zeros((1, layout.sensor_history_steps - 1, layout.sensor_frame_dim)),
            action_history=torch.zeros((1, layout.action_history_steps, layout.action_dim)),
            preview_points_body_m=torch.zeros((1, layout.preview_point_count, layout.preview_point_dim)),
        )

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_longitudinal_path.py"
)
SPEC = importlib.util.spec_from_file_location("pure_rl_longitudinal_path", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pure_rl_longitudinal_path = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pure_rl_longitudinal_path
SPEC.loader.exec_module(pure_rl_longitudinal_path)


def test_stage_configs_freeze_geometry_and_task_weights() -> None:
    configs = pure_rl_longitudinal_path.LONGITUDINAL_STAGE_CONFIGS

    assert configs["c2a"].absolute_slope_deg_range == (1.5, 4.0)
    assert configs["c2a"].task_probabilities == (0.50, 0.25, 0.25)
    assert configs["c2b"].absolute_slope_deg_range == (2.0, 6.0)
    assert configs["c2b"].task_probabilities == (0.30, 0.35, 0.35)
    assert configs["c2c"].absolute_slope_deg_range == (4.0, 12.0)
    assert configs["c2c"].task_probabilities == (0.25, 0.375, 0.375)

    for stage_id, config in configs.items():
        assert config.stage_id == stage_id
        assert config.entry_length_m_range == (15.0, 20.0)
        assert config.slope_length_m_range == (20.0, 30.0)
        assert config.minimum_recovery_length_m == pytest.approx(15.0)
        assert sum(config.task_probabilities) == pytest.approx(1.0)
        assert config.task_probabilities[1] == config.task_probabilities[2]
        for lower, upper in (
            config.absolute_slope_deg_range,
            config.entry_length_m_range,
            config.slope_length_m_range,
        ):
            assert math.isfinite(lower) and math.isfinite(upper)
            assert 0.0 < lower <= upper


def test_unknown_stage_is_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown longitudinal stage"):
        pure_rl_longitudinal_path.resolve_longitudinal_stage("c2z")


def test_sample_batch_preserves_shape_dtype_device_and_exact_level_slope() -> None:
    generator = torch.Generator(device="cpu").manual_seed(42)
    batch = pure_rl_longitudinal_path.sample_longitudinal_path_batch(
        num_paths=512,
        stage="c2a",
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=generator,
        initial_altitude_m=10.0,
    )

    assert batch.task_id.shape == (512,)
    assert batch.task_id.dtype == torch.int64
    for value in (
        batch.heading_rad,
        batch.signed_slope_rad,
        batch.entry_length_m,
        batch.slope_length_m,
        batch.initial_altitude_m,
    ):
        assert value.shape == (512,)
        assert value.dtype == torch.float64
        assert value.device.type == "cpu"
        assert bool(torch.all(torch.isfinite(value)))

    level = batch.task_id == pure_rl_longitudinal_path.LEVEL_TASK_ID
    climb = batch.task_id == pure_rl_longitudinal_path.CLIMB_TASK_ID
    descent = batch.task_id == pure_rl_longitudinal_path.DESCENT_TASK_ID
    assert bool(torch.any(level) and torch.any(climb) and torch.any(descent))
    torch.testing.assert_close(
        batch.signed_slope_rad[level],
        torch.zeros_like(batch.signed_slope_rad[level]),
        rtol=0.0,
        atol=0.0,
    )
    assert bool(torch.all(batch.signed_slope_rad[climb] > 0.0))
    assert bool(torch.all(batch.signed_slope_rad[descent] < 0.0))
    assert bool(torch.all((0.0 <= batch.heading_rad) & (batch.heading_rad < 2.0 * math.pi)))
    assert bool(torch.all((15.0 <= batch.entry_length_m) & (batch.entry_length_m <= 20.0)))
    assert bool(torch.all((20.0 <= batch.slope_length_m) & (batch.slope_length_m <= 30.0)))
    torch.testing.assert_close(batch.initial_altitude_m, torch.full((512,), 10.0, dtype=torch.float64))


def test_sampling_is_reproducible_and_validates_batch_inputs() -> None:
    first_generator = torch.Generator(device="cpu").manual_seed(7)
    second_generator = torch.Generator(device="cpu").manual_seed(7)
    first = pure_rl_longitudinal_path.sample_longitudinal_path_batch(
        num_paths=8,
        stage=pure_rl_longitudinal_path.LONGITUDINAL_STAGE_CONFIGS["c2c"],
        device="cpu",
        dtype=torch.float32,
        generator=first_generator,
    )
    second = pure_rl_longitudinal_path.sample_longitudinal_path_batch(
        num_paths=8,
        stage="c2c",
        device="cpu",
        dtype=torch.float32,
        generator=second_generator,
    )
    for field_name in first.__dataclass_fields__:
        torch.testing.assert_close(getattr(first, field_name), getattr(second, field_name))

    with pytest.raises(ValueError, match="num_paths"):
        pure_rl_longitudinal_path.sample_longitudinal_path_batch(
            num_paths=0,
            stage="c2a",
            device="cpu",
            dtype=torch.float32,
        )
    with pytest.raises(TypeError, match="floating-point"):
        pure_rl_longitudinal_path.sample_longitudinal_path_batch(
            num_paths=1,
            stage="c2a",
            device="cpu",
            dtype=torch.int64,
        )


def test_custom_rehearsal_stage_may_weight_climb_and_descent_asymmetrically() -> None:
    config = pure_rl_longitudinal_path.PureRLLongitudinalStageConfig(
        stage_id="c2c",
        absolute_slope_deg_range=(4.0, 12.0),
        task_probabilities=(0.0, 2.0 / 3.0, 1.0 / 3.0),
    )

    batch = pure_rl_longitudinal_path.sample_longitudinal_path_batch(
        num_paths=4096,
        stage=config,
        device="cpu",
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(17),
    )

    assert not bool(torch.any(batch.task_id == pure_rl_longitudinal_path.LEVEL_TASK_ID))
    climb_fraction = float(
        torch.mean((batch.task_id == pure_rl_longitudinal_path.CLIMB_TASK_ID).to(torch.float64))
    )
    assert climb_fraction == pytest.approx(2.0 / 3.0, abs=0.03)
    assert float(torch.rad2deg(batch.signed_slope_rad.abs()).min()) >= 4.0
    assert float(torch.rad2deg(batch.signed_slope_rad.abs()).max()) <= 12.0


def test_c2c_rehearsal_may_stratify_half_of_climbs_into_strong_band() -> None:
    config = pure_rl_longitudinal_path.PureRLLongitudinalStageConfig(
        stage_id="c2c",
        absolute_slope_deg_range=(4.0, 12.0),
        task_probabilities=(0.0, 2.0 / 3.0, 1.0 / 3.0),
    )

    batch = pure_rl_longitudinal_path.sample_longitudinal_path_batch(
        num_paths=16_384,
        stage=config,
        device="cpu",
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(29),
        climb_strong_slope_probability=0.5,
        climb_strong_slope_minimum_deg=10.0,
    )

    climb = batch.task_id == pure_rl_longitudinal_path.CLIMB_TASK_ID
    climb_slope_deg = torch.rad2deg(batch.signed_slope_rad[climb])
    strong_fraction = float(torch.mean((climb_slope_deg >= 10.0).to(torch.float64)))
    assert strong_fraction == pytest.approx(0.5, abs=0.02)
    assert float(climb_slope_deg.min()) >= 4.0
    assert float(climb_slope_deg.max()) <= 12.0


def _path_batch(
    *,
    headings_rad: list[float],
    slopes_deg: list[float],
    entry_lengths_m: list[float],
    slope_lengths_m: list[float],
    initial_altitudes_m: list[float],
) -> object:
    dtype = torch.float64
    slopes_rad = torch.deg2rad(torch.tensor(slopes_deg, dtype=dtype))
    task_id = torch.where(
        slopes_rad > 0.0,
        pure_rl_longitudinal_path.CLIMB_TASK_ID,
        torch.where(
            slopes_rad < 0.0,
            pure_rl_longitudinal_path.DESCENT_TASK_ID,
            pure_rl_longitudinal_path.LEVEL_TASK_ID,
        ),
    ).to(torch.int64)
    return pure_rl_longitudinal_path.PureRLLongitudinalPathBatch(
        task_id=task_id,
        heading_rad=torch.tensor(headings_rad, dtype=dtype),
        signed_slope_rad=slopes_rad,
        entry_length_m=torch.tensor(entry_lengths_m, dtype=dtype),
        slope_length_m=torch.tensor(slope_lengths_m, dtype=dtype),
        initial_altitude_m=torch.tensor(initial_altitudes_m, dtype=dtype),
    )


def test_query_piecewise_altitude_and_active_slope_cover_entry_slope_and_recovery() -> None:
    path = _path_batch(
        headings_rad=[0.0, 0.0, 0.0],
        slopes_deg=[4.0, 4.0, 4.0],
        entry_lengths_m=[20.0, 20.0, 20.0],
        slope_lengths_m=[25.0, 25.0, 25.0],
        initial_altitudes_m=[10.0, 10.0, 10.0],
    )
    positions = torch.tensor(
        [[10.0, 0.0, 10.0], [30.0, 0.0, 10.0], [60.0, 0.0, 10.0]],
        dtype=torch.float64,
    )
    query = pure_rl_longitudinal_path.query_longitudinal_path(
        path=path,
        position_world_m=positions,
        ground_velocity_world_mps=torch.tensor([[7.0, 0.0, 0.0]] * 3, dtype=torch.float64),
    )

    expected_at_30 = 10.0 + 10.0 * math.tan(math.radians(4.0))
    expected_at_60 = 10.0 + 25.0 * math.tan(math.radians(4.0))
    assert query.reference_altitude_m[0].item() == pytest.approx(10.0)
    assert query.reference_altitude_m[1].item() == pytest.approx(expected_at_30)
    assert query.reference_altitude_m[2].item() == pytest.approx(expected_at_60)
    assert query.active_slope_rad[1].item() == pytest.approx(math.radians(4.0))
    assert query.active_slope_rad[2].item() == pytest.approx(0.0)
    assert query.reached_recovery.tolist() == [False, False, True]


def test_preview_points_cross_entry_slope_and_recovery_boundaries() -> None:
    path = _path_batch(
        headings_rad=[0.0],
        slopes_deg=[4.0],
        entry_lengths_m=[20.0],
        slope_lengths_m=[25.0],
        initial_altitudes_m=[10.0],
    )
    query = pure_rl_longitudinal_path.query_longitudinal_path(
        path=path,
        position_world_m=torch.tensor([[19.0, 0.0, 10.0]], dtype=torch.float64),
        ground_velocity_world_mps=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64),
        preview_times_s=(0.05, 0.15, 0.65, 2.60, 3.10),
    )

    expected_progress = torch.tensor([[19.5, 20.5, 25.5, 45.0, 50.0]], dtype=torch.float64)
    torch.testing.assert_close(query.preview_points_world_m[:, :, 0], expected_progress)
    expected_altitude = 10.0 + torch.clamp(expected_progress - 20.0, min=0.0, max=25.0) * math.tan(
        math.radians(4.0)
    )
    torch.testing.assert_close(query.preview_points_world_m[:, :, 2], expected_altitude)


def test_path_basis_is_right_handed_orthonormal_for_climb_and_descent() -> None:
    path = _path_batch(
        headings_rad=[0.0, math.pi / 3.0, -math.pi / 2.0],
        slopes_deg=[4.0, -6.0, 8.0],
        entry_lengths_m=[20.0, 20.0, 20.0],
        slope_lengths_m=[25.0, 25.0, 25.0],
        initial_altitudes_m=[10.0, 10.0, 10.0],
    )
    horizontal = torch.stack((torch.cos(path.heading_rad), torch.sin(path.heading_rad)), dim=1)
    positions = torch.zeros((3, 3), dtype=torch.float64)
    positions[:, 0:2] = 30.0 * horizontal
    positions[:, 2] = 10.0
    query = pure_rl_longitudinal_path.query_longitudinal_path(
        path=path,
        position_world_m=positions,
        ground_velocity_world_mps=torch.zeros((3, 3), dtype=torch.float64),
    )

    for basis in (
        query.tangent_world,
        query.lateral_normal_world,
        query.vertical_normal_world,
    ):
        torch.testing.assert_close(
            torch.linalg.vector_norm(basis, dim=1),
            torch.ones(3, dtype=torch.float64),
        )
    torch.testing.assert_close(
        torch.sum(query.tangent_world * query.lateral_normal_world, dim=1),
        torch.zeros(3, dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.sum(query.tangent_world * query.vertical_normal_world, dim=1),
        torch.zeros(3, dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.sum(query.lateral_normal_world * query.vertical_normal_world, dim=1),
        torch.zeros(3, dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.cross(query.tangent_world, query.lateral_normal_world, dim=1),
        query.vertical_normal_world,
        atol=1.0e-15,
        rtol=0.0,
    )


def test_zero_slope_query_matches_c1_straight_line_geometry_for_batch_one_and_many() -> None:
    headings = torch.tensor([0.0, 0.7, -2.1], dtype=torch.float64)
    path = _path_batch(
        headings_rad=headings.tolist(),
        slopes_deg=[0.0, 0.0, 0.0],
        entry_lengths_m=[20.0, 20.0, 20.0],
        slope_lengths_m=[25.0, 25.0, 25.0],
        initial_altitudes_m=[10.0, 11.0, 12.0],
    )
    position = torch.tensor([[4.0, -2.0, 10.5], [3.0, 8.0, 9.0], [-1.0, 2.0, 13.0]], dtype=torch.float64)
    velocity = torch.tensor([[7.0, 0.0, 0.0], [4.0, 3.0, 0.0], [-2.0, 1.0, 0.0]], dtype=torch.float64)
    query = pure_rl_longitudinal_path.query_longitudinal_path(
        path=path,
        position_world_m=position,
        ground_velocity_world_mps=velocity,
    )

    c1_tangent = torch.stack((torch.cos(headings), torch.sin(headings), torch.zeros_like(headings)), dim=1)
    c1_normal = torch.stack((-torch.sin(headings), torch.cos(headings), torch.zeros_like(headings)), dim=1)
    torch.testing.assert_close(query.horizontal_progress_m, torch.sum(position * c1_tangent, dim=1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(query.cross_track_error_m, torch.sum(position * c1_normal, dim=1), rtol=0.0, atol=0.0)
    torch.testing.assert_close(query.height_error_m, position[:, 2] - path.initial_altitude_m, rtol=0.0, atol=0.0)
    torch.testing.assert_close(query.tangent_world, c1_tangent, rtol=0.0, atol=0.0)
    torch.testing.assert_close(query.lateral_normal_world, c1_normal, rtol=0.0, atol=0.0)

    single = pure_rl_longitudinal_path.query_longitudinal_path(
        path=_path_batch(
            headings_rad=[0.3],
            slopes_deg=[0.0],
            entry_lengths_m=[20.0],
            slope_lengths_m=[25.0],
            initial_altitudes_m=[10.0],
        ),
        position_world_m=torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float64),
        ground_velocity_world_mps=torch.zeros((1, 3), dtype=torch.float64),
    )
    assert single.preview_points_world_m.shape == (1, 5, 3)

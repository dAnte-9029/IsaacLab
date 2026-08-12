from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_spatial_path.py"
)
SPEC = importlib.util.spec_from_file_location("pure_rl_spatial_path", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pure_rl_spatial_path = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pure_rl_spatial_path
SPEC.loader.exec_module(pure_rl_spatial_path)


def _sample(*, stage: str, count: int = 256, seed: int = 7, dtype: torch.dtype = torch.float64):
    return pure_rl_spatial_path.sample_spatial_path_batch(
        num_paths=count,
        stage=stage,
        device="cpu",
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )


def test_spatial_stage_configs_match_approved_ranges_and_probabilities() -> None:
    configs = pure_rl_spatial_path.SPATIAL_STAGE_CONFIGS

    assert configs["c3a"].geometry_roll_deg_range == (8.0, 14.0)
    assert configs["c3a"].finite_turn_deg_range == (20.0, 50.0)
    assert configs["c3a"].task_probabilities == (0.15, 0.25, 0.60)
    assert configs["c3a"].event_count_range == (1, 1)

    assert configs["c3b"].geometry_roll_deg_range == (10.0, 17.0)
    assert configs["c3b"].finite_turn_deg_range == (20.0, 60.0)
    assert configs["c3b"].task_probabilities == (0.15, 0.20, 0.15, 0.50)
    assert configs["c3b"].event_count_range == (2, 3)
    assert configs["c3b"].loiter_radius_m_range == (50.0, 80.0)

    assert configs["c3c"].geometry_roll_deg_range == (6.0, 17.0)
    assert configs["c3c"].coupled_slope_deg_range == (1.5, 6.0)
    assert configs["c3c"].task_probabilities == (0.15, 0.20, 0.15, 0.50)
    assert configs["c3c"].event_count_range == (2, 4)

    for config in configs.values():
        assert config.transition_length_m_range == (8.0, 12.0)
        assert config.guard_speed_mps == pytest.approx(12.0)
        assert config.sample_spacing_m == pytest.approx(0.25)
        assert config.path_length_m == pytest.approx(300.0)
        assert config.minimum_altitude_m == pytest.approx(0.05)
        assert sum(config.task_probabilities) == pytest.approx(1.0)

    with pytest.raises(FrozenInstanceError):
        configs["c3a"].guard_speed_mps = 10.0


def test_sampled_batch_exposes_dense_centerline_and_finite_metadata() -> None:
    batch = _sample(stage="c3c", count=64)
    required_fields = {
        "points_world_m",
        "tangent_world",
        "lateral_normal_world",
        "vertical_normal_world",
        "curvature_rad_per_m",
        "slope_rad",
        "turn_activity",
        "final_event_progress_m",
        "task_family_id",
        "turn_sign",
        "vertical_sign",
        "peak_geometry_roll_rad",
        "peak_slope_rad",
    }
    assert required_fields <= {field.name for field in fields(batch)}

    for tensor in (
        batch.points_world_m,
        batch.tangent_world,
        batch.lateral_normal_world,
        batch.vertical_normal_world,
    ):
        assert tensor.shape == (64, 1201, 3)
        assert tensor.dtype == torch.float64
        assert tensor.device.type == "cpu"
        assert bool(torch.all(torch.isfinite(tensor)))
    for tensor in (batch.curvature_rad_per_m, batch.slope_rad, batch.turn_activity):
        assert tensor.shape == (64, 1201)
        assert tensor.dtype == torch.float64
        assert bool(torch.all(torch.isfinite(tensor)))
    for tensor in (
        batch.final_event_progress_m,
        batch.turn_sign,
        batch.vertical_sign,
        batch.peak_geometry_roll_rad,
        batch.peak_slope_rad,
    ):
        assert tensor.shape == (64,)
        assert bool(torch.all(torch.isfinite(tensor)))
    assert batch.task_family_id.shape == (64,)
    assert batch.task_family_id.dtype == torch.int64
    assert batch.event_count.shape == (64,)
    assert batch.event_count.dtype == torch.int64
    assert batch.sample_spacing_m == pytest.approx(0.25)
    assert batch.path_length_m == pytest.approx(300.0)


def test_c3c_current_paths_have_two_to_four_events_and_respect_coupled_demand() -> None:
    batch = _sample(stage="c3c", count=512, seed=17)
    current = batch.task_family_id == pure_rl_spatial_path.CURRENT_SPATIAL_TASK_FAMILY_ID
    assert bool(torch.any(current))
    assert bool(torch.all((batch.event_count[current] >= 2) & (batch.event_count[current] <= 4)))

    demand = torch.square(batch.peak_geometry_roll_rad[current] / math.radians(20.0))
    demand += torch.square(torch.abs(batch.peak_slope_rad[current]) / math.radians(6.0))
    assert bool(torch.all(demand <= 1.0 + 1.0e-12))
    assert bool(
        torch.all(
            torch.any(
                (torch.abs(batch.curvature_rad_per_m[current]) > 0.0)
                & (torch.abs(batch.slope_rad[current]) > 0.0),
                dim=1,
            )
        )
    )
    assert set(batch.turn_sign[current].unique().tolist()) == {-1.0, 1.0}
    assert set(batch.vertical_sign[current].unique().tolist()) == {-1.0, 1.0}


def test_current_c3a_and_c3b_event_contracts_are_isolated_and_sequential() -> None:
    c3a = _sample(stage="c3a", count=512, seed=23)
    c3a_current = c3a.task_family_id == pure_rl_spatial_path.CURRENT_SPATIAL_TASK_FAMILY_ID_C3A
    assert bool(torch.any(c3a_current))
    assert bool(torch.all(c3a.event_count[c3a_current] == 1))
    torch.testing.assert_close(
        c3a.slope_rad[c3a_current],
        torch.zeros_like(c3a.slope_rad[c3a_current]),
        rtol=0.0,
        atol=0.0,
    )

    c3b = _sample(stage="c3b", count=512, seed=29)
    c3b_current = c3b.task_family_id == pure_rl_spatial_path.CURRENT_SPATIAL_TASK_FAMILY_ID
    assert bool(torch.any(c3b_current))
    assert bool(torch.all((c3b.event_count[c3b_current] >= 2) & (c3b.event_count[c3b_current] <= 3)))
    simultaneous = (torch.abs(c3b.curvature_rad_per_m[c3b_current]) > 0.0) & (
        torch.abs(c3b.slope_rad[c3b_current]) > 0.0
    )
    assert not bool(torch.any(simultaneous))

    same_direction = c3b.template_id == pure_rl_spatial_path.C3B_SAME_DIRECTION_TURNS_TEMPLATE_ID
    assert bool(torch.any(same_direction))
    torch.testing.assert_close(
        c3b.slope_rad[same_direction],
        torch.zeros_like(c3b.slope_rad[same_direction]),
        rtol=0.0,
        atol=0.0,
    )
    signed_same_direction = c3b.curvature_rad_per_m[same_direction] * c3b.turn_sign[
        same_direction
    ].unsqueeze(1)
    assert bool(torch.all(signed_same_direction >= 0.0))

    s_turn = c3b.template_id == pure_rl_spatial_path.C3B_S_TURNS_TEMPLATE_ID
    assert bool(torch.any(s_turn))
    torch.testing.assert_close(
        c3b.slope_rad[s_turn],
        torch.zeros_like(c3b.slope_rad[s_turn]),
        rtol=0.0,
        atol=0.0,
    )
    assert bool(torch.all(torch.amax(c3b.curvature_rad_per_m[s_turn], dim=1) > 0.0))
    assert bool(torch.all(torch.amin(c3b.curvature_rad_per_m[s_turn], dim=1) < 0.0))

    three_event = c3b_current & (c3b.event_count == 3)
    finite_turn_changes_deg: list[float] = []
    for curvature in c3b.curvature_rad_per_m[three_event]:
        active_indices = torch.nonzero(curvature != 0.0, as_tuple=False).flatten().tolist()
        for component in _contiguous_components(active_indices):
            values = curvature[component[0] : component[-1] + 1]
            finite_turn_changes_deg.append(math.degrees(float(torch.trapezoid(torch.abs(values), dx=0.25))))
    assert finite_turn_changes_deg
    assert max(finite_turn_changes_deg) > 35.0
    assert max(finite_turn_changes_deg) <= 60.5

    loiter = c3b.template_id == pure_rl_spatial_path.C3B_LOITER_TEMPLATE_ID
    assert bool(torch.any(loiter))
    implied_radius_m = 12.0**2 / (
        9.81 * torch.tan(c3b.peak_geometry_roll_rad[loiter])
    )
    assert bool(torch.all((implied_radius_m >= 50.0) & (implied_radius_m <= 80.0)))
    last_active_progress_m = 0.25 * torch.amax(
        torch.where(
            c3b.curvature_rad_per_m[loiter] != 0.0,
            torch.arange(1201).unsqueeze(0),
            torch.zeros((1, 1201), dtype=torch.int64),
        ),
        dim=1,
    ).to(dtype=torch.float64)
    torch.testing.assert_close(
        c3b.final_event_progress_m[loiter],
        last_active_progress_m,
        atol=0.5,
        rtol=0.0,
    )


def test_dense_path_starts_straight_and_level_with_orthonormal_heading_slope_frame() -> None:
    batch = _sample(stage="c3c", count=96, seed=31)
    torch.testing.assert_close(
        batch.curvature_rad_per_m[:, 0],
        torch.zeros(96, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        batch.slope_rad[:, 0],
        torch.zeros(96, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )

    for basis in (batch.tangent_world, batch.lateral_normal_world, batch.vertical_normal_world):
        torch.testing.assert_close(
            torch.linalg.vector_norm(basis, dim=2),
            torch.ones((96, 1201), dtype=torch.float64),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
    zeros = torch.zeros((96, 1201), dtype=torch.float64)
    torch.testing.assert_close(
        torch.sum(batch.tangent_world * batch.lateral_normal_world, dim=2), zeros, atol=1.0e-12, rtol=0.0
    )
    torch.testing.assert_close(
        torch.sum(batch.tangent_world * batch.vertical_normal_world, dim=2), zeros, atol=1.0e-12, rtol=0.0
    )
    torch.testing.assert_close(
        torch.sum(batch.lateral_normal_world * batch.vertical_normal_world, dim=2),
        zeros,
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.cross(batch.tangent_world, batch.lateral_normal_world, dim=2),
        batch.vertical_normal_world,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_centerline_has_quarter_meter_sampling_and_approximately_300_meter_length() -> None:
    batch = _sample(stage="c3b", count=64, seed=37)
    segment_length_m = torch.linalg.vector_norm(torch.diff(batch.points_world_m, dim=1), dim=2)
    assert float(torch.max(torch.abs(segment_length_m - 0.25))) < 1.0e-5
    path_length_m = torch.sum(segment_length_m, dim=1)
    torch.testing.assert_close(path_length_m, torch.full_like(path_length_m, 300.0), atol=0.01, rtol=0.0)


def test_sampling_is_seed_deterministic_and_different_seeds_change_paths() -> None:
    first = _sample(stage="c3c", count=12, seed=41, dtype=torch.float32)
    second = _sample(stage="c3c", count=12, seed=41, dtype=torch.float32)
    different = _sample(stage="c3c", count=12, seed=42, dtype=torch.float32)

    for field in fields(first):
        torch.testing.assert_close(getattr(first, field.name), getattr(second, field.name))
    assert not torch.equal(first.points_world_m, different.points_world_m)


def test_fixed_evaluation_overrides_are_applied_row_by_row() -> None:
    batch = pure_rl_spatial_path.sample_spatial_path_batch(
        num_paths=2,
        stage="c3c",
        device="cpu",
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(43),
        evaluation_template_id=torch.tensor(
            [
                pure_rl_spatial_path.C3C_COUPLED_TEMPLATE_ID,
                pure_rl_spatial_path.C3C_COUPLED_TEMPLATE_ID,
            ],
            dtype=torch.int64,
        ),
        evaluation_geometry_roll_deg=torch.tensor([8.0, 12.0], dtype=torch.float64),
        evaluation_slope_deg=torch.tensor([3.0, -4.0], dtype=torch.float64),
        evaluation_turn_sign=torch.tensor([-1.0, 1.0], dtype=torch.float64),
        evaluation_heading_rad=torch.tensor([0.0, math.pi / 2.0], dtype=torch.float64),
    )

    torch.testing.assert_close(
        batch.template_id,
        torch.full((2,), pure_rl_spatial_path.C3C_COUPLED_TEMPLATE_ID, dtype=torch.int64),
    )
    torch.testing.assert_close(
        batch.peak_geometry_roll_rad,
        torch.deg2rad(torch.tensor([8.0, 12.0], dtype=torch.float64)),
    )
    torch.testing.assert_close(
        batch.peak_slope_rad,
        torch.deg2rad(torch.tensor([3.0, -4.0], dtype=torch.float64)),
    )
    torch.testing.assert_close(batch.turn_sign, torch.tensor([-1.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(
        batch.tangent_world[:, 0],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_dtype_device_and_scalar_or_batched_altitude_are_preserved() -> None:
    scalar = pure_rl_spatial_path.sample_spatial_path_batch(
        num_paths=4,
        stage="c3a",
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(43),
        initial_altitude_m=12.5,
    )
    assert scalar.points_world_m.dtype == torch.float32
    torch.testing.assert_close(scalar.points_world_m[:, 0, 2], torch.full((4,), 12.5))

    altitudes = torch.tensor([3.0, 5.0, 7.0, 9.0], dtype=torch.float64)
    batched = pure_rl_spatial_path.sample_spatial_path_batch(
        num_paths=4,
        stage="c3b",
        device="cpu",
        dtype=torch.float64,
        generator=torch.Generator(device="cpu").manual_seed(47),
        initial_altitude_m=altitudes,
    )
    torch.testing.assert_close(batched.points_world_m[:, 0, 2], altitudes, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_paths", [0, -1, 1.5, True])
def test_invalid_path_count_is_rejected(num_paths: object) -> None:
    with pytest.raises(ValueError, match="num_paths"):
        pure_rl_spatial_path.sample_spatial_path_batch(
            num_paths=num_paths,
            stage="c3a",
            device="cpu",
            dtype=torch.float32,
        )


def test_invalid_stage_dtype_altitude_and_config_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown spatial stage"):
        pure_rl_spatial_path.sample_spatial_path_batch(
            num_paths=1, stage="c3z", device="cpu", dtype=torch.float32
        )
    with pytest.raises(TypeError, match="floating-point"):
        pure_rl_spatial_path.sample_spatial_path_batch(
            num_paths=1, stage="c3a", device="cpu", dtype=torch.int64
        )
    for low_precision_dtype in (torch.float16, torch.bfloat16):
        with pytest.raises(TypeError, match="float32 or float64"):
            pure_rl_spatial_path.sample_spatial_path_batch(
                num_paths=1,
                stage="c3a",
                device="cpu",
                dtype=low_precision_dtype,
            )
    with pytest.raises(ValueError, match="initial_altitude_m"):
        pure_rl_spatial_path.sample_spatial_path_batch(
            num_paths=2,
            stage="c3a",
            device="cpu",
            dtype=torch.float32,
            initial_altitude_m=torch.ones(3),
        )

    invalid_config = pure_rl_spatial_path.PureRLSpatialStageConfig(
        stage_id="c3a",
        task_probabilities=(0.2, 0.2, 0.2),
        geometry_roll_deg_range=(8.0, 14.0),
        finite_turn_deg_range=(20.0, 50.0),
    )
    with pytest.raises(ValueError, match="sum to one"):
        pure_rl_spatial_path.resolve_spatial_stage(invalid_config)


def test_stage_validation_rejects_missing_stage_specific_geometry_bounds() -> None:
    invalid_c3b = pure_rl_spatial_path.PureRLSpatialStageConfig(
        stage_id="c3b",
        task_probabilities=(0.15, 0.20, 0.15, 0.50),
        geometry_roll_deg_range=(10.0, 17.0),
        finite_turn_deg_range=(20.0, 60.0),
        event_count_range=(2, 3),
    )
    with pytest.raises(ValueError, match="loiter_radius_m_range"):
        pure_rl_spatial_path.resolve_spatial_stage(invalid_c3b)

    invalid_c3c = pure_rl_spatial_path.PureRLSpatialStageConfig(
        stage_id="c3c",
        task_probabilities=(0.15, 0.20, 0.15, 0.50),
        geometry_roll_deg_range=(6.0, 17.0),
        finite_turn_deg_range=(20.0, 60.0),
        event_count_range=(2, 4),
    )
    with pytest.raises(ValueError, match="coupled_slope_deg_range"):
        pure_rl_spatial_path.resolve_spatial_stage(invalid_c3c)


def test_c3c_rehearsal_includes_c3b_and_custom_coupled_upper_bound_is_honored() -> None:
    batch = _sample(stage="c3c", count=1024, seed=53)
    earlier = batch.task_family_id == pure_rl_spatial_path.EARLIER_SPATIAL_TASK_FAMILY_ID
    assert bool(torch.any(earlier & (batch.template_id == pure_rl_spatial_path.ISOLATED_TURN_TEMPLATE_ID)))
    assert bool(
        torch.any(
            earlier
            & (
                (batch.template_id == pure_rl_spatial_path.C3B_SAME_DIRECTION_TURNS_TEMPLATE_ID)
                | (batch.template_id == pure_rl_spatial_path.C3B_S_TURNS_TEMPLATE_ID)
            )
        )
    )

    custom = pure_rl_spatial_path.PureRLSpatialStageConfig(
        stage_id="c3c",
        task_probabilities=(0.0, 0.0, 0.0, 1.0),
        geometry_roll_deg_range=(6.0, 17.0),
        finite_turn_deg_range=(20.0, 20.0),
        coupled_slope_deg_range=(1.5, 4.0),
        event_count_range=(2, 4),
    )
    custom_batch = pure_rl_spatial_path.sample_spatial_path_batch(
        num_paths=128,
        stage=custom,
        device="cpu",
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(59),
    )
    assert bool(torch.all(torch.abs(custom_batch.peak_slope_rad) <= math.radians(4.0)))


def test_nonadjacent_centerline_self_intersection_is_rejected_for_non_loiter() -> None:
    batch = _sample(stage="c3a", count=1, seed=61)
    crossing = replace(
        batch,
        points_world_m=_self_intersecting_points(batch.points_world_m),
        template_id=torch.full_like(batch.template_id, pure_rl_spatial_path.ISOLATED_TURN_TEMPLATE_ID),
    )

    with pytest.raises(RuntimeError, match="nonadjacent centerline"):
        pure_rl_spatial_path._validate_generated_batch(
            crossing,
            config=pure_rl_spatial_path.SPATIAL_STAGE_CONFIGS["c3a"],
        )


def test_phase_shifted_parallel_branches_below_clearance_are_rejected() -> None:
    batch = _sample(stage="c3a", count=1, seed=63)
    too_close = replace(
        batch,
        points_world_m=_parallel_close_branch_points(batch.points_world_m),
        template_id=torch.full_like(batch.template_id, pure_rl_spatial_path.ISOLATED_TURN_TEMPLATE_ID),
    )

    with pytest.raises(RuntimeError, match="nonadjacent centerline"):
        pure_rl_spatial_path._validate_generated_batch(
            too_close,
            config=pure_rl_spatial_path.SPATIAL_STAGE_CONFIGS["c3a"],
        )


def test_intentional_c3b_loiter_is_exempt_from_nonadjacent_intersection_rejection() -> None:
    sampled = _sample(stage="c3b", count=128, seed=67)
    loiter_indices = torch.nonzero(
        sampled.template_id == pure_rl_spatial_path.C3B_LOITER_TEMPLATE_ID,
        as_tuple=False,
    ).flatten()
    assert loiter_indices.numel() > 0
    row = int(loiter_indices[0])
    loiter = _select_path_rows(sampled, torch.tensor([row], dtype=torch.int64))
    intentional_periodicity = replace(
        loiter,
        points_world_m=_self_intersecting_points(loiter.points_world_m),
    )

    pure_rl_spatial_path._validate_generated_batch(
        intentional_periodicity,
        config=pure_rl_spatial_path.SPATIAL_STAGE_CONFIGS["c3b"],
    )


def test_full_circle_heading_change_is_exempt_only_for_c3b_loiter() -> None:
    sampled = _sample(stage="c3b", count=128, seed=71)
    loiter_indices = torch.nonzero(
        sampled.template_id == pure_rl_spatial_path.C3B_LOITER_TEMPLATE_ID,
        as_tuple=False,
    ).flatten()
    assert loiter_indices.numel() > 0
    row = int(loiter_indices[0])
    loiter = _select_path_rows(sampled, torch.tensor([row], dtype=torch.int64))
    full_circle_curvature = torch.zeros_like(loiter.curvature_rad_per_m)
    integrated_constant_length_m = 300.0 - 0.5 * 0.25
    full_circle_curvature[:, 1:] = 1.01 * 2.0 * math.pi / integrated_constant_length_m
    periodic_loiter = replace(loiter, curvature_rad_per_m=full_circle_curvature)

    pure_rl_spatial_path._validate_generated_batch(
        periodic_loiter,
        config=pure_rl_spatial_path.SPATIAL_STAGE_CONFIGS["c3b"],
    )

    non_loiter = replace(
        periodic_loiter,
        template_id=torch.full_like(periodic_loiter.template_id, pure_rl_spatial_path.ISOLATED_TURN_TEMPLATE_ID),
    )
    with pytest.raises(RuntimeError, match="full heading revolution"):
        pure_rl_spatial_path._validate_generated_batch(
            non_loiter,
            config=pure_rl_spatial_path.SPATIAL_STAGE_CONFIGS["c3b"],
        )


def test_c3c_low_altitude_rows_are_resampled_once_deterministically() -> None:
    first = _sample(stage="c3c", count=256, seed=8)
    second = _sample(stage="c3c", count=256, seed=8)

    assert float(torch.amin(first.points_world_m[:, :, 2])) >= 0.05
    for field in fields(first):
        torch.testing.assert_close(getattr(first, field.name), getattr(second, field.name))

    with pytest.raises(RuntimeError, match="altitude"):
        pure_rl_spatial_path.sample_spatial_path_batch(
            num_paths=1,
            stage="c3a",
            device="cpu",
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(73),
            initial_altitude_m=-1.0,
        )


def _self_intersecting_points(reference: torch.Tensor) -> torch.Tensor:
    controls = reference.new_tensor(
        [
            [0.0, 0.0, 10.0],
            [20.0, 20.0, 10.0],
            [0.0, 20.0, 10.0],
            [20.0, 0.0, 10.0],
            [40.0, 0.0, 10.0],
        ]
    )
    control_progress = torch.linspace(0.0, 4.0, 1201, device=reference.device, dtype=reference.dtype)
    lower_index = torch.floor(control_progress).to(dtype=torch.int64).clamp(max=3)
    fraction = (control_progress - lower_index.to(dtype=reference.dtype)).unsqueeze(1)
    points = torch.lerp(controls[lower_index], controls[lower_index + 1], fraction)
    return points.unsqueeze(0)


def _parallel_close_branch_points(reference: torch.Tensor) -> torch.Tensor:
    points = torch.empty_like(reference)
    points[:, :, 2] = 10.0
    points[:, 0:301, 0] = torch.linspace(0.0, 75.0, 301, device=reference.device, dtype=reference.dtype)
    points[:, 0:301, 1] = 0.0
    points[:, 300:305, 0] = 75.0
    points[:, 300:305, 1] = torch.linspace(0.0, 1.4, 5, device=reference.device, dtype=reference.dtype)
    points[:, 304:605, 0] = torch.linspace(75.0, 0.0, 301, device=reference.device, dtype=reference.dtype)
    points[:, 304:605, 1] = 1.4
    points[:, 604:, 0] = 0.0
    points[:, 604:, 1] = torch.linspace(1.4, 150.4, 597, device=reference.device, dtype=reference.dtype)
    return points


def _contiguous_components(indices: list[int]) -> list[list[int]]:
    components: list[list[int]] = []
    for index in indices:
        if not components or index != components[-1][-1] + 1:
            components.append([index])
        else:
            components[-1].append(index)
    return components


def _straight_path_batch(
    *,
    count: int,
    point_count: int = 21,
    sample_spacing_m: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> object:
    progress_m = torch.arange(point_count, dtype=dtype) * sample_spacing_m
    points = torch.zeros((count, point_count, 3), dtype=dtype)
    points[:, :, 0] = progress_m
    points[:, :, 2] = 10.0
    tangent = torch.zeros_like(points)
    tangent[:, :, 0] = 1.0
    lateral = torch.zeros_like(points)
    lateral[:, :, 1] = 1.0
    vertical = torch.zeros_like(points)
    vertical[:, :, 2] = 1.0
    zeros_table = torch.zeros((count, point_count), dtype=dtype)
    zeros_row = torch.zeros(count, dtype=dtype)
    zeros_int = torch.zeros(count, dtype=torch.int64)
    return pure_rl_spatial_path.PureRLSpatialPathBatch(
        points_world_m=points,
        tangent_world=tangent,
        lateral_normal_world=lateral,
        vertical_normal_world=vertical,
        curvature_rad_per_m=zeros_table.clone(),
        slope_rad=zeros_table.clone(),
        turn_activity=zeros_table.clone(),
        final_event_progress_m=torch.full((count,), 8.0, dtype=dtype),
        task_family_id=zeros_int.clone(),
        turn_sign=zeros_row.clone(),
        vertical_sign=zeros_row.clone(),
        peak_geometry_roll_rad=zeros_row.clone(),
        peak_slope_rad=zeros_row.clone(),
        event_count=zeros_int.clone(),
        template_id=zeros_int.clone(),
        sample_spacing_m=sample_spacing_m,
        path_length_m=(point_count - 1) * sample_spacing_m,
    )


def _select_path_rows(path: object, row_ids: torch.Tensor) -> object:
    values = {}
    for field in fields(path):
        value = getattr(path, field.name)
        values[field.name] = value.index_select(0, row_ids) if isinstance(value, torch.Tensor) else value
    return pure_rl_spatial_path.PureRLSpatialPathBatch(**values)


def test_query_projects_and_interpolates_straight_table_with_signed_errors() -> None:
    path = _straight_path_batch(count=2)
    progress_table = torch.arange(21, dtype=torch.float64).expand(2, -1)
    path = replace(
        path,
        curvature_rad_per_m=0.1 * progress_table,
        slope_rad=0.01 * progress_table,
        turn_activity=0.02 * progress_table,
    )
    query = pure_rl_spatial_path.query_spatial_path(
        path=path,
        previous_progress_m=torch.tensor([2.0, 2.0], dtype=torch.float64),
        position_world_m=torch.tensor([[2.25, 2.0, 11.0], [2.25, -2.0, 9.0]], dtype=torch.float64),
        ground_velocity_world_mps=torch.tensor([[4.0, 0.0, 0.0]] * 2, dtype=torch.float64),
    )

    torch.testing.assert_close(query.progress_m, torch.tensor([2.25, 2.25], dtype=torch.float64))
    torch.testing.assert_close(
        query.reference_position_world_m,
        torch.tensor([[2.25, 0.0, 10.0], [2.25, 0.0, 10.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(query.horizontal_normal_error_m, torch.tensor([2.0, -2.0], dtype=torch.float64))
    torch.testing.assert_close(query.vertical_normal_error_m, torch.tensor([1.0, -1.0], dtype=torch.float64))
    torch.testing.assert_close(query.active_curvature_rad_per_m, torch.full((2,), 0.225, dtype=torch.float64))
    torch.testing.assert_close(query.active_slope_rad, torch.full((2,), 0.0225, dtype=torch.float64))
    torch.testing.assert_close(query.turn_activity, torch.full((2,), 0.045, dtype=torch.float64))
    with pytest.raises(FrozenInstanceError):
        query.progress_m = torch.zeros_like(query.progress_m)


def test_query_uses_local_window_and_allows_two_meter_backward_progress() -> None:
    path = _straight_path_batch(count=2, point_count=25)
    branch_points = path.points_world_m.clone()
    branch_points[:, :11, 0] = torch.arange(11, dtype=torch.float64)
    branch_points[:, :11, 1] = 0.0
    branch_points[:, 11:, 0] = torch.arange(10, -4, -1, dtype=torch.float64)
    branch_points[:, 11:, 1] = 0.2
    path = replace(path, points_world_m=branch_points)

    query = pure_rl_spatial_path.query_spatial_path(
        path=path,
        previous_progress_m=torch.tensor([3.0, 5.0], dtype=torch.float64),
        position_world_m=torch.tensor([[3.0, 0.19, 10.0], [3.0, 0.0, 10.0]], dtype=torch.float64),
        ground_velocity_world_mps=torch.zeros((2, 3), dtype=torch.float64),
    )

    assert query.progress_m[0].item() == pytest.approx(3.0)
    assert query.progress_m[1].item() == pytest.approx(3.0)


def test_query_preview_uses_tangent_speed_clamps_default_times_and_path_end() -> None:
    path = _straight_path_batch(count=3, point_count=11)
    query = pure_rl_spatial_path.query_spatial_path(
        path=path,
        previous_progress_m=torch.tensor([2.0, 2.0, 9.8], dtype=torch.float64),
        position_world_m=torch.tensor(
            [[2.0, 0.0, 10.0], [2.0, 0.0, 10.0], [9.8, 0.0, 10.0]], dtype=torch.float64
        ),
        ground_velocity_world_mps=torch.tensor(
            [[-4.0, 0.0, 0.0], [6.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=torch.float64
        ),
    )

    preview_times = torch.tensor([0.12, 0.24, 0.36, 0.48, 0.60], dtype=torch.float64)
    torch.testing.assert_close(query.preview_points_world_m[0, :, 0], 2.0 + preview_times)
    torch.testing.assert_close(query.preview_points_world_m[1, :, 0], 2.0 + 6.0 * preview_times)
    torch.testing.assert_close(query.preview_points_world_m[2, :, 0], torch.full((5,), 10.0, dtype=torch.float64))
    assert query.preview_points_world_m.shape == (3, 5, 3)


def test_query_returns_orthonormal_frames_for_generated_level_slope_turn_and_coupled_cases() -> None:
    sampled = _sample(stage="c3c", count=256, seed=79)
    level = (sampled.curvature_rad_per_m == 0.0) & (sampled.slope_rad == 0.0)
    slope = (sampled.curvature_rad_per_m == 0.0) & (sampled.slope_rad != 0.0)
    turn = (sampled.curvature_rad_per_m != 0.0) & (sampled.slope_rad == 0.0)
    coupled = (sampled.curvature_rad_per_m != 0.0) & (sampled.slope_rad != 0.0)
    case_masks = (level, slope, turn, coupled)
    assert all(bool(torch.any(mask)) for mask in case_masks)
    row_indices = [int(torch.nonzero(mask, as_tuple=False)[0, 0]) for mask in case_masks]
    table_indices = [int(torch.nonzero(mask, as_tuple=False)[0, 1]) for mask in case_masks]
    path = _select_path_rows(sampled, torch.tensor(row_indices, dtype=torch.int64))
    table_ids = torch.tensor(table_indices, dtype=torch.int64)
    positions = path.points_world_m[torch.arange(4), table_ids]
    query = pure_rl_spatial_path.query_spatial_path(
        path=path,
        previous_progress_m=table_ids.to(dtype=torch.float64) * path.sample_spacing_m,
        position_world_m=positions,
        ground_velocity_world_mps=torch.zeros((4, 3), dtype=torch.float64),
    )

    for basis in (query.tangent_world, query.lateral_normal_world, query.vertical_normal_world):
        torch.testing.assert_close(torch.linalg.vector_norm(basis, dim=1), torch.ones(4, dtype=torch.float64))
    zeros = torch.zeros(4, dtype=torch.float64)
    torch.testing.assert_close(
        torch.sum(query.tangent_world * query.lateral_normal_world, dim=1), zeros, atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        torch.sum(query.tangent_world * query.vertical_normal_world, dim=1), zeros, atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        torch.sum(query.lateral_normal_world * query.vertical_normal_world, dim=1), zeros, atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        torch.cross(query.tangent_world, query.lateral_normal_world, dim=1),
        query.vertical_normal_world,
        atol=1e-12,
        rtol=0.0,
    )


def test_constant_slope_spatial_query_matches_longitudinal_query_on_path() -> None:
    longitudinal_path = _load_longitudinal_module()
    dtype = torch.float64
    slope_rad = math.radians(5.0)
    spacing = 0.1
    progress = torch.arange(201, dtype=dtype) * spacing
    path = _straight_path_batch(count=1, point_count=201, sample_spacing_m=spacing)
    tangent = torch.tensor([math.cos(slope_rad), 0.0, math.sin(slope_rad)], dtype=dtype)
    lateral = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    vertical = torch.tensor([-math.sin(slope_rad), 0.0, math.cos(slope_rad)], dtype=dtype)
    points = torch.zeros_like(path.points_world_m)
    points[0] = torch.tensor([0.0, 0.0, 10.0], dtype=dtype) + progress.unsqueeze(1) * tangent
    path = replace(
        path,
        points_world_m=points,
        tangent_world=tangent.reshape(1, 1, 3).expand(1, 201, 3).clone(),
        lateral_normal_world=lateral.reshape(1, 1, 3).expand(1, 201, 3).clone(),
        vertical_normal_world=vertical.reshape(1, 1, 3).expand(1, 201, 3).clone(),
        slope_rad=torch.full((1, 201), slope_rad, dtype=dtype),
    )
    position = points[:, 50]
    velocity = 8.0 * tangent.unsqueeze(0)
    spatial_query = pure_rl_spatial_path.query_spatial_path(
        path=path,
        previous_progress_m=torch.tensor([5.0], dtype=dtype),
        position_world_m=position,
        ground_velocity_world_mps=velocity,
    )
    longitudinal_batch = longitudinal_path.PureRLLongitudinalPathBatch(
        task_id=torch.tensor([longitudinal_path.CLIMB_TASK_ID], dtype=torch.int64),
        heading_rad=torch.zeros(1, dtype=dtype),
        signed_slope_rad=torch.full((1,), slope_rad, dtype=dtype),
        entry_length_m=torch.zeros(1, dtype=dtype),
        slope_length_m=torch.full((1,), 100.0, dtype=dtype),
        initial_altitude_m=torch.full((1,), 10.0, dtype=dtype),
    )
    longitudinal_query = longitudinal_path.query_longitudinal_path(
        path=longitudinal_batch,
        position_world_m=position,
        ground_velocity_world_mps=velocity,
    )

    torch.testing.assert_close(spatial_query.reference_position_world_m, position, atol=2e-14, rtol=0.0)
    torch.testing.assert_close(
        spatial_query.preview_points_world_m,
        longitudinal_query.preview_points_world_m,
        atol=2e-12,
        rtol=0.0,
    )


def test_query_reached_all_events_boundary_is_inclusive() -> None:
    path = replace(
        _straight_path_batch(count=2),
        final_event_progress_m=torch.tensor([4.0, 4.0], dtype=torch.float64),
    )
    query = pure_rl_spatial_path.query_spatial_path(
        path=path,
        previous_progress_m=torch.tensor([4.0, 4.0], dtype=torch.float64),
        position_world_m=torch.tensor([[4.0, 0.0, 10.0], [3.999, 0.0, 10.0]], dtype=torch.float64),
        ground_velocity_world_mps=torch.zeros((2, 3), dtype=torch.float64),
    )
    assert query.reached_all_events.tolist() == [True, False]


def test_partial_write_updates_only_selected_rows_and_every_batched_field() -> None:
    destination = _straight_path_batch(count=4)
    source = _straight_path_batch(count=2)
    source_values = {}
    for index, field in enumerate(fields(source), start=1):
        value = getattr(source, field.name)
        if isinstance(value, torch.Tensor):
            fill = index if value.dtype == torch.int64 else float(index)
            source_values[field.name] = torch.full_like(value, fill)
        else:
            source_values[field.name] = value
    source = pure_rl_spatial_path.PureRLSpatialPathBatch(**source_values)
    before = {
        field.name: getattr(destination, field.name).clone()
        for field in fields(destination)
        if isinstance(getattr(destination, field.name), torch.Tensor)
    }
    destination_identity = id(destination)
    env_ids = torch.tensor([1, 3], dtype=torch.int64)

    pure_rl_spatial_path.write_spatial_path_batch_rows_(destination=destination, env_ids=env_ids, source=source)

    assert id(destination) == destination_identity
    untouched = torch.tensor([0, 2], dtype=torch.int64)
    for field in fields(destination):
        value = getattr(destination, field.name)
        if not isinstance(value, torch.Tensor):
            continue
        torch.testing.assert_close(value.index_select(0, env_ids), getattr(source, field.name))
        torch.testing.assert_close(value.index_select(0, untouched), before[field.name].index_select(0, untouched))


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("previous_progress_m", torch.zeros((1, 1), dtype=torch.float64)),
        ("position_world_m", torch.zeros((1, 2), dtype=torch.float64)),
        ("ground_velocity_world_mps", torch.zeros((1, 3), dtype=torch.float32)),
    ],
)
def test_query_rejects_invalid_tensor_shapes_and_dtypes(keyword: str, value: torch.Tensor) -> None:
    arguments = {
        "path": _straight_path_batch(count=1),
        "previous_progress_m": torch.zeros(1, dtype=torch.float64),
        "position_world_m": torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float64),
        "ground_velocity_world_mps": torch.zeros((1, 3), dtype=torch.float64),
    }
    arguments[keyword] = value
    with pytest.raises((TypeError, ValueError), match=keyword):
        pure_rl_spatial_path.query_spatial_path(**arguments)


@pytest.mark.parametrize("bounds", [(-1.0, 12.0), (13.0, 12.0), (math.nan, 12.0), (1.0, math.inf)])
def test_query_rejects_invalid_preview_speed_bounds(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="Preview-speed bounds"):
        pure_rl_spatial_path.query_spatial_path(
            path=_straight_path_batch(count=1),
            previous_progress_m=torch.zeros(1, dtype=torch.float64),
            position_world_m=torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float64),
            ground_velocity_world_mps=torch.zeros((1, 3), dtype=torch.float64),
            minimum_preview_speed_mps=bounds[0],
            maximum_preview_speed_mps=bounds[1],
        )


@pytest.mark.parametrize(
    "preview_times_s",
    [(), (0.12, 0.24, 0.36, 0.48), (0.12, 0.24, 0.36, 0.48, math.nan), (0.12, 0.24, 0.24, 0.48, 0.60)],
)
def test_query_rejects_invalid_preview_times(preview_times_s: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="preview_times_s"):
        pure_rl_spatial_path.query_spatial_path(
            path=_straight_path_batch(count=1),
            previous_progress_m=torch.zeros(1, dtype=torch.float64),
            position_world_m=torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float64),
            ground_velocity_world_mps=torch.zeros((1, 3), dtype=torch.float64),
            preview_times_s=preview_times_s,
        )


def test_query_and_partial_write_reject_device_metadata_and_env_id_contract_mismatches() -> None:
    path = _straight_path_batch(count=2)
    with pytest.raises(ValueError, match="position_world_m"):
        pure_rl_spatial_path.query_spatial_path(
            path=path,
            previous_progress_m=torch.zeros(2, dtype=torch.float64),
            position_world_m=torch.empty((2, 3), dtype=torch.float64, device="meta"),
            ground_velocity_world_mps=torch.zeros((2, 3), dtype=torch.float64),
        )

    source = _straight_path_batch(count=1)
    invalid_env_ids = (
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int64),
    )
    for env_ids in invalid_env_ids:
        with pytest.raises((ValueError, IndexError)):
            pure_rl_spatial_path.write_spatial_path_batch_rows_(destination=path, env_ids=env_ids, source=source)
    with pytest.raises(ValueError, match="unique"):
        pure_rl_spatial_path.write_spatial_path_batch_rows_(
            destination=path,
            env_ids=torch.tensor([0, 0], dtype=torch.int64),
            source=_straight_path_batch(count=2),
        )
    with pytest.raises(ValueError, match="source batch length"):
        pure_rl_spatial_path.write_spatial_path_batch_rows_(
            destination=path,
            env_ids=torch.tensor([0, 1], dtype=torch.int64),
            source=source,
        )
    with pytest.raises(ValueError, match="sample spacing"):
        pure_rl_spatial_path.write_spatial_path_batch_rows_(
            destination=path,
            env_ids=torch.tensor([0], dtype=torch.int64),
            source=replace(source, sample_spacing_m=0.5, path_length_m=10.0),
        )


def _load_longitudinal_module():
    module_path = MODULE_PATH.with_name("pure_rl_longitudinal_path.py")
    spec = importlib.util.spec_from_file_location("pure_rl_longitudinal_path_for_spatial_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

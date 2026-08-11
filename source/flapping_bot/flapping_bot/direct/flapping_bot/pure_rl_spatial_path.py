"""Batched centerline generation for the PureRL spatial curriculum."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

Tensor = torch.Tensor

REHEARSAL_C1_TASK_FAMILY_ID = 0
REHEARSAL_C2C_TASK_FAMILY_ID = 1
EARLIER_SPATIAL_TASK_FAMILY_ID = 2
CURRENT_SPATIAL_TASK_FAMILY_ID_C3A = 2
CURRENT_SPATIAL_TASK_FAMILY_ID = 3

STRAIGHT_TEMPLATE_ID = 0
LONGITUDINAL_TEMPLATE_ID = 1
ISOLATED_TURN_TEMPLATE_ID = 2
C3B_SAME_DIRECTION_TURNS_TEMPLATE_ID = 3
C3B_S_TURNS_TEMPLATE_ID = 4
C3B_TURN_THEN_CLIMB_TEMPLATE_ID = 5
C3B_TURN_THEN_DESCENT_TEMPLATE_ID = 6
C3B_CLIMB_THEN_TURN_TEMPLATE_ID = 7
C3B_DESCENT_THEN_TURN_TEMPLATE_ID = 8
C3B_LOITER_TEMPLATE_ID = 9
C3C_COUPLED_TEMPLATE_ID = 10

_MAX_EVENTS = 4
_EVENT_NONE = 0
_EVENT_TURN = 1
_EVENT_VERTICAL = 2
_EVENT_COUPLED = 3
_GRAVITY_MPS2 = 9.81
_INTERSECTION_VALIDATION_STRIDE = 8
_INTERSECTION_LOCAL_SAMPLE_RADIUS = 4
_MINIMUM_NONADJACENT_CLEARANCE_M = 1.5


@dataclass(frozen=True)
class PureRLSpatialStageConfig:
    """Immutable sampling bounds for one spatial curriculum stage."""

    stage_id: str
    task_probabilities: tuple[float, ...]
    geometry_roll_deg_range: tuple[float, float]
    finite_turn_deg_range: tuple[float, float]
    coupled_slope_deg_range: tuple[float, float] = (0.0, 0.0)
    event_count_range: tuple[int, int] = (1, 1)
    transition_length_m_range: tuple[float, float] = (8.0, 12.0)
    guard_speed_mps: float = 12.0
    sample_spacing_m: float = 0.25
    path_length_m: float = 300.0
    vertical_slope_deg_range: tuple[float, float] = (2.0, 8.0)
    loiter_radius_m_range: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class PureRLSpatialPathBatch:
    """Dense batched path tables and row-level geometry metadata.

    Vector quantities use a z-up world frame. Curvature is horizontal heading
    change per meter, slope is flight-path angle, and all leading dimensions
    index independent paths.
    """

    points_world_m: Tensor
    tangent_world: Tensor
    lateral_normal_world: Tensor
    vertical_normal_world: Tensor
    curvature_rad_per_m: Tensor
    slope_rad: Tensor
    turn_activity: Tensor
    final_event_progress_m: Tensor
    task_family_id: Tensor
    turn_sign: Tensor
    vertical_sign: Tensor
    peak_geometry_roll_rad: Tensor
    peak_slope_rad: Tensor
    event_count: Tensor
    template_id: Tensor


SPATIAL_STAGE_CONFIGS: dict[str, PureRLSpatialStageConfig] = {
    "c3a": PureRLSpatialStageConfig(
        stage_id="c3a",
        task_probabilities=(0.15, 0.25, 0.60),
        geometry_roll_deg_range=(8.0, 14.0),
        finite_turn_deg_range=(20.0, 50.0),
    ),
    "c3b": PureRLSpatialStageConfig(
        stage_id="c3b",
        task_probabilities=(0.15, 0.20, 0.15, 0.50),
        geometry_roll_deg_range=(10.0, 17.0),
        finite_turn_deg_range=(20.0, 60.0),
        event_count_range=(2, 3),
        loiter_radius_m_range=(50.0, 80.0),
    ),
    "c3c": PureRLSpatialStageConfig(
        stage_id="c3c",
        task_probabilities=(0.15, 0.20, 0.15, 0.50),
        geometry_roll_deg_range=(6.0, 17.0),
        finite_turn_deg_range=(20.0, 20.0),
        coupled_slope_deg_range=(1.5, 6.0),
        event_count_range=(2, 4),
    ),
}


def resolve_spatial_stage(stage: str | PureRLSpatialStageConfig) -> PureRLSpatialStageConfig:
    """Resolve a registered spatial stage or validate an explicit config."""

    if isinstance(stage, str):
        try:
            config = SPATIAL_STAGE_CONFIGS[stage]
        except KeyError as error:
            raise ValueError(f"Unknown spatial stage: {stage!r}.") from error
    elif isinstance(stage, PureRLSpatialStageConfig):
        config = stage
    else:
        raise TypeError("stage must be a spatial stage identifier or PureRLSpatialStageConfig.")
    _validate_stage_config(config)
    return config


def sample_spatial_path_batch(
    *,
    num_paths: int,
    stage: str | PureRLSpatialStageConfig,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
    initial_altitude_m: float | Tensor = 10.0,
) -> PureRLSpatialPathBatch:
    """Sample dense spatial centerlines using fixed-size batched Torch operations."""

    if isinstance(num_paths, bool) or not isinstance(num_paths, int) or num_paths <= 0:
        raise ValueError("num_paths must be a positive integer.")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype.")
    resolved_device = torch.device(device)
    config = resolve_spatial_stage(stage)
    altitude = _resolve_initial_altitude(
        initial_altitude_m,
        count=num_paths,
        device=resolved_device,
        dtype=dtype,
    )

    task_family_id = _sample_task_families(
        count=num_paths,
        probabilities=config.task_probabilities,
        device=resolved_device,
        dtype=dtype,
        generator=generator,
    )
    event_type, event_count, template_id, event_turn_sign, event_vertical_sign = _sample_event_contracts(
        task_family_id=task_family_id,
        config=config,
        device=resolved_device,
        dtype=dtype,
        generator=generator,
    )
    event_roll_rad, event_slope_rad = _sample_event_amplitudes(
        task_family_id=task_family_id,
        event_type=event_type,
        template_id=template_id,
        event_vertical_sign=event_vertical_sign,
        config=config,
        device=resolved_device,
        dtype=dtype,
        generator=generator,
    )
    event_start_m, transition_length_m, plateau_length_m, event_end_m = _sample_event_layout(
        event_type=event_type,
        event_count=event_count,
        template_id=template_id,
        event_roll_rad=event_roll_rad,
        config=config,
        device=resolved_device,
        dtype=dtype,
        generator=generator,
    )

    point_count = int(round(config.path_length_m / config.sample_spacing_m)) + 1
    progress_m = torch.arange(point_count, device=resolved_device, dtype=dtype) * config.sample_spacing_m
    event_profile = _build_event_profile(
        progress_m=progress_m,
        event_start_m=event_start_m,
        transition_length_m=transition_length_m,
        plateau_length_m=plateau_length_m,
        event_type=event_type,
    )
    signed_curvature = (
        _GRAVITY_MPS2
        * torch.tan(event_roll_rad)
        / config.guard_speed_mps**2
        * event_turn_sign
    )
    curvature_rad_per_m = torch.sum(signed_curvature.unsqueeze(2) * event_profile, dim=1)
    slope_rad = torch.sum(event_slope_rad.unsqueeze(2) * event_profile, dim=1)
    turn_event = (event_type == _EVENT_TURN) | (event_type == _EVENT_COUPLED)
    turn_activity = torch.amax(event_profile * turn_event.unsqueeze(2), dim=1).clamp(0.0, 1.0)

    initial_heading_rad = 2.0 * math.pi * torch.rand(
        num_paths,
        device=resolved_device,
        dtype=dtype,
        generator=generator,
    )
    heading_delta_rad = (
        0.5
        * (curvature_rad_per_m[:, :-1] + curvature_rad_per_m[:, 1:])
        * config.sample_spacing_m
    )
    heading_rad = initial_heading_rad.unsqueeze(1) + torch.cat(
        (
            torch.zeros((num_paths, 1), device=resolved_device, dtype=dtype),
            torch.cumsum(heading_delta_rad, dim=1),
        ),
        dim=1,
    )
    tangent_world, lateral_normal_world, vertical_normal_world = _heading_slope_frame(
        heading_rad=heading_rad,
        slope_rad=slope_rad,
    )
    segment_delta_world_m = (
        0.5
        * (tangent_world[:, :-1, :] + tangent_world[:, 1:, :])
        * config.sample_spacing_m
    )
    relative_points_world_m = torch.cat(
        (
            torch.zeros((num_paths, 1, 3), device=resolved_device, dtype=dtype),
            torch.cumsum(segment_delta_world_m, dim=1),
        ),
        dim=1,
    )
    relative_points_world_m[:, :, 2] += altitude.unsqueeze(1)

    turn_mask = turn_event
    vertical_mask = (event_type == _EVENT_VERTICAL) | (event_type == _EVENT_COUPLED)
    turn_sign = _first_active_value(event_turn_sign, turn_mask)
    vertical_sign = _first_active_value(event_vertical_sign, vertical_mask)
    peak_geometry_roll_rad = torch.amax(torch.abs(event_roll_rad), dim=1)
    peak_slope_magnitude_rad = torch.amax(torch.abs(event_slope_rad), dim=1)
    peak_slope_rad = vertical_sign * peak_slope_magnitude_rad
    final_event_progress_m = torch.amax(event_end_m, dim=1)

    batch = PureRLSpatialPathBatch(
        points_world_m=relative_points_world_m,
        tangent_world=tangent_world,
        lateral_normal_world=lateral_normal_world,
        vertical_normal_world=vertical_normal_world,
        curvature_rad_per_m=curvature_rad_per_m,
        slope_rad=slope_rad,
        turn_activity=turn_activity,
        final_event_progress_m=final_event_progress_m,
        task_family_id=task_family_id,
        turn_sign=turn_sign,
        vertical_sign=vertical_sign,
        peak_geometry_roll_rad=peak_geometry_roll_rad,
        peak_slope_rad=peak_slope_rad,
        event_count=event_count,
        template_id=template_id,
    )
    _validate_generated_batch(batch, config=config)
    return batch


def _sample_task_families(
    *,
    count: int,
    probabilities: tuple[float, ...],
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    draw = torch.rand(count, device=device, dtype=dtype, generator=generator)
    thresholds = torch.tensor(probabilities, device=device, dtype=dtype).cumsum(dim=0)[:-1]
    return torch.bucketize(draw, thresholds).to(dtype=torch.int64)


def _sample_event_contracts(
    *,
    task_family_id: Tensor,
    config: PureRLSpatialStageConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    count = task_family_id.shape[0]
    event_type = torch.zeros((count, _MAX_EVENTS), device=device, dtype=torch.int64)
    event_count = torch.zeros(count, device=device, dtype=torch.int64)
    template_id = torch.full((count,), STRAIGHT_TEMPLATE_ID, device=device, dtype=torch.int64)
    random_turn_sign = _sample_sign((count, _MAX_EVENTS), device=device, dtype=dtype, generator=generator)
    random_vertical_sign = _sample_sign((count, _MAX_EVENTS), device=device, dtype=dtype, generator=generator)

    c2_rows = task_family_id == REHEARSAL_C2C_TASK_FAMILY_ID
    event_type[c2_rows, 0] = _EVENT_VERTICAL
    event_count[c2_rows] = 1
    template_id[c2_rows] = LONGITUDINAL_TEMPLATE_ID

    earlier_spatial_rows = task_family_id == EARLIER_SPATIAL_TASK_FAMILY_ID
    if config.stage_id == "c3a":
        earlier_spatial_rows = torch.zeros_like(earlier_spatial_rows)
        current_rows = task_family_id == CURRENT_SPATIAL_TASK_FAMILY_ID_C3A
    else:
        current_rows = task_family_id == CURRENT_SPATIAL_TASK_FAMILY_ID
    event_type[earlier_spatial_rows, 0] = _EVENT_TURN
    event_count[earlier_spatial_rows] = 1
    template_id[earlier_spatial_rows] = ISOLATED_TURN_TEMPLATE_ID

    if config.stage_id == "c3a":
        event_type[current_rows, 0] = _EVENT_TURN
        event_count[current_rows] = 1
        template_id[current_rows] = ISOLATED_TURN_TEMPLATE_ID
    elif config.stage_id == "c3b":
        sampled_count = torch.randint(
            config.event_count_range[0],
            config.event_count_range[1] + 1,
            (count,),
            device=device,
            generator=generator,
        )
        event_count[current_rows] = sampled_count[current_rows]
        local_template = torch.randint(0, 7, (count,), device=device, generator=generator)
        template_id[current_rows] = local_template[current_rows] + C3B_SAME_DIRECTION_TURNS_TEMPLATE_ID
        two_event_rows = current_rows & (
            (template_id == C3B_SAME_DIRECTION_TURNS_TEMPLATE_ID)
            | (template_id == C3B_S_TURNS_TEMPLATE_ID)
            | (template_id == C3B_LOITER_TEMPLATE_ID)
        )
        event_count[two_event_rows] = 2
        active = torch.arange(_MAX_EVENTS, device=device).unsqueeze(0) < event_count.unsqueeze(1)
        event_type[current_rows & active[:, 0], 0] = _EVENT_TURN
        event_type[current_rows & active[:, 1], 1] = _EVENT_TURN
        third_rows = current_rows & active[:, 2]
        random_third_type = torch.where(
            torch.rand(count, device=device, dtype=dtype, generator=generator) < 0.5,
            _EVENT_TURN,
            _EVENT_VERTICAL,
        )
        event_type[third_rows, 2] = random_third_type[third_rows]
        _apply_c3b_template_contracts(
            event_type=event_type,
            turn_sign=random_turn_sign,
            vertical_sign=random_vertical_sign,
            template_id=template_id,
            current_rows=current_rows,
        )
    else:
        sampled_count = torch.randint(
            config.event_count_range[0],
            config.event_count_range[1] + 1,
            (count,),
            device=device,
            generator=generator,
        )
        event_count[current_rows] = sampled_count[current_rows]
        template_id[current_rows] = C3C_COUPLED_TEMPLATE_ID
        active = torch.arange(_MAX_EVENTS, device=device).unsqueeze(0) < event_count.unsqueeze(1)
        event_type[current_rows, 0] = _EVENT_COUPLED
        later_type = torch.randint(1, 4, (count, _MAX_EVENTS), device=device, generator=generator)
        later_active = current_rows.unsqueeze(1) & active
        event_type[:, 1:] = torch.where(later_active[:, 1:], later_type[:, 1:], event_type[:, 1:])

        earlier_c3b_rows = earlier_spatial_rows & (
            torch.rand(count, device=device, dtype=dtype, generator=generator) < 0.5
        )
        event_count[earlier_c3b_rows] = 2
        earlier_c3b_template = torch.randint(0, 2, (count,), device=device, generator=generator)
        template_id[earlier_c3b_rows] = (
            earlier_c3b_template[earlier_c3b_rows] + C3B_SAME_DIRECTION_TURNS_TEMPLATE_ID
        )
        event_type[earlier_c3b_rows, 0:2] = _EVENT_TURN
        _apply_c3b_template_contracts(
            event_type=event_type,
            turn_sign=random_turn_sign,
            vertical_sign=random_vertical_sign,
            template_id=template_id,
            current_rows=earlier_c3b_rows,
        )

    active = torch.arange(_MAX_EVENTS, device=device).unsqueeze(0) < event_count.unsqueeze(1)
    event_type = torch.where(active, event_type, torch.zeros_like(event_type))
    return event_type, event_count, template_id, random_turn_sign, random_vertical_sign


def _apply_c3b_template_contracts(
    *,
    event_type: Tensor,
    turn_sign: Tensor,
    vertical_sign: Tensor,
    template_id: Tensor,
    current_rows: Tensor,
) -> None:
    same = current_rows & (template_id == C3B_SAME_DIRECTION_TURNS_TEMPLATE_ID)
    s_turn = current_rows & (template_id == C3B_S_TURNS_TEMPLATE_ID)
    turn_climb = current_rows & (template_id == C3B_TURN_THEN_CLIMB_TEMPLATE_ID)
    turn_descent = current_rows & (template_id == C3B_TURN_THEN_DESCENT_TEMPLATE_ID)
    climb_turn = current_rows & (template_id == C3B_CLIMB_THEN_TURN_TEMPLATE_ID)
    descent_turn = current_rows & (template_id == C3B_DESCENT_THEN_TURN_TEMPLATE_ID)
    loiter = current_rows & (template_id == C3B_LOITER_TEMPLATE_ID)

    event_type[same | s_turn | loiter, 0:3] = _EVENT_TURN
    turn_sign[same | loiter, :] = turn_sign[same | loiter, 0].unsqueeze(1)
    turn_sign[s_turn, 1] = -turn_sign[s_turn, 0]
    turn_sign[s_turn, 2] = turn_sign[s_turn, 0]
    for rows, first_type, second_type, vertical_value in (
        (turn_climb, _EVENT_TURN, _EVENT_VERTICAL, 1.0),
        (turn_descent, _EVENT_TURN, _EVENT_VERTICAL, -1.0),
        (climb_turn, _EVENT_VERTICAL, _EVENT_TURN, 1.0),
        (descent_turn, _EVENT_VERTICAL, _EVENT_TURN, -1.0),
    ):
        event_type[rows, 0] = first_type
        event_type[rows, 1] = second_type
        vertical_slot = 1 if second_type == _EVENT_VERTICAL else 0
        vertical_sign[rows, vertical_slot] = vertical_value


def _sample_event_amplitudes(
    *,
    task_family_id: Tensor,
    event_type: Tensor,
    template_id: Tensor,
    event_vertical_sign: Tensor,
    config: PureRLSpatialStageConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    count = task_family_id.shape[0]
    turn_mask = (event_type == _EVENT_TURN) | (event_type == _EVENT_COUPLED)
    vertical_mask = (event_type == _EVENT_VERTICAL) | (event_type == _EVENT_COUPLED)
    roll_deg = _sample_uniform(
        (count, _MAX_EVENTS),
        config.geometry_roll_deg_range,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    earlier_rows = task_family_id == EARLIER_SPATIAL_TASK_FAMILY_ID
    if config.stage_id != "c3a":
        earlier_roll_deg = _sample_uniform(
            (count, _MAX_EVENTS),
            (8.0, 14.0),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        roll_deg = torch.where(earlier_rows.unsqueeze(1), earlier_roll_deg, roll_deg)
        earlier_c3b_rows = earlier_rows & (
            (template_id == C3B_SAME_DIRECTION_TURNS_TEMPLATE_ID)
            | (template_id == C3B_S_TURNS_TEMPLATE_ID)
        )
        earlier_c3b_roll_deg = _sample_uniform(
            (count, _MAX_EVENTS),
            (10.0, 17.0),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        roll_deg = torch.where(earlier_c3b_rows.unsqueeze(1), earlier_c3b_roll_deg, roll_deg)

    loiter_rows = template_id == C3B_LOITER_TEMPLATE_ID
    if config.stage_id == "c3b":
        radius_m = _sample_uniform(
            (count, 1),
            config.loiter_radius_m_range,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        loiter_roll_rad = torch.atan(config.guard_speed_mps**2 / (_GRAVITY_MPS2 * radius_m))
        roll_deg = torch.where(loiter_rows.unsqueeze(1), torch.rad2deg(loiter_roll_rad).expand_as(roll_deg), roll_deg)

    current_c3c = (config.stage_id == "c3c") & (task_family_id == CURRENT_SPATIAL_TASK_FAMILY_ID)
    if isinstance(current_c3c, Tensor) and bool(torch.any(current_c3c)):
        row_roll_deg = _sample_uniform(
            (count, 1),
            config.geometry_roll_deg_range,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        roll_deg = torch.where(current_c3c.unsqueeze(1), row_roll_deg.expand_as(roll_deg), roll_deg)

    event_roll_rad = torch.deg2rad(roll_deg) * turn_mask
    slope_deg = _sample_uniform(
        (count, _MAX_EVENTS),
        config.vertical_slope_deg_range,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if isinstance(current_c3c, Tensor) and bool(torch.any(current_c3c)):
        row_roll_fraction = row_roll_deg[:, 0] / 20.0
        elliptical_maximum_slope_deg = 6.0 * torch.sqrt(
            torch.clamp(1.0 - torch.square(row_roll_fraction), min=0.0)
        )
        maximum_coupled_slope_deg = torch.minimum(
            elliptical_maximum_slope_deg,
            torch.full_like(elliptical_maximum_slope_deg, config.coupled_slope_deg_range[1]),
        )
        minimum_slope_deg = config.coupled_slope_deg_range[0]
        row_slope_deg = minimum_slope_deg + (
            maximum_coupled_slope_deg - minimum_slope_deg
        ) * torch.rand(count, device=device, dtype=dtype, generator=generator)
        slope_deg = torch.where(current_c3c.unsqueeze(1), row_slope_deg.unsqueeze(1), slope_deg)
    event_slope_rad = torch.deg2rad(slope_deg) * event_vertical_sign * vertical_mask
    return event_roll_rad, event_slope_rad


def _sample_event_layout(
    *,
    event_type: Tensor,
    event_count: Tensor,
    template_id: Tensor,
    event_roll_rad: Tensor,
    config: PureRLSpatialStageConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    count = event_type.shape[0]
    active = event_type != _EVENT_NONE
    transition_length_m = _sample_uniform(
        (count, _MAX_EVENTS),
        config.transition_length_m_range,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    turn_mask = (event_type == _EVENT_TURN) | (event_type == _EVENT_COUPLED)
    curvature_magnitude = _GRAVITY_MPS2 * torch.tan(event_roll_rad) / config.guard_speed_mps**2

    lower_by_row = torch.full((count,), config.finite_turn_deg_range[0], device=device, dtype=dtype)
    upper_by_row = torch.full((count,), config.finite_turn_deg_range[1], device=device, dtype=dtype)
    if config.stage_id == "c3c":
        earlier_c3a = template_id == ISOLATED_TURN_TEMPLATE_ID
        earlier_c3b = (template_id == C3B_SAME_DIRECTION_TURNS_TEMPLATE_ID) | (
            template_id == C3B_S_TURNS_TEMPLATE_ID
        )
        lower_by_row = torch.where(earlier_c3a | earlier_c3b, 20.0, lower_by_row)
        upper_by_row = torch.where(earlier_c3a, 50.0, upper_by_row)
        upper_by_row = torch.where(earlier_c3b, 60.0, upper_by_row)
    heading_change_deg = lower_by_row.unsqueeze(1) + (
        upper_by_row.unsqueeze(1) - lower_by_row.unsqueeze(1)
    ) * torch.rand((count, _MAX_EVENTS), device=device, dtype=dtype, generator=generator)
    safe_curvature = torch.where(turn_mask, curvature_magnitude, torch.ones_like(curvature_magnitude))
    turn_plateau_length_m = torch.deg2rad(heading_change_deg) / safe_curvature - transition_length_m
    vertical_plateau_length_m = _sample_uniform(
        (count, _MAX_EVENTS),
        (20.0, 30.0),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    plateau_length_m = torch.where(turn_mask, turn_plateau_length_m, vertical_plateau_length_m)
    plateau_length_m = torch.where(active, plateau_length_m, torch.zeros_like(plateau_length_m))

    loiter_rows = template_id == C3B_LOITER_TEMPLATE_ID
    if bool(torch.any(loiter_rows)):
        transition_length_m[loiter_rows, 1] = transition_length_m[loiter_rows, 0]
        plateau_length_m[loiter_rows, 0] = 130.0 - transition_length_m[loiter_rows, 0]
        plateau_length_m[loiter_rows, 1] = 130.0 - 2.0 * transition_length_m[loiter_rows, 0]

    total_event_length_m = torch.where(
        active,
        plateau_length_m + 2.0 * transition_length_m,
        torch.zeros_like(plateau_length_m),
    )
    gap_length_m = _sample_uniform(
        (count, _MAX_EVENTS),
        (8.0, 12.0),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    initial_entry_m = _sample_uniform(
        (count,),
        (15.0, 20.0),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    preceding_extent_m = total_event_length_m + gap_length_m
    event_start_m = initial_entry_m.unsqueeze(1) + torch.cumsum(preceding_extent_m, dim=1) - preceding_extent_m
    event_start_m = torch.where(active, event_start_m, torch.zeros_like(event_start_m))
    event_end_m = torch.where(active, event_start_m + total_event_length_m, torch.zeros_like(event_start_m))
    if bool(torch.any(loiter_rows)):
        event_start_m[loiter_rows, 1] = event_start_m[loiter_rows, 0] + 130.0
        event_end_m[loiter_rows, 0] = (
            event_start_m[loiter_rows, 0] + 130.0 + transition_length_m[loiter_rows, 0]
        )
        event_end_m[loiter_rows, 1] = event_start_m[loiter_rows, 0] + 260.0
    return event_start_m, transition_length_m, plateau_length_m, event_end_m


def _build_event_profile(
    *,
    progress_m: Tensor,
    event_start_m: Tensor,
    transition_length_m: Tensor,
    plateau_length_m: Tensor,
    event_type: Tensor,
) -> Tensor:
    progress = progress_m.reshape(1, 1, -1)
    start = event_start_m.unsqueeze(2)
    transition = transition_length_m.unsqueeze(2)
    exit_start = start + transition + plateau_length_m.unsqueeze(2)
    entry_u = torch.clamp((progress - start) / transition, min=0.0, max=1.0)
    exit_u = torch.clamp((progress - exit_start) / transition, min=0.0, max=1.0)
    profile = _quintic_smoothstep(entry_u) * (1.0 - _quintic_smoothstep(exit_u))
    return profile * (event_type != _EVENT_NONE).unsqueeze(2)


def _quintic_smoothstep(value: Tensor) -> Tensor:
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def _heading_slope_frame(*, heading_rad: Tensor, slope_rad: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    cos_heading = torch.cos(heading_rad)
    sin_heading = torch.sin(heading_rad)
    cos_slope = torch.cos(slope_rad)
    sin_slope = torch.sin(slope_rad)
    zeros = torch.zeros_like(heading_rad)
    tangent_world = torch.stack(
        (cos_slope * cos_heading, cos_slope * sin_heading, sin_slope),
        dim=2,
    )
    lateral_normal_world = torch.stack((-sin_heading, cos_heading, zeros), dim=2)
    vertical_normal_world = torch.stack(
        (-sin_slope * cos_heading, -sin_slope * sin_heading, cos_slope),
        dim=2,
    )
    return tangent_world, lateral_normal_world, vertical_normal_world


def _first_active_value(values: Tensor, active: Tensor) -> Tensor:
    first_index = torch.argmax(active.to(dtype=torch.int64), dim=1, keepdim=True)
    selected = torch.gather(values, 1, first_index).squeeze(1)
    return torch.where(torch.any(active, dim=1), selected, torch.zeros_like(selected))


def _sample_sign(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    draw = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    return torch.where(draw < 0.5, -torch.ones_like(draw), torch.ones_like(draw))


def _sample_uniform(
    shape: tuple[int, ...],
    bounds: tuple[float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    lower, upper = bounds
    return lower + (upper - lower) * torch.rand(shape, device=device, dtype=dtype, generator=generator)


def _resolve_initial_altitude(
    initial_altitude_m: float | Tensor,
    *,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    altitude = torch.as_tensor(initial_altitude_m, device=device, dtype=dtype)
    if altitude.ndim == 0:
        altitude = altitude.expand(count).clone()
    elif altitude.shape != (count,):
        raise ValueError("initial_altitude_m must be scalar or have shape (num_paths,).")
    if not bool(torch.all(torch.isfinite(altitude))):
        raise ValueError("initial_altitude_m must contain only finite values.")
    return altitude


def _validate_stage_config(config: PureRLSpatialStageConfig) -> None:
    expected_probability_count = {"c3a": 3, "c3b": 4, "c3c": 4}
    if config.stage_id not in expected_probability_count:
        raise ValueError(f"Unknown spatial stage: {config.stage_id!r}.")
    if len(config.task_probabilities) != expected_probability_count[config.stage_id]:
        raise ValueError("task_probabilities has the wrong number of stage families.")
    if any(not math.isfinite(value) or value < 0.0 for value in config.task_probabilities):
        raise ValueError("task_probabilities must contain finite non-negative values.")
    if not math.isclose(sum(config.task_probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("task_probabilities must sum to one.")
    for name, bounds, allow_zero in (
        ("geometry_roll_deg_range", config.geometry_roll_deg_range, False),
        ("finite_turn_deg_range", config.finite_turn_deg_range, False),
        ("coupled_slope_deg_range", config.coupled_slope_deg_range, True),
        ("transition_length_m_range", config.transition_length_m_range, False),
        ("vertical_slope_deg_range", config.vertical_slope_deg_range, False),
        ("loiter_radius_m_range", config.loiter_radius_m_range, True),
    ):
        _validate_range(name, bounds, allow_zero=allow_zero)
    minimum_events, maximum_events = config.event_count_range
    if (
        isinstance(minimum_events, bool)
        or isinstance(maximum_events, bool)
        or not isinstance(minimum_events, int)
        or not isinstance(maximum_events, int)
        or minimum_events < 1
        or maximum_events < minimum_events
        or maximum_events > _MAX_EVENTS
    ):
        raise ValueError("event_count_range must contain ordered integers between one and four.")
    for name, value in (
        ("guard_speed_mps", config.guard_speed_mps),
        ("sample_spacing_m", config.sample_spacing_m),
        ("path_length_m", config.path_length_m),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    intervals = config.path_length_m / config.sample_spacing_m
    if not math.isclose(intervals, round(intervals), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("path_length_m must be an integer multiple of sample_spacing_m.")
    if config.stage_id == "c3c":
        if config.coupled_slope_deg_range[0] <= 0.0:
            raise ValueError("c3c coupled_slope_deg_range must be positive.")
        maximum_roll_fraction = config.geometry_roll_deg_range[1] / 20.0
        available_slope_deg = 6.0 * math.sqrt(max(0.0, 1.0 - maximum_roll_fraction**2))
        if config.coupled_slope_deg_range[0] > available_slope_deg:
            raise ValueError("c3c coupled ranges do not admit the approved elliptical demand bound.")
    if config.stage_id == "c3b" and config.loiter_radius_m_range[0] <= 0.0:
        raise ValueError("c3b loiter_radius_m_range must be positive.")


def _validate_range(name: str, bounds: tuple[float, float], *, allow_zero: bool) -> None:
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain exactly two bounds.")
    lower, upper = bounds
    minimum = 0.0 if allow_zero else torch.finfo(torch.float64).tiny
    if not math.isfinite(lower) or not math.isfinite(upper) or lower < minimum or upper < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain finite {qualifier} ordered bounds.")


def _validate_generated_batch(batch: PureRLSpatialPathBatch, *, config: PureRLSpatialStageConfig) -> None:
    float_tensors = (
        batch.points_world_m,
        batch.tangent_world,
        batch.lateral_normal_world,
        batch.vertical_normal_world,
        batch.curvature_rad_per_m,
        batch.slope_rad,
        batch.turn_activity,
        batch.final_event_progress_m,
        batch.turn_sign,
        batch.vertical_sign,
        batch.peak_geometry_roll_rad,
        batch.peak_slope_rad,
    )
    if any(not bool(torch.all(torch.isfinite(value))) for value in float_tensors):
        raise RuntimeError("Generated spatial path contains a non-finite value.")
    if bool(torch.any(batch.final_event_progress_m > config.path_length_m)):
        raise RuntimeError("Generated spatial path extends beyond the centerline table.")
    absolute_heading_change_rad = torch.trapezoid(
        torch.abs(batch.curvature_rad_per_m),
        dx=config.sample_spacing_m,
        dim=1,
    )
    if bool(torch.any(absolute_heading_change_rad >= 2.0 * math.pi)):
        raise RuntimeError("Generated bounded template can self-intersect after a full heading revolution.")
    _validate_nonadjacent_centerline_clearance(batch)
    if not bool(torch.all(batch.curvature_rad_per_m[:, 0] == 0.0)) or not bool(
        torch.all(batch.slope_rad[:, 0] == 0.0)
    ):
        raise RuntimeError("Generated spatial paths must begin straight and level.")
    if config.stage_id == "c3b":
        current = batch.task_family_id == CURRENT_SPATIAL_TASK_FAMILY_ID
        simultaneous = (batch.curvature_rad_per_m[current] != 0.0) & (batch.slope_rad[current] != 0.0)
        if bool(torch.any(simultaneous)):
            raise RuntimeError("C3b current-stage paths must contain sequential, non-coupled events.")
    if config.stage_id == "c3c":
        current = batch.task_family_id == CURRENT_SPATIAL_TASK_FAMILY_ID
        demand = torch.square(batch.peak_geometry_roll_rad[current] / math.radians(20.0))
        demand += torch.square(torch.abs(batch.peak_slope_rad[current]) / math.radians(6.0))
        if bool(torch.any(demand > 1.0 + 16.0 * torch.finfo(batch.points_world_m.dtype).eps)):
            raise RuntimeError("C3c current-stage coupled demand exceeds the approved bound.")


def _validate_nonadjacent_centerline_clearance(batch: PureRLSpatialPathBatch) -> None:
    """Reject close nonlocal branches on a generation-time validation grid."""

    checked_rows = batch.template_id != C3B_LOITER_TEMPLATE_ID
    if not bool(torch.any(checked_rows)):
        return
    sampled_points = batch.points_world_m[checked_rows, ::_INTERSECTION_VALIDATION_STRIDE, :]
    if sampled_points.dtype in (torch.float16, torch.bfloat16):
        sampled_points = sampled_points.to(dtype=torch.float32)
    pairwise_distance_m = torch.cdist(sampled_points, sampled_points)
    sample_count = sampled_points.shape[1]
    sample_index = torch.arange(sample_count, device=sampled_points.device)
    nonadjacent = (
        torch.abs(sample_index.unsqueeze(0) - sample_index.unsqueeze(1))
        > _INTERSECTION_LOCAL_SAMPLE_RADIUS
    )
    too_close = pairwise_distance_m < _MINIMUM_NONADJACENT_CLEARANCE_M
    if bool(torch.any(too_close & nonadjacent.unsqueeze(0))):
        raise RuntimeError(
            "Generated nonadjacent centerline branches are closer than "
            f"{_MINIMUM_NONADJACENT_CLEARANCE_M:.1f} m."
        )

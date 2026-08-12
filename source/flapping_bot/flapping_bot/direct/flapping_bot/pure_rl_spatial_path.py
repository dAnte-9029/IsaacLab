"""Batched centerline generation for the PureRL spatial curriculum."""

from __future__ import annotations

from dataclasses import dataclass, fields
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
_INTERSECTION_VALIDATION_STRIDE = 4
_INTERSECTION_LOCAL_SAMPLE_RADIUS = 8
_MINIMUM_NONADJACENT_CLEARANCE_M = 1.5
_CLEARANCE_VALIDATION_THRESHOLD_M = 2.5
_CLEARANCE_VALIDATION_BATCH_SIZE = 32
PURE_RL_PREVIEW_TIMES_S: tuple[float, ...] = (0.12, 0.24, 0.36, 0.48, 0.60)
_LOCAL_PROJECTION_BEHIND_M = 2.0
_LOCAL_PROJECTION_AHEAD_M = 4.0


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
    minimum_altitude_m: float = 0.05
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
    sample_spacing_m: float
    path_length_m: float


@dataclass(frozen=True)
class PureRLSpatialPathQuery:
    """Current route-frame geometry and time-based world-frame preview."""

    progress_m: Tensor
    reference_position_world_m: Tensor
    horizontal_normal_error_m: Tensor
    vertical_normal_error_m: Tensor
    tangent_world: Tensor
    lateral_normal_world: Tensor
    vertical_normal_world: Tensor
    active_curvature_rad_per_m: Tensor
    active_slope_rad: Tensor
    turn_activity: Tensor
    preview_points_world_m: Tensor
    reached_all_events: Tensor


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
    evaluation_template_id: Tensor | None = None,
    evaluation_geometry_roll_deg: Tensor | None = None,
    evaluation_slope_deg: Tensor | None = None,
    evaluation_turn_sign: Tensor | None = None,
    evaluation_heading_rad: Tensor | None = None,
) -> PureRLSpatialPathBatch:
    """Sample dense paths, replacing invalid rows in one bounded resampling pass."""

    if isinstance(num_paths, bool) or not isinstance(num_paths, int) or num_paths <= 0:
        raise ValueError("num_paths must be a positive integer.")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be the floating-point dtype float32 or float64.")
    resolved_device = torch.device(device)
    config = resolve_spatial_stage(stage)
    altitude = _resolve_initial_altitude(
        initial_altitude_m,
        count=num_paths,
        device=resolved_device,
        dtype=dtype,
    )
    evaluation_overrides = _resolve_evaluation_overrides(
        count=num_paths,
        config=config,
        device=resolved_device,
        dtype=dtype,
        template_id=evaluation_template_id,
        geometry_roll_deg=evaluation_geometry_roll_deg,
        slope_deg=evaluation_slope_deg,
        turn_sign=evaluation_turn_sign,
        heading_rad=evaluation_heading_rad,
    )
    batch = _sample_spatial_path_batch_once(
        num_paths=num_paths,
        config=config,
        device=resolved_device,
        dtype=dtype,
        generator=generator,
        initial_altitude_m=altitude,
        evaluation_overrides=evaluation_overrides,
    )
    invalid_rows = _resampleable_invalid_rows(batch, config=config)
    if bool(torch.any(invalid_rows)):
        invalid_ids = torch.nonzero(invalid_rows, as_tuple=False).flatten()
        replacement = _sample_spatial_path_batch_once(
            num_paths=invalid_ids.numel(),
            config=config,
            device=resolved_device,
            dtype=dtype,
            generator=generator,
            initial_altitude_m=altitude.index_select(0, invalid_ids),
            evaluation_overrides=(
                None
                if evaluation_overrides is None
                else tuple(value.index_select(0, invalid_ids) for value in evaluation_overrides)
            ),
        )
        replacement_invalid = _resampleable_invalid_rows(replacement, config=config)
        if bool(torch.any(replacement_invalid)):
            _validate_generated_batch(replacement, config=config)
            raise RuntimeError("Spatial path replacement remained invalid after one resampling pass.")
        batch = _replace_batch_rows(batch, row_ids=invalid_ids, replacement=replacement)
    _validate_generated_batch(batch, config=config)
    return batch


def query_spatial_path(
    *,
    path: PureRLSpatialPathBatch,
    previous_progress_m: Tensor,
    position_world_m: Tensor,
    ground_velocity_world_mps: Tensor,
    minimum_preview_speed_mps: float = 1.0,
    maximum_preview_speed_mps: float = 12.0,
    preview_times_s: tuple[float, ...] = PURE_RL_PREVIEW_TIMES_S,
) -> PureRLSpatialPathQuery:
    """Project positions onto a bounded local path window and query route geometry."""

    _validate_path_batch(path)
    count = path.points_world_m.shape[0]
    _validate_query_tensor(
        "previous_progress_m",
        previous_progress_m,
        expected_shape=(count,),
        reference=path.points_world_m,
    )
    _validate_query_tensor(
        "position_world_m",
        position_world_m,
        expected_shape=(count, 3),
        reference=path.points_world_m,
    )
    _validate_query_tensor(
        "ground_velocity_world_mps",
        ground_velocity_world_mps,
        expected_shape=(count, 3),
        reference=path.points_world_m,
    )
    minimum_speed = float(minimum_preview_speed_mps)
    maximum_speed = float(maximum_preview_speed_mps)
    if (
        not math.isfinite(minimum_speed)
        or not math.isfinite(maximum_speed)
        or minimum_speed < 0.0
        or maximum_speed < minimum_speed
    ):
        raise ValueError("Preview-speed bounds must satisfy finite 0 <= minimum <= maximum.")
    _validate_preview_times(preview_times_s)

    point_count = path.points_world_m.shape[1]
    spacing = path.sample_spacing_m
    previous_segment = torch.floor(
        torch.clamp(previous_progress_m, min=0.0, max=path.path_length_m) / spacing
    ).to(dtype=torch.int64)
    previous_segment.clamp_(max=point_count - 2)
    behind_count = math.ceil(_LOCAL_PROJECTION_BEHIND_M / spacing)
    ahead_count = math.ceil(_LOCAL_PROJECTION_AHEAD_M / spacing)
    offsets = torch.arange(
        -behind_count,
        ahead_count + 1,
        device=path.points_world_m.device,
        dtype=torch.int64,
    )
    segment_indices = (previous_segment.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
        min=0,
        max=point_count - 2,
    )
    row_indices = torch.arange(count, device=path.points_world_m.device).unsqueeze(1)
    segment_start = path.points_world_m[row_indices, segment_indices]
    segment_end = path.points_world_m[row_indices, segment_indices + 1]
    segment_delta = segment_end - segment_start
    denominator = torch.sum(segment_delta * segment_delta, dim=2)
    projection_fraction = torch.sum(
        (position_world_m.unsqueeze(1) - segment_start) * segment_delta,
        dim=2,
    ) / torch.clamp(denominator, min=torch.finfo(path.points_world_m.dtype).tiny)
    projection_fraction.clamp_(min=0.0, max=1.0)
    projected_points = segment_start + projection_fraction.unsqueeze(2) * segment_delta
    squared_distance = torch.sum((position_world_m.unsqueeze(1) - projected_points) ** 2, dim=2)
    closest_offset = torch.argmin(squared_distance, dim=1)
    closest_segment = segment_indices[row_indices.squeeze(1), closest_offset]
    closest_fraction = projection_fraction[row_indices.squeeze(1), closest_offset]
    progress_m = (closest_segment.to(dtype=path.points_world_m.dtype) + closest_fraction) * spacing

    reference_position_world_m = _interpolate_path_table(path.points_world_m, progress_m, path=path)
    tangent_raw = _interpolate_path_table(path.tangent_world, progress_m, path=path)
    lateral_raw = _interpolate_path_table(path.lateral_normal_world, progress_m, path=path)
    tangent_world, lateral_normal_world, vertical_normal_world = _orthonormalize_route_frame(
        tangent_raw=tangent_raw,
        lateral_raw=lateral_raw,
    )
    active_curvature_rad_per_m = _interpolate_path_table(path.curvature_rad_per_m, progress_m, path=path)
    active_slope_rad = _interpolate_path_table(path.slope_rad, progress_m, path=path)
    turn_activity = _interpolate_path_table(path.turn_activity, progress_m, path=path)
    displacement_world_m = position_world_m - reference_position_world_m
    horizontal_normal_error_m = torch.sum(displacement_world_m * lateral_normal_world, dim=1)
    vertical_normal_error_m = torch.sum(displacement_world_m * vertical_normal_world, dim=1)

    tangent_speed_mps = torch.sum(ground_velocity_world_mps * tangent_world, dim=1)
    preview_speed_mps = torch.clamp(tangent_speed_mps, min=minimum_speed, max=maximum_speed)
    preview_times = path.points_world_m.new_tensor(preview_times_s)
    preview_progress_m = progress_m.unsqueeze(1) + preview_speed_mps.unsqueeze(1) * preview_times
    preview_progress_m.clamp_(max=path.path_length_m)
    preview_points_world_m = _interpolate_path_table(path.points_world_m, preview_progress_m, path=path)

    return PureRLSpatialPathQuery(
        progress_m=progress_m,
        reference_position_world_m=reference_position_world_m,
        horizontal_normal_error_m=horizontal_normal_error_m,
        vertical_normal_error_m=vertical_normal_error_m,
        tangent_world=tangent_world,
        lateral_normal_world=lateral_normal_world,
        vertical_normal_world=vertical_normal_world,
        active_curvature_rad_per_m=active_curvature_rad_per_m,
        active_slope_rad=active_slope_rad,
        turn_activity=turn_activity,
        preview_points_world_m=preview_points_world_m,
        reached_all_events=progress_m >= path.final_event_progress_m,
    )


def write_spatial_path_batch_rows_(
    *,
    destination: PureRLSpatialPathBatch,
    env_ids: Tensor,
    source: PureRLSpatialPathBatch,
) -> None:
    """Copy every batched path field into selected destination rows in place."""

    _validate_path_batch(destination)
    _validate_path_batch(source, require_nonempty=False)
    if not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1 or env_ids.dtype != torch.int64:
        raise ValueError("env_ids must be an int64 tensor with shape (K,).")
    if env_ids.device != destination.points_world_m.device:
        raise ValueError("env_ids must be on the destination device.")
    if source.points_world_m.shape[0] != env_ids.shape[0]:
        raise ValueError("source batch length must match env_ids.")
    if bool(torch.any((env_ids < 0) | (env_ids >= destination.points_world_m.shape[0]))):
        raise IndexError("env_ids contains an out-of-range environment index.")
    if torch.unique(env_ids).numel() != env_ids.numel():
        raise ValueError("env_ids must contain unique environment indices.")
    if source.sample_spacing_m != destination.sample_spacing_m:
        raise ValueError("source and destination must share sample spacing.")
    if source.path_length_m != destination.path_length_m:
        raise ValueError("source and destination must share path length metadata.")

    destination_fields = _batched_path_field_names(destination)
    for name in destination_fields:
        destination_value = getattr(destination, name)
        source_value = getattr(source, name)
        if source_value.shape[1:] != destination_value.shape[1:]:
            raise ValueError(f"source path.{name} must match destination trailing shape.")
        if source_value.dtype != destination_value.dtype or source_value.device != destination_value.device:
            raise ValueError(f"source path.{name} must match destination dtype and device.")
    for name in destination_fields:
        getattr(destination, name).index_copy_(0, env_ids, getattr(source, name))


def _resolve_evaluation_overrides(
    *,
    count: int,
    config: PureRLSpatialStageConfig,
    device: torch.device,
    dtype: torch.dtype,
    template_id: Tensor | None,
    geometry_roll_deg: Tensor | None,
    slope_deg: Tensor | None,
    turn_sign: Tensor | None,
    heading_rad: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None:
    values = (template_id, geometry_roll_deg, slope_deg, turn_sign, heading_rad)
    configured = tuple(value is not None for value in values)
    if any(configured) and not all(configured):
        raise ValueError("Spatial evaluation overrides must be configured together.")
    if not any(configured):
        return None
    assert all(value is not None for value in values)
    resolved = tuple(value for value in values if value is not None)
    for name, value in zip(
        ("template_id", "geometry_roll_deg", "slope_deg", "turn_sign", "heading_rad"),
        resolved,
    ):
        if not isinstance(value, torch.Tensor) or value.shape != (count,):
            raise ValueError(f"evaluation_{name} must be a tensor with shape ({count},).")
        if value.device != device:
            raise ValueError(f"evaluation_{name} must be on the requested device.")
    resolved_template, resolved_roll, resolved_slope, resolved_turn, resolved_heading = resolved
    if resolved_template.dtype != torch.int64:
        raise ValueError("evaluation_template_id must use dtype int64.")
    for name, value in (
        ("geometry_roll_deg", resolved_roll),
        ("slope_deg", resolved_slope),
        ("turn_sign", resolved_turn),
        ("heading_rad", resolved_heading),
    ):
        if value.dtype != dtype:
            raise ValueError(f"evaluation_{name} must use the requested floating dtype.")
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"evaluation_{name} must contain finite values.")

    allowed_templates = {
        "c3a": (ISOLATED_TURN_TEMPLATE_ID,),
        "c3b": tuple(range(C3B_SAME_DIRECTION_TURNS_TEMPLATE_ID, C3B_LOITER_TEMPLATE_ID + 1)),
        "c3c": (C3C_COUPLED_TEMPLATE_ID,),
    }[config.stage_id]
    allowed = torch.zeros_like(resolved_template, dtype=torch.bool)
    for allowed_template in allowed_templates:
        allowed |= resolved_template == allowed_template
    if not bool(torch.all(allowed)):
        raise ValueError(f"evaluation_template_id contains a template not valid for {config.stage_id}.")
    if not bool(torch.all((resolved_turn == -1.0) | (resolved_turn == 1.0))):
        raise ValueError("evaluation_turn_sign values must be -1 or 1.")
    if not bool(
        torch.all(
            (resolved_roll >= config.geometry_roll_deg_range[0])
            & (resolved_roll <= config.geometry_roll_deg_range[1])
        )
    ):
        raise ValueError("evaluation_geometry_roll_deg is outside the active stage range.")
    if config.stage_id == "c3a" and not bool(torch.all(resolved_slope == 0.0)):
        raise ValueError("C3a evaluation paths require zero slope.")
    if config.stage_id == "c3b" and not bool(torch.all(torch.abs(resolved_slope) <= 8.0)):
        raise ValueError("C3b evaluation slopes must not exceed 8 degrees.")
    if config.stage_id == "c3c":
        slope_magnitude = torch.abs(resolved_slope)
        if not bool(
            torch.all(
                (slope_magnitude >= config.coupled_slope_deg_range[0])
                & (slope_magnitude <= config.coupled_slope_deg_range[1])
            )
        ):
            raise ValueError("C3c evaluation slope magnitude is outside the coupled stage range.")
        coupled_demand = torch.square(resolved_roll / 20.0) + torch.square(slope_magnitude / 6.0)
        if not bool(torch.all(coupled_demand <= 1.0)):
            raise ValueError("C3c evaluation severity violates the coupled demand bound.")
    return resolved_template, resolved_roll, resolved_slope, resolved_turn, resolved_heading


def _apply_evaluation_event_contracts(
    *,
    event_type: Tensor,
    event_count: Tensor,
    template_id: Tensor,
    event_turn_sign: Tensor,
    event_vertical_sign: Tensor,
    evaluation_template_id: Tensor,
    evaluation_slope_deg: Tensor,
    evaluation_turn_sign: Tensor,
    config: PureRLSpatialStageConfig,
) -> None:
    event_type.zero_()
    event_count.zero_()
    template_id.copy_(evaluation_template_id)
    event_turn_sign.copy_(evaluation_turn_sign.unsqueeze(1).expand_as(event_turn_sign))
    vertical_sign = torch.sign(evaluation_slope_deg)
    event_vertical_sign.copy_(vertical_sign.unsqueeze(1).expand_as(event_vertical_sign))
    if config.stage_id == "c3a":
        event_count.fill_(1)
        event_type[:, 0] = _EVENT_TURN
        return
    if config.stage_id == "c3c":
        event_count.fill_(2)
        event_type[:, 0:2] = _EVENT_COUPLED
        return

    event_count.fill_(2)
    rows = torch.ones_like(template_id, dtype=torch.bool)
    _apply_c3b_template_contracts(
        event_type=event_type,
        turn_sign=event_turn_sign,
        vertical_sign=event_vertical_sign,
        template_id=template_id,
        current_rows=rows,
    )
    active = torch.arange(_MAX_EVENTS, device=event_type.device).unsqueeze(0) < event_count.unsqueeze(1)
    event_type.copy_(torch.where(active, event_type, torch.zeros_like(event_type)))


def _sample_spatial_path_batch_once(
    *,
    num_paths: int,
    config: PureRLSpatialStageConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
    initial_altitude_m: Tensor,
    evaluation_overrides: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None,
) -> PureRLSpatialPathBatch:
    """Sample one unchecked batch with fixed-size batched Torch operations."""

    resolved_device = device
    altitude = initial_altitude_m

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
    if evaluation_overrides is not None:
        evaluation_template_id, _, evaluation_slope_deg, evaluation_turn_sign, _ = evaluation_overrides
        task_family_id.fill_(
            CURRENT_SPATIAL_TASK_FAMILY_ID_C3A
            if config.stage_id == "c3a"
            else CURRENT_SPATIAL_TASK_FAMILY_ID
        )
        _apply_evaluation_event_contracts(
            event_type=event_type,
            event_count=event_count,
            template_id=template_id,
            event_turn_sign=event_turn_sign,
            event_vertical_sign=event_vertical_sign,
            evaluation_template_id=evaluation_template_id,
            evaluation_slope_deg=evaluation_slope_deg,
            evaluation_turn_sign=evaluation_turn_sign,
            config=config,
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
    if evaluation_overrides is not None:
        _, evaluation_roll_deg, evaluation_slope_deg, _, _ = evaluation_overrides
        turn_mask = (event_type == _EVENT_TURN) | (event_type == _EVENT_COUPLED)
        vertical_mask = (event_type == _EVENT_VERTICAL) | (event_type == _EVENT_COUPLED)
        event_roll_rad = torch.deg2rad(evaluation_roll_deg).unsqueeze(1) * turn_mask
        event_slope_rad = torch.deg2rad(evaluation_slope_deg).unsqueeze(1) * vertical_mask
    event_start_m, transition_length_m, plateau_length_m, event_end_m = _sample_event_layout(
        event_type=event_type,
        event_count=event_count,
        template_id=template_id,
        event_roll_rad=event_roll_rad,
        config=config,
        device=resolved_device,
        dtype=dtype,
        generator=generator,
        deterministic=evaluation_overrides is not None,
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

    if evaluation_overrides is None:
        initial_heading_rad = 2.0 * math.pi * torch.rand(
            num_paths,
            device=resolved_device,
            dtype=dtype,
            generator=generator,
        )
    else:
        initial_heading_rad = evaluation_overrides[4]
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

    return PureRLSpatialPathBatch(
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
        sample_spacing_m=config.sample_spacing_m,
        path_length_m=config.path_length_m,
    )


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
    deterministic: bool = False,
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
    if deterministic:
        transition_length_m.fill_(sum(config.transition_length_m_range) / 2.0)
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
    if deterministic:
        heading_change_deg = 0.5 * (lower_by_row + upper_by_row).unsqueeze(1).expand_as(
            heading_change_deg
        )
    safe_curvature = torch.where(turn_mask, curvature_magnitude, torch.ones_like(curvature_magnitude))
    turn_plateau_length_m = torch.deg2rad(heading_change_deg) / safe_curvature - transition_length_m
    vertical_plateau_length_m = _sample_uniform(
        (count, _MAX_EVENTS),
        (20.0, 30.0),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if deterministic:
        vertical_plateau_length_m.fill_(25.0)
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
    if deterministic:
        gap_length_m.fill_(10.0)
    initial_entry_m = _sample_uniform(
        (count,),
        (15.0, 20.0),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if deterministic:
        initial_entry_m.fill_(17.5)
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
        ("minimum_altitude_m", config.minimum_altitude_m),
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
    _validate_path_batch(batch)
    if batch.sample_spacing_m != config.sample_spacing_m or batch.path_length_m != config.path_length_m:
        raise RuntimeError("Generated spatial path table metadata does not match its stage config.")
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
    altitude_invalid = torch.any(batch.points_world_m[:, :, 2] < config.minimum_altitude_m, dim=1)
    if bool(torch.any(altitude_invalid)):
        raise RuntimeError(
            f"Generated spatial path falls below the {config.minimum_altitude_m:.2f} m minimum altitude."
        )
    absolute_heading_change_rad = torch.trapezoid(
        torch.abs(batch.curvature_rad_per_m),
        dx=config.sample_spacing_m,
        dim=1,
    )
    non_loiter = batch.template_id != C3B_LOITER_TEMPLATE_ID
    if bool(torch.any((absolute_heading_change_rad >= 2.0 * math.pi) & non_loiter)):
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

    if bool(torch.any(_nonadjacent_clearance_invalid_rows(batch))):
        raise RuntimeError(
            "Generated nonadjacent centerline branches do not guarantee "
            f"{_MINIMUM_NONADJACENT_CLEARANCE_M:.1f} m clearance."
        )


def _resampleable_invalid_rows(
    batch: PureRLSpatialPathBatch,
    *,
    config: PureRLSpatialStageConfig,
) -> Tensor:
    altitude_invalid = torch.any(batch.points_world_m[:, :, 2] < config.minimum_altitude_m, dim=1)
    return altitude_invalid | _nonadjacent_clearance_invalid_rows(batch)


def _nonadjacent_clearance_invalid_rows(batch: PureRLSpatialPathBatch) -> Tensor:
    invalid_rows = torch.zeros_like(batch.template_id, dtype=torch.bool)
    checked_rows = batch.template_id != C3B_LOITER_TEMPLATE_ID
    if not bool(torch.any(checked_rows)):
        return invalid_rows
    checked_ids = torch.nonzero(checked_rows, as_tuple=False).flatten()
    sampled_points = batch.points_world_m[checked_rows, ::_INTERSECTION_VALIDATION_STRIDE, :]
    sample_count = sampled_points.shape[1]
    sample_index = torch.arange(sample_count, device=sampled_points.device)
    nonadjacent = (
        torch.abs(sample_index.unsqueeze(0) - sample_index.unsqueeze(1))
        > _INTERSECTION_LOCAL_SAMPLE_RADIUS
    )
    checked_invalid_chunks: list[Tensor] = []
    for points_chunk in torch.split(sampled_points, _CLEARANCE_VALIDATION_BATCH_SIZE, dim=0):
        pairwise_distance_m = torch.cdist(points_chunk, points_chunk)
        too_close = pairwise_distance_m < _CLEARANCE_VALIDATION_THRESHOLD_M
        checked_invalid_chunks.append(torch.any(too_close & nonadjacent.unsqueeze(0), dim=(1, 2)))
    invalid_rows.index_copy_(0, checked_ids, torch.cat(checked_invalid_chunks))
    return invalid_rows


def _replace_batch_rows(
    batch: PureRLSpatialPathBatch,
    *,
    row_ids: Tensor,
    replacement: PureRLSpatialPathBatch,
) -> PureRLSpatialPathBatch:
    values: dict[str, Tensor | float] = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            updated = value.clone()
            updated.index_copy_(0, row_ids, getattr(replacement, field.name))
            values[field.name] = updated
        else:
            if value != getattr(replacement, field.name):
                raise ValueError("Replacement spatial path metadata must match the destination batch.")
            values[field.name] = value
    return PureRLSpatialPathBatch(**values)


def _validate_path_batch(path: PureRLSpatialPathBatch, *, require_nonempty: bool = True) -> None:
    if not isinstance(path, PureRLSpatialPathBatch):
        raise TypeError("path must be a PureRLSpatialPathBatch.")
    points = path.points_world_m
    if (
        not isinstance(points, torch.Tensor)
        or points.ndim != 3
        or points.shape[2] != 3
        or points.shape[1] < 2
    ):
        raise ValueError("path.points_world_m must have shape (N, P, 3) with P >= 2.")
    if require_nonempty and points.shape[0] == 0:
        raise ValueError("path batch must be non-empty.")
    if points.dtype not in (torch.float32, torch.float64):
        raise TypeError("path floating-point dtype must be float32 or float64.")
    count, point_count, _ = points.shape
    for name in ("tangent_world", "lateral_normal_world", "vertical_normal_world"):
        _validate_path_tensor(path, name=name, shape=(count, point_count, 3), reference=points)
    for name in ("curvature_rad_per_m", "slope_rad", "turn_activity"):
        _validate_path_tensor(path, name=name, shape=(count, point_count), reference=points)
    for name in (
        "final_event_progress_m",
        "turn_sign",
        "vertical_sign",
        "peak_geometry_roll_rad",
        "peak_slope_rad",
    ):
        _validate_path_tensor(path, name=name, shape=(count,), reference=points)
    for name in ("task_family_id", "event_count", "template_id"):
        value = getattr(path, name)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (count,)
            or value.dtype != torch.int64
            or value.device != points.device
        ):
            raise ValueError(f"path.{name} must be int64 on the path device with shape ({count},).")
    if (
        isinstance(path.sample_spacing_m, bool)
        or not isinstance(path.sample_spacing_m, (int, float))
        or not math.isfinite(path.sample_spacing_m)
        or path.sample_spacing_m <= 0.0
    ):
        raise ValueError("path.sample_spacing_m must be finite and positive.")
    expected_length_m = (point_count - 1) * float(path.sample_spacing_m)
    if (
        isinstance(path.path_length_m, bool)
        or not isinstance(path.path_length_m, (int, float))
        or not math.isfinite(path.path_length_m)
        or not math.isclose(float(path.path_length_m), expected_length_m, rel_tol=0.0, abs_tol=1.0e-9)
    ):
        raise ValueError("path.path_length_m must match the represented table length.")


def _validate_path_tensor(
    path: PureRLSpatialPathBatch,
    *,
    name: str,
    shape: tuple[int, ...],
    reference: Tensor,
) -> None:
    value = getattr(path, name)
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or value.dtype != reference.dtype
        or value.device != reference.device
    ):
        raise ValueError(f"path.{name} must match path points shape, dtype, and device.")


def _validate_query_tensor(
    name: str,
    value: Tensor,
    *,
    expected_shape: tuple[int, ...],
    reference: Tensor,
) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}.")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64.")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must match path dtype and device.")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_preview_times(preview_times_s: tuple[float, ...]) -> None:
    if len(preview_times_s) != len(PURE_RL_PREVIEW_TIMES_S):
        raise ValueError("preview_times_s must contain exactly five entries.")
    converted = tuple(float(value) for value in preview_times_s)
    if any(not math.isfinite(value) or value <= 0.0 for value in converted):
        raise ValueError("preview_times_s entries must be finite and positive.")
    if any(later <= earlier for earlier, later in zip(converted, converted[1:])):
        raise ValueError("preview_times_s must be strictly increasing.")


def _interpolate_path_table(table: Tensor, progress_m: Tensor, *, path: PureRLSpatialPathBatch) -> Tensor:
    point_count = table.shape[1]
    scaled_progress = progress_m / path.sample_spacing_m
    lower_index = torch.floor(scaled_progress).to(dtype=torch.int64).clamp(min=0, max=point_count - 2)
    fraction = scaled_progress - lower_index.to(dtype=progress_m.dtype)
    row_shape = (table.shape[0],) + (1,) * (progress_m.ndim - 1)
    row_index = torch.arange(table.shape[0], device=table.device).reshape(row_shape).expand_as(lower_index)
    lower_value = table[row_index, lower_index]
    upper_value = table[row_index, lower_index + 1]
    if table.ndim == 3:
        fraction = fraction.unsqueeze(-1)
    return torch.lerp(lower_value, upper_value, fraction)


def _orthonormalize_route_frame(*, tangent_raw: Tensor, lateral_raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    lateral_normal_world = torch.nn.functional.normalize(lateral_raw, dim=1)
    tangent_orthogonal = tangent_raw - torch.sum(
        tangent_raw * lateral_normal_world,
        dim=1,
        keepdim=True,
    ) * lateral_normal_world
    tangent_world = torch.nn.functional.normalize(tangent_orthogonal, dim=1)
    vertical_normal_world = torch.cross(tangent_world, lateral_normal_world, dim=1)
    return tangent_world, lateral_normal_world, vertical_normal_world


def _batched_path_field_names(path: PureRLSpatialPathBatch) -> tuple[str, ...]:
    return tuple(field.name for field in fields(path) if isinstance(getattr(path, field.name), torch.Tensor))

"""Pure Tensor geometry contract for the PureRL longitudinal curriculum."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

Tensor = torch.Tensor

LEVEL_TASK_ID = 0
CLIMB_TASK_ID = 1
DESCENT_TASK_ID = 2


@dataclass(frozen=True)
class PureRLLongitudinalStageConfig:
    """Immutable sampling bounds for one longitudinal curriculum stage."""

    stage_id: str
    absolute_slope_deg_range: tuple[float, float]
    task_probabilities: tuple[float, float, float]
    entry_length_m_range: tuple[float, float] = (15.0, 20.0)
    slope_length_m_range: tuple[float, float] = (20.0, 30.0)
    minimum_recovery_length_m: float = 15.0


@dataclass(frozen=True)
class PureRLLongitudinalPathBatch:
    """Batched sampled path parameters in world z-up coordinates."""

    task_id: Tensor
    heading_rad: Tensor
    signed_slope_rad: Tensor
    entry_length_m: Tensor
    slope_length_m: Tensor
    initial_altitude_m: Tensor


@dataclass(frozen=True)
class PureRLLongitudinalPathQuery:
    """Current path-relative geometry and time-based world-frame preview."""

    horizontal_progress_m: Tensor
    reference_altitude_m: Tensor
    height_error_m: Tensor
    cross_track_error_m: Tensor
    active_slope_rad: Tensor
    tangent_world: Tensor
    lateral_normal_world: Tensor
    vertical_normal_world: Tensor
    preview_points_world_m: Tensor
    reached_recovery: Tensor


LONGITUDINAL_STAGE_CONFIGS: dict[str, PureRLLongitudinalStageConfig] = {
    "c2a": PureRLLongitudinalStageConfig(
        stage_id="c2a",
        absolute_slope_deg_range=(1.5, 4.0),
        task_probabilities=(0.50, 0.25, 0.25),
    ),
    "c2b": PureRLLongitudinalStageConfig(
        stage_id="c2b",
        absolute_slope_deg_range=(2.0, 6.0),
        task_probabilities=(0.30, 0.35, 0.35),
    ),
    "c2c": PureRLLongitudinalStageConfig(
        stage_id="c2c",
        absolute_slope_deg_range=(4.0, 12.0),
        task_probabilities=(0.25, 0.375, 0.375),
    ),
}

DEFAULT_PREVIEW_TIMES_S: tuple[float, ...] = (0.12, 0.24, 0.36, 0.48, 0.60)


def resolve_longitudinal_stage(
    stage: str | PureRLLongitudinalStageConfig,
) -> PureRLLongitudinalStageConfig:
    """Resolve a registered stage identifier or validate an immutable config."""

    if isinstance(stage, str):
        try:
            return LONGITUDINAL_STAGE_CONFIGS[stage]
        except KeyError as error:
            raise ValueError(f"Unknown longitudinal stage: {stage!r}.") from error
    if not isinstance(stage, PureRLLongitudinalStageConfig):
        raise TypeError("stage must be a stage identifier or PureRLLongitudinalStageConfig.")
    _validate_stage_config(stage)
    return stage


def sample_longitudinal_path_batch(
    *,
    num_paths: int,
    stage: str | PureRLLongitudinalStageConfig,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
    initial_altitude_m: float | Tensor = 10.0,
    climb_strong_slope_probability: float = 0.0,
    climb_strong_slope_minimum_deg: float = 10.0,
) -> PureRLLongitudinalPathBatch:
    """Sample independent path parameters with batched Torch operations."""

    count = int(num_paths)
    if count <= 0 or count != num_paths:
        raise ValueError("num_paths must be a positive integer.")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype.")
    resolved_device = torch.device(device)
    config = resolve_longitudinal_stage(stage)
    _validate_stage_config(config)
    strong_climb_probability = float(climb_strong_slope_probability)
    strong_climb_minimum_deg = float(climb_strong_slope_minimum_deg)
    minimum_slope_deg, maximum_slope_deg = config.absolute_slope_deg_range
    if not math.isfinite(strong_climb_probability) or not 0.0 <= strong_climb_probability <= 1.0:
        raise ValueError("climb_strong_slope_probability must be finite and lie in [0, 1].")
    if strong_climb_probability > 0.0 and (
        not math.isfinite(strong_climb_minimum_deg)
        or not minimum_slope_deg < strong_climb_minimum_deg < maximum_slope_deg
    ):
        raise ValueError(
            "climb_strong_slope_minimum_deg must lie strictly inside the stage slope range."
        )

    task_draw = torch.rand(count, device=resolved_device, dtype=dtype, generator=generator)
    level_threshold = config.task_probabilities[LEVEL_TASK_ID]
    climb_threshold = level_threshold + config.task_probabilities[CLIMB_TASK_ID]
    task_id = torch.where(
        task_draw < level_threshold,
        LEVEL_TASK_ID,
        torch.where(task_draw < climb_threshold, CLIMB_TASK_ID, DESCENT_TASK_ID),
    ).to(dtype=torch.int64)

    heading_rad = 2.0 * math.pi * torch.rand(
        count,
        device=resolved_device,
        dtype=dtype,
        generator=generator,
    )
    absolute_slope_deg = minimum_slope_deg + (maximum_slope_deg - minimum_slope_deg) * torch.rand(
        count, device=resolved_device, dtype=dtype, generator=generator
    )
    if strong_climb_probability > 0.0:
        strong_climb = (
            torch.rand(count, device=resolved_device, dtype=dtype, generator=generator)
            < strong_climb_probability
        )
        climb_slope_draw = torch.rand(
            count, device=resolved_device, dtype=dtype, generator=generator
        )
        stratified_climb_slope_deg = torch.where(
            strong_climb,
            strong_climb_minimum_deg
            + (maximum_slope_deg - strong_climb_minimum_deg) * climb_slope_draw,
            minimum_slope_deg
            + (strong_climb_minimum_deg - minimum_slope_deg) * climb_slope_draw,
        )
        absolute_slope_deg = torch.where(
            task_id == CLIMB_TASK_ID,
            stratified_climb_slope_deg,
            absolute_slope_deg,
        )
    absolute_slope_rad = torch.deg2rad(absolute_slope_deg)
    signed_slope_rad = torch.where(
        task_id == CLIMB_TASK_ID,
        absolute_slope_rad,
        torch.where(task_id == DESCENT_TASK_ID, -absolute_slope_rad, torch.zeros_like(absolute_slope_rad)),
    )
    entry_length_m = _sample_uniform_range(
        count=count,
        bounds=config.entry_length_m_range,
        device=resolved_device,
        dtype=dtype,
        generator=generator,
    )
    slope_length_m = _sample_uniform_range(
        count=count,
        bounds=config.slope_length_m_range,
        device=resolved_device,
        dtype=dtype,
        generator=generator,
    )
    initial_altitude = torch.as_tensor(initial_altitude_m, device=resolved_device, dtype=dtype)
    if initial_altitude.ndim == 0:
        initial_altitude = initial_altitude.expand(count).clone()
    elif initial_altitude.shape != (count,):
        raise ValueError("initial_altitude_m must be scalar or have shape (num_paths,).")

    return PureRLLongitudinalPathBatch(
        task_id=task_id,
        heading_rad=heading_rad,
        signed_slope_rad=signed_slope_rad,
        entry_length_m=entry_length_m,
        slope_length_m=slope_length_m,
        initial_altitude_m=initial_altitude,
    )


def query_longitudinal_path(
    *,
    path: PureRLLongitudinalPathBatch,
    position_world_m: Tensor,
    ground_velocity_world_mps: Tensor,
    minimum_preview_speed_mps: float = 1.0,
    maximum_preview_speed_mps: float = 12.0,
    preview_times_s: tuple[float, ...] = DEFAULT_PREVIEW_TIMES_S,
) -> PureRLLongitudinalPathQuery:
    """Query piecewise level/slope/level paths in a z-up world frame.

    Positions are relative to the environment origin. The path starts at
    horizontal world origin and ``initial_altitude_m`` and extends without a
    terminal horizontal clamp after its recovery transition.
    """

    _validate_path_batch(path)
    count = path.heading_rad.shape[0]
    _validate_float_matrix(
        "position_world_m",
        position_world_m,
        rows=count,
        columns=3,
        reference=path.heading_rad,
    )
    _validate_float_matrix(
        "ground_velocity_world_mps",
        ground_velocity_world_mps,
        rows=count,
        columns=3,
        reference=path.heading_rad,
    )
    minimum_speed = float(minimum_preview_speed_mps)
    maximum_speed = float(maximum_preview_speed_mps)
    if not math.isfinite(minimum_speed) or not math.isfinite(maximum_speed):
        raise ValueError("Preview-speed bounds must be finite.")
    if minimum_speed < 0.0 or maximum_speed < minimum_speed:
        raise ValueError("Preview-speed bounds must satisfy 0 <= minimum <= maximum.")
    if len(preview_times_s) != 5:
        raise ValueError("preview_times_s must contain exactly five entries.")
    if any((not math.isfinite(float(value))) or float(value) <= 0.0 for value in preview_times_s):
        raise ValueError("preview_times_s entries must be finite and positive.")
    if any(float(later) <= float(earlier) for earlier, later in zip(preview_times_s, preview_times_s[1:])):
        raise ValueError("preview_times_s must be strictly increasing.")

    cos_heading = torch.cos(path.heading_rad)
    sin_heading = torch.sin(path.heading_rad)
    zeros = torch.zeros_like(cos_heading)
    horizontal_tangent = torch.stack((cos_heading, sin_heading, zeros), dim=1)
    lateral_normal_world = torch.stack((-sin_heading, cos_heading, zeros), dim=1)
    horizontal_progress_m = torch.sum(position_world_m * horizontal_tangent, dim=1)
    cross_track_error_m = torch.sum(position_world_m * lateral_normal_world, dim=1)
    slope_end_progress_m = path.entry_length_m + path.slope_length_m
    on_slope = (horizontal_progress_m >= path.entry_length_m) & (
        horizontal_progress_m < slope_end_progress_m
    )
    active_slope_rad = torch.where(on_slope, path.signed_slope_rad, zeros)
    reference_altitude_m = _reference_altitude_m(path, horizontal_progress_m)
    height_error_m = position_world_m[:, 2] - reference_altitude_m

    cos_slope = torch.cos(active_slope_rad)
    sin_slope = torch.sin(active_slope_rad)
    tangent_world = torch.stack(
        (cos_slope * cos_heading, cos_slope * sin_heading, sin_slope),
        dim=1,
    )
    vertical_normal_world = torch.stack(
        (-sin_slope * cos_heading, -sin_slope * sin_heading, cos_slope),
        dim=1,
    )

    horizontal_speed_mps = torch.sum(ground_velocity_world_mps * horizontal_tangent, dim=1)
    preview_speed_mps = torch.clamp(horizontal_speed_mps, min=minimum_speed, max=maximum_speed)
    preview_times = path.heading_rad.new_tensor(preview_times_s)
    preview_progress_m = horizontal_progress_m.unsqueeze(1) + preview_speed_mps.unsqueeze(1) * preview_times
    preview_xy_world_m = preview_progress_m.unsqueeze(2) * horizontal_tangent[:, None, 0:2]
    preview_altitude_m = _reference_altitude_m(path, preview_progress_m)
    preview_points_world_m = torch.cat((preview_xy_world_m, preview_altitude_m.unsqueeze(2)), dim=2)

    return PureRLLongitudinalPathQuery(
        horizontal_progress_m=horizontal_progress_m,
        reference_altitude_m=reference_altitude_m,
        height_error_m=height_error_m,
        cross_track_error_m=cross_track_error_m,
        active_slope_rad=active_slope_rad,
        tangent_world=tangent_world,
        lateral_normal_world=lateral_normal_world,
        vertical_normal_world=vertical_normal_world,
        preview_points_world_m=preview_points_world_m,
        reached_recovery=horizontal_progress_m >= slope_end_progress_m,
    )


def write_longitudinal_path_batch_rows_(
    *,
    destination: PureRLLongitudinalPathBatch,
    env_ids: Tensor,
    source: PureRLLongitudinalPathBatch,
) -> None:
    """Copy a sampled path batch into selected destination environments in place."""

    _validate_path_batch(destination)
    _validate_path_batch(source)
    if not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1 or env_ids.dtype != torch.int64:
        raise ValueError("env_ids must be an int64 tensor with shape (K,).")
    if env_ids.device != destination.heading_rad.device:
        raise ValueError("env_ids must be on the destination device.")
    if source.heading_rad.shape[0] != env_ids.shape[0]:
        raise ValueError("source batch length must match env_ids.")
    if source.heading_rad.dtype != destination.heading_rad.dtype or source.heading_rad.device != destination.heading_rad.device:
        raise ValueError("source and destination must share floating-point dtype and device.")
    if bool(torch.any((env_ids < 0) | (env_ids >= destination.heading_rad.shape[0]))):
        raise IndexError("env_ids contains an out-of-range environment index.")

    destination.task_id.index_copy_(0, env_ids, source.task_id)
    destination.heading_rad.index_copy_(0, env_ids, source.heading_rad)
    destination.signed_slope_rad.index_copy_(0, env_ids, source.signed_slope_rad)
    destination.entry_length_m.index_copy_(0, env_ids, source.entry_length_m)
    destination.slope_length_m.index_copy_(0, env_ids, source.slope_length_m)
    destination.initial_altitude_m.index_copy_(0, env_ids, source.initial_altitude_m)


def _reference_altitude_m(path: PureRLLongitudinalPathBatch, horizontal_progress_m: Tensor) -> Tensor:
    """Evaluate the path altitude for ``(N,)`` or ``(N, K)`` progress."""

    trailing_dims = horizontal_progress_m.ndim - 1
    entry_length_m = path.entry_length_m.reshape((-1,) + (1,) * trailing_dims)
    slope_length_m = path.slope_length_m.reshape((-1,) + (1,) * trailing_dims)
    initial_altitude_m = path.initial_altitude_m.reshape((-1,) + (1,) * trailing_dims)
    signed_slope_rad = path.signed_slope_rad.reshape((-1,) + (1,) * trailing_dims)
    slope_progress_m = torch.clamp(horizontal_progress_m - entry_length_m, min=0.0)
    slope_progress_m = torch.minimum(slope_progress_m, slope_length_m)
    return initial_altitude_m + slope_progress_m * torch.tan(signed_slope_rad)


def _sample_uniform_range(
    *,
    count: int,
    bounds: tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    lower, upper = bounds
    return lower + (upper - lower) * torch.rand(
        count,
        device=device,
        dtype=dtype,
        generator=generator,
    )


def _validate_stage_config(config: PureRLLongitudinalStageConfig) -> None:
    if not config.stage_id:
        raise ValueError("stage_id must be non-empty.")
    for name, bounds in (
        ("absolute_slope_deg_range", config.absolute_slope_deg_range),
        ("entry_length_m_range", config.entry_length_m_range),
        ("slope_length_m_range", config.slope_length_m_range),
    ):
        if len(bounds) != 2:
            raise ValueError(f"{name} must contain exactly two bounds.")
        lower, upper = bounds
        if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0.0 or upper < lower:
            raise ValueError(f"{name} must contain finite positive ordered bounds.")
    probabilities = config.task_probabilities
    if len(probabilities) != 3 or any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("task_probabilities must contain three finite non-negative values.")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("task_probabilities must sum to one.")
    if not math.isfinite(config.minimum_recovery_length_m) or config.minimum_recovery_length_m < 15.0:
        raise ValueError("minimum_recovery_length_m must be finite and at least 15 m.")


def _validate_path_batch(path: PureRLLongitudinalPathBatch) -> None:
    if not isinstance(path, PureRLLongitudinalPathBatch):
        raise TypeError("path must be a PureRLLongitudinalPathBatch.")
    reference = path.heading_rad
    if not isinstance(reference, torch.Tensor) or reference.ndim != 1 or not torch.is_floating_point(reference):
        raise ValueError("path.heading_rad must be a floating-point tensor with shape (N,).")
    if reference.numel() == 0:
        raise ValueError("path batch must be non-empty.")
    if not isinstance(path.task_id, torch.Tensor) or path.task_id.shape != reference.shape:
        raise ValueError("path.task_id must have shape (N,).")
    if path.task_id.dtype != torch.int64 or path.task_id.device != reference.device:
        raise ValueError("path.task_id must be int64 on the path device.")
    for name in (
        "signed_slope_rad",
        "entry_length_m",
        "slope_length_m",
        "initial_altitude_m",
    ):
        value = getattr(path, name)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
        ):
            raise ValueError(f"path.{name} must match path.heading_rad shape, dtype, and device.")


def _validate_float_matrix(
    name: str,
    value: Tensor,
    *,
    rows: int,
    columns: int,
    reference: Tensor,
) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != (rows, columns):
        raise ValueError(f"{name} must have shape ({rows}, {columns}).")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point.")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must match path dtype and device.")

"""Versioned continuous 3-D paths, with sampled kinematic screening only.

Coordinates are episode-local z-up meters. ``s`` is three-dimensional arc
length; ``heading_rate_rad_per_m`` means d(heading)/ds, not horizontal
projected curvature. No aerodynamic or actuator feasibility is claimed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class ContinuousCurveContract:
    """Frozen engineering screening bounds; not an identified flight envelope."""

    version: str = "pure_rl_continuous_curve_v1"
    sample_spacing_m: float = 0.25
    path_length_m: float = 300.0
    completion_progress_m: float = 140.0
    episode_duration_s: float = 20.0
    entry_length_m: float = 20.0
    minimum_design_speed_mps: float = 7.0
    maximum_design_speed_mps: float = 12.0
    minimum_reference_altitude_m: float = 4.0
    maximum_reference_altitude_m: float = 50.0
    maximum_geometry_roll_rad: float = math.radians(14.0)
    maximum_slope_rad: float = math.radians(8.0)
    maximum_joint_demand: float = 0.9
    maximum_heading_rate_derivative_rad_per_m2: float = 0.002
    maximum_slope_derivative_rad_per_m: float = 0.017
    maximum_geometry_roll_rate_rad_s: float = 0.35
    maximum_normal_acceleration_mps2: float = 4.0
    minimum_nonlocal_sample_distance_m: float = 4.0
    nonlocal_exclusion_arc_m: float = 8.0
    clearance_sample_spacing_m: float = 1.0
    turn_activity_scale_rad_per_m: float = 0.003


CONTINUOUS_CURVE_CONTRACT = ContinuousCurveContract()


@dataclass(frozen=True)
class ContinuousCurveSamplingConfig:
    """Candidate ranges; accepted distribution is conditioned on the screen."""

    heading_amplitude_range: tuple[float, float] = (0.005, 0.009)
    slope_amplitude_deg_range: tuple[float, float] = (2.0, 4.0)
    wavelength_m_range: tuple[float, float] = (80.0, 120.0)
    harmonic_fraction_range: tuple[float, float] = (0.0, 0.25)
    chirp_range: tuple[float, float] = (0.0, 0.05)
    max_attempts: int = 32

    def validate(self) -> None:
        """Reject malformed ranges before drawing random numbers."""
        bounds = ((self.heading_amplitude_range, 0.0, math.inf),
                  (self.slope_amplitude_deg_range, 0.0, 8.0),
                  (self.wavelength_m_range, 60.0, 120.0),
                  (self.harmonic_fraction_range, 0.0, 0.35),
                  (self.chirp_range, 0.0, 0.15))
        for interval, lower, upper in bounds:
            if (len(interval) != 2 or not all(math.isfinite(x) for x in interval)
                    or not lower <= interval[0] <= interval[1] <= upper):
                raise ValueError(f"Invalid continuous-curve sampling range: {interval}")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")


class ContinuousCurveScreenError(ValueError):
    """A valid recipe failed sampled screening; only these rows may be retried."""

    def __init__(self, rejected_indices: tuple[int, ...], curve_ids: list[str]) -> None:
        self.rejected_indices = rejected_indices
        super().__init__(f"Continuous curve kinematic screen rejected: {curve_ids}")


@dataclass(frozen=True)
class ContinuousCurveSpec:
    """Serializable curve recipe; angles in radians and distances in meters."""

    curve_id: str = "coupled"
    heading_rate_bias_rad_per_m: float = 0.0
    heading_rate_amplitude_rad_per_m: float = 0.009
    slope_bias_rad: float = 0.0
    slope_amplitude_rad: float = math.radians(4.0)
    heading_wavelength_m: float = 90.0
    slope_wavelength_m: float = 90.0
    heading_phase_rad: float = 0.0
    slope_phase_rad: float = 0.0
    second_harmonic_fraction: float = 0.15
    chirp: float = 0.0
    horizontal_mirror: int = 1
    vertical_mirror: int = 1
    initial_heading_rad: float = 0.0
    initial_altitude_m: float = 10.0
    initial_flap_phase_rad: float = 0.0


@dataclass(frozen=True)
class ContinuousCurveBatch:
    """Dense tensors with shape (N, P[, 3]), on the requested device/dtype."""

    points_world_m: torch.Tensor
    tangent_world: torch.Tensor
    lateral_normal_world: torch.Tensor
    vertical_normal_world: torch.Tensor
    heading_rate_rad_per_m: torch.Tensor
    slope_rad: torch.Tensor
    turn_activity: torch.Tensor
    heading_rate_derivative_rad_per_m2: torch.Tensor
    slope_derivative_rad_per_m: torch.Tensor


def _validate_specs(specs: Sequence[ContinuousCurveSpec]) -> None:
    if not specs:
        raise ValueError("At least one continuous curve is required.")
    for spec in specs:
        if not isinstance(spec, ContinuousCurveSpec):
            raise TypeError("Expected ContinuousCurveSpec entries.")
        if not isinstance(spec.curve_id, str) or not spec.curve_id.strip():
            raise ValueError("curve_id must be nonempty.")
        for field in fields(spec):
            if field.name == "curve_id":
                continue
            value = getattr(spec, field.name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite numeric data.")
        if spec.horizontal_mirror not in (-1, 1) or spec.vertical_mirror not in (-1, 1):
            raise ValueError("Mirrors must be -1 or +1.")
        if not 60.0 <= spec.heading_wavelength_m <= 120.0 or not 60.0 <= spec.slope_wavelength_m <= 120.0:
            raise ValueError("Base wavelengths must lie in [60, 120] m.")
        if not 0.0 <= spec.second_harmonic_fraction <= 0.35 or not 0.0 <= spec.chirp <= 0.15:
            raise ValueError("Harmonic fraction or chirp is outside the v1 domain.")


def build_continuous_curves(
    specs: Sequence[ContinuousCurveSpec], *, device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ContinuousCurveBatch:
    """Integrate unit tangents with the trapezoidal rule, then screen every row.

    Recipes are host metadata; all geometry operations are batched Torch.
    Rejected recipes raise instead of clipping or silently replacing a path.
    """

    _validate_specs(specs)
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("Continuous curves require float32 or float64.")
    c = CONTINUOUS_CURVE_CONTRACT
    s = torch.arange(round(c.path_length_m / c.sample_spacing_m) + 1, device=device, dtype=dtype)[None, :]
    s = s * c.sample_spacing_m

    def column(name: str) -> torch.Tensor:
        return torch.tensor([getattr(spec, name) for spec in specs], device=device, dtype=dtype)[:, None]

    u = (s / c.entry_length_m).clamp(0.0, 1.0)
    envelope = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
    envelope_derivative = 30.0 * u**2 * (1.0 - u)**2 / c.entry_length_m

    def profile(prefix: str, bias: str, amplitude: str, mirror: str) -> tuple[torch.Tensor, torch.Tensor]:
        wavelength = column(prefix + "_wavelength_m")
        phase = column(prefix + "_phase_rad")
        cycles = s / wavelength
        angle = 2.0 * math.pi * (cycles + 0.5 * column("chirp") * cycles**2) + phase
        angle_derivative = 2.0 * math.pi / wavelength * (1.0 + column("chirp") * cycles)
        harmonic = column("second_harmonic_fraction")
        wave = torch.sin(angle) + harmonic * torch.sin(2.0 * angle)
        wave_derivative = (torch.cos(angle) + 2.0 * harmonic * torch.cos(2.0 * angle)) * angle_derivative
        raw = column(bias) + column(amplitude) * wave
        value = column(mirror) * envelope * raw
        derivative = column(mirror) * (envelope_derivative * raw + envelope * column(amplitude) * wave_derivative)
        return value, derivative

    k, dk = profile("heading", "heading_rate_bias_rad_per_m", "heading_rate_amplitude_rad_per_m", "horizontal_mirror")
    gamma, dgamma = profile("slope", "slope_bias_rad", "slope_amplitude_rad", "vertical_mirror")
    heading = column("initial_heading_rad") + torch.cat(
        (torch.zeros_like(k[:, :1]), torch.cumsum(0.5 * (k[:, :-1] + k[:, 1:]) * c.sample_spacing_m, dim=1)), dim=1
    )
    ch, sh, cg, sg = torch.cos(heading), torch.sin(heading), torch.cos(gamma), torch.sin(gamma)
    tangent = torch.stack((cg * ch, cg * sh, sg), dim=-1)
    lateral = torch.stack((-sh, ch, torch.zeros_like(ch)), dim=-1)
    vertical = torch.stack((-sg * ch, -sg * sh, cg), dim=-1)
    delta = 0.5 * (tangent[:, :-1] + tangent[:, 1:]) * c.sample_spacing_m
    points = torch.cat((torch.zeros_like(tangent[:, :1]), torch.cumsum(delta, dim=1)), dim=1)
    points[:, :, 2] += column("initial_altitude_m")
    # Smooth through curvature zero, with no division by a path-specific peak.
    activity = k.square() / (k.square() + c.turn_activity_scale_rad_per_m**2)
    batch = ContinuousCurveBatch(points, tangent, lateral, vertical, k, gamma, activity, dk, dgamma)
    reports = screen_continuous_curves(batch)
    invalid = [spec.curve_id for spec, report in zip(specs, reports, strict=True) if not report["screen_passed"]]
    if invalid:
        raise ContinuousCurveScreenError(
            tuple(i for i, report in enumerate(reports) if not report["screen_passed"]), invalid)
    return batch


def screen_continuous_curves(batch: ContinuousCurveBatch) -> list[dict[str, float | bool | str]]:
    """Report sampled geometric and constant-design-speed kinematic checks.

    Checks cover the entire 300 m table, including the entry and post-completion
    continuation. The speed band is a planning assumption, not a speed governor.
    """

    c = CONTINUOUS_CURVE_CONTRACT
    k, gamma = batch.heading_rate_rad_per_m, batch.slope_rad
    dk, dgamma = batch.heading_rate_derivative_rad_per_m2, batch.slope_derivative_rad_per_m
    speed_squared = c.maximum_design_speed_mps**2
    roll_proxy = torch.atan(speed_squared * k / 9.81)
    roll_rate_proxy = c.maximum_design_speed_mps * speed_squared / 9.81 * dk / (1.0 + (speed_squared * k / 9.81)**2)
    demand = (roll_proxy / math.radians(20.0)).square() + (gamma / math.radians(10.0)).square()
    acceleration = speed_squared * torch.sqrt((torch.cos(gamma) * k).square() + dgamma.square())
    stride = round(c.clearance_sample_spacing_m / c.sample_spacing_m)
    sampled = batch.points_world_m[:, ::stride]
    index = torch.arange(sampled.shape[1], device=sampled.device)
    nonlocal_mask = (index[:, None] - index[None, :]).abs() * c.clearance_sample_spacing_m > c.nonlocal_exclusion_arc_m
    clearances = []
    for chunk in sampled.split(16):
        distances = torch.cdist(chunk, chunk)
        clearances.append(distances.masked_fill(~nonlocal_mask, torch.inf).amin(dim=(1, 2)))
    metrics = {
        "minimum_altitude_m": batch.points_world_m[:, :, 2].amin(dim=1),
        "maximum_altitude_m": batch.points_world_m[:, :, 2].amax(dim=1),
        "maximum_geometry_roll_rad": roll_proxy.abs().amax(dim=1),
        "maximum_slope_rad": gamma.abs().amax(dim=1),
        "maximum_joint_demand": demand.amax(dim=1),
        "maximum_heading_rate_derivative_rad_per_m2": dk.abs().amax(dim=1),
        "maximum_slope_derivative_rad_per_m": dgamma.abs().amax(dim=1),
        "maximum_geometry_roll_rate_rad_s": roll_rate_proxy.abs().amax(dim=1),
        "maximum_normal_acceleration_mps2": acceleration.amax(dim=1),
        "minimum_nonlocal_sample_distance_m": torch.cat(clearances),
    }
    finite = torch.stack([torch.isfinite(getattr(batch, f.name)).flatten(1).all(dim=1) for f in fields(batch)]).all(dim=0)
    passed = finite.clone()
    for name, value in metrics.items():
        bound_name = {
            "minimum_altitude_m": "minimum_reference_altitude_m",
            "maximum_altitude_m": "maximum_reference_altitude_m",
        }.get(name, name)
        bound = getattr(c, bound_name)
        passed &= value >= bound if name.startswith("minimum") else value <= bound
    return [
        {"screen_passed": bool(passed[i]), "qualification": "sampled_kinematic_screen_only",
         **{name: float(value[i]) for name, value in metrics.items()}}
        for i in range(k.shape[0])
    ]


def continuous_curve_diagnostic_grid() -> tuple[ContinuousCurveSpec, ...]:
    """Return the frozen 96-case diagnostic grid; not a final generalization test."""

    profiles = (
        ContinuousCurveSpec(curve_id="coupled"),
        ContinuousCurveSpec(curve_id="quadrature", heading_rate_amplitude_rad_per_m=0.011,
                            slope_amplitude_rad=math.radians(5.0), slope_phase_rad=math.pi / 2.0),
        ContinuousCurveSpec(curve_id="biased", heading_rate_bias_rad_per_m=0.007,
                            heading_rate_amplitude_rad_per_m=0.002, slope_amplitude_rad=math.radians(3.0),
                            heading_wavelength_m=110.0, slope_wavelength_m=110.0),
    )
    return tuple(
        replace(profile, curve_id=f"{profile.curve_id}_h{h:+d}_v{v:+d}_yaw{yaw}_phase{phase}",
                horizontal_mirror=h, vertical_mirror=v, initial_heading_rad=yaw * math.pi / 2.0,
                initial_flap_phase_rad=phase * math.pi)
        for profile in profiles for h in (-1, 1) for v in (-1, 1) for yaw in range(4) for phase in range(2)
    )


def sample_continuous_curve_specs(
    num_paths: int, *, generator: torch.Generator | None = None, device: torch.device | str = "cpu",
    initial_altitude_m: float | torch.Tensor = 10.0,
    config: ContinuousCurveSamplingConfig = ContinuousCurveSamplingConfig(),
) -> tuple[ContinuousCurveSpec, ...]:
    """Sample bounded recipes, preserving independent mirrors on single resets.

    This samples candidates, not accepted training probabilities. The builder
    fails closed on a rejected candidate; no hidden family resampling occurs.
    """

    if isinstance(num_paths, bool) or not isinstance(num_paths, int) or num_paths <= 0:
        raise ValueError("num_paths must be positive.")
    config.validate()
    random = torch.rand((num_paths, 11), generator=generator, device=device).cpu().tolist()
    altitude = torch.as_tensor(initial_altitude_m, device=device).expand(num_paths).cpu().tolist()
    def draw(interval: tuple[float, float], unit: float) -> float:
        return interval[0] + (interval[1] - interval[0]) * unit
    return tuple(
        ContinuousCurveSpec(
            curve_id=f"sample_{i}", heading_rate_amplitude_rad_per_m=draw(config.heading_amplitude_range, row[0]),
            slope_amplitude_rad=math.radians(draw(config.slope_amplitude_deg_range, row[1])),
            heading_wavelength_m=draw(config.wavelength_m_range, row[2]),
            slope_wavelength_m=draw(config.wavelength_m_range, row[3]),
            heading_phase_rad=2.0 * math.pi * row[4], slope_phase_rad=2.0 * math.pi * row[5],
            horizontal_mirror=1 if row[6] >= 0.5 else -1,
            vertical_mirror=1 if row[7] >= 0.5 else -1,
            initial_heading_rad=2.0 * math.pi * row[8], initial_altitude_m=altitude[i],
            second_harmonic_fraction=draw(config.harmonic_fraction_range, row[9]),
            chirp=draw(config.chirp_range, row[10]),
        ) for i, row in enumerate(random)
    )


def sample_accepted_continuous_curve_specs(
    num_paths: int, *, generator: torch.Generator | None = None,
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
    initial_altitude_m: float | torch.Tensor = 10.0,
    config: ContinuousCurveSamplingConfig = ContinuousCurveSamplingConfig(),
    diagnostics: dict[str, int] | None = None,
) -> tuple[ContinuousCurveSpec, ...]:
    """Resample only rejected rows, with a bounded attempt count per row.

    Same seed and reset call sequence reproduce recipes. No fallback curve or
    clipping hides an exhausted sampling budget. This certifies only the screen.
    """
    config.validate()
    def record(key: str, count: int) -> None:
        if diagnostics is not None:
            diagnostics[key] = diagnostics.get(key, 0) + count
    specs = list(sample_continuous_curve_specs(num_paths, generator=generator, device=device,
                                              initial_altitude_m=initial_altitude_m, config=config))
    record('candidate_count', num_paths)
    pending = list(range(num_paths))
    altitude = torch.as_tensor(initial_altitude_m, device=device).expand(num_paths)
    for attempt in range(config.max_attempts):
        try:
            build_continuous_curves(tuple(specs[i] for i in pending), device=device, dtype=dtype)
            record('accepted_count', num_paths)
            return tuple(specs)
        except ContinuousCurveScreenError as error:
            pending = [pending[i] for i in error.rejected_indices]
            record('rejected_count', len(pending))
        if attempt + 1 < config.max_attempts:
            replacements = sample_continuous_curve_specs(
                len(pending), generator=generator, device=device, initial_altitude_m=altitude[pending], config=config)
            record('candidate_count', len(pending))
            for i, recipe in zip(pending, replacements, strict=True):
                specs[i] = replace(recipe, curve_id=f"sample_{i}")
    record('exhausted_count', len(pending))
    raise RuntimeError(f"Continuous curve sampling exhausted {config.max_attempts} attempts for rows {pending}")


def save_continuous_curves(
    path: str | Path, specs: Sequence[ContinuousCurveSpec], *, provenance: Mapping[str, object] | None = None,
) -> None:
    """Save validated recipes and float64 screening reports without overwriting.

    Recipes and generator version are authoritative; dense tables are rebuilt
    on the consumer's explicit device/dtype instead of serializing Torch objects.
    """

    _validate_specs(specs)
    if len({s.curve_id for s in specs}) != len(specs):
        raise ValueError("Continuous curve IDs must be unique within a file.")
    batch = build_continuous_curves(specs, dtype=torch.float64)
    document = {"contract": asdict(CONTINUOUS_CURVE_CONTRACT), "curves": [asdict(s) for s in specs],
                "screening": screen_continuous_curves(batch)}
    if provenance is not None:
        document["provenance"] = dict(provenance)
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_continuous_curves(path: str | Path) -> tuple[ContinuousCurveSpec, ...]:
    """Load only the exact v1 recipe schema and recompute its screening checks."""

    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or document.get("contract") != asdict(CONTINUOUS_CURVE_CONTRACT):
        raise ValueError("Continuous curve file has an incompatible contract.")
    rows = document.get("curves")
    names = {f.name for f in fields(ContinuousCurveSpec)}
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or set(row) != names for row in rows):
        raise ValueError("Continuous curve file must contain complete recipe records.")
    specs = tuple(ContinuousCurveSpec(**row) for row in rows)
    if len({s.curve_id for s in specs}) != len(specs):
        raise ValueError("Continuous curve IDs must be unique within a file.")
    build_continuous_curves(specs, dtype=torch.float64)
    return specs

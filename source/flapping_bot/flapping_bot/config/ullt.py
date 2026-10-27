"""Explicit options for the standalone Izraelevitz (2017) backend."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ULLTConfig:
    """Analytical priors and numerical guards; no environment defaults change."""

    model: str = "one_state"
    air_density_kg_m3: float = 1.225
    lift_slope_per_rad: float = 2.0 * math.pi
    profile_drag_coefficient: float = 0.0
    include_added_mass: bool = True
    added_mass_factor: float = 1.0
    minimum_speed_mps: float = 1.0e-6
    singularity_distance_m: float = 1.0e-9
    geometry_tolerance: float = 1.0e-5
    correction_fit_end: float = 15.0
    correction_fit_samples: int = 301
    maximum_pole_step: float = 0.5
    maximum_substeps: int = 256

    def __post_init__(self) -> None:
        if self.model not in {"quasi_steady", "one_state", "four_state", "six_state"}:
            raise ValueError(f"Unsupported ULLT model: {self.model}")
        for name in (
            "air_density_kg_m3",
            "lift_slope_per_rad",
            "added_mass_factor",
            "minimum_speed_mps",
            "singularity_distance_m",
            "geometry_tolerance",
            "correction_fit_end",
            "maximum_pole_step",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.profile_drag_coefficient) or self.profile_drag_coefficient < 0:
            raise ValueError("profile_drag_coefficient must be finite and nonnegative")
        if (
            not isinstance(self.correction_fit_samples, int)
            or isinstance(self.correction_fit_samples, bool)
            or not isinstance(self.maximum_substeps, int)
            or isinstance(self.maximum_substeps, bool)
            or self.correction_fit_samples < 3
            or self.maximum_substeps < 1
        ):
            raise ValueError("Invalid fit sample count or substep limit")

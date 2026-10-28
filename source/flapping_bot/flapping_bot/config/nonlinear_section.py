"""Explicit, uncalibrated flat-plate closure hypotheses for ULLT version A."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FlatPlatePolarConfig:
    """Blend limits (degrees) and separated sin(2 alpha) lift amplitude.

    These are physical hypotheses, not measured stall angles or fitted polars.
    The separated normal-force scale is twice ``separated_lift_amplitude``.
    """

    transition_start_deg: float = 15.0
    transition_end_deg: float = 30.0
    separated_lift_amplitude: float = 1.0

    def __post_init__(self) -> None:
        values = (self.transition_start_deg, self.transition_end_deg, self.separated_lift_amplitude)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Flat-plate polar parameters must be finite")
        if not 0 < self.transition_start_deg < self.transition_end_deg < 90:
            raise ValueError("Require 0 < transition start < transition end < 90 degrees")
        if self.separated_lift_amplitude <= 0:
            raise ValueError("Separated lift amplitude must be positive")


def flat_plate_prior(name: str) -> FlatPlatePolarConfig:
    """Return fixed pre-experiment hypotheses; neither is a calibrated airfoil."""
    if name == "early":
        return FlatPlatePolarConfig(15, 30, 1.0)
    if name == "late":
        return FlatPlatePolarConfig(20, 40, 1.2)
    raise ValueError(f"Unknown flat-plate prior: {name}")

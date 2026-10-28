"""Smooth static flat-plate polar and analytic slope for nonlinear lifting line.

Separated shape: Wang, Goosen & van Keulen (2016), Eqs. 2.6, 2.9--2.10,
DOI 10.1017/jfm.2016.413. Only its angular shape is borrowed. Its whole-wing
aspect-ratio coefficient is NOT applied again inside the lifting-line model.
Blending thresholds and amplitudes are explicitly uncalibrated hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from flapping_bot.config.nonlinear_section import FlatPlatePolarConfig


@dataclass(frozen=True)
class FlatPlatePolar:
    """Dimensionless coefficients, derivative per radian, all input-shaped tensors."""

    cl: torch.Tensor
    cd: torch.Tensor
    dcl_dalpha: torch.Tensor
    separation_weight: torch.Tensor


def flat_plate_polar(
    alpha_rad: torch.Tensor,
    config: FlatPlatePolarConfig,
    lift_slope_per_rad: float = 2 * math.pi,
    baseline_cd: float = 0.0,
) -> FlatPlatePolar:
    """Evaluate a symmetric C2 blend without fitting, NACA data or dynamic stall.

    Attached Cl=Cla*sin(alpha) exactly represents the existing normalwash law,
    with derivative Cla at zero. Separated Cl=A*sin(2 alpha), Cd=Cd0+2A*sin²(alpha).
    Quintic smoothstep blends coefficients over |alpha| in [start,end]. Values
    stay bounded; static coefficient bounds do not bound all transient loads.
    """
    if not math.isfinite(lift_slope_per_rad) or lift_slope_per_rad <= 0:
        raise ValueError("Positive finite lift slope required")
    if not math.isfinite(baseline_cd) or baseline_cd < 0:
        raise ValueError("Nonnegative finite baseline drag required")
    start, end = math.radians(config.transition_start_deg), math.radians(config.transition_end_deg)
    z = ((alpha_rad.abs() - start) / (end - start)).clamp(0, 1)
    blend = z**3 * (10 + z * (-15 + 6 * z))
    db = 30 * z.square() * (1 - z).square() * alpha_rad.sign() / (end - start)
    attached = lift_slope_per_rad * alpha_rad.sin()
    separated = config.separated_lift_amplitude * (2 * alpha_rad).sin()
    cl = attached + blend * (separated - attached)
    derivative = (
        lift_slope_per_rad * alpha_rad.cos()
        + db * (separated - attached)
        + blend * (2 * config.separated_lift_amplitude * (2 * alpha_rad).cos() - lift_slope_per_rad * alpha_rad.cos())
    )
    cd = baseline_cd + blend * (2 * config.separated_lift_amplitude * alpha_rad.sin().square())
    return FlatPlatePolar(cl, cd, derivative, blend)

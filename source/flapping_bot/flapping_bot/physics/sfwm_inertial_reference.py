"""Amini SFWM longitudinal inertial reference for the measured flapping bot.

This module implements only Eqs. (16)--(18) from Amini et al. (2020) as an
offline validation reference. It does not apply forces or moments to PhysX.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .measured_mass_properties import (
    BODY_LINK_MASS_PROPERTIES,
    LEFT_WING_LINK_MASS_PROPERTIES,
)
from .multibody_mass_distribution import project_diagonal_inertia_to_triangle_cone

# Amini gamma increases toward upstroke in body FRD. The URDF neutral wing
# center lies at body-FLU z<0 (FRD z>0), hence its paper-frame mean is negative.
MEASURED_WING_NEUTRAL_DIHEDRAL_RAD = -0.019391


@dataclass(frozen=True)
class AminiSfwmLongitudinalParams:
    """Parameters for the Amini SFWM longitudinal inertial terms.

    All positions use the paper body FRD frame. ``point_o`` is the
    body-attached point coincident with the three-body system COM at
    ``gamma_mean_rad``.
    """

    total_mass_kg: float
    body_mass_kg: float
    wing_mass_each_kg: float
    wing_cg_from_pivot_x_m: float
    wing_cg_from_pivot_y_m: float
    wing_pivot_from_o_x_m: float
    wing_pivot_from_o_z_m: float
    body_cg_from_o_z_m: float
    body_inertia_yy_kg_m2: float
    wing_inertia_yy_kg_m2: float
    wing_inertia_zz_kg_m2: float
    gamma_mean_rad: float
    point_o_in_base_flu_m: tuple[float, float, float]

    @property
    def wing_to_weight_ratio(self) -> float:
        """Return Amini's dimensionless ``rho = 2*m_wing/M``."""

        return 2.0 * self.wing_mass_each_kg / self.total_mass_kg

    @property
    def constant_pitch_inertia_kg_m2(self) -> float:
        """Return the constant SFWM pitch inertia from Amini Eq. (18)."""

        rho = self.wing_to_weight_ratio
        gamma_mean = self.gamma_mean_rad
        l_x = self.wing_cg_from_pivot_x_m
        l_y = self.wing_cg_from_pivot_y_m
        rp_x = self.wing_pivot_from_o_x_m
        rp_z = self.wing_pivot_from_o_z_m
        r_z = self.body_cg_from_o_z_m
        rotational = (
            self.body_inertia_yy_kg_m2
            + 2.0 * self.wing_inertia_yy_kg_m2 * (1.0 - math.sin(gamma_mean) ** 2)
            + 2.0 * self.wing_inertia_zz_kg_m2 * math.sin(gamma_mean) ** 2
        )
        parallel_axis = self.total_mass_kg * (
            rho / (1.0 - rho) * (l_x + rp_x) ** 2
            + rho * (rp_z - l_y * math.sin(gamma_mean)) ** 2
            + (1.0 - rho) * r_z**2
        )
        return rotational + parallel_axis


@dataclass(frozen=True)
class AminiSfwmInertialResponse:
    """Longitudinal SFWM inertial accelerations in the paper body FRD frame."""

    vertical_acceleration_frd_m_s2: torch.Tensor
    pitch_angular_acceleration_frd_rad_s2: torch.Tensor
    wing_vertical_kinematic_term_rad_s2: torch.Tensor


def measured_flapping_bot_amini_params() -> AminiSfwmLongitudinalParams:
    """Map the accepted measured three-body properties to Amini geometry."""

    body = BODY_LINK_MASS_PROPERTIES
    wing = LEFT_WING_LINK_MASS_PROPERTIES
    total_mass = body.mass_kg + 2.0 * wing.mass_kg
    gamma_mean = MEASURED_WING_NEUTRAL_DIHEDRAL_RAD
    wing_x = wing.com_m[0]
    wing_y = abs(wing.com_m[1])
    neutral_wing_z_flu = wing_y * math.sin(gamma_mean)
    point_o_x_flu = (body.mass_kg * body.com_m[0] + 2.0 * wing.mass_kg * wing_x) / total_mass
    point_o_z_flu = (
        body.mass_kg * body.com_m[2] + 2.0 * wing.mass_kg * neutral_wing_z_flu
    ) / total_mass
    body_runtime_inertia = project_diagonal_inertia_to_triangle_cone(body)

    # FRD and FLU share x. Their z components have opposite signs.
    pivot_from_o_x = -point_o_x_flu
    pivot_from_o_z_frd = point_o_z_flu
    body_cg_from_o_z_frd = -(body.com_m[2] - point_o_z_flu)
    return AminiSfwmLongitudinalParams(
        total_mass_kg=total_mass,
        body_mass_kg=body.mass_kg,
        wing_mass_each_kg=wing.mass_kg,
        wing_cg_from_pivot_x_m=wing_x,
        wing_cg_from_pivot_y_m=wing_y,
        wing_pivot_from_o_x_m=pivot_from_o_x,
        wing_pivot_from_o_z_m=pivot_from_o_z_frd,
        body_cg_from_o_z_m=body_cg_from_o_z_frd,
        body_inertia_yy_kg_m2=body_runtime_inertia[1][1],
        wing_inertia_yy_kg_m2=wing.inertia_kg_m2[1][1],
        wing_inertia_zz_kg_m2=wing.inertia_kg_m2[2][2],
        gamma_mean_rad=gamma_mean,
        point_o_in_base_flu_m=(point_o_x_flu, 0.0, point_o_z_flu),
    )


def compute_amini_sfwm_inertial_response(
    *,
    gamma_rad: torch.Tensor,
    gamma_dot_rad_s: torch.Tensor,
    gamma_ddot_rad_s2: torch.Tensor,
    params: AminiSfwmLongitudinalParams,
) -> AminiSfwmInertialResponse:
    """Evaluate Amini Eqs. (16)--(17) for actual wing motion.

    Args:
        gamma_rad: Symmetric physical wing angle, shape ``(...)``, rad.
        gamma_dot_rad_s: Wing angular rate with the same shape, rad/s.
        gamma_ddot_rad_s2: Wing angular acceleration, rad/s^2.
        params: Frozen geometry and mass parameters.

    Returns:
        Vertical acceleration ``a_fz`` in body FRD ``+z`` (down), pitch
        angular acceleration ``alpha_fy`` about body FRD ``+y`` (right), and
        their common wing-kinematic term. Shapes match ``gamma_rad``.
    """

    if gamma_rad.shape != gamma_dot_rad_s.shape or gamma_rad.shape != gamma_ddot_rad_s2.shape:
        raise ValueError("gamma, gamma_dot and gamma_ddot must have identical shapes.")
    common = gamma_ddot_rad_s2 * torch.cos(gamma_rad) - gamma_dot_rad_s.square() * torch.sin(gamma_rad)
    vertical = params.wing_to_weight_ratio * params.wing_cg_from_pivot_y_m * common
    pitch = (
        -2.0
        * params.wing_cg_from_pivot_y_m
        * (params.wing_cg_from_pivot_x_m + params.wing_pivot_from_o_x_m)
        * params.wing_mass_each_kg
        / params.constant_pitch_inertia_kg_m2
        * common
    )
    return AminiSfwmInertialResponse(
        vertical_acceleration_frd_m_s2=vertical,
        pitch_angular_acceleration_frd_rad_s2=pitch,
        wing_vertical_kinematic_term_rad_s2=common,
    )

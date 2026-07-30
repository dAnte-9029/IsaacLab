"""Kinematic and wrench helpers for measured-wing multibody aerodynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .qsm_delaurier1993 import translate_wrench_moment

Tensor = torch.Tensor

COMMANDED_BASE_EQUIVALENT = "commanded_base_equivalent"
ACTUAL_PER_WING_LINK = "actual_per_wing_link"
WING_AERO_COUPLING_MODES = (
    COMMANDED_BASE_EQUIVALENT,
    ACTUAL_PER_WING_LINK,
)


def validate_wing_aero_coupling_mode(mode: str) -> str:
    """Return a supported wing-aerodynamic coupling mode."""

    normalized = str(mode).strip()
    if normalized not in WING_AERO_COUPLING_MODES:
        raise ValueError(
            f"Unsupported wing_aero_coupling_mode={mode!r}; expected one of {WING_AERO_COUPLING_MODES}."
        )
    return normalized


@dataclass(frozen=True)
class PhysicalWingFlapKinematics:
    """Actual left/right flap kinematics in a common physical sign convention.

    Each tensor has shape ``(..., 2)`` ordered left then right. Position is in
    rad, velocity in rad/s and acceleration in rad/s^2. Positive motion is
    upstroke for both wings.
    """

    position_rad: Tensor
    velocity_rad_s: Tensor
    acceleration_rad_s2: Tensor


def map_opposed_joint_states_to_physical_wing_kinematics(
    *,
    joint_position_rad: Tensor,
    joint_velocity_rad_s: Tensor,
    joint_acceleration_rad_s2: Tensor,
    left_joint_mid_rad: float,
    right_joint_mid_rad: float,
) -> PhysicalWingFlapKinematics:
    """Map opposed URDF joint coordinates to common-sign physical wing motion.

    Inputs have shape ``(..., 2)`` ordered left then right. The left joint uses
    ``q_left - q_left_mid`` while the mirrored right joint uses
    ``-(q_right - q_right_mid)``.
    """

    if (
        joint_position_rad.shape != joint_velocity_rad_s.shape
        or joint_position_rad.shape != joint_acceleration_rad_s2.shape
        or joint_position_rad.ndim < 1
        or joint_position_rad.shape[-1] != 2
    ):
        raise ValueError("Joint position, velocity and acceleration must share shape (..., 2).")
    if (
        joint_position_rad.device != joint_velocity_rad_s.device
        or joint_position_rad.device != joint_acceleration_rad_s2.device
        or joint_position_rad.dtype != joint_velocity_rad_s.dtype
        or joint_position_rad.dtype != joint_acceleration_rad_s2.dtype
    ):
        raise ValueError("Joint kinematics must share device and dtype.")

    position = torch.stack(
        (
            joint_position_rad[..., 0] - float(left_joint_mid_rad),
            -(joint_position_rad[..., 1] - float(right_joint_mid_rad)),
        ),
        dim=-1,
    )
    velocity = torch.stack(
        (joint_velocity_rad_s[..., 0], -joint_velocity_rad_s[..., 1]),
        dim=-1,
    )
    acceleration = torch.stack(
        (joint_acceleration_rad_s2[..., 0], -joint_acceleration_rad_s2[..., 1]),
        dim=-1,
    )
    return PhysicalWingFlapKinematics(
        position_rad=position,
        velocity_rad_s=velocity,
        acceleration_rad_s2=acceleration,
    )


def translate_wing_root_wrench_to_com_link(
    *,
    force_link_n: Tensor,
    moment_link_about_wing_origin_nm: Tensor,
    wing_com_position_link_m: Tensor,
) -> Tensor:
    """Translate per-wing moments from link origin to link COM.

    All inputs use each wing link's FLU Cartesian frame and shape ``(..., 2,
    3)``. Force is in N, moment in N m, and COM position is measured from the
    wing-link origin in m. The returned moment is about the wing COM in N m,
    which is the reference expected when PhysX applies a wrench without an
    explicit position.
    """

    if (
        force_link_n.shape != moment_link_about_wing_origin_nm.shape
        or force_link_n.shape != wing_com_position_link_m.shape
        or force_link_n.ndim < 2
        or force_link_n.shape[-2:] != (2, 3)
    ):
        raise ValueError("Wing wrench and COM inputs must share shape (..., 2, 3).")
    flat_force = force_link_n.reshape(-1, 3)
    flat_moment = moment_link_about_wing_origin_nm.reshape(-1, 3)
    flat_com = wing_com_position_link_m.reshape(-1, 3)
    zeros = torch.zeros_like(flat_com)
    return translate_wrench_moment(
        flat_force,
        flat_moment,
        zeros,
        flat_com,
    ).reshape_as(moment_link_about_wing_origin_nm)


__all__ = [
    "ACTUAL_PER_WING_LINK",
    "COMMANDED_BASE_EQUIVALENT",
    "WING_AERO_COUPLING_MODES",
    "PhysicalWingFlapKinematics",
    "map_opposed_joint_states_to_physical_wing_kinematics",
    "translate_wing_root_wrench_to_com_link",
    "validate_wing_aero_coupling_mode",
]

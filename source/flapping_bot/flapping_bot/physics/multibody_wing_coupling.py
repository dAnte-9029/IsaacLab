"""Kinematic and wrench helpers for measured-wing multibody aerodynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .qsm_delaurier1993 import translate_wrench_moment

Tensor = torch.Tensor

COMMANDED_BASE_EQUIVALENT = "commanded_base_equivalent"
ACTUAL_MOTION_BASE_EQUIVALENT = "actual_motion_base_equivalent"
ACTUAL_PER_WING_LINK = "actual_per_wing_link"
PRESCRIBED_PER_WING_LINK = "prescribed_per_wing_link"
IDEAL_TORQUE_PER_WING_LINK = "ideal_torque_per_wing_link"
SINUSOIDAL_PHASE_PER_WING_LINK = "sinusoidal_phase_per_wing_link"
IDEAL_INVERSE_DYNAMICS_PER_WING_LINK = "ideal_inverse_dynamics_per_wing_link"
NATIVE_HOLONOMIC_PER_WING_LINK = "native_holonomic_per_wing_link"
WING_AERO_COUPLING_MODES = (
    COMMANDED_BASE_EQUIVALENT,
    ACTUAL_MOTION_BASE_EQUIVALENT,
    ACTUAL_PER_WING_LINK,
    PRESCRIBED_PER_WING_LINK,
    IDEAL_TORQUE_PER_WING_LINK,
    SINUSOIDAL_PHASE_PER_WING_LINK,
    IDEAL_INVERSE_DYNAMICS_PER_WING_LINK,
    NATIVE_HOLONOMIC_PER_WING_LINK,
)

ACTUAL_JOINT_ACCELERATION = "actual_joint_acceleration"
PRESCRIBED_ACCELERATION = "prescribed_acceleration"
ZERO_ACCELERATION = "zero_acceleration"
WING_AERO_ACCELERATION_SOURCES = (
    ACTUAL_JOINT_ACCELERATION,
    PRESCRIBED_ACCELERATION,
    ZERO_ACCELERATION,
)

FULL_WING_LINK_WRENCH = "full_wing_link_wrench"
WING_LINK_FORCE_ONLY = "wing_link_force_only"
WING_LINK_MOMENT_ONLY = "wing_link_moment_only"
WING_LINK_AERO_LOAD_MODES = (
    FULL_WING_LINK_WRENCH,
    WING_LINK_FORCE_ONLY,
    WING_LINK_MOMENT_ONLY,
)


def build_prescribed_dof_position_limits(
    *,
    nominal_limits_rad: Tensor,
    target_position_rad: Tensor,
    joint_id: int,
    half_width_rad: float = 0.0,
) -> Tensor:
    """Return articulation limits with one joint constrained about a target.

    Args:
        nominal_limits_rad: Joint limits with shape ``(N, J, 2)`` in rad.
        target_position_rad: Per-environment target with shape ``(N,)`` in rad.
        joint_id: Articulation joint index to prescribe.
        half_width_rad: Nonnegative half-width of the allowed position band.
    """

    if nominal_limits_rad.ndim != 3 or nominal_limits_rad.shape[-1] != 2:
        raise ValueError("nominal_limits_rad must have shape (N, J, 2).")
    if target_position_rad.shape != nominal_limits_rad.shape[:1]:
        raise ValueError("target_position_rad must have shape (N,).")
    if nominal_limits_rad.device != target_position_rad.device:
        raise ValueError("Nominal limits and target positions must share a device.")
    if nominal_limits_rad.dtype != target_position_rad.dtype:
        raise ValueError("Nominal limits and target positions must share a dtype.")
    if joint_id < 0 or joint_id >= nominal_limits_rad.shape[1]:
        raise ValueError(f"joint_id={joint_id} is outside the articulation joint range.")
    if half_width_rad < 0.0:
        raise ValueError("half_width_rad must be nonnegative.")

    limits = nominal_limits_rad.clone()
    half_width = float(half_width_rad)
    limits[:, joint_id, 0] = target_position_rad - half_width
    limits[:, joint_id, 1] = target_position_rad + half_width
    return limits


def validate_wing_aero_coupling_mode(mode: str) -> str:
    """Return a supported wing-aerodynamic coupling mode."""

    normalized = str(mode).strip()
    if normalized not in WING_AERO_COUPLING_MODES:
        raise ValueError(
            f"Unsupported wing_aero_coupling_mode={mode!r}; expected one of {WING_AERO_COUPLING_MODES}."
        )
    return normalized


def validate_wing_aero_acceleration_source(source: str) -> str:
    """Return a supported DeLaurier flap-acceleration input source."""

    normalized = str(source).strip()
    if normalized not in WING_AERO_ACCELERATION_SOURCES:
        raise ValueError(
            "Unsupported wing_aero_acceleration_source="
            f"{source!r}; expected one of {WING_AERO_ACCELERATION_SOURCES}."
        )
    return normalized


def validate_wing_link_aero_load_mode(mode: str) -> str:
    """Return a supported per-wing aerodynamic load ablation mode."""

    normalized = str(mode).strip()
    if normalized not in WING_LINK_AERO_LOAD_MODES:
        raise ValueError(
            f"Unsupported wing_link_aero_load_mode={mode!r}; "
            f"expected one of {WING_LINK_AERO_LOAD_MODES}."
        )
    return normalized


def resolve_wing_aero_acceleration(
    *,
    source: str,
    actual_acceleration_rad_s2: Tensor,
    prescribed_acceleration_rad_s2: Tensor,
) -> Tensor:
    """Select the physical flap-acceleration input used by DeLaurier.

    Both inputs have identical shape, device, and dtype. They express left and
    right physical upstroke-positive flap acceleration in rad/s^2. The zero
    mode removes only the acceleration-dependent heave input; it does not
    change actual position, velocity, drive targets, or PhysX state.
    """

    acceleration_source = validate_wing_aero_acceleration_source(source)
    if (
        actual_acceleration_rad_s2.shape != prescribed_acceleration_rad_s2.shape
        or actual_acceleration_rad_s2.device != prescribed_acceleration_rad_s2.device
        or actual_acceleration_rad_s2.dtype != prescribed_acceleration_rad_s2.dtype
    ):
        raise ValueError("Actual and prescribed wing accelerations must share shape, device and dtype.")
    if acceleration_source == ACTUAL_JOINT_ACCELERATION:
        return actual_acceleration_rad_s2
    if acceleration_source == PRESCRIBED_ACCELERATION:
        return prescribed_acceleration_rad_s2
    return torch.zeros_like(actual_acceleration_rad_s2)


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
    "ACTUAL_JOINT_ACCELERATION",
    "ACTUAL_MOTION_BASE_EQUIVALENT",
    "COMMANDED_BASE_EQUIVALENT",
    "FULL_WING_LINK_WRENCH",
    "IDEAL_TORQUE_PER_WING_LINK",
    "IDEAL_INVERSE_DYNAMICS_PER_WING_LINK",
    "NATIVE_HOLONOMIC_PER_WING_LINK",
    "PRESCRIBED_ACCELERATION",
    "PRESCRIBED_PER_WING_LINK",
    "SINUSOIDAL_PHASE_PER_WING_LINK",
    "WING_AERO_COUPLING_MODES",
    "WING_LINK_AERO_LOAD_MODES",
    "WING_LINK_FORCE_ONLY",
    "WING_LINK_MOMENT_ONLY",
    "WING_AERO_ACCELERATION_SOURCES",
    "ZERO_ACCELERATION",
    "PhysicalWingFlapKinematics",
    "build_prescribed_dof_position_limits",
    "map_opposed_joint_states_to_physical_wing_kinematics",
    "resolve_wing_aero_acceleration",
    "translate_wing_root_wrench_to_com_link",
    "validate_wing_aero_acceleration_source",
    "validate_wing_aero_coupling_mode",
    "validate_wing_link_aero_load_mode",
]

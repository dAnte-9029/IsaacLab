"""Minimal ideal generalized-effort drive for the measured wing pair."""

from __future__ import annotations

import math

import torch

from .measured_mass_properties import RIGHT_WING_LINK_MASS_PROPERTIES

_RIGHT_WING_COM_LINK_M = RIGHT_WING_LINK_MASS_PROPERTIES.com_m
_RIGHT_WING_INERTIA_COM_KG_M2 = RIGHT_WING_LINK_MASS_PROPERTIES.inertia_kg_m2
SINGLE_WING_HINGE_INERTIA_KG_M2 = (
    _RIGHT_WING_INERTIA_COM_KG_M2[0][0]
    + RIGHT_WING_LINK_MASS_PROPERTIES.mass_kg
    * (
        _RIGHT_WING_COM_LINK_M[1] * _RIGHT_WING_COM_LINK_M[1]
        + _RIGHT_WING_COM_LINK_M[2] * _RIGHT_WING_COM_LINK_M[2]
    )
)
IDEAL_TORQUE_EQUIVALENT_INERTIA_KG_M2 = 2.0 * SINGLE_WING_HINGE_INERTIA_KG_M2
IDEAL_TORQUE_NATURAL_FREQUENCY_HZ = 150.0
IDEAL_TORQUE_DAMPING_RATIO = 1.0


def ideal_torque_drive_gains(
    *,
    equivalent_inertia_kg_m2: float,
    natural_frequency_hz: float,
    damping_ratio: float,
    physics_dt_s: float | None = None,
) -> tuple[float, float]:
    """Return inertia-scaled continuous or discrete critical gains."""

    inertia = float(equivalent_inertia_kg_m2)
    frequency = float(natural_frequency_hz)
    damping = float(damping_ratio)
    if inertia <= 0.0:
        raise ValueError("equivalent_inertia_kg_m2 must be positive.")
    if frequency <= 0.0:
        raise ValueError("natural_frequency_hz must be positive.")
    if damping < 0.0:
        raise ValueError("damping_ratio must be nonnegative.")
    omega_n = 2.0 * math.pi * frequency
    if physics_dt_s is not None:
        dt = float(physics_dt_s)
        if dt <= 0.0:
            raise ValueError("physics_dt_s must be positive.")
        if not math.isclose(damping, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Discrete pole-matched gains currently require damping_ratio=1.")
        discrete_pole = math.exp(-omega_n * dt)
        stiffness = inertia * (1.0 - discrete_pole) ** 2 / (dt * dt)
        velocity_gain = inertia * (1.0 - discrete_pole * discrete_pole) / dt
        return stiffness, velocity_gain
    return inertia * omega_n * omega_n, 2.0 * damping * inertia * omega_n


def compute_ideal_torque_drive_effort(
    *,
    position_rad: torch.Tensor,
    velocity_rad_s: torch.Tensor,
    reference_position_rad: torch.Tensor,
    reference_velocity_rad_s: torch.Tensor,
    reference_acceleration_rad_s2: torch.Tensor,
    equivalent_inertia_kg_m2: float,
    natural_frequency_hz: float,
    damping_ratio: float,
    effort_limit_nm: float,
    physics_dt_s: float | None = None,
) -> torch.Tensor:
    """Compute inertia-feedforward plus PD effort for one common wing DOF."""

    tensors = (
        position_rad,
        velocity_rad_s,
        reference_position_rad,
        reference_velocity_rad_s,
        reference_acceleration_rad_s2,
    )
    if any(tensor.shape != position_rad.shape for tensor in tensors):
        raise ValueError("Drive state and reference tensors must share shape.")
    if any(
        tensor.device != position_rad.device or tensor.dtype != position_rad.dtype
        for tensor in tensors
    ):
        raise ValueError("Drive state and reference tensors must share device and dtype.")
    effort_limit = float(effort_limit_nm)
    if effort_limit <= 0.0:
        raise ValueError("effort_limit_nm must be positive.")
    stiffness, damping = ideal_torque_drive_gains(
        equivalent_inertia_kg_m2=equivalent_inertia_kg_m2,
        natural_frequency_hz=natural_frequency_hz,
        damping_ratio=damping_ratio,
        physics_dt_s=physics_dt_s,
    )
    effort = (
        float(equivalent_inertia_kg_m2) * reference_acceleration_rad_s2
        + stiffness * (reference_position_rad - position_rad)
        + damping * (reference_velocity_rad_s - velocity_rad_s)
    )
    return torch.clamp(effort, min=-effort_limit, max=effort_limit)


def compute_common_aerodynamic_hinge_torque(
    *,
    force_link_n: torch.Tensor,
    moment_link_about_com_nm: torch.Tensor,
    wing_com_position_link_m: torch.Tensor,
) -> torch.Tensor:
    """Return aerodynamic generalized torque for opposed wing coordinates."""

    joint_torque = compute_aerodynamic_joint_hinge_torques(
        force_link_n=force_link_n,
        moment_link_about_com_nm=moment_link_about_com_nm,
        wing_com_position_link_m=wing_com_position_link_m,
    )
    return joint_torque[..., 0] - joint_torque[..., 1]


def compute_aerodynamic_joint_hinge_torques(
    *,
    force_link_n: torch.Tensor,
    moment_link_about_com_nm: torch.Tensor,
    wing_com_position_link_m: torch.Tensor,
) -> torch.Tensor:
    """Return left/right aerodynamic hinge torque in each URDF joint sign.

    Inputs have shape ``(...,2,3)`` in each wing link FLU frame. Force is in N,
    moment is about the wing COM in N m, and COM position is measured from the
    wing-link hinge origin in m. The returned ``(...,2)`` tensor is the moment
    about each local ``+x`` revolute axis in N m, ordered left then right.
    """

    if (
        force_link_n.shape != moment_link_about_com_nm.shape
        or force_link_n.shape != wing_com_position_link_m.shape
        or force_link_n.ndim < 2
        or force_link_n.shape[-2:] != (2, 3)
    ):
        raise ValueError("Wing force, moment and COM tensors must share shape (..., 2, 3).")
    moment_about_hinge = moment_link_about_com_nm + torch.linalg.cross(
        wing_com_position_link_m,
        force_link_n,
        dim=-1,
    )
    return moment_about_hinge[..., 0]


def apply_quintic_amplitude_ramp(
    *,
    position_rad: torch.Tensor,
    velocity_rad_s: torch.Tensor,
    acceleration_rad_s2: torch.Tensor,
    elapsed_time_s: torch.Tensor,
    duration_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a C2 quintic startup envelope to prescribed kinematics."""

    tensors = (
        position_rad,
        velocity_rad_s,
        acceleration_rad_s2,
        elapsed_time_s,
        duration_s,
    )
    if any(tensor.shape != position_rad.shape for tensor in tensors):
        raise ValueError("Kinematics, elapsed time and duration tensors must share shape.")
    if any(
        tensor.device != position_rad.device or tensor.dtype != position_rad.dtype
        for tensor in tensors
    ):
        raise ValueError("Kinematics, elapsed time and duration tensors must share device and dtype.")
    if torch.any(duration_s <= 0.0):
        raise ValueError("duration_s must be positive.")
    normalized_time = torch.clamp(elapsed_time_s / duration_s, min=0.0, max=1.0)
    s2 = normalized_time * normalized_time
    s3 = s2 * normalized_time
    s4 = s3 * normalized_time
    s5 = s4 * normalized_time
    envelope = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
    envelope_rate = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) / duration_s
    envelope_acceleration = (
        60.0 * normalized_time - 180.0 * s2 + 120.0 * s3
    ) / (duration_s * duration_s)
    active = elapsed_time_s < duration_s
    envelope_rate = torch.where(active, envelope_rate, torch.zeros_like(envelope_rate))
    envelope_acceleration = torch.where(
        active,
        envelope_acceleration,
        torch.zeros_like(envelope_acceleration),
    )
    return (
        envelope * position_rad,
        envelope * velocity_rad_s + envelope_rate * position_rad,
        envelope * acceleration_rad_s2
        + 2.0 * envelope_rate * velocity_rad_s
        + envelope_acceleration * position_rad,
    )


__all__ = [
    "SINGLE_WING_HINGE_INERTIA_KG_M2",
    "IDEAL_TORQUE_EQUIVALENT_INERTIA_KG_M2",
    "IDEAL_TORQUE_NATURAL_FREQUENCY_HZ",
    "IDEAL_TORQUE_DAMPING_RATIO",
    "ideal_torque_drive_gains",
    "compute_ideal_torque_drive_effort",
    "compute_common_aerodynamic_hinge_torque",
    "compute_aerodynamic_joint_hinge_torques",
    "apply_quintic_amplitude_ramp",
]

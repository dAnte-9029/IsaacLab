"""Startup helpers for flapping-wing reset and release behavior."""

from __future__ import annotations

import math

import torch

Tensor = torch.Tensor

MECHANICAL_SINE_NEUTRAL_UPSTROKE = "mechanical_sine_neutral_upstroke"
LEGACY_COSINE_ENDPOINT_ZERO = "legacy_cosine_endpoint_zero"
FLAP_PHASE_CONVENTIONS = frozenset(
    {
        MECHANICAL_SINE_NEUTRAL_UPSTROKE,
        LEGACY_COSINE_ENDPOINT_ZERO,
    }
)


def compute_prescribed_flap_kinematics(
    *,
    phase: Tensor,
    freq_hz: Tensor,
    amplitude_rad: float,
    convention: str = MECHANICAL_SINE_NEUTRAL_UPSTROKE,
    minimum_active_frequency_hz: float = 1.0e-3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return prescribed flap position, velocity and acceleration.

    The default engineering convention is ``q=A*sin(phase)``: phase zero is
    the neutral pose and positive velocity starts the upstroke. The historical
    cosine convention remains selectable for baseline reproduction.
    """

    if not isinstance(phase, torch.Tensor) or not isinstance(freq_hz, torch.Tensor):
        raise TypeError("phase and freq_hz must be torch tensors.")
    if phase.shape != freq_hz.shape:
        raise ValueError("phase and freq_hz must have the same shape.")
    resolved_convention = str(convention)
    if resolved_convention not in FLAP_PHASE_CONVENTIONS:
        supported = ", ".join(sorted(FLAP_PHASE_CONVENTIONS))
        raise ValueError(f"Unsupported flap phase convention {resolved_convention!r}; expected one of: {supported}.")
    if minimum_active_frequency_hz < 0.0:
        raise ValueError("minimum_active_frequency_hz must be non-negative.")

    amplitude = float(amplitude_rad)
    omega = 2.0 * math.pi * freq_hz
    if resolved_convention == MECHANICAL_SINE_NEUTRAL_UPSTROKE:
        q = amplitude * torch.sin(phase)
        qd = amplitude * omega * torch.cos(phase)
        qdd = -amplitude * omega.square() * torch.sin(phase)
    else:
        q = amplitude * torch.cos(phase)
        qd = -amplitude * omega * torch.sin(phase)
        qdd = -amplitude * omega.square() * torch.cos(phase)

    flap_off = freq_hz <= float(minimum_active_frequency_hz)
    if torch.any(flap_off):
        q = torch.where(flap_off, torch.zeros_like(q), q)
        qd = torch.where(flap_off, torch.zeros_like(qd), qd)
        qdd = torch.where(flap_off, torch.zeros_like(qdd), qdd)
    return q, qd, qdd


def advance_flap_phase(
    *,
    phase: Tensor,
    freq_hz: Tensor,
    physics_dt_s: float,
    freeze_steps: Tensor | None = None,
) -> Tensor:
    """Advance flap phase, optionally holding frozen environments fixed.

    When ``freeze_steps`` is positive for an environment, its flap phase is held
    constant so release does not occur at an arbitrary phase accumulated during
    the reset hold window.
    """

    if physics_dt_s <= 0.0:
        raise ValueError("physics_dt_s must be positive.")

    phase_next = phase.clone()
    if freeze_steps is None:
        free_mask = torch.ones_like(freq_hz, dtype=torch.bool)
    else:
        free_mask = freeze_steps <= 0

    if torch.any(free_mask):
        phase_step = 2.0 * math.pi * freq_hz[free_mask] * float(physics_dt_s)
        phase_next[free_mask] = torch.remainder(phase[free_mask] + phase_step, 2.0 * math.pi)
    return phase_next


def map_symmetric_flap_coordinate_to_joint_space(
    *,
    flap_position_rad: Tensor,
    flap_velocity_rad_s: Tensor,
    left_joint_mid_rad: float,
    right_joint_mid_rad: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Map one physical flap coordinate to mirrored URDF wing joints.

    The project URDF gives both revolute joints a joint-frame ``+x`` axis,
    while the left mesh extends along link ``+y`` and the right mesh along
    link ``-y``. Consequently a physically symmetric pose uses opposite joint
    coordinates. Positive ``flap_position_rad`` raises both representative
    span points toward body-FLU ``+z``. Shapes of all returned position and
    velocity tensors match the corresponding input tensor.

    Args:
        flap_position_rad: Engineering flap coordinate ``q``, in rad.
        flap_velocity_rad_s: Its derivative, in rad/s.
        left_joint_mid_rad: Left joint coordinate at the symmetric mid-pose.
        right_joint_mid_rad: Right joint coordinate at the symmetric mid-pose.

    Returns:
        ``(left_position, right_position, left_velocity, right_velocity)`` in
        URDF joint coordinates.
    """

    if not isinstance(flap_position_rad, torch.Tensor) or not isinstance(flap_velocity_rad_s, torch.Tensor):
        raise TypeError("flap_position_rad and flap_velocity_rad_s must be torch tensors.")
    if flap_position_rad.shape != flap_velocity_rad_s.shape:
        raise ValueError("flap_position_rad and flap_velocity_rad_s must have the same shape.")
    left_position = float(left_joint_mid_rad) + flap_position_rad
    right_position = float(right_joint_mid_rad) - flap_position_rad
    left_velocity = flap_velocity_rad_s
    right_velocity = -flap_velocity_rad_s
    return left_position, right_position, left_velocity, right_velocity

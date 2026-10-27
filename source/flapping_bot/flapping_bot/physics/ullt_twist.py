"""Prescribed LE-axis twist for the CSV/URDF wing, with exact time derivatives.

This is a kinematic prior, not an identified aeroelastic structural model.
Vectors use neutral body FLU, lengths m, angles rad, time s; tensors (B,N,3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .ullt_state_space import ULLTGeometry, ULLTMotion

Jet = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class PrescribedTwistConfig:
    """Positive theta is nose-up about negative LE span (right to left)."""

    amplitude_deg: float = 0.0
    kappa: float = 0.0
    phase_offset_deg: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.amplitude_deg, self.kappa, self.phase_offset_deg)):
            raise ValueError("Twist parameters must be finite")
        if not 0 <= self.amplitude_deg <= 40 or not -1 <= self.kappa <= 1 or abs(self.phase_offset_deg) > 90:
            raise ValueError("Screen bounds: amplitude [0,40] deg, kappa [-1,1], phase [-90,90] deg")


def _constant(x: torch.Tensor) -> Jet:
    return x, torch.zeros_like(x), torch.zeros_like(x)


def _add(a: Jet, b: Jet, scale: float = 1.0) -> Jet:
    return tuple(x + scale * y for x, y in zip(a, b))


def _mul(a: Jet, b: Jet) -> Jet:
    return a[0] * b[0], a[1] * b[0] + a[0] * b[1], a[2] * b[0] + 2 * a[1] * b[1] + a[0] * b[2]


def _dot(a: Jet, b: Jet) -> Jet:
    return tuple(x.sum(-1, keepdim=True) for x in _mul(a, b))


def _cross(a: Jet, b: Jet) -> Jet:
    c = torch.linalg.cross
    return c(a[0], b[0]), c(a[1], b[0]) + c(a[0], b[1]), c(a[2], b[0]) + 2 * c(a[1], b[1]) + c(a[0], b[2])


def _power(a: Jet, p: float) -> Jet:
    if bool((a[0] <= 0).any()):
        raise ValueError("Degenerate twisted section")
    return a[0] ** p, p * a[0] ** (p - 1) * a[1], p * a[0] ** (p - 1) * a[2] + p * (p - 1) * a[0] ** (p - 2) * a[1] ** 2


def _unit(a: Jet) -> Jet:
    return _mul(a, _power(_dot(a, a), -0.5))


def twist_wing(
    neutral: ULLTGeometry,
    hinge_m: torch.Tensor,
    phase_rad: torch.Tensor,
    phase_rate_rad_s: torch.Tensor,
    phase_acceleration_rad_s2: torch.Tensor,
    config: PrescribedTwistConfig,
) -> tuple[ULLTGeometry, ULLTMotion]:
    """Deform one wing before flapping/body rotation, in neutral FLU axes.

    CSV LE is straight along the URDF-rolled y axis; chord is originally +x.
    Shared quarter-chord endpoints receive their own continuous span twist.
    Chord is projected perpendicular to the deformed lifting line, as in the
    original swept-section adapter. All projected-point/frame derivatives are
    included. Motion contains relative flow=-point velocity, not freestream.
    """
    batch = len(phase_rad)
    root = hinge_m.view(1, 1, 3)
    endpoints = torch.cat((neutral.edge_start_w_m, neutral.edge_end_w_m), 1) - root
    transverse = endpoints.clone()
    transverse[..., 0] = 0
    radii = torch.linalg.vector_norm(transverse, dim=-1)
    tip = radii[0].argmax()
    axis = transverse[:, tip : tip + 1] / radii[:, tip : tip + 1, None]
    axis = axis * torch.sign(axis[..., 1:2])  # positive span on both wings
    radius = radii.max()
    if bool(
        (torch.linalg.vector_norm(torch.linalg.cross(transverse, axis.expand_as(transverse)), dim=-1) > 1e-8).any()
    ):
        raise ValueError("Twist adapter requires a straight leading edge in the neutral yz plane")
    xaxis = torch.zeros_like(neutral.quarter_chord_w_m)
    xaxis[..., 0] = 1
    physical_chord = neutral.chord_m / neutral.chord_forward_w[..., 0]

    def rotation_jet(point: torch.Tensor, vector: torch.Tensor | None = None) -> Jet:
        rel = point - root
        station = (rel * axis).sum(-1, keepdim=True)
        s = station.abs() / radius
        shape = (1 - config.kappa) * s + config.kappa * s.square()
        angle = phase_rad[:, None, None] - math.radians(config.phase_offset_deg)
        # -A*sin(phi_D-psi)=A*cos(phi_C-psi); positive pitch rotates about -span.
        amplitude = math.radians(config.amplitude_deg) * shape
        theta = amplitude * angle.cos()
        td = -amplitude * angle.sin() * phase_rate_rad_s[:, None, None]
        tdd = -amplitude * (
            angle.cos() * phase_rate_rad_s[:, None, None].square()
            + angle.sin() * phase_acceleration_rad_s2[:, None, None]
        )
        arm = rel - station * axis if vector is None else vector
        rotation_axis = -axis.expand(batch, arm.shape[1], -1)
        arm = arm.expand(batch, -1, -1)
        parallel = (arm * rotation_axis).sum(-1, keepdim=True) * rotation_axis
        rotated = parallel + (arm - parallel) * theta.cos() + torch.linalg.cross(rotation_axis, arm) * theta.sin()
        omega, alpha = rotation_axis * td, rotation_axis * tdd
        velocity = torch.linalg.cross(omega, rotated)
        acceleration = torch.linalg.cross(alpha, rotated) + torch.linalg.cross(omega, velocity)
        position = rotated if vector is not None else rotated + station * axis + root
        return position, velocity, acceleration

    start, end, quarter = (
        rotation_jet(p) for p in (neutral.edge_start_w_m, neutral.edge_end_w_m, neutral.quarter_chord_w_m)
    )
    span = _unit(_add(end, start, -1))
    physical_forward = rotation_jet(neutral.quarter_chord_w_m, xaxis)
    projected = _add(physical_forward, _mul(_dot(physical_forward, span), span), -1)
    forward = _unit(projected)
    normal = _cross(forward, span)
    chord = physical_chord * torch.linalg.vector_norm(projected[0], dim=-1)
    three_quarter = _add(quarter, _mul(_constant(physical_chord[..., None]), projected), -0.5)
    omega = 0.5 * sum(torch.linalg.cross(a[0], a[1]) for a in (forward, span, normal))
    alpha = 0.5 * sum(torch.linalg.cross(a[0], a[2]) for a in (forward, span, normal))
    return (
        ULLTGeometry(start[0], end[0], quarter[0], forward[0], chord),
        ULLTMotion(-three_quarter[1], -start[1], -end[1], three_quarter[2], omega, alpha),
    )

"""Simple tail aerodynamic model for control surfaces (elevator + rudder).

This module intentionally keeps the model low-dimensional and stable for RL:
- Uses body-frame kinematics (root linear/angular velocity).
- Tail force depends on *deflection angle* (joint position), not only joint velocity.
- Returns net body-frame wrench to apply at the base.

Coordinate convention: body frame is right-handed (x forward, y left, z up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _as_vec3(x: Sequence[float], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    t = torch.as_tensor(x, device=device, dtype=dtype).flatten()
    if t.numel() != 3:
        raise ValueError("Expected a length-3 vector.")
    return t


def _normalize(v: Tensor, eps: float = 1.0e-9) -> Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def _rotate_about_axis(v: Tensor, axis: Tensor, angle: Tensor) -> Tensor:
    """Rotate vectors v about a fixed axis by angle (Rodrigues), vectorized.

    Args:
        v: (N,3)
        axis: (3,) unit vector
        angle: (N,) radians
    """
    a = _normalize(axis.view(1, 3).expand_as(v))
    ang = angle.view(-1, 1)
    c = torch.cos(ang)
    s = torch.sin(ang)
    # Rodrigues rotation formula: v' = v c + (a×v) s + a(a·v)(1-c)
    axv = torch.linalg.cross(a, v)
    adv = torch.sum(a * v, dim=-1, keepdim=True)
    return v * c + axv * s + a * adv * (1.0 - c)


@dataclass
class TailSurfaceCfg:
    """Parameters for one tail surface."""

    name: str
    area: float  # m^2
    lever_arm_body: tuple[float, float, float]  # m, from base origin to aero center in body frame
    span_axis_body: tuple[float, float, float]  # unit-ish, body frame
    chord_axis_body: tuple[float, float, float]  # unit-ish, body frame (chord direction reference)
    cl_alpha_per_rad: float = 3.5  # 2D wing ~2*pi, reduced for finite surface
    cd0: float = 0.02
    cd_k: float = 0.08  # induced-drag-like quadratic on CL
    alpha_limit_deg: float = 30.0
    deflection_sign: float = 1.0  # multiply commanded deflection before rotating chord


@dataclass
class TailAeroCfg:
    """Configuration for elevator + rudder tail model."""

    air_density: float = 1.225  # kg/m^3
    elevator: TailSurfaceCfg = field(
        default_factory=lambda: TailSurfaceCfg(
            name="elevator",
            area=0.012,
            lever_arm_body=(-0.55, 0.0, 0.0),
            span_axis_body=(0.0, 1.0, 0.0),
            chord_axis_body=(-1.0, 0.0, 0.0),
            cl_alpha_per_rad=3.5,
            cd0=0.02,
            cd_k=0.08,
            alpha_limit_deg=30.0,
            # With span=+y and chord=-x, positive rotation about +y yields negative AOA.
            # Use deflection_sign=-1 so positive elevator command produces positive AOA.
            deflection_sign=-1.0,
        )
    )
    rudder: TailSurfaceCfg = field(
        default_factory=lambda: TailSurfaceCfg(
            name="rudder",
            area=0.010,
            lever_arm_body=(-0.55, 0.0, 0.0),
            span_axis_body=(0.0, 0.0, 1.0),
            chord_axis_body=(-1.0, 0.0, 0.0),
            cl_alpha_per_rad=3.0,
            cd0=0.02,
            cd_k=0.08,
            alpha_limit_deg=30.0,
            # With span=+z and chord=-x, positive rotation about +z yields negative AOA.
            # Use deflection_sign=-1 so positive rudder command produces positive AOA.
            deflection_sign=-1.0,
        )
    )


class TailAeroModel:
    """Compute body-frame tail aerodynamic wrench for elevator and rudder."""

    def __init__(self, cfg: TailAeroCfg, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)

    def _surface_wrench(
        self,
        surf: TailSurfaceCfg,
        *,
        root_lin_vel_b: Tensor,
        root_ang_vel_b: Tensor,
        deflection_rad: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (F_b, tau_b) for one surface, body frame, shape (N,3)."""
        device = self.device
        dtype = root_lin_vel_b.dtype

        r_b = _as_vec3(surf.lever_arm_body, device=device, dtype=dtype).view(1, 3)
        r_b = r_b.expand(root_lin_vel_b.shape[0], 3)

        span = _as_vec3(surf.span_axis_body, device=device, dtype=dtype)
        span = _normalize(span).view(1, 3).expand_as(r_b)

        chord0 = _as_vec3(surf.chord_axis_body, device=device, dtype=dtype)
        chord0 = _normalize(chord0).view(1, 3).expand_as(r_b)

        # Point velocity in body frame: v = v_root + ω × r
        v_point = root_lin_vel_b + torch.linalg.cross(root_ang_vel_b, r_b)

        # Project velocity into the surface plane (orthogonal to span axis).
        v_proj = v_point - torch.sum(v_point * span, dim=1, keepdim=True) * span
        speed = torch.linalg.norm(v_proj, dim=1)
        # Unit-ish flow direction in the surface plane. For very small speeds this becomes ~0, which is fine since q~0.
        v_dir = F.normalize(v_proj, dim=1, eps=1.0e-9)

        # Rotate chord about span by deflection.
        ang = float(surf.deflection_sign) * deflection_rad
        chord = _rotate_about_axis(chord0, span[0], ang)

        # Incoming flow direction in the surface plane.
        v_in = -v_dir

        # Signed AOA in [-pi, pi] measured about span axis.
        dot = torch.sum(v_in * chord, dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        cross_vec = torch.linalg.cross(chord, v_in)
        sin = torch.sum(cross_vec * span, dim=1)
        alpha = torch.atan2(sin, dot)
        alpha_lim = math.radians(float(surf.alpha_limit_deg))
        alpha = torch.clamp(alpha, -alpha_lim, alpha_lim)

        # Lift/drag coefficients.
        cl = float(surf.cl_alpha_per_rad) * alpha
        cd = float(surf.cd0) + float(surf.cd_k) * (cl * cl)

        q = 0.5 * float(self.cfg.air_density) * (speed * speed)
        lift = (q * float(surf.area) * cl).unsqueeze(1)
        drag = (q * float(surf.area) * cd).unsqueeze(1)

        # Directions in body frame.
        lift_dir = F.normalize(torch.linalg.cross(v_dir, span), dim=1, eps=1.0e-9)
        drag_dir = -v_dir

        F_b = lift * lift_dir + drag * drag_dir
        tau_b = torch.linalg.cross(r_b, F_b)
        return F_b, tau_b

    def compute_wrench(
        self,
        *,
        root_lin_vel_b: Tensor,
        root_ang_vel_b: Tensor,
        elevator_rad: Tensor,
        rudder_rad: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return net tail wrench in body frame.

        Args:
            root_lin_vel_b: (N,3) base linear velocity in body frame.
            root_ang_vel_b: (N,3) base angular velocity in body frame.
            elevator_rad: (N,) elevator deflection.
            rudder_rad: (N,) rudder deflection.
        """
        if elevator_rad.ndim != 1 or rudder_rad.ndim != 1:
            raise ValueError("elevator_rad and rudder_rad must be 1D tensors.")
        if elevator_rad.shape[0] != root_lin_vel_b.shape[0]:
            raise ValueError("Batch size mismatch in tail aero inputs.")

        F_e, tau_e = self._surface_wrench(
            self.cfg.elevator, root_lin_vel_b=root_lin_vel_b, root_ang_vel_b=root_ang_vel_b, deflection_rad=elevator_rad
        )
        F_r, tau_r = self._surface_wrench(
            self.cfg.rudder, root_lin_vel_b=root_lin_vel_b, root_ang_vel_b=root_ang_vel_b, deflection_rad=rudder_rad
        )
        return (F_e + F_r), (tau_e + tau_r)

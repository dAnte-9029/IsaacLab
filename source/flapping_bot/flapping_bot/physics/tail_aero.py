"""Low-order tail aerodynamic model for the latest five-surface tail geometry.

This module keeps the tail model simple enough for control and RL use, while
grounding its geometry in the latest robot tail layout:
- fixed horizontal tail
- left elevon
- right elevon
- fixed vertical tail
- rudder

The default geometry uses the latest user-provided tail polygons and URDF hinge
locations. Nominal lift slopes are computed from a finite-wing formula instead
of manual tuning. Drag and angle limits remain explicit assumptions.

Coordinate convention: body frame is right-handed (x forward, y left, z up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
Vec2 = tuple[float, float]
Polygon2D = tuple[Vec2, ...]

_RIGHT_ELEVON_POLYGON_MM: Polygon2D = (
    (0.0, 140.0),
    (85.0, 140.0),
    (300.0, 0.0),
    (255.0, -80.0),
    (45.0, -80.0),
    (0.0, 0.0),
)

_VERTICAL_TAIL_POLYGON_MM: Polygon2D = (
    (-108.0, 0.0),
    (-108.0, 60.0),
    (0.0, 170.0),
    (70.0, 170.0),
    (70.0, 50.0),
    (0.0, 0.0),
)

_HINGE_RIGHT_TAIL_BODY = (-0.5365, -0.15, -0.00335)
_HINGE_LEFT_TAIL_BODY = (-0.5365, 0.15, -0.00335)
_HINGE_RUDDER_BODY = (-0.5045, 0.0, -0.00085)

_TAIL_SECTION_LIFT_SLOPE_PER_RAD = 2.0 * math.pi
_HORIZONTAL_EFFICIENCY = 0.85
_VERTICAL_EFFICIENCY = 0.80
_TAIL_CD0 = 0.02
_TAIL_ALPHA_LIMIT_DEG = 25.0


def _as_vec3(x: Sequence[float], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    t = torch.as_tensor(x, device=device, dtype=dtype).flatten()
    if t.numel() != 3:
        raise ValueError("Expected a length-3 vector.")
    return t


def _normalize(v: Tensor, eps: float = 1.0e-9) -> Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def _rotate_about_axis(v: Tensor, axis: Tensor, angle: Tensor) -> Tensor:
    """Rotate vectors ``v`` about a fixed axis by ``angle`` using Rodrigues."""

    a = _normalize(axis.view(1, 3).expand_as(v))
    ang = angle.view(-1, 1)
    c = torch.cos(ang)
    s = torch.sin(ang)
    axv = torch.linalg.cross(a, v)
    adv = torch.sum(a * v, dim=-1, keepdim=True)
    return v * c + axv * s + a * adv * (1.0 - c)


def _finite_wing_lift_slope(aspect_ratio: float, *, efficiency: float) -> float:
    """Return the small-angle finite-wing lift slope in rad^-1."""

    if aspect_ratio <= 0.0:
        raise ValueError("Aspect ratio must be positive.")
    a0 = _TAIL_SECTION_LIFT_SLOPE_PER_RAD
    return a0 / (1.0 + a0 / (math.pi * efficiency * aspect_ratio))


def _induced_drag_factor(aspect_ratio: float, *, efficiency: float) -> float:
    """Return the quadratic induced-drag factor in ``cd = cd0 + k * cl^2``."""

    if aspect_ratio <= 0.0:
        raise ValueError("Aspect ratio must be positive.")
    return 1.0 / (math.pi * efficiency * aspect_ratio)


def _clip_polygon(poly: Polygon2D, *, axis_index: int, keep_ge_zero: bool, eps: float = 1.0e-9) -> Polygon2D:
    """Clip a polygon against the half-space ``coord >= 0`` or ``coord <= 0``."""

    def inside(p: Vec2) -> bool:
        coord = p[axis_index]
        return coord >= -eps if keep_ge_zero else coord <= eps

    def intersect(a: Vec2, b: Vec2) -> Vec2:
        av = a[axis_index]
        bv = b[axis_index]
        if abs(bv - av) < eps:
            return a
        t = -av / (bv - av)
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    out: list[Vec2] = []
    prev = poly[-1]
    prev_inside = inside(prev)
    for curr in poly:
        curr_inside = inside(curr)
        if curr_inside:
            if not prev_inside:
                out.append(intersect(prev, curr))
            out.append(curr)
        elif prev_inside:
            out.append(intersect(prev, curr))
        prev = curr
        prev_inside = curr_inside

    if not out:
        raise ValueError("Polygon clipping produced an empty polygon.")

    deduped: list[Vec2] = []
    for p in out:
        if not deduped or abs(deduped[-1][0] - p[0]) > eps or abs(deduped[-1][1] - p[1]) > eps:
            deduped.append(p)
    if len(deduped) > 1 and abs(deduped[0][0] - deduped[-1][0]) <= eps and abs(deduped[0][1] - deduped[-1][1]) <= eps:
        deduped.pop()
    return tuple(deduped)


def _polyline_intersections(poly: Polygon2D, value: float, *, sweep_axis: int) -> list[float]:
    """Intersect a polygon with a vertical or horizontal sampling line."""

    other_axis = 1 - sweep_axis
    intersections: list[float] = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        av = a[sweep_axis]
        bv = b[sweep_axis]
        if abs(bv - av) < 1.0e-12:
            continue
        lo = min(av, bv)
        hi = max(av, bv)
        if value < lo or value >= hi:
            continue
        t = (value - av) / (bv - av)
        intersections.append(a[other_axis] + t * (b[other_axis] - a[other_axis]))
    intersections.sort()
    return intersections


def _horizontal_surface_center_mm(poly: Polygon2D, *, samples: int = 4096) -> tuple[float, float]:
    """Return ``(s_bar_mm, c_ac_mm)`` for a horizontal-tail surface polygon."""

    s_min = min(p[0] for p in poly)
    s_max = max(p[0] for p in poly)
    ds = (s_max - s_min) / float(samples)

    area_acc = 0.0
    s_acc = 0.0
    c_acc = 0.0
    for idx in range(samples):
        s = s_min + (idx + 0.5) * ds
        cs = _polyline_intersections(poly, s, sweep_axis=0)
        for c_lo, c_hi in zip(cs[::2], cs[1::2]):
            chord = c_hi - c_lo
            if chord <= 0.0:
                continue
            c_qc = c_hi - 0.25 * chord
            area_acc += chord
            s_acc += s * chord
            c_acc += c_qc * chord

    if area_acc <= 0.0:
        raise ValueError("Failed to compute horizontal surface center from polygon.")
    return (s_acc / area_acc, c_acc / area_acc)


def _vertical_surface_center_mm(poly: Polygon2D, *, samples: int = 4096) -> tuple[float, float]:
    """Return ``(h_bar_mm, a_ac_mm)`` for a vertical-tail surface polygon."""

    h_min = min(p[1] for p in poly)
    h_max = max(p[1] for p in poly)
    dh = (h_max - h_min) / float(samples)

    area_acc = 0.0
    h_acc = 0.0
    a_acc = 0.0
    for idx in range(samples):
        h = h_min + (idx + 0.5) * dh
        aa = _polyline_intersections(poly, h, sweep_axis=1)
        for a_lo, a_hi in zip(aa[::2], aa[1::2]):
            chord = a_hi - a_lo
            if chord <= 0.0:
                continue
            a_qc = a_lo + 0.25 * chord
            area_acc += chord
            h_acc += h * chord
            a_acc += a_qc * chord

    if area_acc <= 0.0:
        raise ValueError("Failed to compute vertical surface center from polygon.")
    return (h_acc / area_acc, a_acc / area_acc)


def _make_surface(
    *,
    name: str,
    area_m2: float,
    lever_arm_body: tuple[float, float, float],
    span_axis_body: tuple[float, float, float],
    chord_axis_body: tuple[float, float, float],
    span_m: float,
    mean_chord_m: float,
    efficiency: float,
    incidence_rad: float = 0.0,
    alpha_limit_deg: float = _TAIL_ALPHA_LIMIT_DEG,
    deflection_sign: float = 1.0,
) -> "TailSurfaceCfg":
    aspect_ratio = (span_m * span_m) / area_m2
    return TailSurfaceCfg(
        name=name,
        area=area_m2,
        lever_arm_body=lever_arm_body,
        span_axis_body=span_axis_body,
        chord_axis_body=chord_axis_body,
        span_m=span_m,
        mean_chord_m=mean_chord_m,
        incidence_rad=incidence_rad,
        cl_alpha_per_rad=_finite_wing_lift_slope(aspect_ratio, efficiency=efficiency),
        cd0=_TAIL_CD0,
        cd_k=_induced_drag_factor(aspect_ratio, efficiency=efficiency),
        alpha_limit_deg=alpha_limit_deg,
        deflection_sign=deflection_sign,
    )


def _default_fixed_horizontal_surface() -> "TailSurfaceCfg":
    fixed_right = _clip_polygon(_RIGHT_ELEVON_POLYGON_MM, axis_index=1, keep_ge_zero=True)
    _, c_ac_mm = _horizontal_surface_center_mm(fixed_right)
    return _make_surface(
        name="fixed_horizontal",
        area_m2=0.0539,
        lever_arm_body=(_HINGE_RIGHT_TAIL_BODY[0] + c_ac_mm / 1000.0, 0.0, _HINGE_RIGHT_TAIL_BODY[2]),
        span_axis_body=(0.0, 1.0, 0.0),
        chord_axis_body=(-1.0, 0.0, 0.0),
        span_m=0.600,
        mean_chord_m=0.0539 / 0.600,
        efficiency=_HORIZONTAL_EFFICIENCY,
        incidence_rad=0.0,
        deflection_sign=1.0,
    )


def _default_right_elevon_surface() -> "TailSurfaceCfg":
    elevon = _clip_polygon(_RIGHT_ELEVON_POLYGON_MM, axis_index=1, keep_ge_zero=False)
    s_ac_mm, c_ac_mm = _horizontal_surface_center_mm(elevon)
    return _make_surface(
        name="right_elevon",
        area_m2=0.02039,
        lever_arm_body=(
            _HINGE_RIGHT_TAIL_BODY[0] + c_ac_mm / 1000.0,
            -s_ac_mm / 1000.0,
            _HINGE_RIGHT_TAIL_BODY[2],
        ),
        span_axis_body=(0.0, 1.0, 0.0),
        chord_axis_body=(-1.0, 0.0, 0.0),
        span_m=0.300,
        mean_chord_m=0.02039 / 0.300,
        efficiency=_HORIZONTAL_EFFICIENCY,
        incidence_rad=0.0,
        deflection_sign=-1.0,
    )


def _default_left_elevon_surface() -> "TailSurfaceCfg":
    elevon = _clip_polygon(_RIGHT_ELEVON_POLYGON_MM, axis_index=1, keep_ge_zero=False)
    s_ac_mm, c_ac_mm = _horizontal_surface_center_mm(elevon)
    return _make_surface(
        name="left_elevon",
        area_m2=0.02039,
        lever_arm_body=(
            _HINGE_LEFT_TAIL_BODY[0] + c_ac_mm / 1000.0,
            s_ac_mm / 1000.0,
            _HINGE_LEFT_TAIL_BODY[2],
        ),
        span_axis_body=(0.0, 1.0, 0.0),
        chord_axis_body=(-1.0, 0.0, 0.0),
        span_m=0.300,
        mean_chord_m=0.02039 / 0.300,
        efficiency=_HORIZONTAL_EFFICIENCY,
        incidence_rad=0.0,
        deflection_sign=-1.0,
    )


def _default_fixed_vertical_surface() -> "TailSurfaceCfg":
    fixed_vertical = _clip_polygon(_VERTICAL_TAIL_POLYGON_MM, axis_index=0, keep_ge_zero=False)
    h_ac_mm, a_ac_mm = _vertical_surface_center_mm(fixed_vertical)
    return _make_surface(
        name="fixed_vertical",
        area_m2=0.01265,
        lever_arm_body=(
            _HINGE_RUDDER_BODY[0] - a_ac_mm / 1000.0,
            0.0,
            _HINGE_RUDDER_BODY[2] + h_ac_mm / 1000.0,
        ),
        span_axis_body=(0.0, 0.0, 1.0),
        chord_axis_body=(-1.0, 0.0, 0.0),
        span_m=0.170,
        mean_chord_m=0.01265 / 0.170,
        efficiency=_VERTICAL_EFFICIENCY,
        incidence_rad=0.0,
        deflection_sign=1.0,
    )


def _default_rudder_surface() -> "TailSurfaceCfg":
    rudder = _clip_polygon(_VERTICAL_TAIL_POLYGON_MM, axis_index=0, keep_ge_zero=True)
    h_ac_mm, a_ac_mm = _vertical_surface_center_mm(rudder)
    return _make_surface(
        name="rudder",
        area_m2=0.0101404,
        lever_arm_body=(
            _HINGE_RUDDER_BODY[0] - a_ac_mm / 1000.0,
            0.0,
            _HINGE_RUDDER_BODY[2] + h_ac_mm / 1000.0,
        ),
        span_axis_body=(0.0, 0.0, 1.0),
        chord_axis_body=(-1.0, 0.0, 0.0),
        span_m=0.170,
        mean_chord_m=0.0101404 / 0.170,
        efficiency=_VERTICAL_EFFICIENCY,
        incidence_rad=0.0,
        deflection_sign=-1.0,
    )


@dataclass
class TailSurfaceCfg:
    """Parameters for one tail surface."""

    name: str
    area: float  # m^2
    lever_arm_body: tuple[float, float, float]  # m, from base origin to aerodynamic center in body frame
    span_axis_body: tuple[float, float, float]  # unit-ish, body frame
    chord_axis_body: tuple[float, float, float]  # unit-ish, body frame
    span_m: float
    mean_chord_m: float
    incidence_rad: float = 0.0
    cl_alpha_per_rad: float = 3.5
    cd0: float = 0.02
    cd_k: float = 0.08
    alpha_limit_deg: float = 25.0
    deflection_sign: float = 1.0


@dataclass
class TailAeroCfg:
    """Configuration for the five-surface tail aerodynamic model."""

    air_density: float = 1.225
    fixed_horizontal: TailSurfaceCfg = field(default_factory=_default_fixed_horizontal_surface)
    left_elevon: TailSurfaceCfg = field(default_factory=_default_left_elevon_surface)
    right_elevon: TailSurfaceCfg = field(default_factory=_default_right_elevon_surface)
    fixed_vertical: TailSurfaceCfg = field(default_factory=_default_fixed_vertical_surface)
    rudder: TailSurfaceCfg = field(default_factory=_default_rudder_surface)


class TailAeroModel:
    """Compute the body-frame aerodynamic wrench for the latest tail layout."""

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
        device = self.device
        dtype = root_lin_vel_b.dtype

        r_b = _as_vec3(surf.lever_arm_body, device=device, dtype=dtype).view(1, 3).expand(root_lin_vel_b.shape[0], 3)
        span = _normalize(_as_vec3(surf.span_axis_body, device=device, dtype=dtype)).view(1, 3).expand_as(r_b)
        chord0 = _normalize(_as_vec3(surf.chord_axis_body, device=device, dtype=dtype)).view(1, 3).expand_as(r_b)

        v_point = root_lin_vel_b + torch.linalg.cross(root_ang_vel_b, r_b)
        v_proj = v_point - torch.sum(v_point * span, dim=1, keepdim=True) * span
        speed = torch.linalg.norm(v_proj, dim=1)
        v_dir = F.normalize(v_proj, dim=1, eps=1.0e-9)

        ang = torch.as_tensor(float(surf.incidence_rad), device=device, dtype=dtype) + float(surf.deflection_sign) * deflection_rad
        chord = _rotate_about_axis(chord0, span[0], ang)
        v_in = -v_dir

        dot = torch.sum(v_in * chord, dim=1).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        cross_vec = torch.linalg.cross(chord, v_in)
        sin = torch.sum(cross_vec * span, dim=1)
        alpha = torch.atan2(sin, dot)
        alpha_lim = math.radians(float(surf.alpha_limit_deg))
        alpha = torch.clamp(alpha, -alpha_lim, alpha_lim)

        cl = float(surf.cl_alpha_per_rad) * alpha
        cd = float(surf.cd0) + float(surf.cd_k) * (cl * cl)

        q = 0.5 * float(self.cfg.air_density) * (speed * speed)
        lift = (q * float(surf.area) * cl).unsqueeze(1)
        drag = (q * float(surf.area) * cd).unsqueeze(1)

        lift_dir = F.normalize(torch.linalg.cross(v_dir, span), dim=1, eps=1.0e-9)
        drag_dir = -v_dir

        force_b = lift * lift_dir + drag * drag_dir
        torque_b = torch.linalg.cross(r_b, force_b)
        return force_b, torque_b

    def compute_wrench(
        self,
        *,
        root_lin_vel_b: Tensor,
        root_ang_vel_b: Tensor,
        left_elevon_rad: Tensor | None = None,
        right_elevon_rad: Tensor | None = None,
        rudder_rad: Tensor,
        elevator_rad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return net tail wrench in body frame.

        The preferred interface passes split left/right elevon deflections.
        ``elevator_rad`` is retained only as a compatibility fallback and maps
        to symmetric left/right elevon deflection.
        """

        if rudder_rad.ndim != 1:
            raise ValueError("rudder_rad must be a 1D tensor.")
        batch_size = root_lin_vel_b.shape[0]
        if rudder_rad.shape[0] != batch_size:
            raise ValueError("Batch size mismatch in tail aero inputs.")

        if elevator_rad is not None:
            if left_elevon_rad is not None or right_elevon_rad is not None:
                raise ValueError("Pass either split elevon inputs or elevator_rad, not both.")
            left_elevon_rad = elevator_rad
            right_elevon_rad = elevator_rad

        if left_elevon_rad is None or right_elevon_rad is None:
            raise ValueError("left_elevon_rad and right_elevon_rad must be provided.")
        if left_elevon_rad.ndim != 1 or right_elevon_rad.ndim != 1:
            raise ValueError("left_elevon_rad and right_elevon_rad must be 1D tensors.")
        if left_elevon_rad.shape[0] != batch_size or right_elevon_rad.shape[0] != batch_size:
            raise ValueError("Batch size mismatch in tail aero inputs.")

        zero = torch.zeros_like(rudder_rad)
        total_force = torch.zeros_like(root_lin_vel_b)
        total_torque = torch.zeros_like(root_lin_vel_b)

        for surf, deflection in (
            (self.cfg.fixed_horizontal, zero),
            (self.cfg.left_elevon, left_elevon_rad),
            (self.cfg.right_elevon, right_elevon_rad),
            (self.cfg.fixed_vertical, zero),
            (self.cfg.rudder, rudder_rad),
        ):
            force_b, torque_b = self._surface_wrench(
                surf,
                root_lin_vel_b=root_lin_vel_b,
                root_ang_vel_b=root_ang_vel_b,
                deflection_rad=deflection,
            )
            total_force = total_force + force_b
            total_torque = total_torque + torque_b

        return total_force, total_torque

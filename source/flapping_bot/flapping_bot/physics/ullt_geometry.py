"""Geometry and prescribed rigid-motion adapters for standalone ULLT."""

from __future__ import annotations

import math
import torch
import xml.etree.ElementTree as ET
from pathlib import Path

from .ullt_state_space import ULLTGeometry, ULLTMotion
from .wing_geom_csv import _interp1d_torch, infer_span_from_x_mid, load_wing_geom_csv

Tensor = torch.Tensor


def rotation_xyz(angles_rad: Tensor) -> Tensor:
    """Return Rz(yaw) Ry(pitch) Rx(roll), shape (...,3,3), for (...,3) angles."""
    roll, pitch, yaw = angles_rad.unbind(-1)
    cr, sr, cp, sp, cy, sy = roll.cos(), roll.sin(), pitch.cos(), pitch.sin(), yaw.cos(), yaw.sin()
    return torch.stack(
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ),
        -1,
    ).reshape(*angles_rad.shape[:-1], 3, 3)


def elliptic_geometry(
    *,
    num_sections: int = 9,
    aspect_ratio: float = 3.0,
    mid_chord_m: float = 1.0,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ULLTGeometry:
    """Paper Section III: elliptic chord with straight quarter-chord line.

    Cosine-spaced edges; midpoint collocation. Analytic span is pi*AR*c_mid/4.
    """
    if num_sections < 1 or batch_size < 1 or aspect_ratio <= 0 or mid_chord_m <= 0:
        raise ValueError("Positive geometry dimensions required")
    span = math.pi * aspect_ratio * mid_chord_m / 4
    y = -span / 2 * torch.cos(torch.linspace(0, math.pi, num_sections + 1, device=device, dtype=dtype))
    mid = (y[1:] + y[:-1]) / 2

    def point(value: Tensor) -> Tensor:
        return (
            torch.stack((torch.zeros_like(value), value, torch.zeros_like(value)), -1)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

    chord = mid_chord_m * torch.sqrt(1 - (2 * mid / span).square())
    forward = torch.zeros((batch_size, num_sections, 3), device=device, dtype=dtype)
    forward[..., 0] = 1
    return ULLTGeometry(point(y[:-1]), point(y[1:]), point(mid), forward, chord.unsqueeze(0).repeat(batch_size, 1))


def rigid_motion(
    reference: ULLTGeometry,
    *,
    rotation_w_from_reference: Tensor,
    origin_w_m: Tensor,
    origin_velocity_w_mps: Tensor,
    origin_acceleration_w_mps2: Tensor,
    angular_velocity_w_rad_s: Tensor,
    angular_acceleration_w_rad_s2: Tensor,
) -> tuple[ULLTGeometry, ULLTMotion]:
    """Prescribe whole-wing rigid motion through still air; exact spatial kinematics.

    Rotation is (B,3,3); other arguments (B,3). Input geometry coordinates are
    relative to the rotating origin. Returned motion samples at 3/4 chord and
    trailer edges; body rotations are never reinterpreted as incidence offsets.
    """
    rotation = rotation_w_from_reference

    def rotate(value: Tensor) -> Tensor:
        return torch.einsum("bij,bnj->bni", rotation, value)

    def place(value: Tensor) -> Tensor:
        return rotate(value) + origin_w_m[:, None]

    geometry = ULLTGeometry(
        place(reference.edge_start_w_m),
        place(reference.edge_end_w_m),
        place(reference.quarter_chord_w_m),
        rotate(reference.chord_forward_w),
        reference.chord_m,
    )
    omega = angular_velocity_w_rad_s[:, None].expand_as(geometry.quarter_chord_w_m)
    alpha = angular_acceleration_w_rad_s2[:, None].expand_as(omega)

    def flow_at(point: Tensor) -> Tensor:
        return -(origin_velocity_w_mps[:, None] + torch.linalg.cross(omega, point - origin_w_m[:, None]))

    r3 = geometry.three_quarter_chord_w_m - origin_w_m[:, None]
    acceleration = (
        origin_acceleration_w_mps2[:, None]
        + torch.linalg.cross(alpha, r3)
        + torch.linalg.cross(omega, torch.linalg.cross(omega, r3))
    )
    return geometry, ULLTMotion(
        flow_at(geometry.three_quarter_chord_w_m),
        flow_at(geometry.edge_start_w_m),
        flow_at(geometry.edge_end_w_m),
        acceleration,
        omega,
        alpha,
    )


def load_csv_urdf_geometry(
    csv_path: Path,
    urdf_path: Path,
    *,
    sections_per_wing: int = 20,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ULLTGeometry:
    """Place the entire root-local CSV at the URDF's neutral left/right wing poses.

    No root-gap subtraction. Front tube is LE, with section quarter-chord
    points aft of it. Preserve CSV station/chord interpolation; convert chord
    to perpendicular-to-local-lifting-line length for the swept-section theory.
    Returned panel order is right tip to right root, then left root to left tip.
    The empty center contains no panel, but remains in the spatial interaction.
    """
    if sections_per_wing < 1:
        raise ValueError("sections_per_wing must be positive")
    xs, cs, _ = load_wing_geom_csv(Path(csv_path))
    xp, cp = torch.tensor(xs, device=device, dtype=dtype), torch.tensor(cs, device=device, dtype=dtype)
    if bool((cp <= 0).any()) or bool((xp[1:] <= xp[:-1]).any()):
        raise ValueError("CSV must have positive chords and strictly increasing stations")
    radius = infer_span_from_x_mid(xs)
    edges = torch.linspace(0, radius, sections_per_wing + 1, device=device, dtype=dtype)
    mid = (edges[1:] + edges[:-1]) / 2
    c_edge, c_mid = _interp1d_torch(xp, cp, edges), _interp1d_torch(xp, cp, mid)
    root = ET.parse(urdf_path).getroot()
    pieces = []
    for name, sign in (("right_wing", -1), ("left_wing", 1)):
        joint = root.find(f"joint[@name='{name}']")
        if joint is None or joint.find("origin") is None:
            raise ValueError(f"URDF is missing {name} joint origin")
        origin = joint.find("origin")
        position = xp.new_tensor([float(v) for v in origin.attrib["xyz"].split()])
        angles = xp.new_tensor([float(v) for v in origin.attrib.get("rpy", "0 0 0").split()])
        rot = rotation_xyz(angles)
        local_edges = torch.stack((-0.25 * c_edge, sign * edges, torch.zeros_like(edges)), -1)
        local_mid = torch.stack((-0.25 * c_mid, sign * mid, torch.zeros_like(mid)), -1)
        start, end = (local_edges[:-1], local_edges[1:]) if sign > 0 else (local_edges[1:], local_edges[:-1])
        span_axis = (end - start) / torch.linalg.vector_norm(end - start, dim=-1, keepdim=True)
        chord_axis = torch.zeros_like(span_axis)
        chord_axis[:, 0] = 1
        chord_perp = chord_axis - (chord_axis * span_axis).sum(-1, keepdim=True) * span_axis
        projection = torch.linalg.vector_norm(chord_perp, dim=-1)
        chord_axis = chord_perp / projection[:, None]
        values = [
            start @ rot.T + position,
            end @ rot.T + position,
            local_mid @ rot.T + position,
            chord_axis @ rot.T,
            c_mid * projection,
        ]
        if sign < 0:
            values = [v.flip(0) for v in values]
        pieces.append(values)
    return ULLTGeometry(*(torch.cat((pieces[0][i], pieces[1][i]), 0).unsqueeze(0) for i in range(5)))


def uniform_translation(geometry: ULLTGeometry, speed_mps: float) -> ULLTMotion:
    """Constant +x body translation through still air, zero acceleration."""
    flow = torch.zeros_like(geometry.quarter_chord_w_m)
    flow[..., 0] = -speed_mps
    zero = torch.zeros_like(flow)
    return ULLTMotion(flow, flow.clone(), flow.clone(), zero, zero.clone(), zero.clone())


def opposed_flapping_motion(
    reference: ULLTGeometry,
    hinge_origins_w_m: Tensor,
    *,
    time_s: float,
    speed_mps: float,
    frequency_hz: float,
    amplitude_rad: float,
) -> tuple[ULLTGeometry, ULLTMotion]:
    """Prescribe rigid mirrored flapping around URDF +x axes, without twist.

    Reference panel order is right wing then left, with equal panel counts.
    Hinge origins (B,2,3) have the same right/left order. q_left=A*sin(2*pi*f*t),
    q_right=-q_left. Neutral URDF wing rotations are already in reference.
    Translation through still air is along +x. This is not a free-body solve.
    """
    batch, count = reference.chord_m.shape
    if count % 2 or hinge_origins_w_m.shape != (batch, 2, 3):
        raise ValueError("Opposed flapping needs equal right/left panel counts and (B,2,3) hinges")
    if (
        frequency_hz < 0
        or amplitude_rad < 0
        or not all(math.isfinite(v) for v in (time_s, speed_mps, frequency_hz, amplitude_rad))
    ):
        raise ValueError("Invalid prescribed-flapping parameters")
    omega = 2 * math.pi * frequency_hz
    phase = omega * time_s
    geometries, motions = [], []
    for wing, sign in ((0, -1), (1, 1)):
        sl = slice(wing * count // 2, (wing + 1) * count // 2)
        origin = hinge_origins_w_m[:, wing]
        local = ULLTGeometry(
            *(
                v[:, sl] - origin[:, None]
                for v in (reference.edge_start_w_m, reference.edge_end_w_m, reference.quarter_chord_w_m)
            ),
            reference.chord_forward_w[:, sl],
            reference.chord_m[:, sl],
        )
        angles = reference.chord_m.new_zeros((batch, 3))
        angles[:, 0] = sign * amplitude_rad * math.sin(phase)
        angular_velocity, angular_acceleration = torch.zeros_like(angles), torch.zeros_like(angles)
        angular_velocity[:, 0] = sign * amplitude_rad * omega * math.cos(phase)
        angular_acceleration[:, 0] = -sign * amplitude_rad * omega**2 * math.sin(phase)
        position, velocity = origin.clone(), torch.zeros_like(origin)
        position[:, 0] += speed_mps * time_s
        velocity[:, 0] = speed_mps
        g, m = rigid_motion(
            local,
            rotation_w_from_reference=rotation_xyz(angles),
            origin_w_m=position,
            origin_velocity_w_mps=velocity,
            origin_acceleration_w_mps2=torch.zeros_like(origin),
            angular_velocity_w_rad_s=angular_velocity,
            angular_acceleration_w_rad_s2=angular_acceleration,
        )
        geometries.append(g)
        motions.append(m)
    return (
        ULLTGeometry(
            *(torch.cat((getattr(geometries[0], name), getattr(geometries[1], name)), 1) for name in vars(reference))
        ),
        ULLTMotion(
            *(torch.cat((getattr(motions[0], name), getattr(motions[1], name)), 1) for name in vars(motions[0]))
        ),
    )

"""Circle path geometry helpers for loiter guidance."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def navigate_circle(
    center_xy: Tensor,
    radius_m: float,
    vehicle_pos: Tensor,
    *,
    clockwise: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return path tangent, closest point, and radial error for circle following."""
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive.")

    center = center_xy.view(1, 2)
    rel = vehicle_pos - center
    dist = torch.linalg.norm(rel, dim=1, keepdim=True)
    dist_safe = torch.clamp(dist, min=1.0e-6)

    radial_unit = rel / dist_safe
    default_radial = torch.tensor([1.0, 0.0], dtype=vehicle_pos.dtype, device=vehicle_pos.device).view(1, 2)
    valid = dist > 1.0e-6
    radial_unit = torch.where(valid, radial_unit, default_radial.expand_as(radial_unit))

    closest_point = center + radial_unit * float(radius_m)

    if clockwise:
        unit_tangent = torch.stack((radial_unit[:, 1], -radial_unit[:, 0]), dim=1)
    else:
        unit_tangent = torch.stack((-radial_unit[:, 1], radial_unit[:, 0]), dim=1)

    radial_error = dist.squeeze(1) - float(radius_m)
    return unit_tangent, closest_point, radial_error

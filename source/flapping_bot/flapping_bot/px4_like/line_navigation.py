"""Line-path projection helpers."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def navigate_line(point_on_line_1: Tensor, point_on_line_2: Tensor, vehicle_pos: Tensor) -> tuple[Tensor, Tensor]:
    """Return path tangent and closest points for line following.

    Args:
        point_on_line_1: (2,) line start point.
        point_on_line_2: (2,) line end point.
        vehicle_pos: (N, 2) vehicle positions.
    """
    line_segment = point_on_line_2 - point_on_line_1
    seg_norm = torch.linalg.norm(line_segment)
    if float(seg_norm.item()) <= 1.0e-9:
        raise ValueError("Line segment must have non-zero length.")

    unit_tangent = (line_segment / seg_norm).view(1, 2).expand(vehicle_pos.shape[0], 2)
    point_1_to_vehicle = vehicle_pos - point_on_line_1.view(1, 2)
    projection = torch.sum(point_1_to_vehicle * unit_tangent, dim=1, keepdim=True)
    closest_point = point_on_line_1.view(1, 2) + projection * unit_tangent
    return unit_tangent, closest_point

"""Helpers for equivalent wing application points used in wrench aggregation."""

from __future__ import annotations

import torch

from .qsm_wang2016 import WingGeometry

Tensor = torch.Tensor


def compute_area_weighted_quarter_chord_link_points(wing_geom: WingGeometry) -> Tensor:
    """Return left/right equivalent wing application points in wing-link frames."""
    strip_area = wing_geom.c * wing_geom.dx
    area_sum = torch.clamp(torch.sum(strip_area), min=1.0e-9)
    weights = strip_area / area_sum

    span_center = torch.sum(weights * wing_geom.x_mid)
    # Quarter-chord is defined from the leading edge, while WingGeometry.d_hat locates the
    # pitching axis from the leading edge. In the wing link frame +x points toward the leading
    # edge, so the application point relative to the pitching axis is (d_hat - 0.25) * c.
    quarter_chord_x = torch.sum(weights * ((wing_geom.d_hat - 0.25) * wing_geom.c))
    zero = torch.zeros((), device=wing_geom.x_mid.device, dtype=wing_geom.x_mid.dtype)

    left = torch.stack((quarter_chord_x, span_center, zero))
    right = torch.stack((quarter_chord_x, -span_center, zero))
    return torch.stack((left, right), dim=0)

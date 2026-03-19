"""Generic PX4-like path-tracking controller skeleton."""

from __future__ import annotations

import torch

from .straight_line_controller import PX4LikeStraightLineControllerCfg


class PX4LikePathTrackingControllerCfg(PX4LikeStraightLineControllerCfg):
    """Configuration for the generic path-tracking controller."""


class PX4LikePathTrackingController:
    """Minimal generic controller that consumes unified path queries."""

    def __init__(self, cfg: PX4LikePathTrackingControllerCfg, device: torch.device):
        self.cfg = cfg
        self.device = device

    def compute_actions_from_query(self, **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return placeholder actions and diagnostics for path tracking."""
        pos_local = kwargs.get("pos_local")
        batch_size = 1 if pos_local is None else int(pos_local.shape[0])
        actions = torch.zeros((batch_size, 4), device=self.device)
        diag = {"course_sp": torch.zeros((batch_size,), device=self.device)}
        return actions, diag

from __future__ import annotations

from dataclasses import dataclass

from .mission_primitives import Mission


@dataclass(frozen=True)
class PathManagerCfg:
    """Configuration for the path manager skeleton."""

    max_roll_deg: float = 35.0
    max_flight_path_angle_deg: float = 10.0


@dataclass(frozen=True)
class PathQuery:
    """Minimal path-query output for downstream controllers."""

    curvature_m_inv: float
    progress_s: float
    preview_points_xyz: list[tuple[float, float, float]]


class PathManager:
    """Minimal path manager skeleton for generic path tracking."""

    def __init__(self, cfg: PathManagerCfg, mission: Mission):
        self.cfg = cfg
        self.mission = mission

    def query(
        self,
        *,
        position_xy: tuple[float, float],
        altitude_m: float,
        speed_mps: float,
    ) -> PathQuery:
        del position_xy, altitude_m, speed_mps
        return PathQuery(
            curvature_m_inv=0.0,
            progress_s=0.0,
            preview_points_xyz=[(0.0, 0.0, 0.0)] * 5,
        )

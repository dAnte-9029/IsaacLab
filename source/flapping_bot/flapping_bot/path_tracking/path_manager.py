from __future__ import annotations

from dataclasses import dataclass
import math

from .mission_primitives import Mission


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_to_2pi(angle_rad: float) -> float:
    wrapped = math.fmod(angle_rad, 2.0 * math.pi)
    if wrapped < 0.0:
        wrapped += 2.0 * math.pi
    return wrapped


def _cross2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _dot2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _left_normal(unit_tangent: tuple[float, float]) -> tuple[float, float]:
    return (-unit_tangent[1], unit_tangent[0])


@dataclass(frozen=True)
class PathManagerCfg:
    """Configuration for the primitive path manager."""

    max_roll_deg: float = 35.0
    max_flight_path_angle_deg: float = 10.0
    straight_length_m: float = 40.0
    turn_radius_m: float = 20.0
    loiter_radius_m: float = 20.0
    turn_sweep_deg: float = 90.0
    loiter_turns: float = 1.0
    climb_delta_m: float = 3.0
    initial_position_xy: tuple[float, float] = (0.0, 0.0)
    initial_heading_deg: float = 0.0
    initial_altitude_m: float = 10.0
    preview_times_s: tuple[float, ...] = (0.2, 0.4, 0.7, 1.2, 2.0)


@dataclass(frozen=True)
class PathQuery:
    """Local path-query output for controllers and RL observations."""

    closest_point_xyz: tuple[float, float, float]
    tangent_xy: tuple[float, float]
    curvature_m_inv: float
    progress_s: float
    height_sp_m: float
    lateral_error_m: float
    preview_points_xyz: list[tuple[float, float, float]]


@dataclass(frozen=True)
class _StraightSegment:
    s_start: float
    s_end: float
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    start_alt_m: float
    end_alt_m: float
    unit_tangent_xy: tuple[float, float]

    @property
    def length_m(self) -> float:
        return self.s_end - self.s_start


@dataclass(frozen=True)
class _ArcSegment:
    kind: str
    s_start: float
    s_end: float
    center_xy: tuple[float, float]
    radius_m: float
    start_angle_rad: float
    sweep_angle_rad: float
    turn_direction: int
    altitude_m: float

    @property
    def arc_length_m(self) -> float:
        return self.s_end - self.s_start


@dataclass(frozen=True)
class _QueryCandidate:
    closest_point_xyz: tuple[float, float, float]
    tangent_xy: tuple[float, float]
    curvature_m_inv: float
    progress_s: float
    lateral_error_m: float
    distance_sq_xy: float


class PathManager:
    """Primitive-aware path manager for synthetic path-tracking missions."""

    _BACKWARD_PROGRESS_TOL_M = 2.0
    _BACKWARD_PROGRESS_PENALTY = 1.0e3

    def __init__(self, cfg: PathManagerCfg, mission: Mission):
        if len(cfg.preview_times_s) != 5:
            raise ValueError("preview_times_s must contain exactly five look-ahead times.")
        if len(mission.segments) == 0:
            raise ValueError("mission must contain at least one segment.")

        self.cfg = cfg
        self.mission = mission
        self._segments = self._build_segments()
        self.total_length_m = self._segments[-1].s_end
        self._last_progress_s = 0.0

    def reset(self) -> None:
        """Reset internal progress hysteresis."""
        self._last_progress_s = 0.0

    def query(
        self,
        *,
        position_xy: tuple[float, float],
        altitude_m: float,
        speed_mps: float,
    ) -> PathQuery:
        best: _QueryCandidate | None = None
        best_score = float("inf")

        for segment in self._segments:
            candidate = self._query_segment(segment, position_xy)
            backward_m = max(0.0, self._last_progress_s - candidate.progress_s - self._BACKWARD_PROGRESS_TOL_M)
            score = candidate.distance_sq_xy + self._BACKWARD_PROGRESS_PENALTY * backward_m * backward_m
            if score < best_score or (
                math.isclose(score, best_score, rel_tol=0.0, abs_tol=1.0e-9)
                and best is not None
                and candidate.progress_s > best.progress_s
            ):
                best = candidate
                best_score = score

        if best is None:
            raise RuntimeError("path query failed to evaluate any segment.")

        progress_s = max(self._last_progress_s, best.progress_s)
        self._last_progress_s = progress_s
        closest_point_xyz = self.sample(progress_s)
        preview_speed_mps = max(float(speed_mps), 1.0)
        preview_points_xyz = [
            self.sample(min(progress_s + preview_speed_mps * preview_time_s, self.total_length_m))
            for preview_time_s in self.cfg.preview_times_s
        ]

        return PathQuery(
            closest_point_xyz=closest_point_xyz,
            tangent_xy=best.tangent_xy,
            curvature_m_inv=best.curvature_m_inv,
            progress_s=progress_s,
            height_sp_m=closest_point_xyz[2],
            lateral_error_m=best.lateral_error_m,
            preview_points_xyz=preview_points_xyz,
        )

    def sample(self, progress_s: float) -> tuple[float, float, float]:
        """Sample a point on the path at scalar progress ``s``."""
        clamped_progress = _clamp(float(progress_s), 0.0, self.total_length_m)
        for segment in self._segments:
            if clamped_progress <= segment.s_end + 1.0e-9:
                return self._sample_segment(segment, clamped_progress)
        return self._sample_segment(self._segments[-1], self.total_length_m)

    def _build_segments(self) -> list[_StraightSegment | _ArcSegment]:
        heading_rad = math.radians(float(self.cfg.initial_heading_deg))
        position_xy = tuple(float(v) for v in self.cfg.initial_position_xy)
        altitude_m = float(self.cfg.initial_altitude_m)
        progress_s = 0.0
        climb_sign = 1.0
        turn_sign = 1
        segments: list[_StraightSegment | _ArcSegment] = []

        for mission_segment in self.mission.segments:
            if mission_segment.kind == "straight":
                length_m = max(float(self.cfg.straight_length_m), 1.0e-6)
                unit_tangent_xy = (math.cos(heading_rad), math.sin(heading_rad))
                end_xy = (
                    position_xy[0] + length_m * unit_tangent_xy[0],
                    position_xy[1] + length_m * unit_tangent_xy[1],
                )
                max_alt_delta_m = length_m * math.tan(math.radians(float(self.cfg.max_flight_path_angle_deg)))
                if mission_segment.altitude_changes:
                    altitude_delta_m = climb_sign * min(float(self.cfg.climb_delta_m), max_alt_delta_m)
                    climb_sign *= -1.0
                else:
                    altitude_delta_m = 0.0
                end_altitude_m = altitude_m + altitude_delta_m
                segments.append(
                    _StraightSegment(
                        s_start=progress_s,
                        s_end=progress_s + length_m,
                        start_xy=position_xy,
                        end_xy=end_xy,
                        start_alt_m=altitude_m,
                        end_alt_m=end_altitude_m,
                        unit_tangent_xy=unit_tangent_xy,
                    )
                )
                progress_s += length_m
                position_xy = end_xy
                altitude_m = end_altitude_m
                continue

            if mission_segment.kind not in {"turn", "loiter"}:
                raise ValueError(f"unsupported mission segment kind: {mission_segment.kind}")

            radius_m = float(self.cfg.turn_radius_m if mission_segment.kind == "turn" else self.cfg.loiter_radius_m)
            radius_m = max(radius_m, 1.0e-6)
            unit_tangent_xy = (math.cos(heading_rad), math.sin(heading_rad))
            left_normal_xy = _left_normal(unit_tangent_xy)
            direction = turn_sign
            turn_sign *= -1
            center_xy = (
                position_xy[0] + direction * radius_m * left_normal_xy[0],
                position_xy[1] + direction * radius_m * left_normal_xy[1],
            )
            start_angle_rad = math.atan2(position_xy[1] - center_xy[1], position_xy[0] - center_xy[0])
            sweep_angle_rad = direction * (
                math.radians(float(self.cfg.turn_sweep_deg))
                if mission_segment.kind == "turn"
                else 2.0 * math.pi * float(self.cfg.loiter_turns)
            )
            arc_length_m = abs(sweep_angle_rad) * radius_m
            segments.append(
                _ArcSegment(
                    kind=mission_segment.kind,
                    s_start=progress_s,
                    s_end=progress_s + arc_length_m,
                    center_xy=center_xy,
                    radius_m=radius_m,
                    start_angle_rad=start_angle_rad,
                    sweep_angle_rad=sweep_angle_rad,
                    turn_direction=direction,
                    altitude_m=altitude_m,
                )
            )
            progress_s += arc_length_m
            end_angle_rad = start_angle_rad + sweep_angle_rad
            position_xy = (
                center_xy[0] + radius_m * math.cos(end_angle_rad),
                center_xy[1] + radius_m * math.sin(end_angle_rad),
            )
            heading_rad = heading_rad + sweep_angle_rad

        return segments

    def _query_segment(
        self,
        segment: _StraightSegment | _ArcSegment,
        position_xy: tuple[float, float],
    ) -> _QueryCandidate:
        if isinstance(segment, _StraightSegment):
            rel_xy = (position_xy[0] - segment.start_xy[0], position_xy[1] - segment.start_xy[1])
            along_m = _clamp(_dot2d(rel_xy, segment.unit_tangent_xy), 0.0, segment.length_m)
            frac = along_m / max(segment.length_m, 1.0e-6)
            closest_xy = (
                segment.start_xy[0] + along_m * segment.unit_tangent_xy[0],
                segment.start_xy[1] + along_m * segment.unit_tangent_xy[1],
            )
            closest_alt_m = segment.start_alt_m + frac * (segment.end_alt_m - segment.start_alt_m)
            offset_xy = (position_xy[0] - closest_xy[0], position_xy[1] - closest_xy[1])
            return _QueryCandidate(
                closest_point_xyz=(closest_xy[0], closest_xy[1], closest_alt_m),
                tangent_xy=segment.unit_tangent_xy,
                curvature_m_inv=0.0,
                progress_s=segment.s_start + along_m,
                lateral_error_m=_cross2d(segment.unit_tangent_xy, offset_xy),
                distance_sq_xy=_dot2d(offset_xy, offset_xy),
            )

        raw_angle_rad = math.atan2(position_xy[1] - segment.center_xy[1], position_xy[0] - segment.center_xy[0])
        delta_candidates_rad = self._arc_progress_candidates(segment, raw_angle_rad)
        target_progress_rad = _clamp(
            (self._last_progress_s - segment.s_start) / max(segment.radius_m, 1.0e-6),
            0.0,
            abs(segment.sweep_angle_rad),
        )
        delta_rad = min(delta_candidates_rad, key=lambda value: (abs(value - target_progress_rad), -value))
        angle_on_path_rad = segment.start_angle_rad + segment.turn_direction * delta_rad
        closest_xy = (
            segment.center_xy[0] + segment.radius_m * math.cos(angle_on_path_rad),
            segment.center_xy[1] + segment.radius_m * math.sin(angle_on_path_rad),
        )
        tangent_xy = (
            -segment.turn_direction * math.sin(angle_on_path_rad),
            segment.turn_direction * math.cos(angle_on_path_rad),
        )
        offset_xy = (position_xy[0] - closest_xy[0], position_xy[1] - closest_xy[1])
        return _QueryCandidate(
            closest_point_xyz=(closest_xy[0], closest_xy[1], segment.altitude_m),
            tangent_xy=tangent_xy,
            curvature_m_inv=segment.turn_direction / max(segment.radius_m, 1.0e-6),
            progress_s=segment.s_start + segment.radius_m * delta_rad,
            lateral_error_m=_cross2d(tangent_xy, offset_xy),
            distance_sq_xy=_dot2d(offset_xy, offset_xy),
        )

    def _arc_progress_candidates(self, segment: _ArcSegment, raw_angle_rad: float) -> list[float]:
        total_sweep_rad = abs(segment.sweep_angle_rad)
        if segment.turn_direction > 0:
            base_delta_rad = _wrap_to_2pi(raw_angle_rad - segment.start_angle_rad)
        else:
            base_delta_rad = _wrap_to_2pi(segment.start_angle_rad - raw_angle_rad)

        if total_sweep_rad < 2.0 * math.pi - 1.0e-9:
            return [_clamp(base_delta_rad, 0.0, total_sweep_rad)]

        candidates: list[float] = []
        turns = max(1, int(math.ceil(total_sweep_rad / (2.0 * math.pi))))
        for turn_idx in range(turns + 1):
            candidate_rad = base_delta_rad + 2.0 * math.pi * turn_idx
            if candidate_rad <= total_sweep_rad + 1.0e-9:
                candidates.append(candidate_rad)
        if not candidates:
            candidates.append(_clamp(base_delta_rad, 0.0, total_sweep_rad))
        return candidates

    def _sample_segment(self, segment: _StraightSegment | _ArcSegment, progress_s: float) -> tuple[float, float, float]:
        if isinstance(segment, _StraightSegment):
            along_m = _clamp(progress_s - segment.s_start, 0.0, segment.length_m)
            frac = along_m / max(segment.length_m, 1.0e-6)
            return (
                segment.start_xy[0] + along_m * segment.unit_tangent_xy[0],
                segment.start_xy[1] + along_m * segment.unit_tangent_xy[1],
                segment.start_alt_m + frac * (segment.end_alt_m - segment.start_alt_m),
            )

        arc_progress_rad = _clamp((progress_s - segment.s_start) / max(segment.radius_m, 1.0e-6), 0.0, abs(segment.sweep_angle_rad))
        angle_rad = segment.start_angle_rad + segment.turn_direction * arc_progress_rad
        return (
            segment.center_xy[0] + segment.radius_m * math.cos(angle_rad),
            segment.center_xy[1] + segment.radius_m * math.sin(angle_rad),
            segment.altitude_m,
        )

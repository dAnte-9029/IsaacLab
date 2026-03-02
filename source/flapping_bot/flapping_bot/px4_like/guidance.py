"""PX4 NPFG-like guidance blocks in torch.

This module ports the key formulas used in PX4's fixed-wing NPFG stack:
- Directional guidance from path geometry to course/lateral-feedforward setpoints.
- Airspeed-direction heading controller to map heading error to lateral acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

Tensor = torch.Tensor


def _cross2d(a: Tensor, b: Tensor) -> Tensor:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _dot2d(a: Tensor, b: Tensor) -> Tensor:
    return torch.sum(a * b, dim=1)


@dataclass
class DirectionalGuidanceSettings:
    """Configuration parameters for directional guidance."""

    period: float = 10.0
    damping: float = 0.7071
    enable_period_lb: bool = True
    enable_period_ub: bool = True
    roll_time_const: float = 0.35
    switch_distance_multiplier: float = 0.32
    period_safety_factor: float = 1.5


@dataclass
class DirectionalGuidanceOutput:
    """Guidance outputs for one control step."""

    course_setpoint: Tensor
    lateral_acceleration_feedforward: Tensor
    signed_track_error: Tensor
    track_error_bound: Tensor


class DirectionalGuidance:
    """Torch implementation of PX4 directional guidance formulas."""

    NPFG_EPSILON = 1.0e-6
    MIN_RADIUS = 0.5
    AIRSPEED_BUFFER = 1.5

    def __init__(self, settings: DirectionalGuidanceSettings | None = None):
        self.settings = settings or DirectionalGuidanceSettings()

    def guide_to_path(
        self,
        curr_pos_local: Tensor,
        ground_vel: Tensor,
        wind_vel: Tensor,
        unit_path_tangent: Tensor,
        position_on_path: Tensor,
        path_curvature: float | Tensor,
    ) -> DirectionalGuidanceOutput:
        """Compute NPFG guidance outputs.

        Args:
            curr_pos_local: (N, 2) local position.
            ground_vel: (N, 2) ground-relative velocity.
            wind_vel: (N, 2) wind velocity estimate in local frame.
            unit_path_tangent: (N, 2) unit tangent of target path.
            position_on_path: (N, 2) closest point on path.
            path_curvature: scalar or (N,) path curvature.
        """
        ground_speed = torch.linalg.norm(ground_vel, dim=1)
        air_vel = ground_vel - wind_vel
        airspeed = torch.linalg.norm(air_vel, dim=1)
        wind_speed = torch.linalg.norm(wind_vel, dim=1)

        if isinstance(path_curvature, Tensor):
            curvature = path_curvature
        else:
            curvature = torch.full_like(ground_speed, float(path_curvature))

        path_pos_to_vehicle = curr_pos_local - position_on_path
        signed_track_error = _cross2d(unit_path_tangent, path_pos_to_vehicle)

        wind_cross_upt = _cross2d(wind_vel, unit_path_tangent)
        wind_dot_upt = _dot2d(wind_vel, unit_path_tangent)
        feas_on_track = self._bearing_feasibility(wind_cross_upt, wind_dot_upt, airspeed, wind_speed)

        track_error = torch.abs(signed_track_error)
        adapted_period = self._adapt_period(
            ground_speed=ground_speed,
            airspeed=airspeed,
            wind_speed=wind_speed,
            track_error=track_error,
            path_curvature=curvature,
            wind_vel=wind_vel,
            unit_path_tangent=unit_path_tangent,
            feas_on_track=feas_on_track,
        )
        time_const = adapted_period * float(self.settings.damping)
        track_error_bound = self._track_error_bound(ground_speed, time_const)
        normalized_track_error = torch.clamp(track_error / torch.clamp(track_error_bound, min=self.NPFG_EPSILON), 0.0, 1.0)

        look_ahead_ang = 0.5 * math.pi * torch.square(normalized_track_error - 1.0)
        track_proximity = torch.square(torch.sin(look_ahead_ang))

        bearing_vec = self._bearing_vector(unit_path_tangent, look_ahead_ang, signed_track_error)

        wind_cross_bearing = _cross2d(wind_vel, bearing_vec)
        wind_dot_bearing = _dot2d(wind_vel, bearing_vec)
        feas = self._bearing_feasibility(wind_cross_bearing, wind_dot_bearing, airspeed, wind_speed)
        feas_combined = feas * feas_on_track

        lateral_accel_ff = self._lateral_accel_ff(
            unit_path_tangent=unit_path_tangent,
            ground_vel=ground_vel,
            wind_dot_upt=wind_dot_upt,
            wind_cross_upt=wind_cross_upt,
            airspeed=airspeed,
            signed_track_error=signed_track_error,
            path_curvature=curvature,
        )
        lateral_accel_ff = lateral_accel_ff * feas_combined * track_proximity

        course_sp = torch.atan2(bearing_vec[:, 1], bearing_vec[:, 0])

        return DirectionalGuidanceOutput(
            course_setpoint=course_sp,
            lateral_acceleration_feedforward=lateral_accel_ff,
            signed_track_error=signed_track_error,
            track_error_bound=track_error_bound,
        )

    def switch_distance(self, wp_radius: Tensor, track_error_bound: Tensor) -> Tensor:
        """Equivalent of PX4 `switchDistance(wp_radius)` after guidance update."""
        return torch.minimum(
            wp_radius,
            float(self.settings.switch_distance_multiplier) * track_error_bound,
        )

    def _adapt_period(
        self,
        *,
        ground_speed: Tensor,
        airspeed: Tensor,
        wind_speed: Tensor,
        track_error: Tensor,
        path_curvature: Tensor,
        wind_vel: Tensor,
        unit_path_tangent: Tensor,
        feas_on_track: Tensor,
    ) -> Tensor:
        period = ground_speed.new_full(ground_speed.shape, float(self.settings.period))
        if not self.settings.enable_period_lb or float(self.settings.roll_time_const) <= self.NPFG_EPSILON:
            return period

        air_turn_rate = torch.abs(path_curvature * airspeed)
        wind_factor = self._wind_factor(airspeed, wind_speed)

        period_lb_zero_curv = self._period_lower_bound(
            air_turn_rate=torch.zeros_like(air_turn_rate),
            wind_factor=wind_factor,
            feas_on_track=feas_on_track,
        ) * float(self.settings.period_safety_factor)

        period_lb = self._period_lower_bound(
            air_turn_rate=air_turn_rate,
            wind_factor=wind_factor,
            feas_on_track=feas_on_track,
        ) * float(self.settings.period_safety_factor)

        time_const = period_lb_zero_curv * float(self.settings.damping)
        track_error_bound = self._track_error_bound(ground_speed, time_const)
        normalized_track_error = torch.clamp(track_error / torch.clamp(track_error_bound, min=self.NPFG_EPSILON), 0.0, 1.0)
        look_ahead_ang = 0.5 * math.pi * torch.square(normalized_track_error - 1.0)
        track_proximity = torch.square(torch.sin(look_ahead_ang))

        period_lb = period_lb * track_proximity + (1.0 - track_proximity) * period_lb_zero_curv
        period = torch.maximum(period, period_lb)

        if self.settings.enable_period_ub:
            period_ub = self._period_upper_bound(air_turn_rate, wind_factor, feas_on_track)
            mask = torch.isfinite(period_ub) & (period > period_ub)
            period_adapted = torch.maximum(period_lb, period_ub)
            period_blend = period_adapted * track_proximity + (1.0 - track_proximity) * period
            period = torch.where(mask, period_blend, period)

        return period

    def _wind_factor(self, airspeed: Tensor, wind_speed: Tensor) -> Tensor:
        invalid = (wind_speed > airspeed) | (airspeed < self.NPFG_EPSILON)
        ratio = torch.clamp(wind_speed / torch.clamp(airspeed, min=self.NPFG_EPSILON), min=0.0, max=1.0)
        approx = 2.0 * (1.0 - torch.sqrt(torch.clamp(1.0 - ratio, min=0.0)))
        return torch.where(invalid, airspeed.new_full(airspeed.shape, 2.0), approx)

    def _period_upper_bound(self, air_turn_rate: Tensor, wind_factor: Tensor, feas_on_track: Tensor) -> Tensor:
        denom = air_turn_rate * wind_factor * feas_on_track
        out = air_turn_rate.new_full(air_turn_rate.shape, float("inf"))
        valid = denom > self.NPFG_EPSILON
        value = 4.0 * math.pi * float(self.settings.damping) / torch.clamp(denom, min=self.NPFG_EPSILON)
        return torch.where(valid, value, out)

    def _period_lower_bound(self, air_turn_rate: Tensor, wind_factor: Tensor, feas_on_track: Tensor) -> Tensor:
        period_lb = math.pi * float(self.settings.roll_time_const) / max(float(self.settings.damping), self.NPFG_EPSILON)
        base = air_turn_rate.new_full(air_turn_rate.shape, period_lb)

        if float(self.settings.damping) < 0.5:
            return base

        windy_curved = 4.0 * math.pi * float(self.settings.roll_time_const) * float(self.settings.damping)
        blend = windy_curved * feas_on_track + (1.0 - feas_on_track) * period_lb
        active = (air_turn_rate * wind_factor) >= self.NPFG_EPSILON
        return torch.where(active, blend, base)

    def _track_error_bound(self, ground_speed: Tensor, time_const: Tensor) -> Tensor:
        fast = ground_speed > 1.0
        bound_fast = ground_speed * time_const
        bound_slow = 0.5 * time_const * (ground_speed * ground_speed + 1.0)
        return torch.where(fast, bound_fast, bound_slow)

    def _bearing_vector(self, unit_path_tangent: Tensor, look_ahead_ang: Tensor, signed_track_error: Tensor) -> Tensor:
        cos_laa = torch.cos(look_ahead_ang)
        sin_laa = torch.sin(look_ahead_ang)
        unit_path_normal = torch.stack((-unit_path_tangent[:, 1], unit_path_tangent[:, 0]), dim=1)
        sign = torch.where(signed_track_error < 0.0, -1.0, 1.0)
        unit_track_error = -(sign.unsqueeze(1)) * unit_path_normal
        return cos_laa.unsqueeze(1) * unit_track_error + sin_laa.unsqueeze(1) * unit_path_tangent

    def _bearing_feasibility(
        self,
        wind_cross_bearing: Tensor,
        wind_dot_bearing: Tensor,
        airspeed: Tensor,
        wind_speed: Tensor,
    ) -> Tensor:
        wind_cross = torch.where(wind_dot_bearing < 0.0, wind_speed, torch.abs(wind_cross_bearing))
        arg = torch.clamp((airspeed - wind_cross) / float(self.AIRSPEED_BUFFER), min=0.0, max=1.0)
        sin_arg = torch.sin(0.5 * math.pi * arg)
        return sin_arg * sin_arg

    def _lateral_accel_ff(
        self,
        *,
        unit_path_tangent: Tensor,
        ground_vel: Tensor,
        wind_dot_upt: Tensor,
        wind_cross_upt: Tensor,
        airspeed: Tensor,
        signed_track_error: Tensor,
        path_curvature: Tensor,
    ) -> Tensor:
        denom = torch.maximum(
            1.0 - path_curvature * signed_track_error,
            torch.abs(path_curvature) * float(self.MIN_RADIUS),
        )
        path_frame_curvature = path_curvature / torch.clamp(denom, min=self.NPFG_EPSILON)
        tangent_ground_speed = torch.clamp(_dot2d(ground_vel, unit_path_tangent), min=0.0)
        path_frame_rate = path_frame_curvature * tangent_ground_speed
        proj_air = self._project_airspeed_on_bearing(airspeed, wind_cross_upt)
        speed_ratio = 1.0 + wind_dot_upt / torch.clamp(proj_air, min=self.NPFG_EPSILON)
        return airspeed * speed_ratio * path_frame_rate

    def _project_airspeed_on_bearing(self, airspeed: Tensor, wind_cross_bearing: Tensor) -> Tensor:
        return torch.sqrt(torch.clamp(airspeed * airspeed - wind_cross_bearing * wind_cross_bearing, min=0.0))


@dataclass
class AirspeedDirectionControllerSettings:
    """Heading-to-lateral-acceleration controller settings."""

    p_gain: float = 0.8885


class AirspeedDirectionController:
    """Compute lateral acceleration from heading error."""

    def __init__(self, settings: AirspeedDirectionControllerSettings | None = None):
        self.settings = settings or AirspeedDirectionControllerSettings()

    def control_heading(self, heading_sp: Tensor, heading: Tensor, airspeed: Tensor) -> Tensor:
        airspeed_vector = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1) * airspeed.unsqueeze(1)
        airspeed_sp_unit = torch.stack((torch.cos(heading_sp), torch.sin(heading_sp)), dim=1)

        dot_air_vel_err = _dot2d(airspeed_vector, airspeed_sp_unit)
        cross_air_vel_err = _cross2d(airspeed_vector, airspeed_sp_unit)

        saturated = float(self.settings.p_gain) * torch.where(cross_air_vel_err < 0.0, -airspeed, airspeed)
        linear = float(self.settings.p_gain) * cross_air_vel_err
        return torch.where(dot_air_vel_err < 0.0, saturated, linear)

"""PX4-like straight-line controller for the flapping straight-flight environment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .guidance import (
    AirspeedDirectionController,
    AirspeedDirectionControllerSettings,
    DirectionalGuidance,
    DirectionalGuidanceSettings,
)
from .line_navigation import navigate_line

Tensor = torch.Tensor


def _wrap_pi(angle_rad: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))


@dataclass
class PX4LikeStraightLineControllerCfg:
    """Controller gains and mission settings."""

    line_start_xy: tuple[float, float] = (0.0, 0.0)
    line_end_xy: tuple[float, float] = (120.0, 0.0)
    wind_xy: tuple[float, float] = (0.0, 0.0)

    height_sp_m: float = 10.0
    pitch_trim_deg: float = 13.0
    max_roll_deg: float = 35.0
    max_pitch_up_deg: float = 25.0
    max_pitch_down_deg: float = 20.0

    roll_kp: float = 3.2
    roll_kd: float = 0.18
    pitch_kp: float = 2.5
    pitch_kd: float = 0.16
    yaw_kp: float = 1.3
    yaw_kd: float = 0.08
    height_kp: float = 0.06
    height_rate_kd: float = 0.02

    freq_trim_hz: float = 3.8
    min_flap_hz: float = 3.0
    max_flap_hz: float = 4.6
    enable_speed_hold: bool = False
    speed_sp_mps: float = 7.0
    speed_kp_hz_per_mps: float = 0.08

    guidance_period_s: float = 10.0
    guidance_damping: float = 0.7071
    guidance_roll_time_const_s: float = 0.35
    heading_p_gain: float = 0.8885


class PX4LikeStraightLineController:
    """Compute environment actions from state using PX4-inspired control structure."""

    def __init__(self, cfg: PX4LikeStraightLineControllerCfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._line_start = torch.tensor(cfg.line_start_xy, dtype=torch.float32, device=device)
        self._line_end = torch.tensor(cfg.line_end_xy, dtype=torch.float32, device=device)
        self._wind_xy = torch.tensor(cfg.wind_xy, dtype=torch.float32, device=device).view(1, 2)

        self._directional_guidance = DirectionalGuidance(
            DirectionalGuidanceSettings(
                period=float(cfg.guidance_period_s),
                damping=float(cfg.guidance_damping),
                enable_period_lb=True,
                enable_period_ub=True,
                roll_time_const=float(cfg.guidance_roll_time_const_s),
            )
        )
        self._heading_controller = AirspeedDirectionController(
            AirspeedDirectionControllerSettings(p_gain=float(cfg.heading_p_gain))
        )

    def compute_actions(
        self,
        *,
        pos_local: Tensor,
        ground_vel_local: Tensor,
        roll: Tensor,
        pitch: Tensor,
        yaw: Tensor,
        ang_vel_body: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Map current vehicle state to `[freq, elevator, rudder, roll]` actions."""
        pos_xy = pos_local[:, 0:2]
        vel_xy = ground_vel_local[:, 0:2]
        wind_xy = self._wind_xy.expand(pos_xy.shape[0], 2).to(dtype=pos_xy.dtype)

        unit_tangent, closest_point = navigate_line(self._line_start.to(pos_xy.dtype), self._line_end.to(pos_xy.dtype), pos_xy)
        guidance = self._directional_guidance.guide_to_path(
            curr_pos_local=pos_xy,
            ground_vel=vel_xy,
            wind_vel=wind_xy,
            unit_path_tangent=unit_tangent,
            position_on_path=closest_point,
            path_curvature=0.0,
        )

        air_vel_xy = vel_xy - wind_xy
        airspeed = torch.linalg.norm(air_vel_xy, dim=1)
        heading = torch.atan2(air_vel_xy[:, 1], air_vel_xy[:, 0])
        lateral_accel_fb = self._heading_controller.control_heading(guidance.course_setpoint, heading, airspeed)
        lateral_accel_sp = lateral_accel_fb + guidance.lateral_acceleration_feedforward
        roll_sp = torch.atan(lateral_accel_sp / 9.81)

        max_roll = math.radians(float(self.cfg.max_roll_deg))
        roll_sp = torch.clamp(roll_sp, -max_roll, max_roll)
        roll_err = _wrap_pi(roll_sp - roll)
        action_roll = torch.clamp(
            float(self.cfg.roll_kp) * roll_err - float(self.cfg.roll_kd) * ang_vel_body[:, 0],
            min=-1.0,
            max=1.0,
        )

        pitch_trim = -math.radians(float(self.cfg.pitch_trim_deg))
        height_err = float(self.cfg.height_sp_m) - pos_local[:, 2]
        pitch_sp = pitch_trim + float(self.cfg.height_kp) * height_err - float(self.cfg.height_rate_kd) * ground_vel_local[:, 2]
        pitch_sp = torch.clamp(
            pitch_sp,
            min=-math.radians(float(self.cfg.max_pitch_up_deg)),
            max=math.radians(float(self.cfg.max_pitch_down_deg)),
        )
        pitch_err = _wrap_pi(pitch_sp - pitch)
        action_elevator = torch.clamp(
            float(self.cfg.pitch_kp) * pitch_err - float(self.cfg.pitch_kd) * ang_vel_body[:, 1],
            min=-1.0,
            max=1.0,
        )

        course_err = _wrap_pi(guidance.course_setpoint - yaw)
        action_rudder = torch.clamp(
            float(self.cfg.yaw_kp) * course_err - float(self.cfg.yaw_kd) * ang_vel_body[:, 2],
            min=-1.0,
            max=1.0,
        )

        if self.cfg.enable_speed_hold:
            ground_speed = torch.linalg.norm(vel_xy, dim=1)
            freq_hz = float(self.cfg.freq_trim_hz) + float(self.cfg.speed_kp_hz_per_mps) * (
                float(self.cfg.speed_sp_mps) - ground_speed
            )
        else:
            freq_hz = torch.full_like(action_roll, float(self.cfg.freq_trim_hz))
        freq_hz = torch.clamp(freq_hz, min=float(self.cfg.min_flap_hz), max=float(self.cfg.max_flap_hz))

        denom = max(float(self.cfg.max_flap_hz) - float(self.cfg.min_flap_hz), 1.0e-6)
        action_freq = 2.0 * (freq_hz - float(self.cfg.min_flap_hz)) / denom - 1.0
        action_freq = torch.clamp(action_freq, -1.0, 1.0)

        actions = torch.stack((action_freq, action_elevator, action_rudder, action_roll), dim=1)

        diag = {
            "course_sp": guidance.course_setpoint,
            "course_err": course_err,
            "signed_track_error": guidance.signed_track_error,
            "track_error_bound": guidance.track_error_bound,
            "roll_sp": roll_sp,
            "pitch_sp": pitch_sp,
            "freq_hz": freq_hz,
            "closest_x": closest_point[:, 0],
            "closest_y": closest_point[:, 1],
        }
        return actions, diag

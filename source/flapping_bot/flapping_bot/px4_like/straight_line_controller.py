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
from .tecs import PX4LikeTECS, PX4LikeTECSCfg

Tensor = torch.Tensor


def _wrap_pi(angle_rad: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))


@dataclass
class PX4LikeStraightLineControllerCfg:
    """Controller gains and mission settings."""

    line_start_xy: tuple[float, float] = (0.0, 0.0)
    line_end_xy: tuple[float, float] = (120.0, 0.0)
    wind_xy: tuple[float, float] = (0.0, 0.0)
    control_dt_s: float = 1.0 / 120.0

    height_sp_m: float = 10.0
    pitch_trim_deg: float = 10.0
    max_roll_deg: float = 45.0
    max_pitch_up_deg: float = 25.0
    max_pitch_down_deg: float = 20.0

    roll_kp: float = 2.5
    roll_kd: float = 0.35
    pitch_kp: float = 2.5
    pitch_kd: float = 0.16
    yaw_kp: float = 0.1
    yaw_kd: float = 0.2
    height_kp: float = 0.06
    height_rate_kd: float = 0.02

    freq_trim_hz: float = 2.5
    min_flap_hz: float = 3.0
    max_flap_hz: float = 4.6
    enable_tecs: bool = True
    tecs_max_climb_rate_mps: float = 3.0
    tecs_min_sink_rate_mps: float = 2.0
    tecs_altitude_error_gain: float = 0.55
    tecs_airspeed_error_gain: float = 0.8
    tecs_pitch_speed_weight: float = 0.8
    tecs_pitch_damping_gain: float = 0.08
    tecs_integrator_gain_pitch: float = 0.12
    tecs_throttle_damping_gain: float = 0.35
    tecs_integrator_gain_throttle: float = 0.22
    tecs_ste_rate_time_const_s: float = 0.4
    tecs_tas_min_mps: float = 5.0
    tecs_tas_error_percentage: float = 0.15
    tecs_detect_underspeed: bool = True
    tecs_throttle_slew_rate_per_s: float = 0.0
    tecs_altitude_filter_tau_s: float = 0.3
    tecs_altitude_rate_filter_tau_s: float = 0.2
    tecs_pitch_sp_filter_tau_s: float = 0.35
    tecs_pitch_sp_rate_limit_deg_s: float = 20.0
    tecs_throttle_sp_filter_tau_s: float = 0.25
    tecs_altitude_hold_error_band_m: float = 0.25
    tecs_altitude_capture_error_m: float = 0.8
    tecs_altitude_capture_time_const_s: float = 1.0
    tecs_airspeed_error_gain_capture_scale: float = 0.35
    tecs_pitch_speed_weight_capture: float = 0.35
    tecs_capture_extra_climb_rate_mps: float = 0.7
    tecs_capture_extra_sink_rate_mps: float = 0.2

    inner_pitch_lpf_tau_s: float = 0.12
    inner_pitch_rate_lpf_tau_s: float = 0.1
    inner_elevon_pitch_rate_limit_per_s: float = 2.0
    inner_elevon_roll_rate_limit_per_s: float = 6.0
    inner_pitch_ki: float = 0.8
    inner_pitch_integrator_limit: float = 0.6
    inner_pitch_integrator_leak_per_s: float = 0.04

    enable_speed_hold: bool = False
    speed_sp_mps: float = 7.0
    speed_kp_hz_per_mps: float = 0.08
    # Optional throttle assist for altitude hold (very simplified TECS-like behavior).
    # freq_hz += kp * height_err + kd * (-vz)
    freq_height_kp_hz_per_m: float = 0.0
    freq_height_rate_kd_hz_per_mps: float = 0.0

    guidance_period_s: float = 6.0
    guidance_damping: float = 0.7071
    guidance_roll_time_const_s: float = 0.25
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
        denom = max(float(cfg.max_flap_hz) - float(cfg.min_flap_hz), 1.0e-6)
        throttle_trim = (float(cfg.freq_trim_hz) - float(cfg.min_flap_hz)) / denom
        throttle_trim = min(max(throttle_trim, 0.0), 1.0)
        self._tecs = PX4LikeTECS(
            PX4LikeTECSCfg(
                height_sp_m=float(cfg.height_sp_m),
                speed_sp_mps=float(cfg.speed_sp_mps),
                pitch_trim_deg=float(cfg.pitch_trim_deg),
                max_pitch_up_deg=float(cfg.max_pitch_up_deg),
                max_pitch_down_deg=float(cfg.max_pitch_down_deg),
                throttle_trim=throttle_trim,
                throttle_min=0.0,
                throttle_max=1.0,
                max_climb_rate_mps=float(cfg.tecs_max_climb_rate_mps),
                min_sink_rate_mps=float(cfg.tecs_min_sink_rate_mps),
                altitude_error_gain=float(cfg.tecs_altitude_error_gain),
                airspeed_error_gain=float(cfg.tecs_airspeed_error_gain),
                pitch_speed_weight=float(cfg.tecs_pitch_speed_weight),
                pitch_damping_gain=float(cfg.tecs_pitch_damping_gain),
                integrator_gain_pitch=float(cfg.tecs_integrator_gain_pitch),
                throttle_damping_gain=float(cfg.tecs_throttle_damping_gain),
                integrator_gain_throttle=float(cfg.tecs_integrator_gain_throttle),
                ste_rate_time_const_s=float(cfg.tecs_ste_rate_time_const_s),
                throttle_slew_rate_per_s=float(cfg.tecs_throttle_slew_rate_per_s),
                tas_min_mps=float(cfg.tecs_tas_min_mps),
                tas_error_percentage=float(cfg.tecs_tas_error_percentage),
                detect_underspeed=bool(cfg.tecs_detect_underspeed),
                airspeed_enabled=True,
                altitude_filter_tau_s=float(cfg.tecs_altitude_filter_tau_s),
                altitude_rate_filter_tau_s=float(cfg.tecs_altitude_rate_filter_tau_s),
                pitch_sp_filter_tau_s=float(cfg.tecs_pitch_sp_filter_tau_s),
                pitch_sp_rate_limit_deg_s=float(cfg.tecs_pitch_sp_rate_limit_deg_s),
                throttle_sp_filter_tau_s=float(cfg.tecs_throttle_sp_filter_tau_s),
                altitude_hold_error_band_m=float(cfg.tecs_altitude_hold_error_band_m),
                altitude_capture_error_m=float(cfg.tecs_altitude_capture_error_m),
                altitude_capture_time_const_s=float(cfg.tecs_altitude_capture_time_const_s),
                airspeed_error_gain_capture_scale=float(cfg.tecs_airspeed_error_gain_capture_scale),
                pitch_speed_weight_capture=float(cfg.tecs_pitch_speed_weight_capture),
                capture_extra_climb_rate_mps=float(cfg.tecs_capture_extra_climb_rate_mps),
                capture_extra_sink_rate_mps=float(cfg.tecs_capture_extra_sink_rate_mps),
            ),
            device=device,
        )
        self._pitch_meas_filt: Tensor | None = None
        self._pitch_rate_filt: Tensor | None = None
        self._action_elevon_pitch_prev: Tensor | None = None
        self._action_elevon_roll_prev: Tensor | None = None
        self._action_elevon_pitch_integ: Tensor | None = None

    def _ensure_inner_states(self, pitch: Tensor, pitch_rate: Tensor) -> None:
        if self._pitch_meas_filt is not None and self._pitch_meas_filt.shape == pitch.shape:
            return
        self._pitch_meas_filt = pitch.clone()
        self._pitch_rate_filt = pitch_rate.clone()
        self._action_elevon_pitch_prev = torch.zeros_like(pitch)
        self._action_elevon_roll_prev = torch.zeros_like(pitch)
        self._action_elevon_pitch_integ = torch.zeros_like(pitch)

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset controller states, mainly TECS integrators/filters."""
        self._tecs.reset(env_ids)
        self._pitch_meas_filt = None
        self._pitch_rate_filt = None
        self._action_elevon_pitch_prev = None
        self._action_elevon_roll_prev = None
        self._action_elevon_pitch_integ = None

    def compute_actions(
        self,
        *,
        pos_local: Tensor,
        ground_vel_local: Tensor,
        wind_vel_local: Tensor | None = None,
        roll: Tensor,
        pitch: Tensor,
        yaw: Tensor,
        ang_vel_body: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Map current vehicle state to `[throttle, rudder, elevon_pitch, elevon_roll]` actions."""
        pos_xy = pos_local[:, 0:2]
        vel_xy = ground_vel_local[:, 0:2]
        if wind_vel_local is None:
            wind_xy = self._wind_xy.expand(pos_xy.shape[0], 2).to(dtype=pos_xy.dtype)
        else:
            if wind_vel_local.ndim != 2 or wind_vel_local.shape[1] != 2:
                raise ValueError("wind_vel_local must have shape [N, 2].")
            wind_xy = wind_vel_local.to(device=pos_xy.device, dtype=pos_xy.dtype)

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
        # Wind correction: DirectionalGuidance outputs a *ground* bearing (course) to converge to the path.
        # The heading controller operates on the airspeed vector direction. Compute the desired air-velocity
        # direction such that (air + wind) points along the desired bearing, i.e. "crab" into the wind.
        bearing_unit = torch.stack((torch.cos(guidance.course_setpoint), torch.sin(guidance.course_setpoint)), dim=1)
        # Solve for the (positive) ground speed along the bearing `k` such that:
        #   v_g = k * bearing_unit
        #   v_a = v_g - wind
        #   ||v_a|| = airspeed
        wind_dot_bearing = (wind_xy * bearing_unit).sum(dim=1)
        wind_sq = (wind_xy * wind_xy).sum(dim=1)
        wind_cross_sq = torch.clamp(wind_sq - wind_dot_bearing * wind_dot_bearing, min=0.0)
        airspeed_safe = torch.clamp(airspeed, min=1.0e-3)
        sqrt_term = torch.sqrt(torch.clamp(airspeed_safe * airspeed_safe - wind_cross_sq, min=0.0))
        ground_speed_along_bearing = torch.clamp(wind_dot_bearing + sqrt_term, min=0.0)
        v_a_sp = bearing_unit * ground_speed_along_bearing.unsqueeze(1) - wind_xy
        heading_sp = torch.atan2(v_a_sp[:, 1], v_a_sp[:, 0])

        lateral_accel_fb = self._heading_controller.control_heading(heading_sp, heading, airspeed)
        lateral_accel_sp = lateral_accel_fb + guidance.lateral_acceleration_feedforward
        roll_sp = -torch.atan(lateral_accel_sp / 9.81)

        dt = float(self.cfg.control_dt_s)
        self._ensure_inner_states(pitch, ang_vel_body[:, 1])
        assert self._pitch_meas_filt is not None
        assert self._pitch_rate_filt is not None
        assert self._action_elevon_pitch_prev is not None
        assert self._action_elevon_roll_prev is not None
        assert self._action_elevon_pitch_integ is not None

        if float(self.cfg.inner_pitch_lpf_tau_s) > 0.0:
            alpha_pitch = dt / (float(self.cfg.inner_pitch_lpf_tau_s) + dt)
            alpha_pitch = min(max(alpha_pitch, 0.0), 1.0)
            self._pitch_meas_filt = self._pitch_meas_filt + alpha_pitch * (pitch - self._pitch_meas_filt)
        else:
            self._pitch_meas_filt = pitch
        if float(self.cfg.inner_pitch_rate_lpf_tau_s) > 0.0:
            alpha_pitch_rate = dt / (float(self.cfg.inner_pitch_rate_lpf_tau_s) + dt)
            alpha_pitch_rate = min(max(alpha_pitch_rate, 0.0), 1.0)
            self._pitch_rate_filt = self._pitch_rate_filt + alpha_pitch_rate * (ang_vel_body[:, 1] - self._pitch_rate_filt)
        else:
            self._pitch_rate_filt = ang_vel_body[:, 1]

        pitch_meas_for_ctrl = self._pitch_meas_filt
        pitch_rate_for_ctrl = self._pitch_rate_filt

        max_roll = math.radians(float(self.cfg.max_roll_deg))
        roll_sp = torch.clamp(roll_sp, -max_roll, max_roll)
        roll_err = _wrap_pi(roll_sp - roll)
        action_roll_raw = torch.clamp(
            float(self.cfg.roll_kp) * roll_err - float(self.cfg.roll_kd) * ang_vel_body[:, 0],
            min=-1.0,
            max=1.0,
        )
        action_roll = action_roll_raw
        if float(self.cfg.inner_elevon_roll_rate_limit_per_s) > 0.0:
            max_roll_delta = float(self.cfg.inner_elevon_roll_rate_limit_per_s) * dt
            action_roll = torch.clamp(
                action_roll,
                min=self._action_elevon_roll_prev - max_roll_delta,
                max=self._action_elevon_roll_prev + max_roll_delta,
            )
            action_roll = torch.clamp(action_roll, min=-1.0, max=1.0)
        self._action_elevon_roll_prev = action_roll

        height_err = float(self.cfg.height_sp_m) - pos_local[:, 2]
        if bool(self.cfg.enable_tecs):
            pitch_sp, throttle_sp, tecs_diag = self._tecs.update(
                dt=float(self.cfg.control_dt_s),
                altitude=pos_local[:, 2],
                altitude_rate=ground_vel_local[:, 2],
                tas=airspeed,
                height_sp_m=float(self.cfg.height_sp_m),
                speed_sp_mps=float(self.cfg.speed_sp_mps),
            )
            freq_hz = float(self.cfg.min_flap_hz) + throttle_sp * (
                float(self.cfg.max_flap_hz) - float(self.cfg.min_flap_hz)
            )
        else:
            tecs_diag = {}
            pitch_trim = -math.radians(float(self.cfg.pitch_trim_deg))
            pitch_sp = (
                pitch_trim
                - float(self.cfg.height_kp) * height_err
                + float(self.cfg.height_rate_kd) * ground_vel_local[:, 2]
            )
            pitch_sp = torch.clamp(
                pitch_sp,
                min=-math.radians(float(self.cfg.max_pitch_up_deg)),
                max=math.radians(float(self.cfg.max_pitch_down_deg)),
            )

            if self.cfg.enable_speed_hold:
                freq_hz = float(self.cfg.freq_trim_hz) + float(self.cfg.speed_kp_hz_per_mps) * (
                    float(self.cfg.speed_sp_mps) - airspeed
                )
            else:
                freq_hz = torch.full_like(action_roll, float(self.cfg.freq_trim_hz))

            if float(self.cfg.freq_height_kp_hz_per_m) != 0.0 or float(self.cfg.freq_height_rate_kd_hz_per_mps) != 0.0:
                freq_hz = freq_hz + float(self.cfg.freq_height_kp_hz_per_m) * height_err + float(
                    self.cfg.freq_height_rate_kd_hz_per_mps
                ) * (-ground_vel_local[:, 2])
            freq_hz = torch.clamp(freq_hz, min=float(self.cfg.min_flap_hz), max=float(self.cfg.max_flap_hz))

        pitch_err = _wrap_pi(pitch_sp - pitch_meas_for_ctrl)
        pitch_pd = float(self.cfg.pitch_kp) * pitch_err - float(self.cfg.pitch_kd) * pitch_rate_for_ctrl
        ki = float(self.cfg.inner_pitch_ki)
        if ki > 0.0:
            leak = max(float(self.cfg.inner_pitch_integrator_leak_per_s), 0.0)
            if leak > 0.0:
                self._action_elevon_pitch_integ = self._action_elevon_pitch_integ * max(0.0, 1.0 - leak * dt)

            integ_input = ki * pitch_err
            raw_with_integ = pitch_pd + self._action_elevon_pitch_integ
            at_upper = raw_with_integ >= (1.0 - 1.0e-4)
            at_lower = raw_with_integ <= (-1.0 + 1.0e-4)
            integ_input = torch.where(at_upper, torch.minimum(integ_input, torch.zeros_like(integ_input)), integ_input)
            integ_input = torch.where(at_lower, torch.maximum(integ_input, torch.zeros_like(integ_input)), integ_input)

            self._action_elevon_pitch_integ = self._action_elevon_pitch_integ + integ_input * dt
            integ_limit = max(float(self.cfg.inner_pitch_integrator_limit), 0.0)
            if integ_limit > 0.0:
                self._action_elevon_pitch_integ = torch.clamp(
                    self._action_elevon_pitch_integ, min=-integ_limit, max=integ_limit
                )

        action_elevon_pitch_raw = torch.clamp(
            pitch_pd + self._action_elevon_pitch_integ,
            min=-1.0,
            max=1.0,
        )
        action_elevon_pitch = action_elevon_pitch_raw
        if float(self.cfg.inner_elevon_pitch_rate_limit_per_s) > 0.0:
            max_pitch_delta = float(self.cfg.inner_elevon_pitch_rate_limit_per_s) * dt
            action_elevon_pitch = torch.clamp(
                action_elevon_pitch,
                min=self._action_elevon_pitch_prev - max_pitch_delta,
                max=self._action_elevon_pitch_prev + max_pitch_delta,
            )
            action_elevon_pitch = torch.clamp(action_elevon_pitch, min=-1.0, max=1.0)
        self._action_elevon_pitch_prev = action_elevon_pitch

        # Fixed-wing convention: roll controls ground-track; yaw/rudder should not fight wind-crab.
        # Drive yaw to align with the airspeed direction (beta≈0), with yaw-rate damping.
        course_err = _wrap_pi(yaw - heading)
        action_rudder = torch.clamp(
            float(self.cfg.yaw_kp) * course_err - float(self.cfg.yaw_kd) * ang_vel_body[:, 2],
            min=-1.0,
            max=1.0,
        )

        denom = max(float(self.cfg.max_flap_hz) - float(self.cfg.min_flap_hz), 1.0e-6)
        action_freq = 2.0 * (freq_hz - float(self.cfg.min_flap_hz)) / denom - 1.0
        action_freq = torch.clamp(action_freq, -1.0, 1.0)

        actions = torch.stack((action_freq, action_rudder, action_elevon_pitch, action_roll), dim=1)

        diag = {
            "course_sp": guidance.course_setpoint,
            "heading_sp": heading_sp,
            "heading": heading,
            "course_err": course_err,
            "signed_track_error": guidance.signed_track_error,
            "track_error_bound": guidance.track_error_bound,
            "roll_sp": roll_sp,
            "pitch_sp": pitch_sp,
            "freq_hz": freq_hz,
            "closest_x": closest_point[:, 0],
            "closest_y": closest_point[:, 1],
            "pitch_meas_filt": pitch_meas_for_ctrl,
            "pitch_rate_filt": pitch_rate_for_ctrl,
            "pitch_err_filt": pitch_err,
            "wind_x": wind_xy[:, 0],
            "wind_y": wind_xy[:, 1],
            "airspeed_xy": airspeed,
            "action_elevon_pitch_raw": action_elevon_pitch_raw,
            "action_elevon_pitch_integ": self._action_elevon_pitch_integ,
            "action_elevon_roll_raw": action_roll_raw,
        }
        diag.update(tecs_diag)
        return actions, diag

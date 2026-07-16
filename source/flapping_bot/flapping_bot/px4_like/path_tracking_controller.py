"""Generic PX4-like path-tracking controller."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .straight_line_controller import (
    PX4LikeStraightLineController,
    PX4LikeStraightLineControllerCfg,
    _wrap_pi,
)

Tensor = torch.Tensor


@dataclass
class PX4LikePathTrackingControllerCfg(PX4LikeStraightLineControllerCfg):
    """Configuration for the generic path-tracking controller."""


class PX4LikePathTrackingController(PX4LikeStraightLineController):
    """Compute actions from a unified path query."""

    def __init__(self, cfg: PX4LikePathTrackingControllerCfg, device: torch.device):
        super().__init__(cfg=cfg, device=device)

    def compute_actions_from_query(
        self,
        *,
        path_query: dict[str, Tensor],
        pos_local: Tensor,
        ground_vel_local: Tensor,
        wind_vel_local: Tensor | None = None,
        roll: Tensor,
        pitch: Tensor,
        yaw: Tensor,
        ang_vel_body: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Map current vehicle state and a local path query to controller actions."""
        for key in ("closest_point_xyz", "tangent_xy", "curvature_m_inv", "height_sp_m"):
            if key not in path_query:
                raise KeyError(f"path_query missing required key: {key}")

        pos_xy = pos_local[:, 0:2]
        vel_xy = ground_vel_local[:, 0:2]
        if wind_vel_local is None:
            wind_xy = self._wind_xy.expand(pos_xy.shape[0], 2).to(dtype=pos_xy.dtype)
        else:
            if wind_vel_local.ndim != 2 or wind_vel_local.shape[1] != 2:
                raise ValueError("wind_vel_local must have shape [N, 2].")
            wind_xy = wind_vel_local.to(device=pos_xy.device, dtype=pos_xy.dtype)

        closest_point_xyz = path_query["closest_point_xyz"].to(device=pos_local.device, dtype=pos_local.dtype)
        closest_point_xy = closest_point_xyz[:, 0:2]

        tangent_xy = path_query["tangent_xy"].to(device=pos_xy.device, dtype=pos_xy.dtype)
        tangent_norm = torch.linalg.norm(tangent_xy, dim=1, keepdim=True).clamp_min(1.0e-6)
        unit_tangent = tangent_xy / tangent_norm

        curvature = path_query["curvature_m_inv"].to(device=pos_xy.device, dtype=pos_xy.dtype)
        height_sp_m = path_query["height_sp_m"].to(device=pos_local.device, dtype=pos_local.dtype)

        guidance = self._directional_guidance.guide_to_path(
            curr_pos_local=pos_xy,
            ground_vel=vel_xy,
            wind_vel=wind_xy,
            unit_path_tangent=unit_tangent,
            position_on_path=closest_point_xy,
            path_curvature=curvature,
        )

        air_vel_xy = vel_xy - wind_xy
        airspeed = torch.linalg.norm(air_vel_xy, dim=1)
        heading_diag = self._resolve_lateral_heading(air_vel_xy=air_vel_xy, yaw=yaw)
        heading_used = heading_diag["heading_used"]
        heading_from_velocity = heading_diag["heading_from_velocity"]
        bearing_unit = torch.stack((torch.cos(guidance.course_setpoint), torch.sin(guidance.course_setpoint)), dim=1)
        wind_dot_bearing = (wind_xy * bearing_unit).sum(dim=1)
        wind_sq = (wind_xy * wind_xy).sum(dim=1)
        wind_cross_sq = torch.clamp(wind_sq - wind_dot_bearing * wind_dot_bearing, min=0.0)
        airspeed_safe = torch.clamp(airspeed, min=1.0e-3)
        sqrt_term = torch.sqrt(torch.clamp(airspeed_safe * airspeed_safe - wind_cross_sq, min=0.0))
        ground_speed_along_bearing = torch.clamp(wind_dot_bearing + sqrt_term, min=0.0)
        v_a_sp = bearing_unit * ground_speed_along_bearing.unsqueeze(1) - wind_xy
        heading_sp = torch.atan2(v_a_sp[:, 1], v_a_sp[:, 0])
        guidance_min_airspeed = self._resolve_guidance_min_airspeed(bearing_unit=bearing_unit, wind_xy=wind_xy)

        lateral_accel_fb = self._heading_controller.control_heading(heading_sp, heading_used, airspeed)
        lateral_accel_sp_unc = lateral_accel_fb + guidance.lateral_acceleration_feedforward
        lateral_guidance_quality_scale = self._resolve_lateral_guidance_quality_scale(
            heading_yaw_correction=heading_diag["heading_yaw_correction"]
        )
        lateral_accel_sp = lateral_accel_sp_unc * lateral_guidance_quality_scale
        roll_sp = -torch.atan(lateral_accel_sp / 9.81)

        dt = float(self.cfg.control_dt_s)
        self._ensure_inner_states(pitch, ang_vel_body[:, 1])
        assert self._pitch_meas_filt is not None
        assert self._pitch_rate_filt is not None
        assert self._action_elevon_pitch_prev is not None
        assert self._action_elevon_roll_prev is not None
        assert self._action_elevon_pitch_integ is not None

        pitch_filter_tau_s = (
            float(self.cfg.inner_pitch_cycle_mean_tau_s)
            if bool(self.cfg.inner_pitch_cycle_mean_enabled)
            else float(self.cfg.inner_pitch_lpf_tau_s)
        )
        pitch_rate_filter_tau_s = (
            float(self.cfg.inner_pitch_rate_cycle_mean_tau_s)
            if bool(self.cfg.inner_pitch_cycle_mean_enabled)
            else float(self.cfg.inner_pitch_rate_lpf_tau_s)
        )
        if pitch_filter_tau_s > 0.0:
            alpha_pitch = dt / (pitch_filter_tau_s + dt)
            alpha_pitch = min(max(alpha_pitch, 0.0), 1.0)
            self._pitch_meas_filt = self._pitch_meas_filt + alpha_pitch * (pitch - self._pitch_meas_filt)
        else:
            self._pitch_meas_filt = pitch
        if pitch_rate_filter_tau_s > 0.0:
            alpha_pitch_rate = dt / (pitch_rate_filter_tau_s + dt)
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

        if bool(self.cfg.enable_tecs):
            tecs_turn = self._resolve_tecs_turn_inputs(
                airspeed=airspeed,
                roll=roll,
                roll_sp=roll_sp,
                guidance_min_airspeed=guidance_min_airspeed,
            )
            pitch_sp, throttle_sp, tecs_diag = self._tecs.update(
                dt=float(self.cfg.control_dt_s),
                altitude=pos_local[:, 2],
                altitude_rate=ground_vel_local[:, 2],
                tas=airspeed,
                height_sp_m=height_sp_m,
                speed_sp_mps=tecs_turn["speed_sp_cmd"],
                load_factor=tecs_turn["load_factor"],
                load_factor_correction=tecs_turn["load_factor_correction"],
            )
            freq_hz = float(self.cfg.min_flap_hz) + throttle_sp * (
                float(self.cfg.max_flap_hz) - float(self.cfg.min_flap_hz)
            )
        else:
            tecs_diag = {}
            tecs_turn = {
                "bank_speed_delta": torch.zeros_like(airspeed),
                "bank_min_airspeed": torch.zeros_like(airspeed),
                "bank_min_delta": torch.zeros_like(airspeed),
            }
            height_err = height_sp_m - pos_local[:, 2]
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
        pitch_tc = max(float(self.cfg.inner_pitch_tc_s), 1.0e-3)
        pitch_rate_sp = torch.clamp(
            pitch_err / pitch_tc,
            min=-math.radians(float(self.cfg.inner_pitch_rate_max_deg_s)),
            max=math.radians(float(self.cfg.inner_pitch_rate_max_deg_s)),
        )
        pitch_pd = float(self.cfg.pitch_kp) * pitch_err + float(self.cfg.pitch_kd) * (pitch_rate_sp - pitch_rate_for_ctrl)
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

        action_elevon_pitch_raw = torch.clamp(pitch_pd + self._action_elevon_pitch_integ, min=-1.0, max=1.0)
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

        course_err = _wrap_pi(yaw - heading_from_velocity)
        # ``course_err`` is current minus desired heading, so both feedback terms
        # must oppose their corresponding positive yaw states.
        action_rudder = torch.clamp(
            -float(self.cfg.yaw_kp) * course_err - float(self.cfg.yaw_kd) * ang_vel_body[:, 2],
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
            "heading": heading_from_velocity,
            "heading_used": heading_used,
            "heading_from_velocity": heading_from_velocity,
            "heading_yaw_correction": heading_diag["heading_yaw_correction"],
            "course_err": course_err,
            "signed_track_error": guidance.signed_track_error,
            "track_error_bound": guidance.track_error_bound,
            "guidance_min_airspeed_mps": guidance_min_airspeed,
            "lateral_guidance_quality_scale": lateral_guidance_quality_scale,
            "lateral_accel_sp_unscaled": lateral_accel_sp_unc,
            "roll_sp": roll_sp,
            "pitch_sp": pitch_sp,
            "freq_hz": freq_hz,
            "pitch_meas_filt": pitch_meas_for_ctrl,
            "pitch_rate_filt": pitch_rate_for_ctrl,
            "pitch_rate_sp": pitch_rate_sp,
            "pitch_err_filt": pitch_err,
            "action_elevon_pitch_raw": action_elevon_pitch_raw,
            "action_elevon_pitch_integ": self._action_elevon_pitch_integ,
            "action_elevon_roll_raw": action_roll_raw,
            "closest_point_xyz": closest_point_xyz,
            "height_sp_m": height_sp_m,
            "curvature_m_inv": curvature,
            "tecs_bank_aware_speed_delta_mps": tecs_turn["bank_speed_delta"],
            "tecs_bank_aware_min_airspeed_mps": tecs_turn["bank_min_airspeed"],
            "tecs_bank_aware_min_airspeed_delta_mps": tecs_turn["bank_min_delta"],
        }
        diag.update(tecs_diag)
        return actions, diag

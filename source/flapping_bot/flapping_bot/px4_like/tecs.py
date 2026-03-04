"""PX4-inspired TECS-style longitudinal controller for flapping flight.

This implementation keeps the same core structure:
1) total-energy rate drives throttle (flapping frequency),
2) energy-balance rate drives pitch.

The sign convention follows this project:
- body/world z is up,
- more nose-up pitch setpoint is more negative pitch angle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

Tensor = torch.Tensor
G = 9.81


@dataclass
class PX4LikeTECSCfg:
    """Configuration for the simplified PX4-like TECS."""

    height_sp_m: float = 10.0
    speed_sp_mps: float = 7.0

    pitch_trim_deg: float = 10.0
    max_pitch_up_deg: float = 25.0
    max_pitch_down_deg: float = 20.0

    throttle_trim: float = 0.45
    throttle_min: float = 0.0
    throttle_max: float = 1.0

    max_climb_rate_mps: float = 3.0
    min_sink_rate_mps: float = 2.0

    altitude_error_gain: float = 0.55
    airspeed_error_gain: float = 0.8
    pitch_speed_weight: float = 0.8

    pitch_damping_gain: float = 0.08
    seb_rate_ff: float = 1.0
    integrator_gain_pitch: float = 0.12

    throttle_damping_gain: float = 0.35
    integrator_gain_throttle: float = 0.22
    ste_rate_time_const_s: float = 0.4
    throttle_slew_rate_per_s: float = 0.0

    tas_min_mps: float = 5.0
    tas_error_percentage: float = 0.15
    detect_underspeed: bool = True
    airspeed_enabled: bool = True

    speed_filter_tau_s: float = 0.25
    speed_rate_filter_tau_s: float = 0.35
    altitude_filter_tau_s: float = 0.3
    altitude_rate_filter_tau_s: float = 0.2

    pitch_sp_filter_tau_s: float = 0.35
    pitch_sp_rate_limit_deg_s: float = 20.0
    throttle_sp_filter_tau_s: float = 0.25


class PX4LikeTECS:
    """Vectorized TECS-like longitudinal controller."""

    def __init__(self, cfg: PX4LikeTECSCfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._initialized = False

        self._pitch_sp: Tensor | None = None
        self._throttle_sp: Tensor | None = None
        self._pitch_integ: Tensor | None = None
        self._throttle_integ: Tensor | None = None
        self._ste_rate_est: Tensor | None = None
        self._tas_filt: Tensor | None = None
        self._tas_rate_filt: Tensor | None = None
        self._tas_prev: Tensor | None = None
        self._altitude_filt: Tensor | None = None
        self._altitude_rate_filt: Tensor | None = None
        self._ratio_underspeed: Tensor | None = None

    def _ensure_state(self, batch_size: int, dtype: torch.dtype) -> None:
        if self._initialized and self._pitch_sp is not None and self._pitch_sp.numel() == batch_size:
            return

        pitch_trim = -math.radians(float(self.cfg.pitch_trim_deg))
        throttle_trim = float(self.cfg.throttle_trim)

        self._pitch_sp = torch.full((batch_size,), pitch_trim, device=self.device, dtype=dtype)
        self._throttle_sp = torch.full((batch_size,), throttle_trim, device=self.device, dtype=dtype)
        self._pitch_integ = torch.zeros((batch_size,), device=self.device, dtype=dtype)
        self._throttle_integ = torch.zeros((batch_size,), device=self.device, dtype=dtype)
        self._ste_rate_est = torch.zeros((batch_size,), device=self.device, dtype=dtype)
        self._tas_filt = torch.full((batch_size,), float(self.cfg.speed_sp_mps), device=self.device, dtype=dtype)
        self._tas_rate_filt = torch.zeros((batch_size,), device=self.device, dtype=dtype)
        self._tas_prev = self._tas_filt.clone()
        self._altitude_filt = torch.full((batch_size,), float(self.cfg.height_sp_m), device=self.device, dtype=dtype)
        self._altitude_rate_filt = torch.zeros((batch_size,), device=self.device, dtype=dtype)
        self._ratio_underspeed = torch.zeros((batch_size,), device=self.device, dtype=dtype)
        self._initialized = True

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset controller states globally or for selected environments."""
        if not self._initialized or self._pitch_sp is None:
            return

        pitch_trim = -math.radians(float(self.cfg.pitch_trim_deg))
        throttle_trim = float(self.cfg.throttle_trim)

        if env_ids is None:
            self._pitch_sp.fill_(pitch_trim)
            self._throttle_sp.fill_(throttle_trim)
            self._pitch_integ.zero_()
            self._throttle_integ.zero_()
            self._ste_rate_est.zero_()
            self._tas_filt.fill_(float(self.cfg.speed_sp_mps))
            self._tas_rate_filt.zero_()
            self._tas_prev.copy_(self._tas_filt)
            self._altitude_filt.fill_(float(self.cfg.height_sp_m))
            self._altitude_rate_filt.zero_()
            self._ratio_underspeed.zero_()
            return

        ids = env_ids.to(device=self.device, dtype=torch.long)
        self._pitch_sp[ids] = pitch_trim
        self._throttle_sp[ids] = throttle_trim
        self._pitch_integ[ids] = 0.0
        self._throttle_integ[ids] = 0.0
        self._ste_rate_est[ids] = 0.0
        self._tas_filt[ids] = float(self.cfg.speed_sp_mps)
        self._tas_rate_filt[ids] = 0.0
        self._tas_prev[ids] = self._tas_filt[ids]
        self._altitude_filt[ids] = float(self.cfg.height_sp_m)
        self._altitude_rate_filt[ids] = 0.0
        self._ratio_underspeed[ids] = 0.0

    def update(
        self,
        *,
        dt: float,
        altitude: Tensor,
        altitude_rate: Tensor,
        tas: Tensor,
        height_sp_m: float | None = None,
        speed_sp_mps: float | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Compute pitch and throttle setpoints."""
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        self._ensure_state(int(altitude.shape[0]), altitude.dtype)
        assert self._pitch_sp is not None
        assert self._throttle_sp is not None
        assert self._pitch_integ is not None
        assert self._throttle_integ is not None
        assert self._ste_rate_est is not None
        assert self._tas_filt is not None
        assert self._tas_rate_filt is not None
        assert self._tas_prev is not None
        assert self._altitude_filt is not None
        assert self._altitude_rate_filt is not None
        assert self._ratio_underspeed is not None

        height_sp = float(self.cfg.height_sp_m if height_sp_m is None else height_sp_m)
        speed_sp = float(self.cfg.speed_sp_mps if speed_sp_mps is None else speed_sp_mps)

        pitch_min = -math.radians(float(self.cfg.max_pitch_up_deg))
        pitch_max = math.radians(float(self.cfg.max_pitch_down_deg))
        pitch_trim = -math.radians(float(self.cfg.pitch_trim_deg))
        tas_min = max(float(self.cfg.tas_min_mps), 1.0e-3)

        throttle_min = float(self.cfg.throttle_min)
        throttle_max = float(self.cfg.throttle_max)
        throttle_trim = float(self.cfg.throttle_trim)
        throttle_trim = min(max(throttle_trim, throttle_min), throttle_max)

        max_climb_rate = max(float(self.cfg.max_climb_rate_mps), 1.0e-3)
        min_sink_rate = max(float(self.cfg.min_sink_rate_mps), 1.0e-3)
        ste_rate_max = max_climb_rate * G
        ste_rate_min = -min_sink_rate * G
        ste_rate_span = max(ste_rate_max - ste_rate_min, 1.0e-3)

        # Airspeed filter and derivative estimate.
        alpha_v = dt / (max(float(self.cfg.speed_filter_tau_s), 0.0) + dt)
        alpha_v = min(max(alpha_v, 0.0), 1.0)
        self._tas_filt = self._tas_filt + alpha_v * (tas - self._tas_filt)
        tas_rate_raw = (self._tas_filt - self._tas_prev) / dt
        self._tas_prev.copy_(self._tas_filt)

        alpha_vdot = dt / (max(float(self.cfg.speed_rate_filter_tau_s), 0.0) + dt)
        alpha_vdot = min(max(alpha_vdot, 0.0), 1.0)
        self._tas_rate_filt = self._tas_rate_filt + alpha_vdot * (tas_rate_raw - self._tas_rate_filt)
        tas_ctrl = torch.clamp(self._tas_filt, min=1.0e-3)

        # Altitude/vz filtering: avoid chasing flapping-period ripple in energy loops.
        alpha_h = dt / (max(float(self.cfg.altitude_filter_tau_s), 0.0) + dt)
        alpha_h = min(max(alpha_h, 0.0), 1.0)
        self._altitude_filt = self._altitude_filt + alpha_h * (altitude - self._altitude_filt)
        alpha_hdot = dt / (max(float(self.cfg.altitude_rate_filter_tau_s), 0.0) + dt)
        alpha_hdot = min(max(alpha_hdot, 0.0), 1.0)
        self._altitude_rate_filt = self._altitude_rate_filt + alpha_hdot * (altitude_rate - self._altitude_rate_filt)

        # Outer loops -> target height/speed rates.
        altitude_rate_sp = (height_sp - self._altitude_filt) * float(self.cfg.altitude_error_gain)
        altitude_rate_sp = torch.clamp(altitude_rate_sp, min=-min_sink_rate, max=max_climb_rate)

        if bool(self.cfg.airspeed_enabled):
            max_tas_rate_sp = 0.5 * ste_rate_max / torch.clamp(tas_ctrl, min=1.0e-3)
            min_tas_rate_sp = 0.5 * ste_rate_min / torch.clamp(tas_ctrl, min=1.0e-3)
            tas_rate_sp = (speed_sp - tas_ctrl) * float(self.cfg.airspeed_error_gain)
            tas_rate_sp = torch.maximum(torch.minimum(tas_rate_sp, max_tas_rate_sp), min_tas_rate_sp)
        else:
            tas_rate_sp = torch.zeros_like(tas_ctrl)

        # Specific energy rates.
        spe_rate_sp = altitude_rate_sp * G
        ske_rate_sp = tas_ctrl * tas_rate_sp
        spe_rate_est = self._altitude_rate_filt * G
        ske_rate_est = tas_ctrl * self._tas_rate_filt

        # Underspeed ratio in [0,1].
        if bool(self.cfg.detect_underspeed) and bool(self.cfg.airspeed_enabled):
            tas_err_bound = float(self.cfg.tas_error_percentage) * max(speed_sp, 1.0e-3)
            tas_soft_bound = tas_err_bound
            tas_fully_underspeed = max(tas_min - tas_err_bound - tas_soft_bound, 0.0)
            tas_start_underspeed = max(tas_min - tas_err_bound, tas_fully_underspeed + 1.0e-6)
            ratio = 1.0 - torch.clamp(
                (tas_ctrl - tas_fully_underspeed) / (tas_start_underspeed - tas_fully_underspeed), min=0.0, max=1.0
            )
            self._ratio_underspeed.copy_(ratio)
        else:
            self._ratio_underspeed.zero_()

        # Pitch control via specific-energy-balance rate.
        pitch_speed_weight = float(min(max(self.cfg.pitch_speed_weight, 0.0), 2.0))
        if bool(self.cfg.airspeed_enabled):
            psw = 2.0 * self._ratio_underspeed + (1.0 - self._ratio_underspeed) * pitch_speed_weight
        else:
            psw = torch.zeros_like(self._ratio_underspeed)

        spe_weight = torch.clamp(2.0 - psw, min=0.0, max=2.0)
        ske_weight = torch.clamp(psw, min=0.0, max=2.0)

        seb_rate_sp = spe_rate_sp * spe_weight - ske_rate_sp * ske_weight
        seb_rate_est = spe_rate_est * spe_weight - ske_rate_est * ske_weight
        seb_rate_err = seb_rate_sp - seb_rate_est

        climb_to_seb = torch.clamp(tas_ctrl, min=tas_min) * G
        pitch_integ_input = seb_rate_err * float(self.cfg.integrator_gain_pitch) / torch.clamp(climb_to_seb, min=1.0e-3)

        at_pitch_min = self._pitch_sp <= (pitch_min + 1.0e-4)
        at_pitch_max = self._pitch_sp >= (pitch_max - 1.0e-4)
        pitch_integ_input = torch.where(at_pitch_min, torch.minimum(pitch_integ_input, torch.zeros_like(pitch_integ_input)), pitch_integ_input)
        pitch_integ_input = torch.where(at_pitch_max, torch.maximum(pitch_integ_input, torch.zeros_like(pitch_integ_input)), pitch_integ_input)
        self._pitch_integ = self._pitch_integ + pitch_integ_input * dt

        seb_rate_corr = seb_rate_err * float(self.cfg.pitch_damping_gain) + float(self.cfg.seb_rate_ff) * seb_rate_sp
        pitch_term = seb_rate_corr / torch.clamp(climb_to_seb, min=1.0e-3) + self._pitch_integ
        pitch_sp_cmd = torch.clamp(pitch_trim - pitch_term, min=pitch_min, max=pitch_max)
        prev_pitch_sp = self._pitch_sp.clone()
        if float(self.cfg.pitch_sp_filter_tau_s) > 0.0:
            alpha_pitch_sp = dt / (float(self.cfg.pitch_sp_filter_tau_s) + dt)
            alpha_pitch_sp = min(max(alpha_pitch_sp, 0.0), 1.0)
            pitch_sp_cmd = prev_pitch_sp + alpha_pitch_sp * (pitch_sp_cmd - prev_pitch_sp)
        if float(self.cfg.pitch_sp_rate_limit_deg_s) > 0.0:
            pitch_delta_max = math.radians(float(self.cfg.pitch_sp_rate_limit_deg_s)) * dt
            pitch_sp_cmd = torch.clamp(
                pitch_sp_cmd,
                min=prev_pitch_sp - pitch_delta_max,
                max=prev_pitch_sp + pitch_delta_max,
            )
        self._pitch_sp = torch.clamp(pitch_sp_cmd, min=pitch_min, max=pitch_max)

        # Throttle control via specific-total-energy rate.
        ste_rate_sp = torch.clamp(spe_rate_sp + ske_rate_sp, min=ste_rate_min, max=ste_rate_max)
        ste_rate_est_raw = spe_rate_est + ske_rate_est
        alpha_ste = dt / (max(float(self.cfg.ste_rate_time_const_s), 0.0) + dt)
        alpha_ste = min(max(alpha_ste, 0.0), 1.0)
        self._ste_rate_est = self._ste_rate_est + alpha_ste * (ste_rate_est_raw - self._ste_rate_est)
        ste_rate_err = ste_rate_sp - self._ste_rate_est

        throttle_above_trim_per_ste = (throttle_max - throttle_trim) / max(ste_rate_max, 1.0e-3)
        throttle_below_trim_per_ste = (throttle_trim - throttle_min) / min(ste_rate_min, -1.0e-3)
        throttle_pred = torch.where(
            ste_rate_sp >= 0.0,
            throttle_trim + ste_rate_sp * throttle_above_trim_per_ste,
            throttle_trim - ste_rate_sp * throttle_below_trim_per_ste,
        )

        ste_to_throttle = 1.0 / ste_rate_span
        if bool(self.cfg.airspeed_enabled):
            throttle_integ_input = (
                ste_rate_err
                * float(self.cfg.integrator_gain_throttle)
                * dt
                * ste_to_throttle
                * (1.0 - self._ratio_underspeed)
            )
            at_throttle_max = self._throttle_sp >= (throttle_max - 1.0e-4)
            at_throttle_min = self._throttle_sp <= (throttle_min + 1.0e-4)
            throttle_integ_input = torch.where(
                at_throttle_max, torch.minimum(throttle_integ_input, torch.zeros_like(throttle_integ_input)), throttle_integ_input
            )
            throttle_integ_input = torch.where(
                at_throttle_min, torch.maximum(throttle_integ_input, torch.zeros_like(throttle_integ_input)), throttle_integ_input
            )
            self._throttle_integ = self._throttle_integ + throttle_integ_input
        else:
            self._throttle_integ.zero_()

        throttle_unc = (
            throttle_pred
            + ste_rate_err * float(self.cfg.throttle_damping_gain) * ste_to_throttle
            + self._throttle_integ
        )
        throttle_unc = self._ratio_underspeed * throttle_max + (1.0 - self._ratio_underspeed) * throttle_unc

        throttle_cmd = torch.clamp(throttle_unc, min=throttle_min, max=throttle_max)
        prev_throttle = self._throttle_sp.clone()
        if float(self.cfg.throttle_sp_filter_tau_s) > 0.0:
            alpha_throttle_sp = dt / (float(self.cfg.throttle_sp_filter_tau_s) + dt)
            alpha_throttle_sp = min(max(alpha_throttle_sp, 0.0), 1.0)
            throttle_cmd = prev_throttle + alpha_throttle_sp * (throttle_cmd - prev_throttle)
        if float(self.cfg.throttle_slew_rate_per_s) > 0.0:
            throttle_inc_limit = dt * (throttle_max - throttle_min) * float(self.cfg.throttle_slew_rate_per_s)
            throttle_cmd = torch.clamp(
                throttle_cmd, min=prev_throttle - throttle_inc_limit, max=prev_throttle + throttle_inc_limit
            )

        self._throttle_sp = torch.clamp(throttle_cmd, min=throttle_min, max=throttle_max)

        diag = {
            "tecs_altitude_rate_sp": altitude_rate_sp,
            "tecs_tas_sp": torch.full_like(tas_ctrl, speed_sp),
            "tecs_tas": tas_ctrl,
            "tecs_tas_rate": self._tas_rate_filt,
            "tecs_altitude_filt": self._altitude_filt,
            "tecs_altitude_rate_filt": self._altitude_rate_filt,
            "tecs_spe_rate_sp": spe_rate_sp,
            "tecs_ske_rate_sp": ske_rate_sp,
            "tecs_spe_rate_est": spe_rate_est,
            "tecs_ske_rate_est": ske_rate_est,
            "tecs_ste_rate_sp": ste_rate_sp,
            "tecs_ste_rate_est": self._ste_rate_est,
            "tecs_seb_rate_sp": seb_rate_sp,
            "tecs_seb_rate_est": seb_rate_est,
            "tecs_ratio_underspeed": self._ratio_underspeed,
            "tecs_throttle_sp": self._throttle_sp,
            "tecs_pitch_sp": self._pitch_sp,
            "tecs_pitch_integ": self._pitch_integ,
            "tecs_throttle_integ": self._throttle_integ,
        }
        return self._pitch_sp, self._throttle_sp, diag

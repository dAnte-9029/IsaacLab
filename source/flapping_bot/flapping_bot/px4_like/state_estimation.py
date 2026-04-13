"""Lightweight sensor simulation and state estimation for flapping PX4-like rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .imu_provider import ImuMeasurement

Tensor = torch.Tensor


def _wrap_pi(angle_rad: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))


def _alpha_from_tau(dt: float, tau: float) -> float:
    if tau <= 0.0:
        return 1.0
    return float(max(0.0, min(1.0, dt / (tau + dt))))


def _angle_lpf(prev_angle: Tensor, measured_angle: Tensor, alpha: Tensor | float) -> Tensor:
    delta = _wrap_pi(measured_angle - prev_angle)
    return _wrap_pi(prev_angle + alpha * delta)


def _body_from_world(roll: Tensor, pitch: Tensor, yaw: Tensor, vec_world: Tensor) -> Tensor:
    """Rotate world-frame vectors into body frame from current Euler attitude."""
    sr = torch.sin(roll)
    cr = torch.cos(roll)
    sp = torch.sin(pitch)
    cp = torch.cos(pitch)
    sy = torch.sin(yaw)
    cy = torch.cos(yaw)

    # Body-to-world matrix R_wb; vector in body frame is R_wb^T * vec_world.
    r11 = cy * cp
    r12 = cy * sp * sr - sy * cr
    r13 = cy * sp * cr + sy * sr
    r21 = sy * cp
    r22 = sy * sp * sr + cy * cr
    r23 = sy * sp * cr - cy * sr
    r31 = -sp
    r32 = cp * sr
    r33 = cp * cr

    x = r11 * vec_world[:, 0] + r21 * vec_world[:, 1] + r31 * vec_world[:, 2]
    y = r12 * vec_world[:, 0] + r22 * vec_world[:, 1] + r32 * vec_world[:, 2]
    z = r13 * vec_world[:, 0] + r23 * vec_world[:, 1] + r33 * vec_world[:, 2]
    return torch.stack((x, y, z), dim=1)


def _world_from_body(roll: Tensor, pitch: Tensor, yaw: Tensor, vec_body: Tensor) -> Tensor:
    """Rotate body-frame vectors into world frame from current Euler attitude."""
    sr = torch.sin(roll)
    cr = torch.cos(roll)
    sp = torch.sin(pitch)
    cp = torch.cos(pitch)
    sy = torch.sin(yaw)
    cy = torch.cos(yaw)

    # Body-to-world matrix R_wb.
    r11 = cy * cp
    r12 = cy * sp * sr - sy * cr
    r13 = cy * sp * cr + sy * sr
    r21 = sy * cp
    r22 = sy * sp * sr + cy * cr
    r23 = sy * sp * cr - cy * sr
    r31 = -sp
    r32 = cp * sr
    r33 = cp * cr

    x = r11 * vec_body[:, 0] + r12 * vec_body[:, 1] + r13 * vec_body[:, 2]
    y = r21 * vec_body[:, 0] + r22 * vec_body[:, 1] + r23 * vec_body[:, 2]
    z = r31 * vec_body[:, 0] + r32 * vec_body[:, 1] + r33 * vec_body[:, 2]
    return torch.stack((x, y, z), dim=1)


def _integrate_euler_from_body_rates(
    roll: Tensor,
    pitch: Tensor,
    yaw: Tensor,
    ang_vel_body: Tensor,
    dt: float,
    *,
    max_pitch_rad: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Integrate Euler attitude with body-rate kinematics."""
    p = ang_vel_body[:, 0]
    q = ang_vel_body[:, 1]
    r = ang_vel_body[:, 2]

    sin_roll = torch.sin(roll)
    cos_roll = torch.cos(roll)
    cos_pitch = torch.cos(pitch)
    cos_pitch_safe = torch.sign(cos_pitch) * torch.clamp(torch.abs(cos_pitch), min=1.0e-3)
    tan_pitch = torch.sin(pitch) / cos_pitch_safe

    roll_dot = p + sin_roll * tan_pitch * q + cos_roll * tan_pitch * r
    pitch_dot = cos_roll * q - sin_roll * r
    yaw_dot = (sin_roll / cos_pitch_safe) * q + (cos_roll / cos_pitch_safe) * r

    roll_next = _wrap_pi(roll + roll_dot * dt)
    pitch_next = torch.clamp(pitch + pitch_dot * dt, min=-max_pitch_rad, max=max_pitch_rad)
    yaw_next = _wrap_pi(yaw + yaw_dot * dt)
    return roll_next, pitch_next, yaw_next


class _DelayBuffer:
    """Fixed-update delay buffer for vectorized sensor samples."""

    def __init__(self, delay_updates: int):
        self.delay_updates = max(int(delay_updates), 0)
        self._queue: list[Tensor] = []

    def reset(self) -> None:
        self._queue.clear()

    def push(self, sample: Tensor) -> Tensor:
        if self.delay_updates <= 0:
            return sample
        self._queue.append(sample.clone())
        if len(self._queue) <= self.delay_updates:
            return self._queue[0]
        return self._queue.pop(0)


@dataclass
class SensorSuiteCfg:
    """Sensor simulation configuration."""

    gps_rate_hz: float = 10.0
    baro_rate_hz: float = 25.0
    airspeed_rate_hz: float = 40.0
    mag_rate_hz: float = 40.0

    gps_delay_s: float = 0.10
    baro_delay_s: float = 0.04
    airspeed_delay_s: float = 0.03
    mag_delay_s: float = 0.03

    gps_pos_noise_std_m: float = 0.20
    gps_vel_noise_std_mps: float = 0.10
    baro_alt_noise_std_m: float = 0.08
    airspeed_noise_std_mps: float = 0.15
    mag_heading_noise_std_deg: float = 1.5
    gyro_noise_std_dps: float = 0.35
    accel_noise_std_mps2: float = 0.35

    gps_pos_bias_std_m: float = 0.05
    gps_vel_bias_std_mps: float = 0.03
    baro_alt_bias_std_m: float = 0.05
    airspeed_bias_std_mps: float = 0.08
    mag_heading_bias_std_deg: float = 1.0
    gyro_bias_std_dps: float = 0.25
    accel_bias_std_mps2: float = 0.08


@dataclass
class StateEstimatorCfg:
    """Estimator gains and filter time constants."""

    roll_pitch_accel_gain: float = 0.05
    roll_pitch_accel_gain_min: float = 0.0
    roll_pitch_accel_gate_sigma_mps2: float = 1.25
    roll_pitch_accel_gate_gyro_dps: float = 90.0
    roll_pitch_accel_hard_gate: bool = True
    roll_pitch_accel_gate_low_g: float = 0.9
    roll_pitch_accel_gate_high_g: float = 1.1
    roll_pitch_accel_gate_lpf_tau_s: float = 0.25
    roll_pitch_accel_correction_mode: str = "gravity_vector"
    attitude_max_pitch_deg: float = 85.0
    yaw_mag_gain: float = 0.08
    gps_pos_lpf_tau_s: float = 0.18
    gps_vel_lpf_tau_s: float = 0.12
    gps_delay_compensation: bool = True
    baro_alt_lpf_tau_s: float = 0.18
    baro_rate_lpf_tau_s: float = 0.15
    baro_delay_compensation: bool = True
    airspeed_lpf_tau_s: float = 0.12
    wind_lpf_tau_s: float = 0.8
    imu_propagation_gain: float = 0.30
    imu_accel_world_lpf_tau_s: float = 0.06
    imu_accel_world_clip_mps2: float = 30.0


class SensorStateEstimator:
    """Vectorized sensor+estimator block for rollout scripts."""

    def __init__(
        self,
        sensor_cfg: SensorSuiteCfg,
        estimator_cfg: StateEstimatorCfg,
        *,
        num_envs: int,
        device: torch.device,
        control_dt_s: float,
        noise_scale: float = 1.0,
        bias_scale: float = 1.0,
    ):
        self.sensor_cfg = sensor_cfg
        self.estimator_cfg = estimator_cfg
        self.num_envs = int(num_envs)
        self.device = device
        self.dt = float(control_dt_s)
        self.noise_scale = float(max(noise_scale, 0.0))
        self.bias_scale = float(max(bias_scale, 0.0))

        self._gps_period_s = 1.0 / max(float(sensor_cfg.gps_rate_hz), 1.0e-3)
        self._baro_period_s = 1.0 / max(float(sensor_cfg.baro_rate_hz), 1.0e-3)
        self._airspeed_period_s = 1.0 / max(float(sensor_cfg.airspeed_rate_hz), 1.0e-3)
        self._mag_period_s = 1.0 / max(float(sensor_cfg.mag_rate_hz), 1.0e-3)

        self._gps_timer_s = self._gps_period_s
        self._baro_timer_s = self._baro_period_s
        self._airspeed_timer_s = self._airspeed_period_s
        self._mag_timer_s = self._mag_period_s

        gps_delay_updates = max(int(round(float(sensor_cfg.gps_delay_s) / self._gps_period_s)), 0)
        baro_delay_updates = max(int(round(float(sensor_cfg.baro_delay_s) / self._baro_period_s)), 0)
        airspeed_delay_updates = max(int(round(float(sensor_cfg.airspeed_delay_s) / self._airspeed_period_s)), 0)
        mag_delay_updates = max(int(round(float(sensor_cfg.mag_delay_s) / self._mag_period_s)), 0)

        self._gps_delay_s_effective = float(gps_delay_updates) * self._gps_period_s
        self._baro_delay_s_effective = float(baro_delay_updates) * self._baro_period_s
        self._airspeed_delay_s_effective = float(airspeed_delay_updates) * self._airspeed_period_s
        self._mag_delay_s_effective = float(mag_delay_updates) * self._mag_period_s

        self._gps_pos_delay = _DelayBuffer(gps_delay_updates)
        self._gps_vel_delay = _DelayBuffer(gps_delay_updates)
        self._baro_delay = _DelayBuffer(baro_delay_updates)
        self._airspeed_delay = _DelayBuffer(airspeed_delay_updates)
        self._mag_delay = _DelayBuffer(mag_delay_updates)

        zeros_3 = torch.zeros((self.num_envs, 3), device=self.device)
        zeros_1 = torch.zeros((self.num_envs,), device=self.device)
        zeros_2 = torch.zeros((self.num_envs, 2), device=self.device)

        self._pos_est = zeros_3.clone()
        self._vel_est = zeros_3.clone()
        self._roll_est = zeros_1.clone()
        self._pitch_est = zeros_1.clone()
        self._yaw_est = zeros_1.clone()
        self._alt_est = zeros_1.clone()
        self._alt_rate_est = zeros_1.clone()
        self._airspeed_est = zeros_1.clone()
        self._wind_est_xy = zeros_2.clone()

        self._gyro_meas = zeros_3.clone()
        self._accel_meas = zeros_3.clone()
        self._accel_gate_lpf = zeros_3.clone()
        self._gps_pos_meas = zeros_3.clone()
        self._gps_vel_meas = zeros_3.clone()
        self._baro_alt_meas = zeros_1.clone()
        self._airspeed_meas = zeros_1.clone()
        self._mag_yaw_meas = zeros_1.clone()
        self._acc_world_est = zeros_3.clone()

        self._baro_alt_prev_meas = zeros_1.clone()
        self._vel_prev_true = zeros_3.clone()
        self._initialized = False

        self._gps_pos_bias = zeros_3.clone()
        self._gps_vel_bias = zeros_3.clone()
        self._baro_bias = zeros_1.clone()
        self._airspeed_bias = zeros_1.clone()
        self._mag_bias = zeros_1.clone()
        self._gyro_bias = zeros_3.clone()
        self._accel_bias = zeros_3.clone()

    def _randn_like(self, ref: Tensor) -> Tensor:
        return torch.randn_like(ref, device=self.device)

    def _sample_biases(self) -> None:
        c = self.sensor_cfg
        s = self.bias_scale
        self._gps_pos_bias = self._randn_like(self._gps_pos_bias) * (float(c.gps_pos_bias_std_m) * s)
        self._gps_vel_bias = self._randn_like(self._gps_vel_bias) * (float(c.gps_vel_bias_std_mps) * s)
        self._baro_bias = self._randn_like(self._baro_bias) * (float(c.baro_alt_bias_std_m) * s)
        self._airspeed_bias = self._randn_like(self._airspeed_bias) * (float(c.airspeed_bias_std_mps) * s)
        self._mag_bias = self._randn_like(self._mag_bias) * (math.radians(float(c.mag_heading_bias_std_deg)) * s)
        self._gyro_bias = self._randn_like(self._gyro_bias) * (math.radians(float(c.gyro_bias_std_dps)) * s)
        self._accel_bias = self._randn_like(self._accel_bias) * (float(c.accel_bias_std_mps2) * s)

    def reset(
        self,
        *,
        pos_local_true: Tensor,
        vel_local_true: Tensor,
        roll_true: Tensor,
        pitch_true: Tensor,
        yaw_true: Tensor,
        airspeed_true: Tensor,
        wind_local_true: Tensor | None = None,
    ) -> None:
        self._sample_biases()

        self._pos_est = pos_local_true.clone()
        self._vel_est = vel_local_true.clone()
        self._roll_est = roll_true.clone()
        self._pitch_est = pitch_true.clone()
        self._yaw_est = yaw_true.clone()
        self._alt_est = pos_local_true[:, 2].clone()
        self._alt_rate_est = vel_local_true[:, 2].clone()
        self._airspeed_est = airspeed_true.clone()
        if wind_local_true is None:
            self._wind_est_xy.zero_()
        else:
            self._wind_est_xy = wind_local_true[:, 0:2].clone()

        self._gps_pos_meas = pos_local_true.clone()
        self._gps_vel_meas = vel_local_true.clone()
        self._baro_alt_meas = pos_local_true[:, 2].clone()
        self._airspeed_meas = airspeed_true.clone()
        self._mag_yaw_meas = yaw_true.clone()

        self._baro_alt_prev_meas = self._baro_alt_meas.clone()
        self._vel_prev_true = vel_local_true.clone()
        self._gyro_meas.zero_()
        self._accel_meas.zero_()
        gravity_world = torch.zeros_like(pos_local_true)
        gravity_world[:, 2] = 9.81
        self._accel_gate_lpf = _body_from_world(roll_true, pitch_true, yaw_true, gravity_world)
        self._acc_world_est.zero_()

        self._gps_timer_s = self._gps_period_s
        self._baro_timer_s = self._baro_period_s
        self._airspeed_timer_s = self._airspeed_period_s
        self._mag_timer_s = self._mag_period_s
        self._gps_pos_delay.reset()
        self._gps_vel_delay.reset()
        self._baro_delay.reset()
        self._airspeed_delay.reset()
        self._mag_delay.reset()
        self._initialized = True

    def step(
        self,
        *,
        pos_local_true: Tensor,
        vel_local_true: Tensor,
        roll_true: Tensor,
        pitch_true: Tensor,
        yaw_true: Tensor,
        ang_vel_body_true: Tensor,
        airspeed_true: Tensor,
        imu_measurement: ImuMeasurement | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        if not self._initialized:
            self.reset(
                pos_local_true=pos_local_true,
                vel_local_true=vel_local_true,
                roll_true=roll_true,
                pitch_true=pitch_true,
                yaw_true=yaw_true,
                airspeed_true=airspeed_true,
            )

        c = self.sensor_cfg
        e = self.estimator_cfg
        noise = self.noise_scale

        vel_acc_world = (vel_local_true - self._vel_prev_true) / max(self.dt, 1.0e-6)
        self._vel_prev_true = vel_local_true.clone()

        # Accelerometer measures specific force f = a - g in body frame.
        specific_force_world = vel_acc_world.clone()
        specific_force_world[:, 2] = specific_force_world[:, 2] + 9.81
        specific_force_body = _body_from_world(roll_true, pitch_true, yaw_true, specific_force_world)

        if imu_measurement is None:
            self._gyro_meas = (
                ang_vel_body_true
                + self._gyro_bias
                + self._randn_like(ang_vel_body_true) * (math.radians(float(c.gyro_noise_std_dps)) * noise)
            )
            self._accel_meas = (
                specific_force_body
                + self._accel_bias
                + self._randn_like(specific_force_body) * (float(c.accel_noise_std_mps2) * noise)
            )
        else:
            self._gyro_meas = imu_measurement.gyro_rad_s.clone()
            self._accel_meas = imu_measurement.accel_mps2.clone()

        roll_pred, pitch_pred, yaw_pred = _integrate_euler_from_body_rates(
            self._roll_est,
            self._pitch_est,
            self._yaw_est,
            self._gyro_meas,
            self.dt,
            max_pitch_rad=math.radians(max(float(e.attitude_max_pitch_deg), 1.0)),
        )

        base_att_gain = float(max(e.roll_pitch_accel_gain, 0.0))
        acc_norm = torch.linalg.norm(self._accel_meas, dim=1)
        alpha_accel_gate = _alpha_from_tau(self.dt, float(e.roll_pitch_accel_gate_lpf_tau_s))
        self._accel_gate_lpf = self._accel_gate_lpf + alpha_accel_gate * (self._accel_meas - self._accel_gate_lpf)
        acc_lpf_norm = torch.linalg.norm(self._accel_gate_lpf, dim=1)
        corr_scale = torch.zeros_like(self._roll_est)
        if base_att_gain > 0.0:
            sigma = max(float(e.roll_pitch_accel_gate_sigma_mps2), 1.0e-3)
            acc_scale = torch.exp(-torch.square(torch.abs(acc_norm - 9.81) / sigma))

            gyro_norm = torch.linalg.norm(self._gyro_meas, dim=1)
            gyro_gate_radps = math.radians(max(float(e.roll_pitch_accel_gate_gyro_dps), 1.0))
            gyro_scale = torch.clamp(1.0 - gyro_norm / gyro_gate_radps, min=0.0, max=1.0)

            corr_scale = acc_scale * gyro_scale
            hard_gate_mask = torch.ones_like(corr_scale, dtype=torch.bool)
            if bool(e.roll_pitch_accel_hard_gate):
                gate_low = max(float(e.roll_pitch_accel_gate_low_g), 0.0) * 9.81
                gate_high = (
                    max(float(e.roll_pitch_accel_gate_high_g), float(e.roll_pitch_accel_gate_low_g) + 1.0e-3)
                    * 9.81
                )
                accel_gate = (
                    (acc_norm >= gate_low)
                    & (acc_norm <= gate_high)
                    & (acc_lpf_norm >= gate_low)
                    & (acc_lpf_norm <= gate_high)
                )
                gyro_gate = gyro_norm <= gyro_gate_radps
                hard_gate_mask = accel_gate & gyro_gate

            min_scale = max(0.0, min(float(e.roll_pitch_accel_gain_min), 1.0))
            corr_scale = torch.clamp(corr_scale, min=min_scale, max=1.0)
            corr_scale = torch.where(hard_gate_mask, corr_scale, torch.zeros_like(corr_scale))
            att_corr_gain = base_att_gain * corr_scale
        else:
            att_corr_gain = torch.zeros_like(self._roll_est)

        correction_mode = str(e.roll_pitch_accel_correction_mode).strip().lower()
        if correction_mode in {"euler", "euler_lpf", "legacy"}:
            roll_acc = torch.atan2(self._accel_meas[:, 1], self._accel_meas[:, 2])
            pitch_acc = torch.atan2(
                -self._accel_meas[:, 0],
                torch.sqrt(torch.clamp(self._accel_meas[:, 1] ** 2 + self._accel_meas[:, 2] ** 2, min=1.0e-6)),
            )
            self._roll_est = _angle_lpf(roll_pred, roll_acc, att_corr_gain)
            self._pitch_est = _angle_lpf(pitch_pred, pitch_acc, att_corr_gain)
        elif correction_mode in {"gravity_vector", "px4_gravity"}:
            up_world = torch.zeros_like(self._accel_meas)
            up_world[:, 2] = 1.0
            predicted_up_body = _body_from_world(roll_pred, pitch_pred, yaw_pred, up_world)
            measured_up_body = self._accel_meas / torch.clamp(acc_norm, min=1.0e-6).unsqueeze(1)
            tilt_correction = torch.cross(measured_up_body, predicted_up_body, dim=1)
            max_pitch_rad = math.radians(max(float(e.attitude_max_pitch_deg), 1.0))
            self._roll_est = _wrap_pi(roll_pred + att_corr_gain * tilt_correction[:, 0])
            self._pitch_est = torch.clamp(
                pitch_pred + att_corr_gain * tilt_correction[:, 1],
                min=-max_pitch_rad,
                max=max_pitch_rad,
            )
        else:
            raise ValueError(f"Unsupported roll_pitch_accel_correction_mode: {e.roll_pitch_accel_correction_mode}")
        self._yaw_est = yaw_pred

        self._mag_timer_s += self.dt
        if self._mag_timer_s >= self._mag_period_s:
            self._mag_timer_s -= self._mag_period_s
            mag_meas_now = _wrap_pi(
                yaw_true
                + self._mag_bias
                + self._randn_like(yaw_true) * (math.radians(float(c.mag_heading_noise_std_deg)) * noise)
            )
            self._mag_yaw_meas = self._mag_delay.push(mag_meas_now)
            self._yaw_est = _angle_lpf(self._yaw_est, self._mag_yaw_meas, float(max(e.yaw_mag_gain, 0.0)))

        accel_world = _world_from_body(self._roll_est, self._pitch_est, self._yaw_est, self._accel_meas)
        accel_world[:, 2] = accel_world[:, 2] - 9.81
        accel_clip = max(float(e.imu_accel_world_clip_mps2), 0.0)
        if accel_clip > 0.0:
            accel_norm = torch.linalg.norm(accel_world, dim=1, keepdim=True)
            accel_scale = torch.clamp(accel_clip / torch.clamp(accel_norm, min=1.0e-6), max=1.0)
            accel_world = accel_world * accel_scale
        alpha_accel_world = _alpha_from_tau(self.dt, float(e.imu_accel_world_lpf_tau_s))
        self._acc_world_est = self._acc_world_est + alpha_accel_world * (accel_world - self._acc_world_est)

        imu_gain = max(float(e.imu_propagation_gain), 0.0)
        vel_pred = self._vel_est + self._acc_world_est * self.dt
        pos_pred = self._pos_est + vel_pred * self.dt
        self._vel_est = self._vel_est + imu_gain * (vel_pred - self._vel_est)
        self._pos_est = self._pos_est + imu_gain * (pos_pred - self._pos_est)
        self._alt_est = self._pos_est[:, 2].clone()
        self._alt_rate_est = self._vel_est[:, 2].clone()

        self._gps_timer_s += self.dt
        if self._gps_timer_s >= self._gps_period_s:
            self._gps_timer_s -= self._gps_period_s
            gps_pos_now = (
                pos_local_true
                + self._gps_pos_bias
                + self._randn_like(pos_local_true) * (float(c.gps_pos_noise_std_m) * noise)
            )
            gps_vel_now = (
                vel_local_true
                + self._gps_vel_bias
                + self._randn_like(vel_local_true) * (float(c.gps_vel_noise_std_mps) * noise)
            )
            self._gps_pos_meas = self._gps_pos_delay.push(gps_pos_now)
            self._gps_vel_meas = self._gps_vel_delay.push(gps_vel_now)
            alpha_gps_pos = _alpha_from_tau(self._gps_period_s, float(e.gps_pos_lpf_tau_s))
            alpha_gps_vel = _alpha_from_tau(self._gps_period_s, float(e.gps_vel_lpf_tau_s))
            gps_pos_for_update = self._gps_pos_meas
            if bool(e.gps_delay_compensation) and (self._gps_delay_s_effective > 0.0):
                gps_pos_for_update = gps_pos_for_update + self._gps_vel_meas * self._gps_delay_s_effective
            self._pos_est = self._pos_est + alpha_gps_pos * (gps_pos_for_update - self._pos_est)
            self._vel_est = self._vel_est + alpha_gps_vel * (self._gps_vel_meas - self._vel_est)

        self._baro_timer_s += self.dt
        if self._baro_timer_s >= self._baro_period_s:
            self._baro_timer_s -= self._baro_period_s
            baro_now = (
                pos_local_true[:, 2]
                + self._baro_bias
                + self._randn_like(pos_local_true[:, 2]) * (float(c.baro_alt_noise_std_m) * noise)
            )
            self._baro_alt_meas = self._baro_delay.push(baro_now)
            baro_rate_raw = (self._baro_alt_meas - self._baro_alt_prev_meas) / max(self._baro_period_s, 1.0e-6)
            self._baro_alt_prev_meas = self._baro_alt_meas.clone()
            alpha_baro_alt = _alpha_from_tau(self._baro_period_s, float(e.baro_alt_lpf_tau_s))
            alpha_baro_rate = _alpha_from_tau(self._baro_period_s, float(e.baro_rate_lpf_tau_s))
            baro_alt_for_update = self._baro_alt_meas
            if bool(e.baro_delay_compensation) and (self._baro_delay_s_effective > 0.0):
                baro_alt_for_update = baro_alt_for_update + self._alt_rate_est * self._baro_delay_s_effective
            self._alt_est = self._alt_est + alpha_baro_alt * (baro_alt_for_update - self._alt_est)
            self._alt_rate_est = self._alt_rate_est + alpha_baro_rate * (baro_rate_raw - self._alt_rate_est)

        self._pos_est[:, 2] = self._alt_est
        self._vel_est[:, 2] = self._alt_rate_est

        self._airspeed_timer_s += self.dt
        if self._airspeed_timer_s >= self._airspeed_period_s:
            self._airspeed_timer_s -= self._airspeed_period_s
            airspeed_now = (
                airspeed_true
                + self._airspeed_bias
                + self._randn_like(airspeed_true) * (float(c.airspeed_noise_std_mps) * noise)
            )
            airspeed_now = torch.clamp(airspeed_now, min=0.0)
            self._airspeed_meas = self._airspeed_delay.push(airspeed_now)
            alpha_airspeed = _alpha_from_tau(self._airspeed_period_s, float(e.airspeed_lpf_tau_s))
            self._airspeed_est = self._airspeed_est + alpha_airspeed * (self._airspeed_meas - self._airspeed_est)

        air_vel_xy_est = self._airspeed_est.unsqueeze(1) * torch.stack(
            (torch.cos(self._yaw_est), torch.sin(self._yaw_est)), dim=1
        )
        alpha_wind = _alpha_from_tau(self.dt, float(e.wind_lpf_tau_s))
        self._wind_est_xy = self._wind_est_xy + alpha_wind * ((self._vel_est[:, 0:2] - air_vel_xy_est) - self._wind_est_xy)

        state = {
            "pos_local": self._pos_est.clone(),
            "ground_vel_local": self._vel_est.clone(),
            "roll": self._roll_est.clone(),
            "pitch": self._pitch_est.clone(),
            "yaw": self._yaw_est.clone(),
            "ang_vel_body": self._gyro_meas.clone(),
            "wind_xy": self._wind_est_xy.clone(),
            "airspeed": self._airspeed_est.clone(),
        }

        diag = {
            "gps_x": self._gps_pos_meas[:, 0].clone(),
            "gps_y": self._gps_pos_meas[:, 1].clone(),
            "gps_z": self._gps_pos_meas[:, 2].clone(),
            "gps_vx": self._gps_vel_meas[:, 0].clone(),
            "gps_vy": self._gps_vel_meas[:, 1].clone(),
            "gps_vz": self._gps_vel_meas[:, 2].clone(),
            "baro_alt": self._baro_alt_meas.clone(),
            "airspeed_meas": self._airspeed_meas.clone(),
            "mag_yaw": self._mag_yaw_meas.clone(),
            "gyro_x": self._gyro_meas[:, 0].clone(),
            "gyro_y": self._gyro_meas[:, 1].clone(),
            "gyro_z": self._gyro_meas[:, 2].clone(),
            "accel_x": self._accel_meas[:, 0].clone(),
            "accel_y": self._accel_meas[:, 1].clone(),
            "accel_z": self._accel_meas[:, 2].clone(),
            "acc_world_x": self._acc_world_est[:, 0].clone(),
            "acc_world_y": self._acc_world_est[:, 1].clone(),
            "acc_world_z": self._acc_world_est[:, 2].clone(),
            "att_corr_gain": att_corr_gain.clone(),
            "att_corr_scale": corr_scale.clone(),
            "accel_norm": acc_norm.clone(),
            "accel_gate_lpf_norm": acc_lpf_norm.clone(),
        }
        return state, diag

from __future__ import annotations

import torch

try:
    from flapping_bot.px4_like.imu_provider import ImuMeasurement
    from flapping_bot.px4_like.state_estimation import SensorStateEstimator, SensorSuiteCfg, StateEstimatorCfg
except ImportError:
    from flapping_bot.flapping_bot.px4_like.imu_provider import ImuMeasurement
    from flapping_bot.flapping_bot.px4_like.state_estimation import (
        SensorStateEstimator,
        SensorSuiteCfg,
        StateEstimatorCfg,
    )


def _make_estimator() -> SensorStateEstimator:
    return SensorStateEstimator(
        sensor_cfg=SensorSuiteCfg(),
        estimator_cfg=StateEstimatorCfg(),
        num_envs=1,
        device=torch.device("cpu"),
        control_dt_s=0.01,
        noise_scale=0.0,
        bias_scale=0.0,
    )


def test_estimator_prefers_external_imu_measurements_when_supplied() -> None:
    estimator = _make_estimator()
    external_imu = ImuMeasurement(
        gyro_rad_s=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        accel_mps2=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
    )

    state, diag = estimator.step(
        pos_local_true=torch.zeros((1, 3), dtype=torch.float32),
        vel_local_true=torch.zeros((1, 3), dtype=torch.float32),
        roll_true=torch.zeros((1,), dtype=torch.float32),
        pitch_true=torch.zeros((1,), dtype=torch.float32),
        yaw_true=torch.zeros((1,), dtype=torch.float32),
        ang_vel_body_true=torch.zeros((1, 3), dtype=torch.float32),
        airspeed_true=torch.zeros((1,), dtype=torch.float32),
        imu_measurement=external_imu,
    )

    assert torch.allclose(state["ang_vel_body"], external_imu.gyro_rad_s)
    assert torch.allclose(diag["gyro_x"], torch.tensor([0.1], dtype=torch.float32))
    assert torch.allclose(diag["gyro_y"], torch.tensor([0.2], dtype=torch.float32))
    assert torch.allclose(diag["gyro_z"], torch.tensor([0.3], dtype=torch.float32))
    assert torch.allclose(diag["accel_x"], torch.tensor([1.0], dtype=torch.float32))
    assert torch.allclose(diag["accel_y"], torch.tensor([2.0], dtype=torch.float32))
    assert torch.allclose(diag["accel_z"], torch.tensor([3.0], dtype=torch.float32))


def test_estimator_uses_internal_synthetic_imu_when_external_not_supplied() -> None:
    estimator = _make_estimator()

    state, diag = estimator.step(
        pos_local_true=torch.zeros((1, 3), dtype=torch.float32),
        vel_local_true=torch.zeros((1, 3), dtype=torch.float32),
        roll_true=torch.zeros((1,), dtype=torch.float32),
        pitch_true=torch.zeros((1,), dtype=torch.float32),
        yaw_true=torch.zeros((1,), dtype=torch.float32),
        ang_vel_body_true=torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32),
        airspeed_true=torch.zeros((1,), dtype=torch.float32),
    )

    assert torch.allclose(state["ang_vel_body"], torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32))
    assert torch.allclose(diag["accel_x"], torch.tensor([0.0], dtype=torch.float32))
    assert torch.allclose(diag["accel_y"], torch.tensor([0.0], dtype=torch.float32))
    assert torch.allclose(diag["accel_z"], torch.tensor([9.81], dtype=torch.float32))


def test_estimator_hard_gates_accel_tilt_correction_under_dynamic_acceleration() -> None:
    estimator = SensorStateEstimator(
        sensor_cfg=SensorSuiteCfg(),
        estimator_cfg=StateEstimatorCfg(
            roll_pitch_accel_gain=1.0,
            roll_pitch_accel_hard_gate=True,
            roll_pitch_accel_gate_low_g=0.9,
            roll_pitch_accel_gate_high_g=1.1,
            roll_pitch_accel_correction_mode="gravity_vector",
        ),
        num_envs=1,
        device=torch.device("cpu"),
        control_dt_s=0.01,
        noise_scale=0.0,
        bias_scale=0.0,
    )
    external_imu = ImuMeasurement(
        gyro_rad_s=torch.zeros((1, 3), dtype=torch.float32),
        accel_mps2=torch.tensor([[0.0, 9.81, 9.81]], dtype=torch.float32),
    )

    state, diag = estimator.step(
        pos_local_true=torch.zeros((1, 3), dtype=torch.float32),
        vel_local_true=torch.zeros((1, 3), dtype=torch.float32),
        roll_true=torch.zeros((1,), dtype=torch.float32),
        pitch_true=torch.zeros((1,), dtype=torch.float32),
        yaw_true=torch.zeros((1,), dtype=torch.float32),
        ang_vel_body_true=torch.zeros((1, 3), dtype=torch.float32),
        airspeed_true=torch.zeros((1,), dtype=torch.float32),
        imu_measurement=external_imu,
    )

    assert torch.allclose(state["roll"], torch.zeros((1,), dtype=torch.float32), atol=1.0e-5)
    assert torch.allclose(state["pitch"], torch.zeros((1,), dtype=torch.float32), atol=1.0e-5)
    assert torch.allclose(diag["att_corr_gain"], torch.zeros((1,), dtype=torch.float32), atol=1.0e-6)


def test_estimator_hard_gate_overrides_minimum_accel_tilt_gain() -> None:
    estimator = SensorStateEstimator(
        sensor_cfg=SensorSuiteCfg(),
        estimator_cfg=StateEstimatorCfg(
            roll_pitch_accel_gain=1.0,
            roll_pitch_accel_gain_min=0.5,
            roll_pitch_accel_hard_gate=True,
            roll_pitch_accel_gate_low_g=0.9,
            roll_pitch_accel_gate_high_g=1.1,
            roll_pitch_accel_correction_mode="gravity_vector",
        ),
        num_envs=1,
        device=torch.device("cpu"),
        control_dt_s=0.01,
        noise_scale=0.0,
        bias_scale=0.0,
    )

    state, diag = estimator.step(
        pos_local_true=torch.zeros((1, 3), dtype=torch.float32),
        vel_local_true=torch.zeros((1, 3), dtype=torch.float32),
        roll_true=torch.zeros((1,), dtype=torch.float32),
        pitch_true=torch.zeros((1,), dtype=torch.float32),
        yaw_true=torch.zeros((1,), dtype=torch.float32),
        ang_vel_body_true=torch.zeros((1, 3), dtype=torch.float32),
        airspeed_true=torch.zeros((1,), dtype=torch.float32),
        imu_measurement=ImuMeasurement(
            gyro_rad_s=torch.zeros((1, 3), dtype=torch.float32),
            accel_mps2=torch.tensor([[0.0, 9.81, 9.81]], dtype=torch.float32),
        ),
    )

    assert torch.allclose(state["roll"], torch.zeros((1,), dtype=torch.float32), atol=1.0e-5)
    assert torch.allclose(diag["att_corr_gain"], torch.zeros((1,), dtype=torch.float32), atol=1.0e-6)


def test_estimator_gravity_vector_correction_tracks_static_tilt() -> None:
    estimator = SensorStateEstimator(
        sensor_cfg=SensorSuiteCfg(),
        estimator_cfg=StateEstimatorCfg(
            roll_pitch_accel_gain=1.0,
            roll_pitch_accel_hard_gate=True,
            roll_pitch_accel_gate_lpf_tau_s=0.0,
            roll_pitch_accel_correction_mode="gravity_vector",
        ),
        num_envs=1,
        device=torch.device("cpu"),
        control_dt_s=0.01,
        noise_scale=0.0,
        bias_scale=0.0,
    )
    roll_rad = torch.tensor([0.10], dtype=torch.float32)
    static_gravity_body = torch.tensor(
        [[0.0, 9.81 * torch.sin(roll_rad).item(), 9.81 * torch.cos(roll_rad).item()]], dtype=torch.float32
    )

    state, diag = estimator.step(
        pos_local_true=torch.zeros((1, 3), dtype=torch.float32),
        vel_local_true=torch.zeros((1, 3), dtype=torch.float32),
        roll_true=torch.zeros((1,), dtype=torch.float32),
        pitch_true=torch.zeros((1,), dtype=torch.float32),
        yaw_true=torch.zeros((1,), dtype=torch.float32),
        ang_vel_body_true=torch.zeros((1, 3), dtype=torch.float32),
        airspeed_true=torch.zeros((1,), dtype=torch.float32),
        imu_measurement=ImuMeasurement(
            gyro_rad_s=torch.zeros((1, 3), dtype=torch.float32),
            accel_mps2=static_gravity_body,
        ),
    )

    assert state["roll"].item() > 0.08
    assert torch.allclose(state["pitch"], torch.zeros((1,), dtype=torch.float32), atol=1.0e-5)
    assert diag["att_corr_gain"].item() > 0.0

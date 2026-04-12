from __future__ import annotations

import pytest
import torch

try:
    from flapping_bot.px4_like.imu_provider import ImuMeasurement, SyntheticImuProvider, build_imu_provider
except ImportError:
    from flapping_bot.flapping_bot.px4_like.imu_provider import ImuMeasurement, SyntheticImuProvider, build_imu_provider

try:
    from flapping_bot.px4_like.isaacsim_imu_adapter import ISAACSIM_IMU_RUNTIME_AVAILABLE, IsaacSimImuProvider
except ImportError:
    from flapping_bot.flapping_bot.px4_like.isaacsim_imu_adapter import (
        ISAACSIM_IMU_RUNTIME_AVAILABLE,
        IsaacSimImuProvider,
    )


def test_synthetic_imu_provider_returns_required_fields() -> None:
    provider = SyntheticImuProvider()
    measurement = provider.build_from_truth(
        ang_vel_body=torch.zeros((2, 3), dtype=torch.float32),
        specific_force_body=torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 9.0]], dtype=torch.float32),
    )

    assert isinstance(measurement, ImuMeasurement)
    assert measurement.gyro_rad_s.shape == (2, 3)
    assert measurement.accel_mps2.shape == (2, 3)
    assert torch.allclose(measurement.gyro_rad_s, torch.zeros((2, 3), dtype=torch.float32))
    assert torch.allclose(
        measurement.accel_mps2,
        torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 9.0]], dtype=torch.float32),
    )


def test_build_imu_provider_returns_synthetic_backend() -> None:
    provider = build_imu_provider("synthetic")
    assert isinstance(provider, SyntheticImuProvider)
    assert provider.backend_name == "synthetic"


def test_build_imu_provider_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        build_imu_provider("bogus")


def test_isaacsim_imu_adapter_imports_safely() -> None:
    provider = IsaacSimImuProvider()
    measurement = provider.build_from_sensor_data(
        ang_vel_b=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        lin_acc_b=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
    )

    assert provider.backend_name == "isaacsim"
    assert provider.runtime_available is ISAACSIM_IMU_RUNTIME_AVAILABLE
    assert isinstance(measurement, ImuMeasurement)
    assert torch.allclose(measurement.gyro_rad_s, torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32))
    assert torch.allclose(measurement.accel_mps2, torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))


def test_isaacsim_imu_provider_can_normalize_live_sensor_data_without_runtime() -> None:
    provider = IsaacSimImuProvider()

    class _FakeSensor:
        class _Data:
            ang_vel_b = torch.tensor([[0.7, 0.8, 0.9]], dtype=torch.float32)
            lin_acc_b = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)

        data = _Data()

    measurement = provider.build_from_sensor(_FakeSensor())
    assert torch.allclose(measurement.gyro_rad_s, torch.tensor([[0.7, 0.8, 0.9]], dtype=torch.float32))
    assert torch.allclose(measurement.accel_mps2, torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32))


def test_build_imu_provider_accepts_isaacsim_backend_name() -> None:
    provider = build_imu_provider("isaacsim")
    assert isinstance(provider, IsaacSimImuProvider)
    assert provider.backend_name == "isaacsim"

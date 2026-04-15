from __future__ import annotations

import pytest
import torch

try:
    from flapping_bot.px4_like.imu_provider import ImuMeasurement, SyntheticImuProvider, build_imu_provider
except ImportError:
    from flapping_bot.flapping_bot.px4_like.imu_provider import ImuMeasurement, SyntheticImuProvider, build_imu_provider

try:
    import flapping_bot.px4_like.isaacsim_imu_adapter as imu_adapter_module
    from flapping_bot.px4_like.isaacsim_imu_adapter import (
        ISAACSIM_IMU_RUNTIME_AVAILABLE,
        IsaacSimImuProvider,
        IsaacSimImuSensorSpec,
        resolve_base_body_com_offset_b,
    )
except ImportError:
    import flapping_bot.flapping_bot.px4_like.isaacsim_imu_adapter as imu_adapter_module
    from flapping_bot.flapping_bot.px4_like.isaacsim_imu_adapter import (
        ISAACSIM_IMU_RUNTIME_AVAILABLE,
        IsaacSimImuProvider,
        IsaacSimImuSensorSpec,
        resolve_base_body_com_offset_b,
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


def test_resolve_base_body_com_offset_b_uses_first_base_body_id() -> None:
    robot = type(
        "FakeRobot",
        (),
        {
            "data": type(
                "FakeData",
                (),
                {
                    "body_com_pos_b": torch.tensor(
                        [[[0.15, -0.02, 0.03], [0.40, 0.50, 0.60]]],
                        dtype=torch.float32,
                    )
                },
            )()
        },
    )()

    offset = resolve_base_body_com_offset_b(robot, [0])

    assert offset == pytest.approx((0.15, -0.02, 0.03))


def test_isaacsim_imu_provider_make_sensor_cfg_forwards_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeOffsetCfg:
        def __init__(self, pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)):
            self.pos = pos
            self.rot = rot

    class FakeImuCfg:
        OffsetCfg = FakeOffsetCfg

        def __init__(self, *, prim_path, update_period, gravity_bias, offset):
            captured["prim_path"] = prim_path
            captured["update_period"] = update_period
            captured["gravity_bias"] = gravity_bias
            captured["offset"] = offset

    monkeypatch.setattr(imu_adapter_module, "ISAACSIM_IMU_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(imu_adapter_module, "ImuCfg", FakeImuCfg)

    provider = IsaacSimImuProvider()
    provider.make_sensor_cfg(
        IsaacSimImuSensorSpec(
            prim_path="/World/envs/env_.*/Robot/base_link",
            update_period=0.02,
            offset_pos_b=(0.15, -0.02, 0.03),
        )
    )

    assert captured["prim_path"] == "/World/envs/env_.*/Robot/base_link"
    assert captured["update_period"] == pytest.approx(0.02)
    assert captured["gravity_bias"] == pytest.approx((0.0, 0.0, 9.81))
    assert captured["offset"].pos == pytest.approx((0.15, -0.02, 0.03))

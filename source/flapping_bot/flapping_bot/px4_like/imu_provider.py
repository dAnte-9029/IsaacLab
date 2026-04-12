"""IMU provider abstraction for synthetic and external backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class ImuMeasurement:
    """Normalized IMU sample in body frame."""

    gyro_rad_s: Tensor
    accel_mps2: Tensor


class ImuProvider:
    """Base IMU provider interface."""

    backend_name: str = "base"

    def build_from_truth(self, *, ang_vel_body: Tensor, specific_force_body: Tensor) -> ImuMeasurement:
        raise NotImplementedError


class SyntheticImuProvider(ImuProvider):
    """Deterministic provider that forwards synthetic truth-derived IMU signals."""

    backend_name = "synthetic"

    def build_from_truth(self, *, ang_vel_body: Tensor, specific_force_body: Tensor) -> ImuMeasurement:
        return ImuMeasurement(
            gyro_rad_s=ang_vel_body.clone(),
            accel_mps2=specific_force_body.clone(),
        )


def build_imu_provider(backend_name: str) -> ImuProvider:
    """Construct one IMU provider from its backend name."""
    normalized = str(backend_name).strip().lower()
    if normalized == "synthetic":
        return SyntheticImuProvider()
    if normalized == "isaacsim":
        from .isaacsim_imu_adapter import IsaacSimImuProvider

        return IsaacSimImuProvider()
    raise ValueError(f"unsupported imu provider backend: {backend_name!r}")

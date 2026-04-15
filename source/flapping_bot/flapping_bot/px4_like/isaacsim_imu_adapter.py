"""Repo-local adapter for Isaac Lab's IMU sensor wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .imu_provider import ImuMeasurement, ImuProvider

try:
    from isaaclab.sensors.imu import Imu, ImuCfg
except Exception as exc:  # pragma: no cover - exercised indirectly through runtime_available flag
    ISAACSIM_IMU_RUNTIME_AVAILABLE = False
    ISAACSIM_IMU_IMPORT_ERROR: Exception | None = exc
    Imu = Any  # type: ignore[assignment]
    ImuCfg = Any  # type: ignore[assignment]
else:
    ISAACSIM_IMU_RUNTIME_AVAILABLE = True
    ISAACSIM_IMU_IMPORT_ERROR = None


@dataclass(frozen=True)
class IsaacSimImuSensorSpec:
    """Minimal sensor specification for one Isaac Lab IMU binding."""

    prim_path: str
    update_period: float = 0.0
    gravity_bias: tuple[float, float, float] = (0.0, 0.0, 9.81)
    offset_pos_b: tuple[float, float, float] = (0.0, 0.0, 0.0)


def resolve_base_body_com_offset_b(robot, base_body_ids) -> tuple[float, float, float]:
    """Return the first base-body COM offset in the base-link frame."""
    if base_body_ids is None or len(base_body_ids) == 0:
        return (0.0, 0.0, 0.0)
    base_body_id = int(base_body_ids[0])
    body_com_pos_b = robot.data.body_com_pos_b
    return tuple(float(v) for v in body_com_pos_b[0, base_body_id, 0:3].tolist())


class IsaacSimImuProvider(ImuProvider):
    """Normalize Isaac Lab IMU sensor output into the rollout IMU contract."""

    backend_name = "isaacsim"

    def __init__(self):
        self.runtime_available = ISAACSIM_IMU_RUNTIME_AVAILABLE

    def build_from_truth(self, *, ang_vel_body, specific_force_body) -> ImuMeasurement:
        return self.build_from_sensor_data(ang_vel_b=ang_vel_body, lin_acc_b=specific_force_body)

    def build_from_sensor_data(self, *, ang_vel_b, lin_acc_b) -> ImuMeasurement:
        return ImuMeasurement(
            gyro_rad_s=ang_vel_b.clone(),
            accel_mps2=lin_acc_b.clone(),
        )

    def build_from_sensor(self, sensor) -> ImuMeasurement:
        """Read one live Isaac Lab IMU sensor sample."""
        return self.build_from_sensor_data(
            ang_vel_b=sensor.data.ang_vel_b,
            lin_acc_b=sensor.data.lin_acc_b,
        )

    def make_sensor_cfg(self, spec: IsaacSimImuSensorSpec):
        """Create an Isaac Lab IMU config when the runtime is available."""
        if not self.runtime_available:
            raise RuntimeError("Isaac Sim IMU runtime is unavailable") from ISAACSIM_IMU_IMPORT_ERROR
        return ImuCfg(
            prim_path=str(spec.prim_path),
            update_period=float(spec.update_period),
            gravity_bias=tuple(float(v) for v in spec.gravity_bias),
            offset=ImuCfg.OffsetCfg(
                pos=tuple(float(v) for v in spec.offset_pos_b),
            ),
        )

    def create_sensor(self, spec: IsaacSimImuSensorSpec):
        """Instantiate one Isaac Lab IMU sensor."""
        return Imu(self.make_sensor_cfg(spec))

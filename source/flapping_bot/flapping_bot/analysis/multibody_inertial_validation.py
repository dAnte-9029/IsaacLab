"""Signal and momentum analysis for the PhysX multibody inertial benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class HarmonicFit:
    """Least-squares fundamental harmonic represented as ``A*sin(w*t+phi)``."""

    amplitude: float
    phase_rad: float
    rms_residual: float
    dc_offset: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def fit_fundamental_harmonic(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    frequency_hz: float,
    nuisance_harmonics: int = 3,
) -> HarmonicFit:
    """Fit a fundamental while absorbing drift and higher harmonics."""

    time = np.asarray(time_s, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    if time.ndim != 1 or signal.shape != time.shape or time.size < 8:
        raise ValueError("time_s and values must be equal one-dimensional arrays with at least eight samples.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        raise ValueError("Harmonic-fit inputs must be finite.")
    omega = 2.0 * math.pi * float(frequency_hz)
    columns = [np.ones_like(time), time - time.mean()]
    for harmonic in range(1, int(nuisance_harmonics) + 1):
        columns.extend(
            [
                np.sin(harmonic * omega * time),
                np.cos(harmonic * omega * time),
            ]
        )
    design = np.column_stack(columns)
    coefficients, _, _, _ = np.linalg.lstsq(design, signal, rcond=None)
    sin_coefficient = float(coefficients[2])
    cos_coefficient = float(coefficients[3])
    fitted = design @ coefficients
    return HarmonicFit(
        amplitude=math.hypot(sin_coefficient, cos_coefficient),
        phase_rad=math.atan2(cos_coefficient, sin_coefficient),
        rms_residual=float(np.sqrt(np.mean(np.square(signal - fitted)))),
        dc_offset=float(coefficients[0]),
    )


def wrapped_phase_difference_deg(phase_a_rad: float, phase_b_rad: float) -> float:
    """Return ``phase_a - phase_b`` wrapped to ``[-180, 180)`` degrees."""

    difference = (float(phase_a_rad) - float(phase_b_rad) + math.pi) % (2.0 * math.pi) - math.pi
    return math.degrees(difference)


def _matrix_from_quat_wxyz(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert normalized wxyz quaternions to rotation matrices."""

    quaternion = quaternion_wxyz / torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True)
    w, x, y, z = torch.unbind(quaternion, dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x.square() + z.square()),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def compute_total_momentum_about_system_com_w(
    *,
    masses_kg: torch.Tensor,
    inertia_about_com_principal_kg_m2: torch.Tensor,
    com_positions_w_m: torch.Tensor,
    com_quaternions_wxyz: torch.Tensor,
    com_linear_velocities_w_m_s: torch.Tensor,
    angular_velocities_w_rad_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute system COM, linear momentum and angular momentum in world axes.

    Args:
        masses_kg: Shape ``(B,)``.
        inertia_about_com_principal_kg_m2: Shape ``(B,3,3)`` in each body's
            principal COM frame.
        com_positions_w_m: Shape ``(B,3)``.
        com_quaternions_wxyz: Principal COM frame orientation, shape ``(B,4)``.
        com_linear_velocities_w_m_s: Shape ``(B,3)``.
        angular_velocities_w_rad_s: Shape ``(B,3)``.

    Returns:
        ``(system_com_w_m, linear_momentum_w_kg_m_s,
        angular_momentum_about_system_com_w_kg_m2_s)``.
    """

    body_count = masses_kg.shape[0]
    if masses_kg.ndim != 1 or body_count == 0:
        raise ValueError("masses_kg must have non-empty shape (B,).")
    expected_shapes = (
        (inertia_about_com_principal_kg_m2, (body_count, 3, 3)),
        (com_positions_w_m, (body_count, 3)),
        (com_quaternions_wxyz, (body_count, 4)),
        (com_linear_velocities_w_m_s, (body_count, 3)),
        (angular_velocities_w_rad_s, (body_count, 3)),
    )
    if any(tensor.shape != shape for tensor, shape in expected_shapes):
        raise ValueError("Momentum inputs have inconsistent body dimensions.")

    total_mass = masses_kg.sum()
    system_com = torch.sum(masses_kg[:, None] * com_positions_w_m, dim=0) / total_mass
    linear_momentum = torch.sum(masses_kg[:, None] * com_linear_velocities_w_m_s, dim=0)
    rotations = _matrix_from_quat_wxyz(com_quaternions_wxyz)
    inertia_w = rotations @ inertia_about_com_principal_kg_m2 @ rotations.transpose(-1, -2)
    spin = torch.einsum("bij,bj->bi", inertia_w, angular_velocities_w_rad_s)
    orbital = torch.cross(
        com_positions_w_m - system_com,
        masses_kg[:, None] * com_linear_velocities_w_m_s,
        dim=-1,
    )
    angular_momentum = torch.sum(spin + orbital, dim=0)
    return system_com, linear_momentum, angular_momentum

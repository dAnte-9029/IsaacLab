"""Short-window offline rigid-body rollout for effective-wrench backends."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


STATE_COLUMNS = [
    "vehicle_local_position.x",
    "vehicle_local_position.y",
    "vehicle_local_position.z",
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
    "vehicle_attitude.q[0]",
    "vehicle_attitude.q[1]",
    "vehicle_attitude.q[2]",
    "vehicle_attitude.q[3]",
    "vehicle_angular_velocity.xyz[0]",
    "vehicle_angular_velocity.xyz[1]",
    "vehicle_angular_velocity.xyz[2]",
]

WRENCH_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]


@dataclass(frozen=True)
class RigidBodyParams:
    """Rigid-body parameters used by the label-generation pipeline."""

    mass_kg: float
    inertia_b_kg_m2: np.ndarray
    gravity_m_s2: float = 9.81

    @classmethod
    def default_flapper(cls) -> "RigidBodyParams":
        return cls(
            mass_kg=0.95,
            inertia_b_kg_m2=np.asarray(
                [
                    [0.000100393269, 0.000004069196, -0.000000389614],
                    [0.000004069196, 0.000172763445, -0.000000237533],
                    [-0.000000389614, -0.000000237533, 0.000077498594],
                ],
                dtype=float,
            ),
            gravity_m_s2=9.81,
        )


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def _rotation_body_to_ned(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _integrate_quaternion_body_rates(q: np.ndarray, omega_b: np.ndarray, dt: float) -> np.ndarray:
    omega_norm = float(np.linalg.norm(omega_b))
    if omega_norm <= 1.0e-12:
        return _normalize_quat(q)
    angle = omega_norm * dt
    axis = omega_b / omega_norm
    delta = np.concatenate(([math.cos(0.5 * angle)], axis * math.sin(0.5 * angle)))
    return _normalize_quat(_quat_multiply(q, delta))


def _required_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame.loc[:, columns].astype(float)


def integrate_wrench_window(
    frame: pd.DataFrame,
    wrench: pd.DataFrame,
    *,
    params: RigidBodyParams | None = None,
    max_omega_radps: float = 200.0,
) -> pd.DataFrame:
    """Integrate a single real-log window with a body-frame effective-wrench sequence.

    Coordinates follow the dataset metadata: body frame is FRD and local frame is NED.
    The wrench is the non-gravity external wrench, so gravity is added during integration.
    """

    params = params or RigidBodyParams.default_flapper()
    if len(frame) != len(wrench):
        raise ValueError("frame and wrench must have the same length.")
    if len(frame) < 1:
        raise ValueError("window must contain at least one sample.")

    state = _required_frame(frame, STATE_COLUMNS)
    wrench_values = _required_frame(wrench, WRENCH_COLUMNS).to_numpy(dtype=float)
    time = frame["time_s"].to_numpy(dtype=float) if "time_s" in frame.columns else np.arange(len(frame), dtype=float)

    pos = state.loc[state.index[0], STATE_COLUMNS[0:3]].to_numpy(dtype=float)
    vel = state.loc[state.index[0], STATE_COLUMNS[3:6]].to_numpy(dtype=float)
    quat = _normalize_quat(state.loc[state.index[0], STATE_COLUMNS[6:10]].to_numpy(dtype=float))
    omega = state.loc[state.index[0], STATE_COLUMNS[10:13]].to_numpy(dtype=float)

    inertia = np.asarray(params.inertia_b_kg_m2, dtype=float)
    inertia_inv = np.linalg.inv(inertia)
    gravity_n = np.asarray([0.0, 0.0, float(params.gravity_m_s2)], dtype=float)

    rows: list[dict[str, float]] = []
    diverged = False
    for idx in range(len(frame)):
        finite_state = np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(quat).all() and np.isfinite(omega).all()
        if not finite_state:
            diverged = True
        rows.append(
            {
                "time_s": float(time[idx]),
                "pred_x": float(pos[0]),
                "pred_y": float(pos[1]),
                "pred_z": float(pos[2]),
                "pred_vx": float(vel[0]),
                "pred_vy": float(vel[1]),
                "pred_vz": float(vel[2]),
                "pred_qw": float(quat[0]),
                "pred_qx": float(quat[1]),
                "pred_qy": float(quat[2]),
                "pred_qz": float(quat[3]),
                "pred_wx": float(omega[0]),
                "pred_wy": float(omega[1]),
                "pred_wz": float(omega[2]),
                "diverged": bool(diverged),
            }
        )
        if idx == len(frame) - 1 or diverged:
            break
        dt = float(time[idx + 1] - time[idx])
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 0.01

        force_b = wrench_values[idx, 0:3]
        moment_b = wrench_values[idx, 3:6]
        rot_nb = _rotation_body_to_ned(quat)
        acc_n = rot_nb @ (force_b / float(params.mass_kg)) + gravity_n
        pos = pos + vel * dt + 0.5 * acc_n * dt * dt
        vel = vel + acc_n * dt

        angular_momentum = inertia @ omega
        alpha = inertia_inv @ (moment_b - np.cross(omega, angular_momentum))
        omega_mid = omega + 0.5 * alpha * dt
        omega = omega + alpha * dt
        if (
            (not np.isfinite(omega).all())
            or (not np.isfinite(omega_mid).all())
            or float(np.linalg.norm(omega)) > float(max_omega_radps)
            or float(np.linalg.norm(omega_mid)) > float(max_omega_radps)
        ):
            diverged = True
            pos[:] = np.nan
            vel[:] = np.nan
            quat[:] = np.nan
            omega[:] = np.nan
            continue
        quat = _integrate_quaternion_body_rates(quat, omega_mid, dt)

    if len(rows) < len(frame):
        for idx in range(len(rows), len(frame)):
            rows.append(
                {
                    "time_s": float(time[idx]),
                    "pred_x": np.nan,
                    "pred_y": np.nan,
                    "pred_z": np.nan,
                    "pred_vx": np.nan,
                    "pred_vy": np.nan,
                    "pred_vz": np.nan,
                    "pred_qw": np.nan,
                    "pred_qx": np.nan,
                    "pred_qy": np.nan,
                    "pred_qz": np.nan,
                    "pred_wx": np.nan,
                    "pred_wy": np.nan,
                    "pred_wz": np.nan,
                    "diverged": True,
                }
            )

    return pd.DataFrame(rows)


def quaternion_angle_error_rad(q_pred: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    q_pred = np.asarray(q_pred, dtype=float)
    q_true = np.asarray(q_true, dtype=float)
    dots = np.abs(np.sum(q_pred * q_true, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return 2.0 * np.arccos(dots)


def rollout_error_summary(real_window: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, float]:
    """Return trajectory errors for one integrated window."""

    real = _required_frame(real_window, STATE_COLUMNS)
    pos_true = real.loc[:, STATE_COLUMNS[0:3]].to_numpy(dtype=float)
    vel_true = real.loc[:, STATE_COLUMNS[3:6]].to_numpy(dtype=float)
    quat_true = real.loc[:, STATE_COLUMNS[6:10]].to_numpy(dtype=float)
    omega_true = real.loc[:, STATE_COLUMNS[10:13]].to_numpy(dtype=float)

    pos_pred = predicted.loc[:, ["pred_x", "pred_y", "pred_z"]].to_numpy(dtype=float)
    vel_pred = predicted.loc[:, ["pred_vx", "pred_vy", "pred_vz"]].to_numpy(dtype=float)
    quat_pred = predicted.loc[:, ["pred_qw", "pred_qx", "pred_qy", "pred_qz"]].to_numpy(dtype=float)
    omega_pred = predicted.loc[:, ["pred_wx", "pred_wy", "pred_wz"]].to_numpy(dtype=float)

    pos_err = np.linalg.norm(pos_pred - pos_true, axis=1)
    vel_err = np.linalg.norm(vel_pred - vel_true, axis=1)
    omega_err = np.linalg.norm(omega_pred - omega_true, axis=1)
    attitude_err = quaternion_angle_error_rad(quat_pred, quat_true)
    finite = np.isfinite(pos_err) & np.isfinite(vel_err) & np.isfinite(omega_err) & np.isfinite(attitude_err)
    diverged = bool(predicted["diverged"].any()) if "diverged" in predicted.columns else False
    if not finite.any():
        return {
            "pos_rmse_m": np.nan,
            "vel_rmse_mps": np.nan,
            "att_rmse_deg": np.nan,
            "omega_rmse_radps": np.nan,
            "final_pos_err_m": np.nan,
            "final_vel_err_mps": np.nan,
            "final_att_err_deg": np.nan,
            "final_omega_err_radps": np.nan,
            "diverged": diverged,
        }
    return {
        "pos_rmse_m": float(np.sqrt(np.mean(pos_err[finite] * pos_err[finite]))),
        "vel_rmse_mps": float(np.sqrt(np.mean(vel_err[finite] * vel_err[finite]))),
        "att_rmse_deg": float(np.degrees(np.sqrt(np.mean(attitude_err[finite] * attitude_err[finite])))),
        "omega_rmse_radps": float(np.sqrt(np.mean(omega_err[finite] * omega_err[finite]))),
        "final_pos_err_m": float(pos_err[-1]) if np.isfinite(pos_err[-1]) else np.nan,
        "final_vel_err_mps": float(vel_err[-1]) if np.isfinite(vel_err[-1]) else np.nan,
        "final_att_err_deg": float(np.degrees(attitude_err[-1])) if np.isfinite(attitude_err[-1]) else np.nan,
        "final_omega_err_radps": float(omega_err[-1]) if np.isfinite(omega_err[-1]) else np.nan,
        "diverged": diverged,
    }

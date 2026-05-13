"""Gain calibration helpers for DeLaurier effective-wrench baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]


@dataclass(frozen=True)
class ParameterBound:
    """Bound and nominal value for one physical calibration parameter."""

    low: float
    high: float
    nominal: float


PHYSICAL_PARAMETER_BOUNDS: dict[str, ParameterBound] = {
    "wing_normal_force_scale": ParameterBound(0.5, 1.8, 1.0),
    "wing_chordwise_force_scale": ParameterBound(0.2, 2.0, 1.0),
    "delaurier_theta_w_deg": ParameterBound(-8.0, 8.0, 0.0),
    "twist_eta_max_deg": ParameterBound(0.0, 25.0, 10.0),
    "delaurier_induced_drag_efficiency": ParameterBound(0.3, 1.2, 0.8),
    "fuselage_drag_cda": ParameterBound(0.0001, 0.02, 0.005),
    "tail_lift_scale": ParameterBound(0.3, 2.0, 1.0),
    "phase_delay_s": ParameterBound(-0.04, 0.04, 0.0),
}

@dataclass(frozen=True)
class PhysicalCalibrationParameters:
    """Interpretable parameter vector for the physical DeLaurier calibration."""

    wing_normal_force_scale: float
    wing_chordwise_force_scale: float
    delaurier_theta_w_deg: float
    twist_eta_max_deg: float
    delaurier_induced_drag_efficiency: float
    fuselage_drag_cda: float
    tail_lift_scale: float
    phase_delay_s: float

    @classmethod
    def nominal(cls) -> "PhysicalCalibrationParameters":
        return cls(**{name: bound.nominal for name, bound in PHYSICAL_PARAMETER_BOUNDS.items()})

    @classmethod
    def from_mapping(cls, values: dict[str, float] | pd.Series) -> "PhysicalCalibrationParameters":
        return cls(**{name: float(values[name]) for name in PHYSICAL_PARAMETER_BOUNDS})

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in PHYSICAL_PARAMETER_BOUNDS}


PhysicalPredictor = Callable[[pd.DataFrame, PhysicalCalibrationParameters], pd.DataFrame]


@dataclass(frozen=True)
class PhysicalCalibrationSearchResult:
    """Result of bounded physical-parameter search."""

    best_parameters: PhysicalCalibrationParameters
    best_objective: float
    trace: pd.DataFrame


@dataclass(frozen=True)
class OfflineDeLaurierConfig:
    """Parameter snapshot matching the current IsaacLab DeLaurier baseline."""

    wing_geom_csv: Path = Path("outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv")
    num_strips: int = 80
    air_density: float = 1.225
    min_airspeed: float = 0.5
    theta_w_deg: float = 0.0
    enable_separation: bool = False
    induced_drag_efficiency: float = 0.0
    fuselage_drag_cda: float = 0.0
    twist_f_ref_hz: float = 4.0
    twist_eta_max_deg: float = 10.0
    twist_eta_limit_deg: float = 10.0
    twist_sign_left: float = 1.0
    twist_sign_right: float = 1.0
    wing_normal_force_scale: float = 1.0
    wing_chordwise_force_scale: float = 1.0
    wing_amplitude_rad: float = math.radians(30.0)
    phase_delay_s: float = 0.0
    elevon_max_deg: float = 41.0
    rudder_max_deg: float = 25.0
    tail_elevator_bias_deg: float = 0.0
    tail_horizontal_tail_incidence_bias_deg: float = 0.0
    tail_fixed_horizontal_effectiveness: float = 0.5
    tail_elevon_effectiveness: float = 1.2
    tail_elevon_alpha_limit_deg: float = 25.0
    tail_horizontal_tail_q_scale: float = 1.0
    tail_force_scale: float = 1.0
    base_com_pos_b: tuple[float, float, float] = (-0.10, 0.0, 0.0)
    left_wing_origin_b: tuple[float, float, float] = (0.0, 0.056, 0.0)
    right_wing_origin_b: tuple[float, float, float] = (0.0, -0.056, 0.0)
    left_wing_origin_roll_rad: float = -0.019391
    right_wing_origin_roll_rad: float = 0.019391
    delaurier_alpha0_rad: float = 0.0
    delaurier_eta_s: float = 0.65
    delaurier_cd_cf: float = 1.95
    delaurier_alpha_stall_min_rad: float = math.radians(-12.0)
    delaurier_alpha_stall_max_rad: float = math.radians(12.0)
    delaurier_xi: float = 0.0
    delaurier_c_mac: float = 0.0
    delaurier_nu: float = 1.5e-5
    delaurier_cd_f: float = 0.028


def parameter_vector_to_config(
    parameters: PhysicalCalibrationParameters,
    base_cfg: OfflineDeLaurierConfig | None = None,
) -> OfflineDeLaurierConfig:
    """Map an interpretable calibration vector to an offline DeLaurier config."""

    from dataclasses import replace

    cfg = base_cfg or OfflineDeLaurierConfig()
    twist_eta = float(parameters.twist_eta_max_deg)
    return replace(
        cfg,
        theta_w_deg=float(parameters.delaurier_theta_w_deg),
        twist_eta_max_deg=twist_eta,
        twist_eta_limit_deg=max(float(cfg.twist_eta_limit_deg), twist_eta),
        induced_drag_efficiency=float(parameters.delaurier_induced_drag_efficiency),
        fuselage_drag_cda=float(parameters.fuselage_drag_cda),
        tail_force_scale=float(parameters.tail_lift_scale),
        wing_normal_force_scale=float(parameters.wing_normal_force_scale),
        wing_chordwise_force_scale=float(parameters.wing_chordwise_force_scale),
        phase_delay_s=float(parameters.phase_delay_s),
    )


def normalized_wrench_objective(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    scale: pd.Series,
    parameters: PhysicalCalibrationParameters,
    *,
    regularization_weight: float = 0.0,
    target_columns: list[str] | None = None,
) -> float:
    """Return normalized MSE plus weak nominal-parameter regularization."""

    pred = _target_frame(predictions, target_columns)
    truth = _target_frame(targets, list(pred.columns))
    scale_series = pd.Series(scale, dtype=float).loc[pred.columns]
    scale_values = np.maximum(scale_series.to_numpy(dtype=float), 1.0e-9)
    residual = (pred.to_numpy(dtype=float) - truth.to_numpy(dtype=float)) / scale_values.reshape(1, -1)
    valid = np.isfinite(residual)
    if not valid.any():
        return float("inf")
    objective = float(np.mean(residual[valid] * residual[valid]))
    if regularization_weight <= 0.0:
        return objective

    penalty_terms: list[float] = []
    for name, bound in PHYSICAL_PARAMETER_BOUNDS.items():
        span = max(float(bound.high) - float(bound.low), 1.0e-12)
        distance = (float(getattr(parameters, name)) - float(bound.nominal)) / span
        penalty_terms.append(distance * distance)
    return objective + float(regularization_weight) * float(np.mean(penalty_terms))


def random_search_physical_calibration(
    calibration_frame: pd.DataFrame,
    targets: pd.DataFrame,
    scale: pd.Series,
    *,
    n_candidates: int,
    seed: int,
    predictor: PhysicalPredictor,
    regularization_weight: float = 0.0,
) -> PhysicalCalibrationSearchResult:
    """Run deterministic bounded random search over physical DeLaurier parameters."""

    rng = np.random.default_rng(seed)
    candidates = [PhysicalCalibrationParameters.nominal()]
    for _ in range(int(n_candidates)):
        values = {
            name: float(rng.uniform(bound.low, bound.high))
            for name, bound in PHYSICAL_PARAMETER_BOUNDS.items()
        }
        candidates.append(PhysicalCalibrationParameters(**values))

    rows: list[dict[str, float | int]] = []
    best_parameters = candidates[0]
    best_objective = float("inf")
    for candidate_id, parameters in enumerate(candidates):
        predictions = predictor(calibration_frame, parameters)
        objective = normalized_wrench_objective(
            predictions,
            targets,
            scale,
            parameters,
            regularization_weight=regularization_weight,
        )
        row: dict[str, float | int] = {"candidate": int(candidate_id), "objective": float(objective)}
        row.update(parameters.as_dict())
        rows.append(row)
        if objective < best_objective:
            best_objective = float(objective)
            best_parameters = parameters

    return PhysicalCalibrationSearchResult(
        best_parameters=best_parameters,
        best_objective=best_objective,
        trace=pd.DataFrame(rows),
    )


def _target_frame(frame: pd.DataFrame, target_columns: list[str] | None = None) -> pd.DataFrame:
    columns = target_columns or TARGET_COLUMNS
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    return frame.loc[:, columns].astype(float)


def fit_channel_gains(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    target_columns: list[str] | None = None,
    eps: float = 1.0e-12,
) -> pd.Series:
    """Fit one least-squares scalar gain per target channel."""
    pred = _target_frame(predictions, target_columns)
    truth = _target_frame(targets, list(pred.columns))
    gains: dict[str, float] = {}
    for column in pred.columns:
        x = pred[column].to_numpy(dtype=float)
        y = truth[column].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        denom = float(np.dot(x[mask], x[mask])) if mask.any() else 0.0
        if denom <= eps:
            gains[column] = 0.0
        else:
            gains[column] = float(np.dot(x[mask], y[mask]) / denom)
    return pd.Series(gains, dtype=float)


def apply_channel_gains(
    predictions: pd.DataFrame,
    gains: pd.Series | dict[str, float],
    *,
    target_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Apply per-channel scalar gains to a prediction frame."""
    pred = _target_frame(predictions, target_columns)
    gain_series = pd.Series(gains, dtype=float)
    missing = [column for column in pred.columns if column not in gain_series.index]
    if missing:
        raise ValueError(f"Missing gains for columns: {missing}")
    return pred.mul(gain_series.loc[pred.columns], axis=1)


def metrics_by_channel(
    targets: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-channel MAE, RMSE, bias, and R2."""
    truth = _target_frame(targets, target_columns)
    pred = _target_frame(predictions, list(truth.columns))
    rows: list[dict[str, float | int | str]] = []
    for column in truth.columns:
        y = truth[column].to_numpy(dtype=float)
        y_hat = pred[column].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(y_hat)
        if not mask.any():
            rows.append({"target": column, "n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan})
            continue
        residual = y_hat[mask] - y[mask]
        ss_res = float(np.sum(residual * residual))
        centered = y[mask] - float(np.mean(y[mask]))
        ss_tot = float(np.sum(centered * centered))
        rows.append(
            {
                "target": column,
                "n": int(mask.sum()),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual * residual))),
                "bias": float(np.mean(residual)),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _first_existing(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Missing any required column from: {candidates}")


def _frame_values(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)


def _relative_air_velocity_b(frame: pd.DataFrame) -> np.ndarray:
    existing = ["relative_air_velocity_b.x", "relative_air_velocity_b.y", "relative_air_velocity_b.z"]
    if all(column in frame.columns for column in existing):
        return _frame_values(frame, existing)

    velocity_b_columns = ["velocity_b.x", "velocity_b.y", "velocity_b.z"]
    if all(column in frame.columns for column in velocity_b_columns):
        velocity_b = _frame_values(frame, velocity_b_columns)
        if all(column in frame.columns for column in ["wind_body.x", "wind_body.y", "wind_body.z"]):
            return velocity_b - _frame_values(frame, ["wind_body.x", "wind_body.y", "wind_body.z"])
        return velocity_b

    quat_columns = [
        "vehicle_attitude.q[0]",
        "vehicle_attitude.q[1]",
        "vehicle_attitude.q[2]",
        "vehicle_attitude.q[3]",
    ]
    velocity_columns = [
        "vehicle_local_position.vx",
        "vehicle_local_position.vy",
        "vehicle_local_position.vz",
    ]
    quat = _frame_values(frame, quat_columns)
    velocity_n = _frame_values(frame, velocity_columns)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.where(norm > 1.0e-8, norm, 1.0)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    rotation_body_to_ned = np.empty((len(frame), 3, 3), dtype=np.float64)
    rotation_body_to_ned[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation_body_to_ned[:, 0, 1] = 2.0 * (x * y - z * w)
    rotation_body_to_ned[:, 0, 2] = 2.0 * (x * z + y * w)
    rotation_body_to_ned[:, 1, 0] = 2.0 * (x * y + z * w)
    rotation_body_to_ned[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation_body_to_ned[:, 1, 2] = 2.0 * (y * z - x * w)
    rotation_body_to_ned[:, 2, 0] = 2.0 * (x * z - y * w)
    rotation_body_to_ned[:, 2, 1] = 2.0 * (y * z + x * w)
    rotation_body_to_ned[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    wind_n = np.zeros_like(velocity_n)
    if "wind.windspeed_north" in frame.columns:
        wind_n[:, 0] = frame["wind.windspeed_north"].to_numpy(dtype=np.float64)
    if "wind.windspeed_east" in frame.columns:
        wind_n[:, 1] = frame["wind.windspeed_east"].to_numpy(dtype=np.float64)
    relative_air_velocity_n = velocity_n - wind_n
    return np.einsum("nji,nj->ni", rotation_body_to_ned, relative_air_velocity_n)


def _body_angular_velocity(frame: pd.DataFrame) -> np.ndarray:
    candidates = [
        ["vehicle_angular_velocity.x", "vehicle_angular_velocity.y", "vehicle_angular_velocity.z"],
        ["vehicle_angular_velocity.xyz[0]", "vehicle_angular_velocity.xyz[1]", "vehicle_angular_velocity.xyz[2]"],
    ]
    for columns in candidates:
        if all(column in frame.columns for column in columns):
            return _frame_values(frame, columns)
    return np.zeros((len(frame), 3), dtype=np.float64)


def _wing_state(frame: pd.DataFrame, *, phase_delay_s: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_column = _first_existing(frame, ("wing_stroke_angle_rad", "q_cmd", "wing_angle_rad"))
    q = frame[q_column].to_numpy(dtype=np.float64)

    if "wing_stroke_velocity_rad_s" in frame.columns:
        qd = frame["wing_stroke_velocity_rad_s"].to_numpy(dtype=np.float64)
    elif "qd_cmd" in frame.columns:
        qd = frame["qd_cmd"].to_numpy(dtype=np.float64)
    else:
        qd = _grouped_gradient(frame, q)

    if "wing_stroke_acceleration_rad_s2" in frame.columns:
        qdd = frame["wing_stroke_acceleration_rad_s2"].to_numpy(dtype=np.float64)
    elif "qdd_cmd" in frame.columns:
        qdd = frame["qdd_cmd"].to_numpy(dtype=np.float64)
    else:
        qdd = _grouped_gradient(frame, qd)

    if abs(float(phase_delay_s)) > 1.0e-12:
        q = _shift_by_time_within_logs(frame, q, float(phase_delay_s))
        qd = _shift_by_time_within_logs(frame, qd, float(phase_delay_s))
        qdd = _shift_by_time_within_logs(frame, qdd, float(phase_delay_s))

    return q, qd, qdd


def _shift_by_time_within_logs(frame: pd.DataFrame, values: np.ndarray, delay_s: float) -> np.ndarray:
    if "time_s" in frame.columns:
        time = frame["time_s"].to_numpy(dtype=np.float64)
    elif "timestamp_us" in frame.columns:
        time = frame["timestamp_us"].to_numpy(dtype=np.float64) * 1.0e-6
    else:
        return values

    shifted = np.empty_like(values, dtype=np.float64)
    if "log_id" not in frame.columns:
        return np.interp(time - delay_s, time, values, left=values[0], right=values[-1])

    log_ids = frame["log_id"].to_numpy()
    for log_id in pd.unique(log_ids):
        mask = log_ids == log_id
        shifted[mask] = np.interp(
            time[mask] - delay_s,
            time[mask],
            values[mask],
            left=float(values[mask][0]),
            right=float(values[mask][-1]),
        )
    return shifted


def _grouped_gradient(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    if "time_s" in frame.columns:
        time = frame["time_s"].to_numpy(dtype=np.float64)
    elif "timestamp_us" in frame.columns:
        time = frame["timestamp_us"].to_numpy(dtype=np.float64) * 1.0e-6
    else:
        return np.gradient(values.astype(np.float64))

    result = np.empty_like(values, dtype=np.float64)
    if "log_id" not in frame.columns:
        return _safe_gradient(values, time)
    log_ids = frame["log_id"].to_numpy()
    for log_id in pd.unique(log_ids):
        mask = log_ids == log_id
        result[mask] = _safe_gradient(values[mask], time[mask])
    return result


def _safe_gradient(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(values, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    valid_time = np.isfinite(time)
    if valid_time.sum() < 2 or np.nanmax(time[valid_time]) <= np.nanmin(time[valid_time]):
        return np.gradient(values)
    return np.gradient(values, time, edge_order=1)


def _rho(frame: pd.DataFrame, default: float) -> np.ndarray:
    if "vehicle_air_data.rho" in frame.columns:
        rho = frame["vehicle_air_data.rho"].to_numpy(dtype=np.float64)
        return np.where(np.isfinite(rho) & (rho > 0.0), rho, default)
    return np.full(len(frame), default, dtype=np.float64)


def _servo_rad(frame: pd.DataFrame, column: str, max_deg: float) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=np.float64)
    command = frame[column].to_numpy(dtype=np.float64)
    command = np.clip(np.nan_to_num(command, nan=0.0), -1.0, 1.0)
    return command * math.radians(max_deg)


def _rotation_x_batch(angle: "object"):
    import torch

    batch = angle.shape[0]
    rotation = torch.zeros((batch, 3, 3), dtype=angle.dtype, device=angle.device)
    c = torch.cos(angle)
    s = torch.sin(angle)
    rotation[:, 0, 0] = 1.0
    rotation[:, 1, 1] = c
    rotation[:, 1, 2] = -s
    rotation[:, 2, 1] = s
    rotation[:, 2, 2] = c
    return rotation


def predict_delaurier_wrench(
    frame: pd.DataFrame,
    *,
    cfg: OfflineDeLaurierConfig | None = None,
    batch_size: int = 8192,
    device: str = "cpu",
) -> pd.DataFrame:
    """Predict body-frame effective wrench with the current IsaacLab DeLaurier baseline."""

    import torch

    from flapping_bot.physics.qsm_delaurier1993 import DeLaurierParams, compute_aero_wrench_delaurier1993
    from flapping_bot.physics.tail_aero import TailAeroCfg, TailAeroModel
    from flapping_bot.physics.wing_equivalent_ac import compute_area_weighted_quarter_chord_link_points
    from flapping_bot.physics.wing_geom_csv import build_wing_geometry_from_csv

    cfg = cfg or OfflineDeLaurierConfig()
    torch_device = torch.device(device)
    wing_geom, _info = build_wing_geometry_from_csv(
        cfg.wing_geom_csv,
        N=int(cfg.num_strips),
        device=torch_device,
        dtype=torch.float32,
        dhat=0.0,
        aspect_ratio=None,
    )
    wing_area = float(torch.sum(wing_geom.c * wing_geom.dx).item())
    wing_application_point_link = compute_area_weighted_quarter_chord_link_points(wing_geom).to(device=torch_device)
    delaurier_params = DeLaurierParams(
        alpha0_rad=float(cfg.delaurier_alpha0_rad),
        eta_s=float(cfg.delaurier_eta_s),
        cd_cf=float(cfg.delaurier_cd_cf),
        alpha_stall_min_rad=float(cfg.delaurier_alpha_stall_min_rad),
        alpha_stall_max_rad=float(cfg.delaurier_alpha_stall_max_rad),
        xi=float(cfg.delaurier_xi),
        c_mac=float(cfg.delaurier_c_mac),
        nu=float(cfg.delaurier_nu),
        cd_f=float(cfg.delaurier_cd_f),
    )
    tail_cfg = TailAeroCfg(
        air_density=float(cfg.air_density),
        horizontal_tail_incidence_bias_deg=float(cfg.tail_horizontal_tail_incidence_bias_deg),
        fixed_horizontal_effectiveness=float(cfg.tail_fixed_horizontal_effectiveness),
        elevon_effectiveness=float(cfg.tail_elevon_effectiveness),
        elevon_alpha_limit_deg=float(cfg.tail_elevon_alpha_limit_deg),
        horizontal_tail_q_scale=float(cfg.tail_horizontal_tail_q_scale),
    )
    tail_model = TailAeroModel(tail_cfg, torch_device)

    v_air_b_np = _relative_air_velocity_b(frame)
    w_b_np = _body_angular_velocity(frame)
    q_np, qd_np, qdd_np = _wing_state(frame, phase_delay_s=float(cfg.phase_delay_s))
    rho_np = _rho(frame, cfg.air_density)
    left_elevon_np = _servo_rad(frame, "servo_left_elevon", cfg.elevon_max_deg) + math.radians(
        cfg.tail_elevator_bias_deg
    )
    right_elevon_np = _servo_rad(frame, "servo_right_elevon", cfg.elevon_max_deg) + math.radians(
        cfg.tail_elevator_bias_deg
    )
    rudder_np = _servo_rad(frame, "servo_rudder", cfg.rudder_max_deg)

    output = np.empty((len(frame), len(TARGET_COLUMNS)), dtype=np.float32)
    x_mid = wing_geom.x_mid
    strip_count = int(x_mid.numel())
    A_l2w_L = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], device=torch_device)
    A_l2w_R = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], device=torch_device)
    A_w2l_pair = torch.stack((A_l2w_L.T, A_l2w_R.T), dim=0).to(dtype=torch.float32)
    base_com_pos_b_pair = torch.as_tensor(cfg.base_com_pos_b, dtype=torch.float32, device=torch_device)
    wing_origin_pair = torch.as_tensor(
        [cfg.left_wing_origin_b, cfg.right_wing_origin_b], dtype=torch.float32, device=torch_device
    )
    origin_roll_pair = torch.as_tensor(
        [cfg.left_wing_origin_roll_rad, cfg.right_wing_origin_roll_rad], dtype=torch.float32, device=torch_device
    )

    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            sl = slice(start, stop)
            n_env = stop - start
            batch_wings = 2 * n_env
            y = x_mid.view(1, strip_count).expand(batch_wings, strip_count)

            v_air_b = torch.as_tensor(v_air_b_np[sl], dtype=torch.float32, device=torch_device)
            w_b = torch.as_tensor(w_b_np[sl], dtype=torch.float32, device=torch_device)
            q = torch.as_tensor(q_np[sl], dtype=torch.float32, device=torch_device)
            qd = torch.as_tensor(qd_np[sl], dtype=torch.float32, device=torch_device)
            qdd = torch.as_tensor(qdd_np[sl], dtype=torch.float32, device=torch_device)
            rho = torch.as_tensor(rho_np[sl], dtype=torch.float32, device=torch_device)
            left_elevon = torch.as_tensor(left_elevon_np[sl], dtype=torch.float32, device=torch_device)
            right_elevon = torch.as_tensor(right_elevon_np[sl], dtype=torch.float32, device=torch_device)
            rudder = torch.as_tensor(rudder_np[sl], dtype=torch.float32, device=torch_device)

            q_w = torch.repeat_interleave(q, 2)
            qd_w = torch.repeat_interleave(qd, 2)
            qdd_w = torch.repeat_interleave(qdd, 2)
            freq_np = frame["flap_frequency_hz"].to_numpy(dtype=np.float64) if "flap_frequency_hz" in frame.columns else None
            if freq_np is None:
                freq = torch.full((n_env,), float(cfg.twist_f_ref_hz), dtype=torch.float32, device=torch_device)
            else:
                freq = torch.as_tensor(freq_np[sl], dtype=torch.float32, device=torch_device)
                freq = torch.nan_to_num(freq, nan=float(cfg.twist_f_ref_hz)).clamp(min=1.0e-6)
            w = torch.repeat_interleave(2.0 * math.pi * freq, 2)

            vx_b = torch.clamp(v_air_b[:, 0], min=1.0e-3)
            theta_a_env = torch.atan2(-v_air_b[:, 2], vx_b)
            theta_a = torch.repeat_interleave(theta_a_env, 2)
            theta_bar = theta_a + math.radians(float(cfg.theta_w_deg))
            U = torch.repeat_interleave(torch.clamp(v_air_b[:, 0], min=float(cfg.min_airspeed)), 2)

            h = -q_w.view(batch_wings, 1) * y
            hdot = -qd_w.view(batch_wings, 1) * y
            hddot = -qdd_w.view(batch_wings, 1) * y

            eta_max = math.radians(float(cfg.twist_eta_max_deg))
            eta_lim = math.radians(float(cfg.twist_eta_limit_deg))
            qd_ref = max(float(cfg.wing_amplitude_rad) * (2.0 * math.pi * float(cfg.twist_f_ref_hz)), 1.0e-6)
            s = torch.clamp(qd_w / qd_ref, min=-1.0, max=1.0)
            sgn = torch.tensor([cfg.twist_sign_left, cfg.twist_sign_right], device=torch_device, dtype=torch.float32).repeat(n_env)
            eta_tip = torch.clamp(eta_max * sgn * s, min=-eta_lim, max=eta_lim)
            k = eta_max / qd_ref
            qddd = -(w * w) * qd_w
            etad_tip = (k * sgn) * qdd_w
            etadd_tip = (k * sgn) * qddd

            theta = (theta_bar + eta_tip).view(batch_wings, 1).expand(batch_wings, strip_count)
            thetad = etad_tip.view(batch_wings, 1).expand(batch_wings, strip_count)
            thetadd = etadd_tip.view(batch_wings, 1).expand(batch_wings, strip_count)
            omega_ref = w.view(batch_wings, 1).expand(batch_wings, strip_count)

            # The current model uses one rho per batch call. Use the batch mean; logged rho varies slowly.
            F_c, _tau_c, _power, _sep = compute_aero_wrench_delaurier1993(
                h,
                hdot,
                hddot,
                theta,
                thetad,
                thetadd,
                wing_geom,
                rho=float(rho.mean().item()),
                U=U.view(batch_wings, 1),
                theta_a=theta_a.view(batch_wings, 1),
                theta_bar=theta_bar.view(batch_wings, 1),
                omega_ref=omega_ref,
                params=delaurier_params,
                enable_separation=bool(cfg.enable_separation),
                return_terms=False,
            )
            F_c = F_c.clone()
            F_c[:, 1] = F_c[:, 1] * float(cfg.wing_normal_force_scale)
            F_c[:, 2] = F_c[:, 2] * float(cfg.wing_chordwise_force_scale)

            A_w2l = A_w2l_pair.repeat(n_env, 1, 1)
            F_l = torch.bmm(A_w2l, F_c.view(batch_wings, 3, 1)).view(batch_wings, 3)

            wing_origin = wing_origin_pair.repeat(n_env, 1)
            angle = (origin_roll_pair.repeat(n_env) + q_w).to(dtype=torch.float32)
            R_link_to_body = _rotation_x_batch(angle)
            application_point = wing_application_point_link.repeat(n_env, 1)
            p_wing_b = wing_origin + torch.bmm(R_link_to_body, application_point.view(batch_wings, 3, 1)).view(batch_wings, 3)
            F_wing_b = torch.bmm(R_link_to_body, F_l.view(batch_wings, 3, 1)).view(batch_wings, 3)
            r_wing_b = p_wing_b - base_com_pos_b_pair.view(1, 3)
            tau_wing_b = torch.linalg.cross(r_wing_b, F_wing_b, dim=1)
            F_b = F_wing_b.view(n_env, 2, 3).sum(dim=1)
            tau_b = tau_wing_b.view(n_env, 2, 3).sum(dim=1)

            e = float(cfg.induced_drag_efficiency)
            if e > 0.0:
                speed = torch.linalg.norm(v_air_b, dim=1)
                speed_safe = torch.clamp(speed, min=1.0e-6)
                v_dir = v_air_b / speed_safe.unsqueeze(1)
                q_dyn = 0.5 * float(cfg.air_density) * speed_safe * speed_safe
                f_para = torch.sum(F_b * v_dir, dim=1, keepdim=True) * v_dir
                lift_mag = torch.linalg.norm(F_b - f_para, dim=1)
                cd_k = 1.0 / (math.pi * float(wing_geom.aspect_ratio) * e)
                drag = cd_k * lift_mag * lift_mag / torch.clamp(q_dyn * wing_area, min=1.0e-6)
                F_b = F_b - drag.unsqueeze(1) * v_dir

            f_tail, tau_tail = tail_model.compute_wrench(
                root_lin_vel_b=v_air_b,
                root_ang_vel_b=w_b,
                left_elevon_rad=left_elevon,
                right_elevon_rad=right_elevon,
                rudder_rad=rudder,
                base_com_pos_b=base_com_pos_b_pair.view(1, 3).expand(n_env, 3),
            )
            F_b = F_b + f_tail * float(cfg.tail_force_scale)
            tau_b = tau_b + tau_tail * float(cfg.tail_force_scale)

            cda = float(cfg.fuselage_drag_cda)
            if cda > 0.0:
                speed = torch.linalg.norm(v_air_b, dim=1)
                speed_safe = torch.clamp(speed, min=1.0e-6)
                F_b = F_b - 0.5 * float(cfg.air_density) * cda * speed.unsqueeze(1) * v_air_b

            wrench = torch.cat((F_b, tau_b), dim=1)
            output[sl, :] = wrench.detach().cpu().numpy().astype(np.float32)

    return pd.DataFrame(output, columns=TARGET_COLUMNS, index=frame.index)

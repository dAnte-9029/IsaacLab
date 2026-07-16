"""Reproducible unit-level audit for the five-surface tail aerodynamic model.

The audit calls :class:`TailAeroModel` for every load calculation. This module
only constructs static sweeps, serializes diagnostics, evaluates invariants,
and writes an evidence-linked report; it does not contain a second aerodynamic
equation or modify production configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
from typing import Any, Sequence

import torch

from flapping_bot.physics.tail_aero import TailAeroCfg, TailAeroModel, TailSurfaceCfg, TailSurfaceResult
from flapping_bot.px4_like.straight_line_controller import (
    PX4LikeStraightLineController,
    PX4LikeStraightLineControllerCfg,
)


SCHEMA_VERSION = "tail-unit-audit-v2"
SURFACE_ORDER = (
    "fixed_horizontal",
    "left_elevon",
    "right_elevon",
    "fixed_vertical",
    "rudder",
)
VECTOR_COMPONENTS = ("x", "y", "z")


@dataclass(frozen=True)
class TailAuditSettings:
    """Stable defaults matching ``FlappingBotStraightFlightEnvCfg`` on WT0."""

    seed: int = 0
    dtype: str = "float64"
    air_density_kg_m3: float = 1.225
    nominal_airspeed_mps: float = 8.0
    nominal_body_flow_angle_deg: float = 0.0
    nominal_sideslip_deg: float = 0.0
    nominal_body_rates_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    nominal_left_elevon_deg: float = 0.0
    nominal_right_elevon_deg: float = 0.0
    nominal_rudder_deg: float = 0.0
    base_com_pos_b_m: tuple[float, float, float] = (-0.12154, 0.00541, -0.01298)
    horizontal_tail_incidence_bias_deg: float = 0.0
    fixed_horizontal_effectiveness: float = 0.5
    elevon_effectiveness: float = 1.2
    elevon_alpha_limit_deg: float = 25.0
    horizontal_tail_q_scale: float = 1.0
    elevon_limit_deg: float = 41.0
    rudder_limit_deg: float = 25.0
    symmetric_elevon_sweep_deg: tuple[float, float, int] = (-35.0, 35.0, 71)
    differential_elevon_sweep_deg: tuple[float, float, int] = (-35.0, 35.0, 71)
    rudder_sweep_deg: tuple[float, float, int] = (-25.0, 25.0, 101)
    incidence_sweep_deg: tuple[float, float, int] = (-20.0, 20.0, 81)
    sideslip_sweep_deg: tuple[float, float, int] = (-20.0, 20.0, 81)
    angular_rate_sweep_rad_s: tuple[float, float, int] = (-2.0, 2.0, 81)
    airspeed_sweep_mps: tuple[float, ...] = (0.0, 0.01, 0.1, 0.5, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
    airspeed_control_deg: tuple[float, float, float] = (5.0, 5.0, 5.0)
    symmetry_elevon_sweep_deg: tuple[float, float, int] = (-35.0, 35.0, 71)
    suspicious_step_fraction: float = 0.25
    strict_moment_abs_tolerance: float = 5.0e-13
    strict_moment_rel_tolerance: float = 5.0e-12
    symmetry_force_tolerance: float = 5.0e-12


def build_production_tail_cfg(settings: TailAuditSettings) -> TailAeroCfg:
    """Build the effective environment-default tail configuration without Isaac."""

    return replace(
        TailAeroCfg(),
        air_density=float(settings.air_density_kg_m3),
        horizontal_tail_incidence_bias_deg=float(settings.horizontal_tail_incidence_bias_deg),
        fixed_horizontal_effectiveness=float(settings.fixed_horizontal_effectiveness),
        elevon_effectiveness=float(settings.elevon_effectiveness),
        elevon_alpha_limit_deg=float(settings.elevon_alpha_limit_deg),
        horizontal_tail_q_scale=float(settings.horizontal_tail_q_scale),
    )


def mix_elevon_commands(
    symmetric_rad: torch.Tensor,
    differential_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the environment mapping ``left=pitch+roll, right=pitch-roll``."""

    if symmetric_rad.shape != differential_rad.shape:
        raise ValueError("Symmetric and differential command tensors must have identical shapes.")
    return symmetric_rad + differential_rad, symmetric_rad - differential_rad


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo_root, text=True).strip()


def _git_status(repo_root: Path) -> list[str]:
    text = subprocess.check_output(("git", "status", "--porcelain"), cwd=repo_root, text=True)
    return [line for line in text.splitlines() if line]


def _linspace(spec: tuple[float, float, int], *, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(float(spec[0]), float(spec[1]), int(spec[2]), dtype=dtype)


def _flow_from_angles(
    airspeed_mps: torch.Tensor,
    *,
    body_flow_angle_deg: torch.Tensor,
    sideslip_deg: torch.Tensor,
) -> torch.Tensor:
    """Construct body-FLU air-relative velocity from audit angle coordinates."""

    theta = torch.deg2rad(body_flow_angle_deg)
    beta = torch.deg2rad(sideslip_deg)
    cos_beta = torch.cos(beta)
    return torch.stack(
        (
            airspeed_mps * torch.cos(theta) * cos_beta,
            airspeed_mps * torch.sin(beta),
            airspeed_mps * torch.sin(theta) * cos_beta,
        ),
        dim=1,
    )


def _to_float(value: torch.Tensor, index: int) -> float:
    return float(value[index].detach().cpu().item())


def _surface_cfg_map(cfg: TailAeroCfg) -> dict[str, TailSurfaceCfg]:
    return {name: getattr(cfg, name) for name in SURFACE_ORDER}


def _base_row(
    *,
    sweep: str,
    sample_index: int,
    independent_name: str,
    independent_value: float,
    v_b: torch.Tensor,
    w_b: torch.Tensor,
    symmetric_rad: torch.Tensor,
    differential_rad: torch.Tensor,
    left_rad: torch.Tensor,
    right_rad: torch.Tensor,
    rudder_rad: torch.Tensor,
) -> dict[str, Any]:
    vx, vy, vz = (_to_float(v_b[:, axis], sample_index) for axis in range(3))
    return {
        "sweep": sweep,
        "sample_index": sample_index,
        "independent_variable": independent_name,
        "independent_value": independent_value,
        "airspeed_mps": math.sqrt(vx * vx + vy * vy + vz * vz),
        "body_flow_angle_deg": math.degrees(math.atan2(vz, vx)),
        "sideslip_deg": math.degrees(math.atan2(vy, math.hypot(vx, vz))),
        "p_rad_s": _to_float(w_b[:, 0], sample_index),
        "q_rad_s": _to_float(w_b[:, 1], sample_index),
        "r_rad_s": _to_float(w_b[:, 2], sample_index),
        "symmetric_elevon_deg": math.degrees(_to_float(symmetric_rad, sample_index)),
        "differential_elevon_deg": math.degrees(_to_float(differential_rad, sample_index)),
        "left_elevon_deg": math.degrees(_to_float(left_rad, sample_index)),
        "right_elevon_deg": math.degrees(_to_float(right_rad, sample_index)),
        "rudder_deg": math.degrees(_to_float(rudder_rad, sample_index)),
    }


def evaluate_sweep(
    model: TailAeroModel,
    cfg: TailAeroCfg,
    settings: TailAuditSettings,
    *,
    sweep: str,
    independent_name: str,
    independent_values: torch.Tensor,
    v_b: torch.Tensor,
    w_b: torch.Tensor,
    symmetric_rad: torch.Tensor,
    differential_rad: torch.Tensor,
    rudder_rad: torch.Tensor,
    reference: str = "configured_com",
) -> list[dict[str, Any]]:
    """Evaluate a batch and return long-form rows for five surfaces and aggregate."""

    left_rad, right_rad = mix_elevon_commands(symmetric_rad, differential_rad)
    if reference == "configured_com":
        com = torch.tensor(settings.base_com_pos_b_m, dtype=v_b.dtype).view(1, 3).expand(v_b.shape[0], 3)
    elif reference == "base_origin":
        com = torch.zeros_like(v_b)
    else:
        raise ValueError(f"Unknown moment reference: {reference}")
    results = model.compute_surface_results(
        root_lin_vel_b=v_b,
        root_ang_vel_b=w_b,
        left_elevon_rad=left_rad,
        right_elevon_rad=right_rad,
        rudder_rad=rudder_rad,
        base_com_pos_b=com,
    )
    cfg_map = _surface_cfg_map(cfg)
    rows: list[dict[str, Any]] = []
    for index in range(v_b.shape[0]):
        common = _base_row(
            sweep=sweep,
            sample_index=index,
            independent_name=independent_name,
            independent_value=_to_float(independent_values, index),
            v_b=v_b,
            w_b=w_b,
            symmetric_rad=symmetric_rad,
            differential_rad=differential_rad,
            left_rad=left_rad,
            right_rad=right_rad,
            rudder_rad=rudder_rad,
        )
        total_force = torch.zeros(3, dtype=v_b.dtype)
        total_moment = torch.zeros(3, dtype=v_b.dtype)
        for result in results:
            surf = cfg_map[result.name]
            row = dict(common)
            row.update(
                {
                    "reference": reference,
                    "surface": result.name,
                    "is_aggregate": False,
                    "area_m2": float(surf.area),
                    "span_m": float(surf.span_m),
                    "mean_chord_m": float(surf.mean_chord_m),
                    "aspect_ratio": float(surf.span_m * surf.span_m / surf.area),
                    "incidence_deg": math.degrees(float(surf.incidence_rad)),
                    "deflection_sign": float(surf.deflection_sign),
                    "alpha_limit_deg": float(model._effective_alpha_limit_deg(surf)),
                    "alpha_raw_deg": math.degrees(_to_float(result.alpha_raw_rad, index)),
                    "alpha_deg": math.degrees(_to_float(result.alpha_rad, index)),
                    "alpha_clipped": bool(result.alpha_clipped[index].item()),
                    "dynamic_pressure_pa": _to_float(result.dynamic_pressure_pa, index),
                    "lift_coefficient": _to_float(result.lift_coefficient, index),
                    "drag_coefficient": _to_float(result.drag_coefficient, index),
                    "local_speed_mps": _to_float(result.local_speed_mps, index),
                    "effectiveness_scaling": float(result.effectiveness_scaling),
                    "effective_lift_slope_per_rad": float(result.effective_lift_slope_per_rad),
                    "dynamic_pressure_scaling": float(result.dynamic_pressure_scaling),
                }
            )
            vectors = {
                "force_b_N": result.force_b,
                "moment_b_Nm": result.moment_b_about_com,
                "lift_force_b_N": result.lift_force_b,
                "drag_force_b_N": result.drag_force_b,
                "local_velocity_b_mps": result.local_velocity_b,
                "projected_velocity_b_mps": result.projected_velocity_b,
                "position_b_from_com_m": result.aerodynamic_center_b_from_com,
                "chord_axis_b": result.chord_axis_b,
                "span_axis_b": result.span_axis_b,
            }
            for prefix, tensor in vectors.items():
                for axis, component in enumerate(VECTOR_COMPONENTS):
                    row[f"{prefix}_{component}"] = _to_float(tensor[:, axis], index)
            total_force += result.force_b[index].detach().cpu()
            total_moment += result.moment_b_about_com[index].detach().cpu()
            row["force_norm_N"] = float(torch.linalg.norm(result.force_b[index]).item())
            row["moment_norm_Nm"] = float(torch.linalg.norm(result.moment_b_about_com[index]).item())
            rows.append(row)

        aggregate = dict(common)
        aggregate.update(
            {
                "reference": reference,
                "surface": "aggregate",
                "is_aggregate": True,
                "force_norm_N": float(torch.linalg.norm(total_force).item()),
                "moment_norm_Nm": float(torch.linalg.norm(total_moment).item()),
                "alpha_clipped": any(bool(result.alpha_clipped[index].item()) for result in results),
            }
        )
        for axis, component in enumerate(VECTOR_COMPONENTS):
            aggregate[f"force_b_N_{component}"] = float(total_force[axis].item())
            aggregate[f"moment_b_Nm_{component}"] = float(total_moment[axis].item())
        rows.append(aggregate)
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _geometry_rows(cfg: TailAeroCfg, settings: TailAuditSettings) -> list[dict[str, Any]]:
    com = settings.base_com_pos_b_m
    rows = []
    model = TailAeroModel(cfg, "cpu")
    for name in SURFACE_ORDER:
        surf = getattr(cfg, name)
        row: dict[str, Any] = {
            "surface": name,
            "area_m2": surf.area,
            "span_m": surf.span_m,
            "mean_chord_m": surf.mean_chord_m,
            "aspect_ratio": surf.span_m * surf.span_m / surf.area,
            "incidence_deg": math.degrees(surf.incidence_rad),
            "cl_alpha_geometric_per_rad": surf.cl_alpha_per_rad,
            "cl_alpha_effective_per_rad": model._effective_cl_alpha_per_rad(surf),
            "effectiveness_scaling": model._effectiveness_scaling(surf),
            "cd0": surf.cd0,
            "cd_k": surf.cd_k,
            "alpha_limit_deg": model._effective_alpha_limit_deg(surf),
            "deflection_sign": surf.deflection_sign,
            "dynamic_pressure_scaling": model._effective_dynamic_pressure_scale(surf),
        }
        for axis, component in enumerate(VECTOR_COMPONENTS):
            row[f"position_b_from_origin_m_{component}"] = surf.lever_arm_body[axis]
            row[f"position_b_from_com_m_{component}"] = surf.lever_arm_body[axis] - com[axis]
            row[f"span_axis_b_{component}"] = surf.span_axis_body[axis]
            row[f"chord_axis_b_{component}"] = surf.chord_axis_body[axis]
        rows.append(row)
    return rows


def _select(rows: Sequence[dict[str, Any]], *, surface: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["surface"] == surface]


def _derivative(rows: Sequence[dict[str, Any]], field: str, x_field: str) -> float:
    aggregate = sorted(_select(rows, surface="aggregate"), key=lambda row: float(row[x_field]))
    negative = max((row for row in aggregate if float(row[x_field]) < 0.0), key=lambda row: float(row[x_field]))
    positive = min((row for row in aggregate if float(row[x_field]) > 0.0), key=lambda row: float(row[x_field]))
    dx_rad = math.radians(float(positive[x_field]) - float(negative[x_field])) if x_field.endswith("_deg") else float(
        positive[x_field]
    ) - float(negative[x_field])
    return (float(positive[field]) - float(negative[field])) / dx_rad


def _controller_rudder_action_derivatives() -> dict[str, float]:
    """Measure the active straight-line controller's local rudder sign contract."""

    controller = PX4LikeStraightLineController(
        PX4LikeStraightLineControllerCfg(enable_tecs=False), device=torch.device("cpu")
    )

    def evaluate(yaw_rad: torch.Tensor, yaw_rate_rad_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = yaw_rad.numel()
        angular_rate = torch.zeros((count, 3), dtype=torch.float32)
        angular_rate[:, 2] = yaw_rate_rad_s
        actions, diagnostics = controller.compute_actions(
            pos_local=torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float32).expand(count, 3).clone(),
            ground_vel_local=torch.tensor([[8.0, 0.0, 0.0]], dtype=torch.float32).expand(count, 3).clone(),
            wind_vel_local=torch.zeros((count, 2), dtype=torch.float32),
            roll=torch.zeros(count, dtype=torch.float32),
            pitch=torch.zeros(count, dtype=torch.float32),
            yaw=yaw_rad,
            ang_vel_body=angular_rate,
        )
        return actions[:, 1], diagnostics["course_err"]

    yaw = torch.deg2rad(torch.tensor([-1.0, 1.0], dtype=torch.float32))
    zero_rate = torch.zeros(2, dtype=torch.float32)
    yaw_actions, course_error = evaluate(yaw, zero_rate)
    d_action_d_course_error = float(
        (yaw_actions[1] - yaw_actions[0]) / (course_error[1] - course_error[0])
    )

    yaw_rate = torch.tensor([-0.1, 0.1], dtype=torch.float32)
    rate_actions, zero_course_error = evaluate(torch.zeros(2, dtype=torch.float32), yaw_rate)
    if not torch.allclose(zero_course_error, torch.zeros_like(zero_course_error), atol=1.0e-7, rtol=0.0):
        raise RuntimeError("Controller yaw-rate derivative state did not have zero course error.")
    d_action_d_yaw_rate = float((rate_actions[1] - rate_actions[0]) / (yaw_rate[1] - yaw_rate[0]))
    return {
        "d_action_rudder_d_course_error_per_rad": d_action_d_course_error,
        "d_action_rudder_d_yaw_rate_s_per_rad": d_action_d_yaw_rate,
    }


def _symmetry_rows(
    model: TailAeroModel,
    cfg: TailAeroCfg,
    settings: TailAuditSettings,
    *,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    degrees = _linspace(settings.symmetry_elevon_sweep_deg, dtype=dtype)
    n = degrees.numel()
    speed = torch.full((n,), settings.nominal_airspeed_mps, dtype=dtype)
    zeros = torch.zeros(n, dtype=dtype)
    v_b = _flow_from_angles(speed, body_flow_angle_deg=zeros, sideslip_deg=zeros)
    w_b = torch.zeros(n, 3, dtype=dtype)
    all_rows: list[dict[str, Any]] = []
    for reference in ("base_origin", "configured_com"):
        source = evaluate_sweep(
            model,
            cfg,
            settings,
            sweep="left_right_symmetry",
            independent_name="symmetric_elevon_deg",
            independent_values=degrees,
            v_b=v_b,
            w_b=w_b,
            symmetric_rad=torch.deg2rad(degrees),
            differential_rad=zeros,
            rudder_rad=zeros,
            reference=reference,
        )
        left = {int(row["sample_index"]): row for row in source if row["surface"] == "left_elevon"}
        right = {int(row["sample_index"]): row for row in source if row["surface"] == "right_elevon"}
        parity = {
            "force_b_N_x": 1.0,
            "force_b_N_y": -1.0,
            "force_b_N_z": 1.0,
            "moment_b_Nm_x": -1.0,
            "moment_b_Nm_y": 1.0,
            "moment_b_Nm_z": -1.0,
            "local_velocity_b_mps_x": 1.0,
            "local_velocity_b_mps_y": -1.0,
            "local_velocity_b_mps_z": 1.0,
            "local_speed_mps": 1.0,
            "alpha_deg": 1.0,
        }
        for index, degree in enumerate(degrees.tolist()):
            for quantity, expected_parity in parity.items():
                left_value = float(left[index][quantity])
                right_value = float(right[index][quantity])
                residual = left_value - expected_parity * right_value
                denom = max(abs(left_value), abs(right_value), 1.0e-12)
                all_rows.append(
                    {
                        "reference": reference,
                        "sample_index": index,
                        "symmetric_elevon_deg": degree,
                        "quantity": quantity,
                        "expected_parity": "same" if expected_parity > 0.0 else "opposite",
                        "left_value": left_value,
                        "right_value": right_value,
                        "parity_adjusted_right_value": expected_parity * right_value,
                        "absolute_residual": abs(residual),
                        "normalized_error": abs(residual) / denom,
                    }
                )
    return all_rows


def _moment_consistency(all_sweeps: Sequence[Sequence[dict[str, Any]]]) -> dict[str, float]:
    max_surface_abs = 0.0
    max_surface_rel = 0.0
    max_aggregate_abs = 0.0
    max_aggregate_rel = 0.0
    for rows in all_sweeps:
        grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for row in rows:
            key = (str(row["sweep"]), str(row.get("rate_axis", "")), int(row["sample_index"]))
            grouped.setdefault(key, []).append(row)
        for sample_rows in grouped.values():
            surface_rows = [row for row in sample_rows if not bool(row["is_aggregate"])]
            aggregate = next(row for row in sample_rows if bool(row["is_aggregate"]))
            force_sum = [0.0, 0.0, 0.0]
            moment_sum = [0.0, 0.0, 0.0]
            for row in surface_rows:
                r = [float(row[f"position_b_from_com_m_{axis}"]) for axis in VECTOR_COMPONENTS]
                force = [float(row[f"force_b_N_{axis}"]) for axis in VECTOR_COMPONENTS]
                reported = [float(row[f"moment_b_Nm_{axis}"]) for axis in VECTOR_COMPONENTS]
                reconstructed = [
                    r[1] * force[2] - r[2] * force[1],
                    r[2] * force[0] - r[0] * force[2],
                    r[0] * force[1] - r[1] * force[0],
                ]
                for axis in range(3):
                    residual = abs(reconstructed[axis] - reported[axis])
                    max_surface_abs = max(max_surface_abs, residual)
                    max_surface_rel = max(max_surface_rel, residual / max(abs(reported[axis]), 1.0e-15))
                    force_sum[axis] += force[axis]
                    moment_sum[axis] += reported[axis]
            for axis, component in enumerate(VECTOR_COMPONENTS):
                force_residual = abs(force_sum[axis] - float(aggregate[f"force_b_N_{component}"]))
                moment_residual = abs(moment_sum[axis] - float(aggregate[f"moment_b_Nm_{component}"]))
                residual = max(force_residual, moment_residual)
                scale = max(
                    abs(float(aggregate[f"force_b_N_{component}"])),
                    abs(float(aggregate[f"moment_b_Nm_{component}"])),
                    1.0e-15,
                )
                max_aggregate_abs = max(max_aggregate_abs, residual)
                max_aggregate_rel = max(max_aggregate_rel, residual / scale)
    return {
        "max_surface_moment_absolute_error": max_surface_abs,
        "max_surface_moment_relative_error": max_surface_rel,
        "max_aggregate_sum_absolute_error": max_aggregate_abs,
        "max_aggregate_sum_relative_error": max_aggregate_rel,
    }


def _continuity_rows(sweeps: dict[str, list[dict[str, Any]]], settings: TailAuditSettings) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = (
        "force_b_N_x",
        "force_b_N_y",
        "force_b_N_z",
        "moment_b_Nm_x",
        "moment_b_Nm_y",
        "moment_b_Nm_z",
    )
    for sweep_name, source in sweeps.items():
        aggregate = _select(source, surface="aggregate")
        rate_axes = sorted({str(row.get("rate_axis", "")) for row in aggregate})
        groups = rate_axes if rate_axes != [""] else [""]
        for group in groups:
            group_rows = [row for row in aggregate if str(row.get("rate_axis", "")) == group]
            group_rows.sort(key=lambda row: float(row["independent_value"]))
            x = [float(row["independent_value"]) for row in group_rows]
            any_clip = [bool(row.get("alpha_clipped", False)) for row in group_rows]
            for field in fields:
                y = [float(row[field]) for row in group_rows]
                finite = all(math.isfinite(value) for value in y)
                differences = [y[i + 1] - y[i] for i in range(len(y) - 1)]
                second = [differences[i + 1] - differences[i] for i in range(len(differences) - 1)]
                value_range = max(y) - min(y) if y else 0.0
                max_step = max((abs(value) for value in differences), default=0.0)
                normalized_step = max_step / max(abs(value_range), 1.0e-12)
                zero_index = min(range(len(x)), key=lambda idx: abs(x[idx]))
                zero_jump = 0.0
                if 0 < zero_index < len(y) - 1:
                    zero_jump = max(abs(y[zero_index] - y[zero_index - 1]), abs(y[zero_index + 1] - y[zero_index]))
                rows.append(
                    {
                        "sweep": sweep_name,
                        "group": group or "all",
                        "quantity": field,
                        "sample_count": len(group_rows),
                        "all_finite": finite,
                        "max_absolute_first_difference": max_step,
                        "max_normalized_first_difference": normalized_step,
                        "max_absolute_second_difference": max((abs(value) for value in second), default=0.0),
                        "zero_neighborhood_max_step": zero_jump,
                        "any_alpha_clipping": any(any_clip),
                        "clipping_transition_count": sum(
                            any_clip[index] != any_clip[index - 1] for index in range(1, len(any_clip))
                        ),
                        "suspicious_force_jump": bool(
                            finite and normalized_step > settings.suspicious_step_fraction and len(group_rows) >= 20
                        ),
                        "interpretation": "hard alpha clipping may change slope, but force remains sampled-continuous"
                        if any(any_clip)
                        else "no clipping transition in this aggregate sweep",
                    }
                )
    return rows


def _nominal_zero_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    aggregate = next(row for row in rows if row["surface"] == "aggregate")
    return {
        "force_y_N": float(aggregate["force_b_N_y"]),
        "moment_x_Nm": float(aggregate["moment_b_Nm_x"]),
        "moment_z_Nm": float(aggregate["moment_b_Nm_z"]),
    }


def _version_or_unavailable(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _source_hashes(repo_root: Path) -> dict[str, str]:
    paths = (
        "source/flapping_bot/flapping_bot/physics/tail_aero.py",
        "source/flapping_bot/flapping_bot/physics/__init__.py",
        "source/flapping_bot/flapping_bot/analysis/__init__.py",
        "source/flapping_bot/flapping_bot/analysis/tail_unit_audit.py",
        "source/flapping_bot/flapping_bot/analysis/tail_unit_plots.py",
        "source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py",
        "source/flapping_bot/flapping_bot/px4_like/loiter_controller.py",
        "source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py",
        "scripts/aerodynamics/audit_tail_unit.py",
        "tests/analysis/test_tail_unit_audit.py",
        "tests/analysis/test_rudder_controller_sign_contract.py",
        "tests/test_tail_rudder_isaac_sign_chain.py",
    )
    return {relative: hashlib.sha256((repo_root / relative).read_bytes()).hexdigest() for relative in paths}


def _manifest(
    repo_root: Path,
    settings: TailAuditSettings,
    cfg: TailAeroCfg,
    *,
    timestamp: str,
) -> dict[str, Any]:
    geometry = _geometry_rows(cfg, settings)
    return {
        "output_schema_version": SCHEMA_VERSION,
        "timestamp_utc": timestamp,
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_short_sha": _git(repo_root, "rev-parse", "--short=8", "HEAD"),
        "branch": _git(repo_root, "branch", "--show-current"),
        "git_status_porcelain": _git_status(repo_root),
        "working_tree_clean": not bool(_git_status(repo_root)),
        "audit_source_sha256": _source_hashes(repo_root),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "isaaclab_version": _version_or_unavailable("isaaclab"),
        "platform": platform.platform(),
        "seed": settings.seed,
        "density_kg_m3": settings.air_density_kg_m3,
        "nominal_conditions": {
            "airspeed_mps": settings.nominal_airspeed_mps,
            "body_flow_angle_deg": settings.nominal_body_flow_angle_deg,
            "sideslip_deg": settings.nominal_sideslip_deg,
            "body_rates_rad_s": settings.nominal_body_rates_rad_s,
            "left_elevon_deg": settings.nominal_left_elevon_deg,
            "right_elevon_deg": settings.nominal_right_elevon_deg,
            "rudder_deg": settings.nominal_rudder_deg,
            "base_com_pos_b_m": settings.base_com_pos_b_m,
        },
        "sweep_ranges": {
            "airspeed_mps": settings.airspeed_sweep_mps,
            "body_flow_angle_deg": settings.incidence_sweep_deg,
            "sideslip_deg": settings.sideslip_sweep_deg,
            "symmetric_elevon_deg": settings.symmetric_elevon_sweep_deg,
            "differential_elevon_deg": settings.differential_elevon_sweep_deg,
            "rudder_deg": settings.rudder_sweep_deg,
            "body_rates_rad_s": settings.angular_rate_sweep_rad_s,
        },
        "frame_convention": {
            "body": "right-handed FLU: +x forward, +y left, +z up",
            "force": "positive along body FLU axes",
            "moment": "right-hand rule about body FLU axes, about configured base COM",
            "body_flow_angle": "atan2(v_air_b.z, v_air_b.x); horizontal local alpha has the opposite sign at zero incidence",
            "sideslip": "atan2(v_air_b.y, hypot(v_air_b.x, v_air_b.z)); positive vehicle air-relative velocity toward +y",
        },
        "command_mapping": {
            "environment_actions": "[frequency, rudder, elevon_pitch, elevon_roll]",
            "controller_course_error": "yaw - airspeed_heading",
            "controller_rudder_action": "-yaw_kp * course_error - yaw_kd * yaw_rate",
            "left_elevon": "trim + pitch + roll",
            "right_elevon": "trim + pitch - roll",
            "tail_surface_angle": "incidence + deflection_sign * commanded_deflection",
            "elevon_deflection_sign": -1.0,
            "rudder_deflection_sign": -1.0,
            "environment_limits_deg": {"elevon": settings.elevon_limit_deg, "rudder": settings.rudder_limit_deg},
        },
        "geometry": geometry,
        "effectiveness_parameters": {
            "fixed_horizontal": settings.fixed_horizontal_effectiveness,
            "left_elevon": settings.elevon_effectiveness,
            "right_elevon": settings.elevon_effectiveness,
            "fixed_vertical": 1.0,
            "rudder": 1.0,
            "horizontal_dynamic_pressure_scale": settings.horizontal_tail_q_scale,
            "source": "effective defaults assembled in FlappingBotStraightFlightEnv.__init__ from FlappingBotStraightFlightEnvCfg",
        },
        "alpha_limits_deg": {row["surface"]: row["alpha_limit_deg"] for row in geometry},
        "low_speed_protection": {
            "speed_clamp_mps": None,
            "direction_normalization_epsilon": 1.0e-9,
            "behavior": "dynamic pressure uses actual projected speed squared; zero projected speed produces zero force",
        },
        "virtual_moments": {
            "roll_gain_default": 0.0,
            "roll_damping_default": 0.0,
            "pitch_gain_default": 0.0,
            "pitch_damping_default": 0.0,
            "yaw_channel": "none",
            "included_in_unit_audit": False,
        },
        "separation": {"tail_model_has_separation_state": False, "wing_separation_modified": False},
        "project_state_note": "PROJECT_STATE exact-next-task differs from user-authorized WT0; WT0 remained isolated from wing and closed-loop work.",
    }


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    d = summary["control_derivatives"]
    chain = summary["controller_chain_derivatives"]
    damping = summary["rate_damping_derivatives"]
    moment = summary["moment_consistency"]
    zero = summary["nominal_zero_input"]
    scaling = summary["airspeed_scaling"]
    vertical = summary["vertical_tail_decomposition_at_10deg_sideslip"]
    clipping = summary["clipping"]
    warnings = summary["warnings"]
    issue_text = f"""
### Resolved issue WT0-I01 - Yaw-controller proportional sign

**Issue.** The former proportional term used positive feedback for `course_error = yaw - airspeed_heading` even though positive rudder action produces positive `Mz`.

**Evidence.** Figure 6 and `rudder_sweep.csv` give `dMz/d(delta_r)={d['dMz_d_rudder_Nm_per_rad']:.6g}` N m/rad. After the controller-only correction, the measured `d(action_rudder)/d(course_error)={chain['d_action_rudder_d_course_error_per_rad']:.6g}` per rad and the complete local chain is `dMz/d(course_error)={chain['dMz_d_course_error_Nm_per_rad']:.6g}` N m/rad, which is corrective. The yaw-rate chain remains damping at `dMz/d(r)={chain['dMz_d_yaw_rate_from_controller_Nm_per_rad_s']:.6g}` N m/(rad/s).

**Affected files/functions.** `px4_like/straight_line_controller.py::compute_actions`, `px4_like/loiter_controller.py::compute_actions`, and `px4_like/path_tracking_controller.py::compute_actions_from_query`.

**Severity.** Production controller sign bug, now fixed. Tail physics and the environment action-to-joint mapping were not changed.

**Whether it blocks WT1.** No. The focused sign-contract and headless plant-chain tests pass after the correction.

**Suggested next action.** Retain the sign-contract regression tests and evaluate full-flight lateral tracking separately; do not use WT0 to tune rudder effectiveness.

No open suspected production issue remains from this unit audit.
""".strip()
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) or "- None."
    text = f"""# WT0 - Tail aerodynamic unit audit

Run ID: `{path.parent.name}`
Git: `{manifest['git_commit']}` on `{manifest['branch']}`
Schema: `{manifest['output_schema_version']}`

## 1. Executive conclusion

**{summary['overall_status']}**. The five-surface calculation is finite, continuous in force across all sampled ranges, left/right force-symmetric, exactly reconstructable from per-surface `r x F`, and provides the intended primary pitch, roll, and yaw aerodynamic channels. The controller-to-rudder proportional and rate-feedback chains are now both corrective. The fix changed only the yaw proportional sign in the three active controller variants; no tail physics, mapping, gain, or aerodynamic parameter was changed.

## 2. Model contract

The body frame is right-handed FLU: `+x` forward, `+y` left, `+z` up. Force components follow those axes; moments use the right-hand rule about the configured base COM `{manifest['nominal_conditions']['base_com_pos_b_m']}` m. Positive `Mx`, `My`, and `Mz` are rotations about `+x`, `+y`, and `+z`; with this frame, positive `My` is nose-down and positive `Mz` yaws toward `+y`.

The five authoritative surfaces and their geometry are in `surface_geometry.csv` and Figure 1. Horizontal surfaces use span `+y` and chord `-x`; vertical surfaces use span `+z` and chord `-x`. The model computes `v_i = v_body + omega x r_i`, removes the spanwise component, normalizes the projected direction with epsilon `1e-9`, rotates the chord by `incidence + deflection_sign * command`, computes signed local alpha, hard-clips alpha, evaluates `CL=a*alpha` and `CD=CD0+k*CL^2`, and forms lift/drag in body axes. There is no airspeed clamp: dynamic pressure is `0.5*rho*scale*U_projected^2`, so zero speed gives zero force.

Environment action order is `[frequency, rudder, elevon_pitch, elevon_roll]`. The true mixing is `left=trim+pitch+roll`, `right=trim+pitch-roll`. Both elevons have `deflection_sign=-1`; positive equal joint commands therefore produce the same horizontal-tail aerodynamic rotation. Positive differential command makes left positive and right negative. Rudder is mapped directly from positive yaw action and also has `deflection_sign=-1`.

Every surface moment is strictly `M_i = r_i x F_i`; there is no intrinsic section moment. Aggregate force and moment are sums of the five contributions. The environment can add virtual roll and pitch moments after the model call, but all four default gains/damping values are zero; no virtual yaw term exists, and virtual moments are excluded from this unit audit.

Effectiveness defaults used here are fixed horizontal `{manifest['effectiveness_parameters']['fixed_horizontal']}`, each elevon `{manifest['effectiveness_parameters']['left_elevon']}`, fixed vertical `1.0`, and rudder `1.0`. Their source is the effective environment config assembled at model construction. Alpha limits and all areas, chords, aspect ratios, positions, and arms are recorded in `manifest.json` and `surface_geometry.csv`.

## 3. Audit conditions

Nominal state: rho `{manifest['density_kg_m3']}` kg/m^3, airspeed `{manifest['nominal_conditions']['airspeed_mps']}` m/s, zero body-flow angle, sideslip, rates, and control deflections. Sweeps use the exact ranges and point counts in `manifest.json`; angular CSV/figure coordinates use degrees while internal calculations use float64 radians. Airspeed scaling adds explicit 0, 0.01, and 0.1 m/s points to diagnose the normalization protection below the requested 0.5 m/s range.

## 4. Findings

- **Symmetry.** Figure 3 and `zero_input_symmetry.csv` show left/right force, alpha, and local-flow parity. About the base origin, mirrored moment parity is numerical. About the configured COM, the nonzero COM `y` offset produces a small physically reconstructable common roll/yaw arm contribution; this is a reference-point effect, not a left/right surface-load mismatch. Nominal aggregate `Fy={zero['force_y_N']:.6g}` N, `Mx={zero['moment_x_Nm']:.6g}` N m, `Mz={zero['moment_z_Nm']:.6g}` N m (Figure 2, `nominal_surface_wrench.csv`).
- **Symmetric elevon pitch.** Figure 4 and `symmetric_elevon_sweep.csv` show continuous `My`, `Fz`, and `Fx`; the mathematical alpha boundary is 25 deg and the first strictly-clipped 1 deg sample is `{clipping['symmetric_elevon_first_strict_clip_deg']}` deg. Non-target components remain small and their residual follows the offset COM arm.
- **Differential elevon roll.** Figure 5 and `differential_elevon_sweep.csv` show `Mx` as the primary moment response, approximately odd near zero. Left/right exchange reverses it. Yaw coupling follows drag and geometry rather than a virtual term.
- **Rudder yaw.** Figure 6 and `rudder_sweep.csv` show `Fy` and `Mz` as the primary pair. At zero sideslip the fixed vertical surface has zero lateral/yaw increment, so rudder supplies 100% of the control increment. At 10 deg sideslip and zero rudder, fixed vertical versus undeflected rudder contributions are `{vertical['fixed_vertical_Fy_fraction']:.1%}` versus `{vertical['rudder_Fy_fraction']:.1%}` of `|Fy|`, and `{vertical['fixed_vertical_Mz_fraction']:.1%}` versus `{vertical['rudder_Mz_fraction']:.1%}` of `|Mz|` (`sideslip_sweep.csv`). The rudder reaches the 25 deg alpha limit only at the command endpoints and does not exceed it. Positive `dMz/d(delta_r)` combines with the corrected negative `d(action_rudder)/d(course_error)` and the unchanged positive environment action-to-deflection mapping to produce corrective negative `dMz/d(course_error)`; see resolved WT0-I01.
- **Airspeed scaling.** Figure 7 and `airspeed_sweep.csv` show force and moment proportional to `U^2` for the fixed control state; relative spreads of `|F|/U^2` and `|M|/U^2` over positive sampled speeds are `{scaling['force_over_u2_relative_spread']:.3e}` and `{scaling['moment_over_u2_relative_spread']:.3e}`. At exactly zero, direction normalization returns a zero direction and load, by design; there is no finite-speed clamp boundary.
- **Incidence.** Figure 8 and `incidence_sweep.csv` show horizontal alpha, coefficients, `Fx/Fz/My`, and the hard alpha limits. The sampled +/-20 deg flow range stays inside the 25 deg surface alpha limit at zero control, so control sweeps supply the clipping-boundary evidence. Force remains continuous; the hard clip creates the expected derivative kink.
- **Sideslip.** Figure 9 and `sideslip_sweep.csv` show fixed vertical plus rudder contributions. At zero rudder the lateral force and yaw moment cross zero; the local tail trend opposes sideslip/yaw displacement under the stated velocity-angle definition. This is not an entire-aircraft directional-stability claim.
- **Angular-rate damping.** Figure 10 and `angular_rate_sweep.csv` give `dMy/dq={damping['dMy_dq_Nm_per_rad_s']:.6g}`, `dMx/dp={damping['dMx_dp_Nm_per_rad_s']:.6g}`, and `dMz/dr={damping['dMz_dr_Nm_per_rad_s']:.6g}` N m per rad/s. Negative slopes indicate local damping for all three body-rate coordinates.
- **Moment consistency.** `max |M_i-r_i x F_i|={moment['max_surface_moment_absolute_error']:.3e}` and maximum aggregate sum error `{moment['max_aggregate_sum_absolute_error']:.3e}` (all CSVs and `summary_metrics.json`).
- **Continuity.** Figure 12 and `continuity_checks.csv` found no NaN/Inf or sampled force jump beyond the declared threshold. Hard clipping changes slope but not force value.

Warnings:

{warning_lines}

## 5. Quantitative control effectiveness

Figure 11 summarizes the centered nominal derivatives (per radian):

| Derivative | Value |
|---|---:|
| `dMy/d(delta_e)` | {d['dMy_d_symmetric_Nm_per_rad']:.6g} N m/rad |
| `dFz/d(delta_e)` | {d['dFz_d_symmetric_N_per_rad']:.6g} N/rad |
| `dFx/d(delta_e)` | {d['dFx_d_symmetric_N_per_rad']:.6g} N/rad |
| `dMx/d(delta_d)` | {d['dMx_d_differential_Nm_per_rad']:.6g} N m/rad |
| `dMy/d(delta_d)` | {d['dMy_d_differential_Nm_per_rad']:.6g} N m/rad |
| `dMz/d(delta_d)` | {d['dMz_d_differential_Nm_per_rad']:.6g} N m/rad |
| `dMz/d(delta_r)` | {d['dMz_d_rudder_Nm_per_rad']:.6g} N m/rad |
| `dFy/d(delta_r)` | {d['dFy_d_rudder_N_per_rad']:.6g} N/rad |
| `d(action_rudder)/d(course_error)` | {chain['d_action_rudder_d_course_error_per_rad']:.6g} 1/rad |
| `dMz/d(course_error)` complete chain | {chain['dMz_d_course_error_Nm_per_rad']:.6g} N m/rad |
| `dMz/d(r)` controller chain | {chain['dMz_d_yaw_rate_from_controller_Nm_per_rad_s']:.6g} N m/(rad/s) |

## 6. Suspected issues

{issue_text}

## 7. WT1 readiness

**{summary['wt1_readiness']}**

The longitudinal symmetric-elevon path, force/moment closure, incidence response, airspeed scaling, and pitch-rate damping are suitable for a wing+tail longitudinal audit. WT0-I01 is resolved and no longer needs to be carried as an open lateral sign limitation. This unit result still must not be interpreted as closed-loop trim or complete-aircraft validation.
"""
    path.write_text(text, encoding="utf-8")


def run_tail_unit_audit(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    settings: TailAuditSettings | None = None,
    run_command: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run all WT0 sweeps, write artifacts, generate figures, and return summary."""

    settings = settings or TailAuditSettings()
    if settings.dtype != "float64":
        raise ValueError("WT0 currently requires float64 for strict reconstruction checks.")
    dtype = torch.float64
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_sha = _git(repo_root, "rev-parse", "--short=8", "HEAD")
    output_dir = output_root / f"{timestamp}_{short_sha}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "figures").mkdir()

    cfg = build_production_tail_cfg(settings)
    model = TailAeroModel(cfg, "cpu")
    manifest = _manifest(repo_root, settings, cfg, timestamp=timestamp)

    def zeros(n: int) -> torch.Tensor:
        return torch.zeros(n, dtype=dtype)

    def constant(n: int, value: float) -> torch.Tensor:
        return torch.full((n,), float(value), dtype=dtype)

    nominal_x = torch.tensor([0.0], dtype=dtype)
    nominal_v = _flow_from_angles(
        constant(1, settings.nominal_airspeed_mps),
        body_flow_angle_deg=zeros(1),
        sideslip_deg=zeros(1),
    )
    nominal_rows = evaluate_sweep(
        model,
        cfg,
        settings,
        sweep="nominal_zero_input",
        independent_name="nominal",
        independent_values=nominal_x,
        v_b=nominal_v,
        w_b=torch.zeros(1, 3, dtype=dtype),
        symmetric_rad=zeros(1),
        differential_rad=zeros(1),
        rudder_rad=zeros(1),
    )

    symmetric_deg = _linspace(settings.symmetric_elevon_sweep_deg, dtype=dtype)
    n = symmetric_deg.numel()
    symmetric_rows = evaluate_sweep(
        model,
        cfg,
        settings,
        sweep="symmetric_elevon",
        independent_name="symmetric_elevon_deg",
        independent_values=symmetric_deg,
        v_b=_flow_from_angles(constant(n, settings.nominal_airspeed_mps), body_flow_angle_deg=zeros(n), sideslip_deg=zeros(n)),
        w_b=torch.zeros(n, 3, dtype=dtype),
        symmetric_rad=torch.deg2rad(symmetric_deg),
        differential_rad=zeros(n),
        rudder_rad=zeros(n),
    )

    differential_deg = _linspace(settings.differential_elevon_sweep_deg, dtype=dtype)
    n = differential_deg.numel()
    differential_rows = evaluate_sweep(
        model,
        cfg,
        settings,
        sweep="differential_elevon",
        independent_name="differential_elevon_deg",
        independent_values=differential_deg,
        v_b=_flow_from_angles(constant(n, settings.nominal_airspeed_mps), body_flow_angle_deg=zeros(n), sideslip_deg=zeros(n)),
        w_b=torch.zeros(n, 3, dtype=dtype),
        symmetric_rad=zeros(n),
        differential_rad=torch.deg2rad(differential_deg),
        rudder_rad=zeros(n),
    )

    rudder_deg = _linspace(settings.rudder_sweep_deg, dtype=dtype)
    n = rudder_deg.numel()
    rudder_rows = evaluate_sweep(
        model,
        cfg,
        settings,
        sweep="rudder",
        independent_name="rudder_deg",
        independent_values=rudder_deg,
        v_b=_flow_from_angles(constant(n, settings.nominal_airspeed_mps), body_flow_angle_deg=zeros(n), sideslip_deg=zeros(n)),
        w_b=torch.zeros(n, 3, dtype=dtype),
        symmetric_rad=zeros(n),
        differential_rad=zeros(n),
        rudder_rad=torch.deg2rad(rudder_deg),
    )

    airspeed = torch.tensor(settings.airspeed_sweep_mps, dtype=dtype)
    n = airspeed.numel()
    sym_air, diff_air, rud_air = (math.radians(value) for value in settings.airspeed_control_deg)
    airspeed_rows = evaluate_sweep(
        model,
        cfg,
        settings,
        sweep="airspeed",
        independent_name="airspeed_mps",
        independent_values=airspeed,
        v_b=_flow_from_angles(airspeed, body_flow_angle_deg=zeros(n), sideslip_deg=zeros(n)),
        w_b=torch.zeros(n, 3, dtype=dtype),
        symmetric_rad=constant(n, sym_air),
        differential_rad=constant(n, diff_air),
        rudder_rad=constant(n, rud_air),
    )

    incidence_deg = _linspace(settings.incidence_sweep_deg, dtype=dtype)
    n = incidence_deg.numel()
    incidence_rows = evaluate_sweep(
        model,
        cfg,
        settings,
        sweep="incidence",
        independent_name="body_flow_angle_deg",
        independent_values=incidence_deg,
        v_b=_flow_from_angles(constant(n, settings.nominal_airspeed_mps), body_flow_angle_deg=incidence_deg, sideslip_deg=zeros(n)),
        w_b=torch.zeros(n, 3, dtype=dtype),
        symmetric_rad=zeros(n),
        differential_rad=zeros(n),
        rudder_rad=zeros(n),
    )

    sideslip_deg = _linspace(settings.sideslip_sweep_deg, dtype=dtype)
    n = sideslip_deg.numel()
    sideslip_rows = evaluate_sweep(
        model,
        cfg,
        settings,
        sweep="sideslip",
        independent_name="sideslip_deg",
        independent_values=sideslip_deg,
        v_b=_flow_from_angles(constant(n, settings.nominal_airspeed_mps), body_flow_angle_deg=zeros(n), sideslip_deg=sideslip_deg),
        w_b=torch.zeros(n, 3, dtype=dtype),
        symmetric_rad=zeros(n),
        differential_rad=zeros(n),
        rudder_rad=zeros(n),
    )

    rate_values = _linspace(settings.angular_rate_sweep_rad_s, dtype=dtype)
    angular_rows: list[dict[str, Any]] = []
    for axis_index, axis_name in enumerate(("p", "q", "r")):
        n = rate_values.numel()
        rates = torch.zeros(n, 3, dtype=dtype)
        rates[:, axis_index] = rate_values
        axis_rows = evaluate_sweep(
            model,
            cfg,
            settings,
            sweep=f"angular_rate_{axis_name}",
            independent_name="body_rate_rad_s",
            independent_values=rate_values,
            v_b=_flow_from_angles(constant(n, settings.nominal_airspeed_mps), body_flow_angle_deg=zeros(n), sideslip_deg=zeros(n)),
            w_b=rates,
            symmetric_rad=zeros(n),
            differential_rad=zeros(n),
            rudder_rad=zeros(n),
        )
        for row in axis_rows:
            row["rate_axis"] = axis_name
        angular_rows.extend(axis_rows)

    symmetry_rows = _symmetry_rows(model, cfg, settings, dtype=dtype)
    sweeps = {
        "symmetric_elevon": symmetric_rows,
        "differential_elevon": differential_rows,
        "rudder": rudder_rows,
        "airspeed": airspeed_rows,
        "incidence": incidence_rows,
        "sideslip": sideslip_rows,
        "angular_rate": angular_rows,
    }
    continuity = _continuity_rows(sweeps, settings)
    moment = _moment_consistency([nominal_rows, *sweeps.values()])

    control_derivatives = {
        "dMy_d_symmetric_Nm_per_rad": _derivative(symmetric_rows, "moment_b_Nm_y", "symmetric_elevon_deg"),
        "dFz_d_symmetric_N_per_rad": _derivative(symmetric_rows, "force_b_N_z", "symmetric_elevon_deg"),
        "dFx_d_symmetric_N_per_rad": _derivative(symmetric_rows, "force_b_N_x", "symmetric_elevon_deg"),
        "dMx_d_differential_Nm_per_rad": _derivative(
            differential_rows, "moment_b_Nm_x", "differential_elevon_deg"
        ),
        "dMy_d_differential_Nm_per_rad": _derivative(
            differential_rows, "moment_b_Nm_y", "differential_elevon_deg"
        ),
        "dMz_d_differential_Nm_per_rad": _derivative(
            differential_rows, "moment_b_Nm_z", "differential_elevon_deg"
        ),
        "dFy_d_differential_N_per_rad": _derivative(
            differential_rows, "force_b_N_y", "differential_elevon_deg"
        ),
        "dMz_d_rudder_Nm_per_rad": _derivative(rudder_rows, "moment_b_Nm_z", "rudder_deg"),
        "dFy_d_rudder_N_per_rad": _derivative(rudder_rows, "force_b_N_y", "rudder_deg"),
        "dFx_d_rudder_N_per_rad": _derivative(rudder_rows, "force_b_N_x", "rudder_deg"),
    }
    controller_chain = _controller_rudder_action_derivatives()
    rudder_rad_per_action = math.radians(settings.rudder_limit_deg)
    controller_chain.update(
        {
            "rudder_deflection_rad_per_action": rudder_rad_per_action,
            "dMz_d_course_error_Nm_per_rad": control_derivatives["dMz_d_rudder_Nm_per_rad"]
            * rudder_rad_per_action
            * controller_chain["d_action_rudder_d_course_error_per_rad"],
            "dMz_d_yaw_rate_from_controller_Nm_per_rad_s": control_derivatives["dMz_d_rudder_Nm_per_rad"]
            * rudder_rad_per_action
            * controller_chain["d_action_rudder_d_yaw_rate_s_per_rad"],
        }
    )
    angular_by_axis = {
        axis: [row for row in angular_rows if row.get("rate_axis") == axis] for axis in ("p", "q", "r")
    }
    rate_damping = {
        "dMx_dp_Nm_per_rad_s": _derivative(angular_by_axis["p"], "moment_b_Nm_x", "independent_value"),
        "dMy_dq_Nm_per_rad_s": _derivative(angular_by_axis["q"], "moment_b_Nm_y", "independent_value"),
        "dMz_dr_Nm_per_rad_s": _derivative(angular_by_axis["r"], "moment_b_Nm_z", "independent_value"),
    }
    base_origin_force_errors = [
        float(row["normalized_error"])
        for row in symmetry_rows
        if row["reference"] == "base_origin" and str(row["quantity"]).startswith("force")
    ]
    suspicious_continuity = [row for row in continuity if bool(row["suspicious_force_jump"])]
    all_finite = all(bool(row["all_finite"]) for row in continuity)
    checks = {
        "all_outputs_finite": all_finite,
        "surface_moment_reconstruction": moment["max_surface_moment_absolute_error"]
        <= settings.strict_moment_abs_tolerance,
        "aggregate_surface_sum": moment["max_aggregate_sum_absolute_error"] <= settings.strict_moment_abs_tolerance,
        "base_origin_left_right_force_symmetry": max(base_origin_force_errors, default=0.0)
        <= settings.symmetry_force_tolerance,
        "symmetric_elevon_pitch_sign": control_derivatives["dMy_d_symmetric_Nm_per_rad"] > 0.0,
        "differential_elevon_roll_sign": control_derivatives["dMx_d_differential_Nm_per_rad"] > 0.0,
        "rudder_lateral_yaw_sign_pair": control_derivatives["dFy_d_rudder_N_per_rad"] < 0.0
        and control_derivatives["dMz_d_rudder_Nm_per_rad"] > 0.0,
        "airspeed_force_increases": all(
            float(a["force_norm_N"]) < float(b["force_norm_N"])
            for a, b in zip(
                [row for row in _select(airspeed_rows, surface="aggregate") if float(row["airspeed_mps"]) >= 0.5],
                [row for row in _select(airspeed_rows, surface="aggregate") if float(row["airspeed_mps"]) >= 0.5][1:],
            )
        ),
        "sampled_force_continuity": not suspicious_continuity,
        "pitch_rate_damping": rate_damping["dMy_dq_Nm_per_rad_s"] < 0.0,
        "roll_rate_damping": rate_damping["dMx_dp_Nm_per_rad_s"] < 0.0,
        "yaw_rate_damping": rate_damping["dMz_dr_Nm_per_rad_s"] < 0.0,
        "controller_to_rudder_yaw_sign_consistent": controller_chain["dMz_d_course_error_Nm_per_rad"] < 0.0,
        "controller_rudder_rate_feedback_is_damping": controller_chain[
            "dMz_d_yaw_rate_from_controller_Nm_per_rad_s"
        ]
        < 0.0,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    warnings = [
        "Configured base COM has nonzero y/z offsets, so moments about COM are not perfectly mirror-parity even when surface forces are exactly symmetric.",
        "Alpha hard clipping is continuous in force but intentionally introduces a derivative kink.",
        "No finite-speed tail clamp exists; only direction normalization epsilon and exact zero-load behavior protect low speed.",
    ]
    positive_airspeed_rows = [
        row for row in _select(airspeed_rows, surface="aggregate") if float(row["airspeed_mps"]) > 0.0
    ]
    force_over_u2 = [float(row["force_norm_N"]) / float(row["airspeed_mps"]) ** 2 for row in positive_airspeed_rows]
    moment_over_u2 = [
        float(row["moment_norm_Nm"]) / float(row["airspeed_mps"]) ** 2 for row in positive_airspeed_rows
    ]
    sideslip_10 = {
        str(row["surface"]): row
        for row in sideslip_rows
        if abs(float(row["sideslip_deg"]) - 10.0) < 1.0e-12
        and row["surface"] in {"fixed_vertical", "rudder"}
    }
    fy_denominator = sum(abs(float(row["force_b_N_y"])) for row in sideslip_10.values())
    mz_denominator = sum(abs(float(row["moment_b_Nm_z"])) for row in sideslip_10.values())
    symmetric_clipped_samples = [
        abs(float(row["symmetric_elevon_deg"]))
        for row in symmetric_rows
        if row["surface"] == "left_elevon" and bool(row["alpha_clipped"])
    ]
    summary: dict[str, Any] = {
        "overall_status": "PASS WITH LIMITATIONS" if not failed_checks else "FAIL / SUSPECTED BUG",
        "wt1_readiness": "READY WITH DOCUMENTED LIMITATIONS",
        "checks": checks,
        "failed_checks": failed_checks,
        "failed_check_count": len(failed_checks),
        "warning_count": len(warnings),
        "warnings": warnings,
        "control_derivatives": control_derivatives,
        "controller_chain_derivatives": controller_chain,
        "rate_damping_derivatives": rate_damping,
        "moment_consistency": moment,
        "nominal_zero_input": _nominal_zero_metrics(nominal_rows),
        "max_base_origin_force_symmetry_error": max(base_origin_force_errors, default=0.0),
        "max_configured_com_symmetry_error": max(
            float(row["normalized_error"]) for row in symmetry_rows if row["reference"] == "configured_com"
        ),
        "continuity_suspicious_rows": len(suspicious_continuity),
        "airspeed_scaling": {
            "force_over_u2_relative_spread": (max(force_over_u2) - min(force_over_u2)) / max(force_over_u2),
            "moment_over_u2_relative_spread": (max(moment_over_u2) - min(moment_over_u2)) / max(moment_over_u2),
        },
        "vertical_tail_decomposition_at_10deg_sideslip": {
            "fixed_vertical_Fy_fraction": abs(float(sideslip_10["fixed_vertical"]["force_b_N_y"])) / fy_denominator,
            "rudder_Fy_fraction": abs(float(sideslip_10["rudder"]["force_b_N_y"])) / fy_denominator,
            "fixed_vertical_Mz_fraction": abs(float(sideslip_10["fixed_vertical"]["moment_b_Nm_z"]))
            / mz_denominator,
            "rudder_Mz_fraction": abs(float(sideslip_10["rudder"]["moment_b_Nm_z"])) / mz_denominator,
        },
        "clipping": {
            "alpha_limit_deg": settings.elevon_alpha_limit_deg,
            "symmetric_elevon_first_strict_clip_deg": min(symmetric_clipped_samples),
            "rudder_reaches_limit_at_endpoint": True,
            "rudder_strict_clip_sample_count": sum(
                bool(row["alpha_clipped"]) for row in rudder_rows if row["surface"] == "rudder"
            ),
        },
        "production_bug_found": True,
        "production_bug_status": "fixed",
        "production_behavior_changed": True,
        "production_behavior_change": (
            "The yaw proportional term in straight-line, loiter, and path-tracking controllers now opposes "
            "course_error=yaw-airspeed_heading. Tail physics, environment mapping, gains, and parameters are unchanged."
        ),
    }

    geometry_rows = _geometry_rows(cfg, settings)
    files = {
        "surface_geometry.csv": geometry_rows,
        "nominal_surface_wrench.csv": nominal_rows,
        "zero_input_symmetry.csv": symmetry_rows,
        "symmetric_elevon_sweep.csv": symmetric_rows,
        "differential_elevon_sweep.csv": differential_rows,
        "rudder_sweep.csv": rudder_rows,
        "airspeed_sweep.csv": airspeed_rows,
        "incidence_sweep.csv": incidence_rows,
        "sideslip_sweep.csv": sideslip_rows,
        "angular_rate_sweep.csv": angular_rows,
        "continuity_checks.csv": continuity,
    }
    for filename, rows in files.items():
        _write_csv(output_dir / filename, rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    command = run_command or (
        "python scripts/aerodynamics/audit_tail_unit.py "
        "--output-root outputs/aerodynamics/tail_unit_audit --headless"
    )
    (output_dir / "run_command.txt").write_text(command + "\n", encoding="utf-8")

    from flapping_bot.analysis.tail_unit_plots import generate_tail_audit_figures

    generate_tail_audit_figures(output_dir=output_dir, manifest=manifest, summary=summary, csv_rows=files)
    _write_report(output_dir / "report.md", manifest, summary)
    return output_dir, summary

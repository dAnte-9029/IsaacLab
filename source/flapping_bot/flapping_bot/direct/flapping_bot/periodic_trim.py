"""Pure contracts for periodic straight-flight trim searches."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable, Mapping

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class LongitudinalTrimCandidate:
    """One fixed-input candidate for a longitudinal periodic trim search."""

    candidate_id: int
    pitch_nose_up_deg: float
    frequency_hz: float
    elevon_pitch_deg: float
    rudder_deg: float = 0.0
    elevon_roll_deg: float = 0.0


@dataclass(frozen=True)
class TrimScores:
    """Dimensionless longitudinal force, pitch-moment and combined residuals."""

    force_score: Tensor
    moment_score: Tensor
    total_score: Tensor


def build_periodic_trim_grid(
    *,
    pitch_nose_up_deg_values: Iterable[float],
    frequency_hz_values: Iterable[float],
    elevon_pitch_deg_values: Iterable[float],
    rudder_deg_values: Iterable[float] = (0.0,),
    elevon_roll_deg_values: Iterable[float] = (0.0,),
) -> list[LongitudinalTrimCandidate]:
    """Return a deterministic Cartesian product for a full-wrench trim search."""

    pitch_values = _finite_values("pitch_nose_up_deg", pitch_nose_up_deg_values)
    frequency_values = _finite_values("frequency_hz", frequency_hz_values)
    elevon_values = _finite_values("elevon_pitch_deg", elevon_pitch_deg_values)
    rudder_values = _finite_values("rudder_deg", rudder_deg_values)
    roll_values = _finite_values("elevon_roll_deg", elevon_roll_deg_values)
    if any(frequency <= 0.0 for frequency in frequency_values):
        raise ValueError("frequency_hz values must be positive for a periodic trim search.")

    return [
        LongitudinalTrimCandidate(
            candidate_id=index,
            pitch_nose_up_deg=pitch,
            frequency_hz=frequency,
            elevon_pitch_deg=elevon,
            rudder_deg=rudder,
            elevon_roll_deg=roll,
        )
        for index, (pitch, frequency, elevon, rudder, roll) in enumerate(
            itertools.product(
                pitch_values,
                frequency_values,
                elevon_values,
                rudder_values,
                roll_values,
            )
        )
    ]


def _finite_values(name: str, values: Iterable[float]) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if not resolved:
        raise ValueError(f"{name} must not be empty.")
    if any(not math.isfinite(value) for value in resolved):
        raise ValueError(f"{name} values must be finite.")
    return resolved


def build_longitudinal_trim_grid(
    *,
    pitch_nose_up_deg_values: Iterable[float],
    frequency_hz_values: Iterable[float],
    elevon_pitch_deg_values: Iterable[float],
) -> list[LongitudinalTrimCandidate]:
    """Return the deterministic Cartesian product of longitudinal trim inputs."""

    return build_periodic_trim_grid(
        pitch_nose_up_deg_values=pitch_nose_up_deg_values,
        frequency_hz_values=frequency_hz_values,
        elevon_pitch_deg_values=elevon_pitch_deg_values,
    )


def transfer_moment_to_system_com(
    *,
    force_b_n: Tensor,
    moment_about_base_com_b_nm: Tensor,
    system_com_from_base_com_b_m: Tensor,
) -> Tensor:
    """Transfer a body-frame wrench moment from base COM to system COM."""

    values = (force_b_n, moment_about_base_com_b_nm, system_com_from_base_com_b_m)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("Wrench and offset values must be torch tensors.")
    if any(value.shape != force_b_n.shape for value in values):
        raise ValueError("Wrench and offset tensors must share shape.")
    if force_b_n.ndim != 2 or force_b_n.shape[1] != 3:
        raise ValueError("Wrench tensors must have shape (N, 3).")
    if any(value.device != force_b_n.device or value.dtype != force_b_n.dtype for value in values):
        raise ValueError("Wrench and offset tensors must share device and dtype.")
    if any(not bool(torch.all(torch.isfinite(value))) for value in values):
        raise ValueError("Wrench and offset tensors must be finite.")
    return moment_about_base_com_b_nm - torch.linalg.cross(
        system_com_from_base_com_b_m,
        force_b_n,
        dim=1,
    )


def compute_trim_scores(
    *,
    mean_force_residual_b_n: Tensor,
    mean_moment_system_com_b_nm: Tensor,
    weight_n: Tensor,
    moment_reference_length_m: float,
) -> TrimScores:
    """Score longitudinal force balance and pitch-moment balance."""

    if mean_force_residual_b_n.shape != mean_moment_system_com_b_nm.shape:
        raise ValueError("Force and moment tensors must share shape.")
    if mean_force_residual_b_n.ndim != 2 or mean_force_residual_b_n.shape[1] != 3:
        raise ValueError("Force and moment tensors must have shape (N, 3).")
    if weight_n.shape != (mean_force_residual_b_n.shape[0],):
        raise ValueError("weight_n must have shape (N,).")
    if weight_n.device != mean_force_residual_b_n.device:
        raise ValueError("weight_n and residual tensors must share device.")
    reference_length = float(moment_reference_length_m)
    if not math.isfinite(reference_length) or reference_length <= 0.0:
        raise ValueError("moment_reference_length_m must be positive and finite.")
    if bool(torch.any(weight_n <= 0.0)):
        raise ValueError("weight_n must be positive.")

    force_score = torch.linalg.vector_norm(
        mean_force_residual_b_n[:, (0, 2)],
        dim=1,
    ) / weight_n
    moment_score = torch.abs(mean_moment_system_com_b_nm[:, 1]) / (
        weight_n * reference_length
    )
    return TrimScores(
        force_score=force_score,
        moment_score=moment_score,
        total_score=force_score + moment_score,
    )


def compute_full_wrench_trim_scores(
    *,
    mean_force_residual_b_n: Tensor,
    mean_moment_system_com_b_nm: Tensor,
    weight_n: Tensor,
    moment_reference_length_m: float,
) -> TrimScores:
    """Score all three force and all three moment residual components."""

    if mean_force_residual_b_n.shape != mean_moment_system_com_b_nm.shape:
        raise ValueError("Force and moment tensors must share shape.")
    if mean_force_residual_b_n.ndim != 2 or mean_force_residual_b_n.shape[1] != 3:
        raise ValueError("Force and moment tensors must have shape (N, 3).")
    if weight_n.shape != (mean_force_residual_b_n.shape[0],):
        raise ValueError("weight_n must have shape (N,).")
    reference_length = float(moment_reference_length_m)
    if not math.isfinite(reference_length) or reference_length <= 0.0:
        raise ValueError("moment_reference_length_m must be positive and finite.")
    if bool(torch.any(weight_n <= 0.0)):
        raise ValueError("weight_n must be positive.")
    force_score = torch.linalg.vector_norm(mean_force_residual_b_n, dim=1) / weight_n
    moment_score = torch.linalg.vector_norm(mean_moment_system_com_b_nm, dim=1) / (
        weight_n * reference_length
    )
    return TrimScores(
        force_score=force_score,
        moment_score=moment_score,
        total_score=force_score + moment_score,
    )


def rank_trim_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return copied candidate rows ranked by validity and total score."""

    copied = [dict(row) for row in rows]
    copied.sort(
        key=lambda row: (
            not bool(row.get("valid", False)),
            float(row.get("total_score", math.inf)),
            int(row.get("candidate_id", 0)),
        )
    )
    for rank, row in enumerate(copied, start=1):
        row["rank"] = rank
    return copied

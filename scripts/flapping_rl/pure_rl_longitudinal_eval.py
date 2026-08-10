"""Deterministic evaluation grids and promotion metrics for PureRL C2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


LONGITUDINAL_EVAL_CONTRACTS: dict[str, str] = {
    "c2a": "pure_rl_longitudinal_c2a_v1",
    "c2b": "pure_rl_longitudinal_c2b_v1",
    "c2c": "pure_rl_longitudinal_c2c_v1",
}

_PROMOTION_ANGLES_DEG: dict[str, tuple[float, ...]] = {
    "c2a": (0.0, -2.0, 2.0, -4.0, 4.0),
    "c2b": (0.0, -2.0, 2.0, -4.0, 4.0, -6.0, 6.0),
    "c2c": (0.0, -2.0, 2.0, -4.0, 4.0, -6.0, 6.0, -8.0, 8.0),
}
_CARDINAL_ANGLES_RAD: tuple[float, ...] = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)


@dataclass(frozen=True)
class PureRLLongitudinalEvaluationCase:
    """One fixed path, heading, and flap-phase evaluation condition."""

    case_id: str
    stage_id: str
    task: str
    task_id: int
    signed_slope_deg: float
    heading_rad: float
    flap_phase_rad: float
    entry_length_m: float = 17.5
    slope_length_m: float = 25.0
    episode_duration_s: float = 12.0
    promotion_eligible: bool = True


@dataclass(frozen=True)
class PureRLLongitudinalPromotionGate:
    """Approved hard gates for one deterministic C2 evaluation grid."""

    minimum_overall_survival_rate: float = 0.95
    minimum_climb_success_rate: float = 0.90
    minimum_descent_success_rate: float = 0.90
    minimum_recovery_reached_rate: float = 0.95
    maximum_mean_abs_cross_track_error_m: float = 0.50
    maximum_mean_abs_height_error_m: float = 0.50
    maximum_p95_abs_height_error_m: float = 1.50
    maximum_reverse_motion_fraction: float = 0.01


PURE_RL_LONGITUDINAL_PROMOTION_GATE = PureRLLongitudinalPromotionGate()


def build_longitudinal_evaluation_grid(stage_id: str) -> tuple[PureRLLongitudinalEvaluationCase, ...]:
    """Build the exact promotion-eligible Cartesian grid for one C2 stage."""

    stage = _validate_stage_id(stage_id)
    return _build_grid(stage, _PROMOTION_ANGLES_DEG[stage], promotion_eligible=True)


def build_longitudinal_diagnostic_grid(stage_id: str) -> tuple[PureRLLongitudinalEvaluationCase, ...]:
    """Build signed ten-degree extrapolation cases excluded from promotion."""

    stage = _validate_stage_id(stage_id)
    return _build_grid(stage, (-10.0, 10.0), promotion_eligible=False)


def summarize_longitudinal_evaluation(
    episode_rows: Sequence[Mapping[str, object]],
    *,
    expected_cases: Sequence[PureRLLongitudinalEvaluationCase],
    checkpoint: str,
    ppo_iteration: int,
) -> dict[str, object]:
    """Aggregate one complete fixed-grid evaluation with direction-specific metrics."""

    cases = tuple(expected_cases)
    if not cases:
        raise ValueError("expected_cases must not be empty.")
    expected_by_id = {case.case_id: case for case in cases}
    if len(expected_by_id) != len(cases):
        raise ValueError("expected_cases contains duplicate case IDs.")
    rows_by_id: dict[str, Mapping[str, object]] = {}
    for row in episode_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("Every episode row requires a non-empty case_id.")
        if case_id in rows_by_id:
            raise ValueError(f"duplicate longitudinal evaluation case: {case_id}")
        rows_by_id[case_id] = row
    if set(rows_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id).difference(rows_by_id))
        unexpected = sorted(set(rows_by_id).difference(expected_by_id))
        raise ValueError(f"Longitudinal evaluation grid must be complete; missing={missing}, unexpected={unexpected}.")

    cross_track_samples: list[float] = []
    height_samples: list[float] = []
    tangent_velocity_samples: list[float] = []
    survived: list[float] = []
    recovered: list[float] = []
    success_by_task: dict[str, list[float]] = {"level": [], "climb": [], "descent": []}
    finite_flags: list[bool] = []
    for case in cases:
        row = rows_by_id[case.case_id]
        if str(row.get("stage_id", "")) != case.stage_id or str(row.get("task", "")) != case.task:
            raise ValueError(f"Case metadata does not match registered grid: {case.case_id}")
        if bool(row.get("promotion_eligible")) != case.promotion_eligible:
            raise ValueError(f"Case promotion eligibility mismatch: {case.case_id}")
        cross_track_samples.extend(_finite_sequence(row, "cross_track_error_m"))
        height_samples.extend(_finite_sequence(row, "height_error_m"))
        tangent_velocity_samples.extend(_finite_sequence(row, "tangent_velocity_mps"))
        survived.append(float(not _strict_bool(row.get("terminated"), name="terminated")))
        recovered.append(float(_strict_bool(row.get("recovery_reached"), name="recovery_reached")))
        success_by_task[case.task].append(float(_strict_bool(row.get("success"), name="success")))
        finite_flags.append(_strict_bool(row.get("finite_metrics"), name="finite_metrics"))

    absolute_cross_track = [abs(value) for value in cross_track_samples]
    absolute_height = [abs(value) for value in height_samples]
    stage_ids = {case.stage_id for case in cases}
    if len(stage_ids) != 1:
        raise ValueError("expected_cases must belong to exactly one stage.")
    stage_id = next(iter(stage_ids))
    result = {
        "checkpoint": str(checkpoint),
        "ppo_iteration": int(ppo_iteration),
        "stage_id": stage_id,
        "evaluation_contract": LONGITUDINAL_EVAL_CONTRACTS[stage_id],
        "grid_complete": True,
        "case_count": len(cases),
        "level_case_count": len(success_by_task["level"]),
        "climb_case_count": len(success_by_task["climb"]),
        "descent_case_count": len(success_by_task["descent"]),
        "overall_survival_rate": _mean(survived),
        "level_success_rate": _mean(success_by_task["level"]),
        "climb_success_rate": _mean(success_by_task["climb"]),
        "descent_success_rate": _mean(success_by_task["descent"]),
        "recovery_reached_rate": _mean(recovered),
        "mean_abs_cross_track_error_m": _mean(absolute_cross_track),
        "mean_abs_height_error_m": _mean(absolute_height),
        "p95_abs_height_error_m": _quantile(absolute_height, 0.95),
        "reverse_motion_fraction": _mean([float(value < 0.0) for value in tangent_velocity_samples]),
        "finite_metrics": all(finite_flags),
    }
    result["promotion_gate_passed"] = row_meets_longitudinal_promotion_gate(result)
    return result


def row_meets_longitudinal_promotion_gate(
    row: Mapping[str, object],
    *,
    gate: PureRLLongitudinalPromotionGate = PURE_RL_LONGITUDINAL_PROMOTION_GATE,
) -> bool:
    """Return whether a complete summary satisfies every approved hard gate."""

    try:
        metrics = {
            name: float(row[name])
            for name in (
                "overall_survival_rate",
                "climb_success_rate",
                "descent_success_rate",
                "recovery_reached_rate",
                "mean_abs_cross_track_error_m",
                "mean_abs_height_error_m",
                "p95_abs_height_error_m",
                "reverse_motion_fraction",
            )
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            return False
        return bool(row.get("grid_complete")) and bool(row.get("finite_metrics")) and (
            int(row.get("climb_case_count", 0)) > 0
            and int(row.get("descent_case_count", 0)) > 0
            and metrics["overall_survival_rate"] >= gate.minimum_overall_survival_rate
            and metrics["climb_success_rate"] >= gate.minimum_climb_success_rate
            and metrics["descent_success_rate"] >= gate.minimum_descent_success_rate
            and metrics["recovery_reached_rate"] >= gate.minimum_recovery_reached_rate
            and metrics["mean_abs_cross_track_error_m"] <= gate.maximum_mean_abs_cross_track_error_m
            and metrics["mean_abs_height_error_m"] <= gate.maximum_mean_abs_height_error_m
            and metrics["p95_abs_height_error_m"] <= gate.maximum_p95_abs_height_error_m
            and metrics["reverse_motion_fraction"] <= gate.maximum_reverse_motion_fraction
        )
    except (KeyError, TypeError, ValueError):
        return False


def _build_grid(
    stage_id: str,
    slopes_deg: Sequence[float],
    *,
    promotion_eligible: bool,
) -> tuple[PureRLLongitudinalEvaluationCase, ...]:
    cases: list[PureRLLongitudinalEvaluationCase] = []
    for slope_deg in slopes_deg:
        task = "level" if slope_deg == 0.0 else ("climb" if slope_deg > 0.0 else "descent")
        task_id = 0 if task == "level" else (1 if task == "climb" else 2)
        for heading_index, heading_rad in enumerate(_CARDINAL_ANGLES_RAD):
            for phase_index, flap_phase_rad in enumerate(_CARDINAL_ANGLES_RAD):
                case_id = (
                    f"{stage_id}_slope_{slope_deg:+05.1f}_h{heading_index}_p{phase_index}_"
                    f"{'promotion' if promotion_eligible else 'diagnostic'}"
                )
                cases.append(
                    PureRLLongitudinalEvaluationCase(
                        case_id=case_id,
                        stage_id=stage_id,
                        task=task,
                        task_id=task_id,
                        signed_slope_deg=float(slope_deg),
                        heading_rad=heading_rad,
                        flap_phase_rad=flap_phase_rad,
                        promotion_eligible=promotion_eligible,
                    )
                )
    return tuple(cases)


def _validate_stage_id(stage_id: str) -> str:
    stage = str(stage_id).strip().lower()
    if stage not in _PROMOTION_ANGLES_DEG:
        raise ValueError(f"Unknown longitudinal evaluation stage: {stage_id!r}.")
    return stage


def _finite_sequence(row: Mapping[str, object], name: str) -> list[float]:
    raw = row.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) == 0:
        raise ValueError(f"{name} must be a non-empty sequence.")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} values must be finite.")
    return values


def _strict_bool(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be boolean or 0/1.")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence.")
    return float(sum(values) / len(values))


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sequence.")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


__all__ = [
    "LONGITUDINAL_EVAL_CONTRACTS",
    "PURE_RL_LONGITUDINAL_PROMOTION_GATE",
    "PureRLLongitudinalEvaluationCase",
    "PureRLLongitudinalPromotionGate",
    "build_longitudinal_diagnostic_grid",
    "build_longitudinal_evaluation_grid",
    "row_meets_longitudinal_promotion_gate",
    "summarize_longitudinal_evaluation",
]

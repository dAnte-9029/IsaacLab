"""Deterministic evaluation grids and promotion metrics for PureRL C3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


SPATIAL_EVAL_CONTRACTS: dict[str, str] = {
    "c3a": "pure_rl_spatial_c3a_v1",
    "c3b": "pure_rl_spatial_c3b_v1",
    "c3c": "pure_rl_spatial_c3c_v1",
}

_CARDINAL_HEADINGS_RAD = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
_CARDINAL_PHASES_RAD = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
_HALF_PHASES_RAD = (0.0, math.pi)
_C3A_ROLL_LEVELS_DEG = (9.0, 11.0, 13.0)
_C3B_TEMPLATE_IDS = tuple(range(3, 10))
_C3C_SEVERITIES = (("low", 8.0, 2.0), ("medium", 12.0, 3.0), ("high", 16.0, 3.0))


@dataclass(frozen=True)
class PureRLSpatialEvaluationCase:
    """One held-out spatial path and reset condition."""

    case_id: str
    stage_id: str
    template_id: int
    geometry_roll_deg: float
    slope_deg: float
    turn_sign: int
    vertical_sign: int
    heading_rad: float
    flap_phase_rad: float
    severity_id: str
    episode_duration_s: float = 20.0


@dataclass(frozen=True)
class PureRLSpatialPromotionGate:
    """Approved hard gates for one deterministic C3 grid."""

    minimum_overall_survival_rate: float = 0.95
    minimum_all_event_completion_rate: float = 0.95
    minimum_overall_success_rate: float = 0.90
    maximum_mean_abs_horizontal_error_m: float = 0.50
    maximum_mean_abs_vertical_error_m: float = 0.50
    maximum_p95_abs_horizontal_error_m: float = 1.50
    maximum_p95_abs_vertical_error_m: float = 1.50
    maximum_reverse_motion_fraction: float = 0.01
    maximum_p95_abs_roll_deg: float = 25.0
    maximum_roll_limit_terminations: int = 0
    minimum_c3a_direction_success_rate: float = 0.90
    minimum_c3b_template_success_rate: float = 0.875
    minimum_c3c_slice_success_rate: float = 0.90


PURE_RL_SPATIAL_PROMOTION_GATE = PureRLSpatialPromotionGate()


def build_spatial_evaluation_grid(stage_id: str) -> tuple[PureRLSpatialEvaluationCase, ...]:
    """Build the exact promotion grid for one C3 stage."""

    stage = _validate_stage_id(stage_id)
    cases: list[PureRLSpatialEvaluationCase] = []
    if stage == "c3a":
        for roll_deg in _C3A_ROLL_LEVELS_DEG:
            for turn_sign in (-1, 1):
                for heading_index, heading in enumerate(_CARDINAL_HEADINGS_RAD):
                    for phase_index, phase in enumerate(_CARDINAL_PHASES_RAD):
                        cases.append(
                            _case(
                                stage=stage,
                                template_id=2,
                                roll_deg=roll_deg,
                                slope_deg=0.0,
                                turn_sign=turn_sign,
                                heading=heading,
                                phase=phase,
                                suffix=f"r{roll_deg:02.0f}_t{turn_sign:+d}_h{heading_index}_p{phase_index}",
                                severity=f"roll_{roll_deg:02.0f}",
                            )
                        )
    elif stage == "c3b":
        for template_id in _C3B_TEMPLATE_IDS:
            slope_magnitude = 0.0 if template_id in (3, 4, 9) else 5.0
            vertical_sign = -1 if template_id in (6, 8) else 1
            for turn_sign in (-1, 1):
                for heading_index, heading in enumerate(_CARDINAL_HEADINGS_RAD):
                    for phase_index, phase in enumerate(_HALF_PHASES_RAD):
                        cases.append(
                            _case(
                                stage=stage,
                                template_id=template_id,
                                roll_deg=13.5,
                                slope_deg=vertical_sign * slope_magnitude,
                                turn_sign=turn_sign,
                                heading=heading,
                                phase=phase,
                                suffix=f"tpl{template_id}_t{turn_sign:+d}_h{heading_index}_p{phase_index}",
                                severity=f"template_{template_id}",
                            )
                        )
    else:
        for severity, roll_deg, slope_magnitude in _C3C_SEVERITIES:
            for turn_sign in (-1, 1):
                for vertical_sign in (-1, 1):
                    for heading_index, heading in enumerate(_CARDINAL_HEADINGS_RAD):
                        for phase_index, phase in enumerate(_HALF_PHASES_RAD):
                            cases.append(
                                _case(
                                    stage=stage,
                                    template_id=10,
                                    roll_deg=roll_deg,
                                    slope_deg=vertical_sign * slope_magnitude,
                                    turn_sign=turn_sign,
                                    heading=heading,
                                    phase=phase,
                                    suffix=(
                                        f"{severity}_t{turn_sign:+d}_v{vertical_sign:+d}_"
                                        f"h{heading_index}_p{phase_index}"
                                    ),
                                    severity=severity,
                                )
                            )
    return tuple(cases)


def summarize_spatial_evaluation(
    episode_rows: Sequence[Mapping[str, object]],
    *,
    expected_cases: Sequence[PureRLSpatialEvaluationCase],
    checkpoint: str,
    ppo_iteration: int,
) -> dict[str, object]:
    """Aggregate one complete spatial grid and its mandatory slices."""

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
            raise ValueError("Every spatial episode row requires a case_id.")
        if case_id in rows_by_id:
            raise ValueError(f"duplicate spatial evaluation case: {case_id}")
        rows_by_id[case_id] = row
    if set(rows_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id).difference(rows_by_id))
        unexpected = sorted(set(rows_by_id).difference(expected_by_id))
        raise ValueError(f"Spatial evaluation grid must be complete; missing={missing}, unexpected={unexpected}.")

    stage_ids = {case.stage_id for case in cases}
    if len(stage_ids) != 1:
        raise ValueError("expected_cases must belong to one spatial stage.")
    stage_id = next(iter(stage_ids))
    horizontal_samples: list[float] = []
    vertical_samples: list[float] = []
    tangent_velocity_samples: list[float] = []
    roll_samples_rad: list[float] = []
    survived: list[float] = []
    completed: list[float] = []
    succeeded: list[float] = []
    roll_limit_count = 0
    finite_flags: list[bool] = []
    slice_successes: dict[str, list[float]] = {}
    for case in cases:
        row = rows_by_id[case.case_id]
        for name, expected in (
            ("stage_id", case.stage_id),
            ("template_id", case.template_id),
            ("turn_sign", case.turn_sign),
            ("vertical_sign", case.vertical_sign),
            ("severity_id", case.severity_id),
        ):
            if row.get(name) != expected:
                raise ValueError(f"Case metadata does not match registered grid: {case.case_id} ({name}).")
        horizontal_samples.extend(_finite_sequence(row, "horizontal_normal_error_m"))
        vertical_samples.extend(_finite_sequence(row, "vertical_normal_error_m"))
        tangent_velocity_samples.extend(_finite_sequence(row, "tangent_velocity_mps"))
        roll_samples_rad.extend(_finite_sequence(row, "roll_rad"))
        survived.append(float(not _strict_bool(row.get("terminated"), name="terminated")))
        completed.append(float(_strict_bool(row.get("events_reached"), name="events_reached")))
        success = float(_strict_bool(row.get("success"), name="success"))
        succeeded.append(success)
        roll_limit_count += int(_strict_bool(row.get("roll_limit_termination"), name="roll_limit_termination"))
        finite_flags.append(_strict_bool(row.get("finite_metrics"), name="finite_metrics"))
        for slice_name in _slice_names(case):
            slice_successes.setdefault(slice_name, []).append(success)

    absolute_horizontal = [abs(value) for value in horizontal_samples]
    absolute_vertical = [abs(value) for value in vertical_samples]
    absolute_roll_deg = [abs(math.degrees(value)) for value in roll_samples_rad]
    result: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "ppo_iteration": int(ppo_iteration),
        "stage_id": stage_id,
        "evaluation_contract": SPATIAL_EVAL_CONTRACTS[stage_id],
        "grid_complete": True,
        "case_count": len(cases),
        "overall_survival_rate": _mean(survived),
        "all_event_completion_rate": _mean(completed),
        "overall_success_rate": _mean(succeeded),
        "mean_abs_horizontal_error_m": _mean(absolute_horizontal),
        "mean_abs_vertical_error_m": _mean(absolute_vertical),
        "p95_abs_horizontal_error_m": _quantile(absolute_horizontal, 0.95),
        "p95_abs_vertical_error_m": _quantile(absolute_vertical, 0.95),
        "reverse_motion_fraction": _mean([float(value < 0.0) for value in tangent_velocity_samples]),
        "p95_abs_roll_deg": _quantile(absolute_roll_deg, 0.95),
        "roll_limit_termination_count": roll_limit_count,
        "finite_metrics": all(finite_flags),
        "slice_success_rates": {name: _mean(values) for name, values in sorted(slice_successes.items())},
    }
    result["promotion_gate_passed"] = row_meets_spatial_promotion_gate(result)
    return result


def row_meets_spatial_promotion_gate(
    row: Mapping[str, object],
    *,
    gate: PureRLSpatialPromotionGate = PURE_RL_SPATIAL_PROMOTION_GATE,
) -> bool:
    """Return whether a complete C3 summary satisfies every hard gate."""

    try:
        stage_id = _validate_stage_id(str(row["stage_id"]))
        metrics = {
            name: float(row[name])
            for name in (
                "overall_survival_rate",
                "all_event_completion_rate",
                "overall_success_rate",
                "mean_abs_horizontal_error_m",
                "mean_abs_vertical_error_m",
                "p95_abs_horizontal_error_m",
                "p95_abs_vertical_error_m",
                "reverse_motion_fraction",
                "p95_abs_roll_deg",
            )
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            return False
        slice_rates = row["slice_success_rates"]
        if not isinstance(slice_rates, Mapping) or not slice_rates:
            return False
        slice_minimum = {
            "c3a": gate.minimum_c3a_direction_success_rate,
            "c3b": gate.minimum_c3b_template_success_rate,
            "c3c": gate.minimum_c3c_slice_success_rate,
        }[stage_id]
        if not all(math.isfinite(float(value)) and float(value) >= slice_minimum for value in slice_rates.values()):
            return False
        return bool(row.get("grid_complete")) and bool(row.get("finite_metrics")) and (
            metrics["overall_survival_rate"] >= gate.minimum_overall_survival_rate
            and metrics["all_event_completion_rate"] >= gate.minimum_all_event_completion_rate
            and metrics["overall_success_rate"] >= gate.minimum_overall_success_rate
            and metrics["mean_abs_horizontal_error_m"] <= gate.maximum_mean_abs_horizontal_error_m
            and metrics["mean_abs_vertical_error_m"] <= gate.maximum_mean_abs_vertical_error_m
            and metrics["p95_abs_horizontal_error_m"] <= gate.maximum_p95_abs_horizontal_error_m
            and metrics["p95_abs_vertical_error_m"] <= gate.maximum_p95_abs_vertical_error_m
            and metrics["reverse_motion_fraction"] <= gate.maximum_reverse_motion_fraction
            and metrics["p95_abs_roll_deg"] <= gate.maximum_p95_abs_roll_deg
            and int(row["roll_limit_termination_count"]) <= gate.maximum_roll_limit_terminations
        )
    except (KeyError, TypeError, ValueError):
        return False


def _case(
    *,
    stage: str,
    template_id: int,
    roll_deg: float,
    slope_deg: float,
    turn_sign: int,
    heading: float,
    phase: float,
    suffix: str,
    severity: str,
) -> PureRLSpatialEvaluationCase:
    return PureRLSpatialEvaluationCase(
        case_id=f"{stage}_{suffix}",
        stage_id=stage,
        template_id=template_id,
        geometry_roll_deg=float(roll_deg),
        slope_deg=float(slope_deg),
        turn_sign=int(turn_sign),
        vertical_sign=int(math.copysign(1, slope_deg)) if slope_deg != 0.0 else 0,
        heading_rad=float(heading),
        flap_phase_rad=float(phase),
        severity_id=severity,
    )


def _slice_names(case: PureRLSpatialEvaluationCase) -> tuple[str, ...]:
    turn = "left" if case.turn_sign < 0 else "right"
    if case.stage_id == "c3a":
        return (turn,)
    if case.stage_id == "c3b":
        return (f"template_{case.template_id}",)
    vertical = "descent" if case.vertical_sign < 0 else "climb"
    return (turn, vertical, f"sign_{case.turn_sign:+d}_{case.vertical_sign:+d}")


def _validate_stage_id(stage_id: str) -> str:
    stage = str(stage_id).strip().lower()
    if stage not in SPATIAL_EVAL_CONTRACTS:
        raise ValueError(f"Unknown spatial evaluation stage: {stage_id!r}.")
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
    "PURE_RL_SPATIAL_PROMOTION_GATE",
    "SPATIAL_EVAL_CONTRACTS",
    "PureRLSpatialEvaluationCase",
    "PureRLSpatialPromotionGate",
    "build_spatial_evaluation_grid",
    "row_meets_spatial_promotion_gate",
    "summarize_spatial_evaluation",
]

"""Lightweight runtime contract helpers for truth/estimated state sources."""

from __future__ import annotations

from dataclasses import dataclass

VALID_CONTROLLER_STATE_SOURCES = ("truth", "estimated", "compare")
VALID_RUNTIME_STATE_SOURCES = ("truth", "estimated")
VALID_IMU_SOURCES = ("synthetic", "isaacsim")


def _normalize_token(value: str | None, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must not be None")
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def resolve_controller_state_source(value: str) -> str:
    """Resolve standalone controller state-source selection."""
    normalized = _normalize_token(value, field_name="state_source")
    if normalized not in VALID_CONTROLLER_STATE_SOURCES:
        raise ValueError(
            f"state_source must be one of {VALID_CONTROLLER_STATE_SOURCES}, got {value!r}"
        )
    return normalized


def resolve_teacher_state_source(value: str) -> str:
    """Resolve teacher runtime state-source selection."""
    normalized = _normalize_token(value, field_name="teacher_state_source")
    if normalized not in VALID_RUNTIME_STATE_SOURCES:
        raise ValueError(
            f"teacher_state_source must be one of {VALID_RUNTIME_STATE_SOURCES}, got {value!r}"
        )
    return normalized


def resolve_policy_state_source(value: str) -> str:
    """Resolve policy runtime state-source selection."""
    normalized = _normalize_token(value, field_name="policy_state_source")
    if normalized not in VALID_RUNTIME_STATE_SOURCES:
        raise ValueError(
            f"policy_state_source must be one of {VALID_RUNTIME_STATE_SOURCES}, got {value!r}"
        )
    return normalized


def resolve_imu_source(value: str) -> str:
    """Resolve IMU provider selection."""
    normalized = _normalize_token(value, field_name="imu_source")
    if normalized not in VALID_IMU_SOURCES:
        raise ValueError(f"imu_source must be one of {VALID_IMU_SOURCES}, got {value!r}")
    return normalized


def teacher_should_use_truth_wind(teacher_state_source: str, teacher_guidance_use_wind_truth: bool) -> bool:
    """Truth wind is only valid when the teacher itself runs on truth state."""
    return bool(teacher_guidance_use_wind_truth) and resolve_teacher_state_source(teacher_state_source) == "truth"


@dataclass(frozen=True)
class TeacherStateInputs:
    """Resolved runtime selection for teacher and policy states."""

    teacher_state_source: str
    policy_state_source: str
    teacher_uses_truth_wind: bool


def resolve_teacher_state_inputs(
    teacher_state_source: str,
    policy_state_source: str,
    teacher_guidance_use_wind_truth: bool,
) -> TeacherStateInputs:
    """Normalize teacher/policy sources and report whether truth-wind is still valid."""
    teacher_source = resolve_teacher_state_source(teacher_state_source)
    policy_source = resolve_policy_state_source(policy_state_source)
    return TeacherStateInputs(
        teacher_state_source=teacher_source,
        policy_state_source=policy_source,
        teacher_uses_truth_wind=teacher_should_use_truth_wind(teacher_source, teacher_guidance_use_wind_truth),
    )


@dataclass(frozen=True)
class RuntimeStateSourceSelection:
    """Resolved state-source contract for one runtime."""

    controller_state_source: str
    teacher_state_source: str
    policy_state_source: str
    imu_source: str


def resolve_runtime_state_source_selection(
    *,
    state_source: str,
    teacher_state_source: str | None = None,
    policy_state_source: str | None = None,
    imu_source: str = "synthetic",
) -> RuntimeStateSourceSelection:
    """Resolve CLI/env state-source settings into one explicit contract."""
    controller_source = resolve_controller_state_source(state_source)
    default_runtime_source = "truth" if controller_source == "truth" else "estimated"
    teacher_source = resolve_teacher_state_source(teacher_state_source or default_runtime_source)
    policy_source = resolve_policy_state_source(policy_state_source or default_runtime_source)
    return RuntimeStateSourceSelection(
        controller_state_source=controller_source,
        teacher_state_source=teacher_source,
        policy_state_source=policy_source,
        imu_source=resolve_imu_source(imu_source),
    )

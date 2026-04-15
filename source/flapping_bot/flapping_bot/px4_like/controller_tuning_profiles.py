"""State-source-specific controller tuning profiles for controller-only PX4-like scripts."""

from __future__ import annotations

from typing import Any


_ESTIMATED_TEACHER_LATERAL_OVERRIDES: dict[str, float] = {
    "max_roll_deg": 35.0,
    "roll_kd": 0.55,
    "tecs_load_factor_use_roll_sp": False,
}

_ESTIMATED_TEACHER_CONTROLLER_KIND_OVERRIDES: dict[str, dict[str, float | bool]] = {
    "loiter": {
        "max_roll_deg": 40.0,
        "roll_kd": 0.45,
        "inner_elevon_roll_rate_limit_per_s": 16.0,
        "guidance_min_ground_speed_mps": 2.0,
        "lateral_guidance_uncertainty_start_deg": 8.0,
        "lateral_guidance_uncertainty_full_deg": 25.0,
        "lateral_guidance_uncertainty_min_scale": 0.6,
    },
    "path_tracking": {
        "heading_p_gain": 1.4,
        "lateral_heading_yaw_blend": 0.75,
        "lateral_heading_yaw_correction_limit_deg": 20.0,
        "inner_elevon_roll_rate_limit_per_s": 16.0,
        "guidance_min_ground_speed_mps": 2.0,
        "lateral_guidance_uncertainty_start_deg": 8.0,
        "lateral_guidance_uncertainty_full_deg": 25.0,
        "lateral_guidance_uncertainty_min_scale": 0.6,
    },
}


def resolve_controller_tuning_profile(*, controller_state_source: str) -> str:
    """Return the tuning profile name for the active controller state source."""
    normalized = str(controller_state_source).strip().lower()
    if normalized == "truth":
        return "truth_baseline"
    if normalized in {"estimated", "compare"}:
        return "estimated_teacher"
    raise ValueError(f"Unsupported controller_state_source: {controller_state_source}")


def apply_controller_tuning_profile(
    base_kwargs: dict[str, Any],
    *,
    controller_state_source: str,
    controller_kind: str | None = None,
    explicit_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Overlay the controller-only tuning profile onto config kwargs."""
    tuned_kwargs = dict(base_kwargs)
    if resolve_controller_tuning_profile(controller_state_source=controller_state_source) == "estimated_teacher":
        tuned_kwargs.update(_ESTIMATED_TEACHER_LATERAL_OVERRIDES)
        if controller_kind is not None:
            normalized_kind = str(controller_kind).strip().lower()
            tuned_kwargs.update(_ESTIMATED_TEACHER_CONTROLLER_KIND_OVERRIDES.get(normalized_kind, {}))
    if explicit_overrides:
        tuned_kwargs.update(explicit_overrides)
    return tuned_kwargs

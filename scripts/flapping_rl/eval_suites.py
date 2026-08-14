"""Shared evaluation-suite definitions for flapping RL checkpoints."""

from __future__ import annotations

import math

try:
    from .pure_rl_longitudinal_eval import (
        LONGITUDINAL_EVAL_CONTRACTS,
        build_longitudinal_diagnostic_grid,
        build_longitudinal_evaluation_grid,
    )
except ImportError:  # pragma: no cover - direct script import path
    from pure_rl_longitudinal_eval import (
        LONGITUDINAL_EVAL_CONTRACTS,
        build_longitudinal_diagnostic_grid,
        build_longitudinal_evaluation_grid,
    )

try:
    from .pure_rl_spatial_eval import SPATIAL_EVAL_CONTRACTS, build_spatial_evaluation_grid
except ImportError:  # pragma: no cover - direct script import path
    from pure_rl_spatial_eval import SPATIAL_EVAL_CONTRACTS, build_spatial_evaluation_grid


_PURE_RL_HEADINGS_RAD = (0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi)
_PURE_RL_PHASES_RAD = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
_PURE_RL_HEADING_PHASE_PAIRS = tuple(
    (heading, phase)
    for heading in _PURE_RL_HEADINGS_RAD
    for phase in _PURE_RL_PHASES_RAD
)

EVAL_SUITE_CHOICES = (
    "straight_standard",
    "single",
    "pure_rl_curriculum1_nowind_v1",
    "pure_rl_curriculum1_nowind_v2",
    "pure_rl_longitudinal_c2a_v1",
    "pure_rl_longitudinal_c2b_v1",
    LONGITUDINAL_EVAL_CONTRACTS["c2c"],
    SPATIAL_EVAL_CONTRACTS["c3a"],
    SPATIAL_EVAL_CONTRACTS["c3b"],
    SPATIAL_EVAL_CONTRACTS["c3c"],
    "path_tracking_standard",
    "path_tracking_truth_nowind_v1",
    "path_tracking_estimated_nowind_v1",
    "path_tracking_truth_primitives_nowind_v1",
    "path_tracking_estimated_primitives_nowind_v1",
)


def get_eval_suite_choices() -> tuple[str, ...]:
    """Return the supported evaluation-suite names."""
    return EVAL_SUITE_CHOICES


def build_eval_cases(eval_suite: str) -> list[dict]:
    """Build the evaluation cases for the requested suite."""
    def _case(
        name: str,
        *,
        wind_enabled: bool,
        wind_xy_mps: tuple[float, float],
        wind_ou_enabled: bool,
        teacher_state_source: str | None = None,
        policy_state_source: str | None = None,
        imu_source: str | None = None,
        wind_ou_tau_s: float = 2.0,
        wind_ou_sigma_xy_mps: tuple[float, float] = (0.0, 0.0),
        mission_seed: int | None = None,
        mission_increment_seed_per_reset: bool | None = None,
        mission_num_segments_min: int | None = None,
        mission_num_segments_max: int | None = None,
        mission_allow_straight: bool | None = None,
        mission_allow_turn: bool | None = None,
        mission_allow_loiter: bool | None = None,
        mission_allow_climb_on_straight: bool | None = None,
        straight_line_heading_schedule_rad: tuple[float, ...] | None = None,
        flap_phase_schedule_rad: tuple[float, ...] | None = None,
        longitudinal_cases=None,
        spatial_cases=None,
    ) -> dict:
        case = {
            "name": name,
            "wind_enabled": wind_enabled,
            "wind_xy_mps": wind_xy_mps,
            "wind_ou_enabled": wind_ou_enabled,
            "wind_ou_tau_s": wind_ou_tau_s,
            "wind_ou_sigma_xy_mps": wind_ou_sigma_xy_mps,
        }
        if teacher_state_source is not None:
            case["teacher_state_source"] = str(teacher_state_source)
        if policy_state_source is not None:
            case["policy_state_source"] = str(policy_state_source)
        if imu_source is not None:
            case["imu_source"] = str(imu_source)
        if mission_seed is not None:
            case["mission_seed"] = mission_seed
        if mission_increment_seed_per_reset is not None:
            case["mission_increment_seed_per_reset"] = mission_increment_seed_per_reset
        if mission_num_segments_min is not None:
            case["mission_num_segments_min"] = mission_num_segments_min
        if mission_num_segments_max is not None:
            case["mission_num_segments_max"] = mission_num_segments_max
        if mission_allow_straight is not None:
            case["mission_allow_straight"] = mission_allow_straight
        if mission_allow_turn is not None:
            case["mission_allow_turn"] = mission_allow_turn
        if mission_allow_loiter is not None:
            case["mission_allow_loiter"] = mission_allow_loiter
        if mission_allow_climb_on_straight is not None:
            case["mission_allow_climb_on_straight"] = mission_allow_climb_on_straight
        if straight_line_heading_schedule_rad is not None:
            case["straight_line_heading_schedule_rad"] = tuple(straight_line_heading_schedule_rad)
        if flap_phase_schedule_rad is not None:
            case["flap_phase_schedule_rad"] = tuple(flap_phase_schedule_rad)
        if longitudinal_cases is not None:
            registered_cases = tuple(longitudinal_cases)
            case["longitudinal_stage_id"] = registered_cases[0].stage_id
            case["longitudinal_case_ids"] = tuple(item.case_id for item in registered_cases)
            case["longitudinal_task_names"] = tuple(item.task for item in registered_cases)
            case["longitudinal_task_schedule"] = tuple(item.task_id for item in registered_cases)
            case["longitudinal_slope_deg_schedule"] = tuple(
                item.signed_slope_deg for item in registered_cases
            )
            case["longitudinal_entry_length_m_schedule"] = tuple(
                item.entry_length_m for item in registered_cases
            )
            case["longitudinal_slope_length_m_schedule"] = tuple(
                item.slope_length_m for item in registered_cases
            )
            case["promotion_eligible"] = all(item.promotion_eligible for item in registered_cases)
        if spatial_cases is not None:
            registered_cases = tuple(spatial_cases)
            case["spatial_stage_id"] = registered_cases[0].stage_id
            case["spatial_case_ids"] = tuple(item.case_id for item in registered_cases)
            case["spatial_template_schedule"] = tuple(item.template_id for item in registered_cases)
            case["spatial_geometry_roll_deg_schedule"] = tuple(
                item.geometry_roll_deg for item in registered_cases
            )
            case["spatial_slope_deg_schedule"] = tuple(item.slope_deg for item in registered_cases)
            case["spatial_turn_sign_schedule"] = tuple(item.turn_sign for item in registered_cases)
            case["promotion_eligible"] = True
        return case

    if eval_suite == "single":
        return [
            _case("single", wind_enabled=False, wind_xy_mps=(0.0, 0.0), wind_ou_enabled=False)
        ]

    if eval_suite in {"pure_rl_curriculum1_nowind_v1", "pure_rl_curriculum1_nowind_v2"}:
        return [
            _case(
                "curriculum1_nowind_fixed_heading_phase",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="estimated",
                policy_state_source="estimated",
                imu_source="synthetic",
                straight_line_heading_schedule_rad=tuple(pair[0] for pair in _PURE_RL_HEADING_PHASE_PAIRS),
                flap_phase_schedule_rad=tuple(pair[1] for pair in _PURE_RL_HEADING_PHASE_PAIRS),
            )
        ]

    if eval_suite in LONGITUDINAL_EVAL_CONTRACTS.values():
        stage_id = next(
            stage for stage, contract in LONGITUDINAL_EVAL_CONTRACTS.items() if contract == eval_suite
        )
        promotion_cases = build_longitudinal_evaluation_grid(stage_id)
        diagnostic_cases = build_longitudinal_diagnostic_grid(stage_id)
        diagnostic_angle = int(max(abs(item.signed_slope_deg) for item in diagnostic_cases))

        def _longitudinal_case(name: str, registered_cases) -> dict:
            return _case(
                name,
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="estimated",
                policy_state_source="estimated",
                imu_source="synthetic",
                straight_line_heading_schedule_rad=tuple(item.heading_rad for item in registered_cases),
                flap_phase_schedule_rad=tuple(item.flap_phase_rad for item in registered_cases),
                longitudinal_cases=registered_cases,
            )

        return [
            _longitudinal_case(f"{stage_id}_promotion_grid", promotion_cases),
            _longitudinal_case(
                f"{stage_id}_signed_{diagnostic_angle}deg_diagnostic",
                diagnostic_cases,
            ),
        ]

    if eval_suite in SPATIAL_EVAL_CONTRACTS.values():
        stage_id = next(
            stage for stage, contract in SPATIAL_EVAL_CONTRACTS.items() if contract == eval_suite
        )
        registered_cases = build_spatial_evaluation_grid(stage_id)
        return [
            _case(
                f"{stage_id}_promotion_grid",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="estimated",
                policy_state_source="estimated",
                imu_source="synthetic",
                straight_line_heading_schedule_rad=tuple(
                    item.heading_rad for item in registered_cases
                ),
                flap_phase_schedule_rad=tuple(item.flap_phase_rad for item in registered_cases),
                spatial_cases=registered_cases,
            )
        ]

    if eval_suite == "path_tracking_standard":
        return [
            _case("path_calm", wind_enabled=False, wind_xy_mps=(0.0, 0.0), wind_ou_enabled=False),
            _case("path_crosswind_steady", wind_enabled=True, wind_xy_mps=(0.0, 2.0), wind_ou_enabled=False),
            _case(
                "path_crosswind_ou",
                wind_enabled=True,
                wind_xy_mps=(0.0, 1.5),
                wind_ou_enabled=True,
                wind_ou_sigma_xy_mps=(0.0, 0.8),
            ),
        ]

    if eval_suite == "path_tracking_truth_nowind_v1":
        return [
            _case(
                "straight_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="truth",
                policy_state_source="truth",
                imu_source="synthetic",
                mission_seed=101,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=1,
                mission_num_segments_max=1,
                mission_allow_straight=True,
                mission_allow_turn=False,
                mission_allow_loiter=False,
                mission_allow_climb_on_straight=False,
            ),
            _case(
                "loiter_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="truth",
                policy_state_source="truth",
                imu_source="synthetic",
                mission_seed=202,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=1,
                mission_num_segments_max=1,
                mission_allow_straight=False,
                mission_allow_turn=False,
                mission_allow_loiter=True,
                mission_allow_climb_on_straight=False,
            ),
            _case(
                "random_mission_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="truth",
                policy_state_source="truth",
                imu_source="synthetic",
                mission_seed=303,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=3,
                mission_num_segments_max=3,
                mission_allow_straight=True,
                mission_allow_turn=True,
                mission_allow_loiter=True,
                mission_allow_climb_on_straight=True,
            ),
        ]

    if eval_suite == "path_tracking_estimated_nowind_v1":
        return [
            _case(
                "straight_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="estimated",
                policy_state_source="estimated",
                imu_source="synthetic",
                mission_seed=101,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=1,
                mission_num_segments_max=1,
                mission_allow_straight=True,
                mission_allow_turn=False,
                mission_allow_loiter=False,
                mission_allow_climb_on_straight=False,
            ),
            _case(
                "loiter_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="estimated",
                policy_state_source="estimated",
                imu_source="synthetic",
                mission_seed=202,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=1,
                mission_num_segments_max=1,
                mission_allow_straight=False,
                mission_allow_turn=False,
                mission_allow_loiter=True,
                mission_allow_climb_on_straight=False,
            ),
            _case(
                "random_mission_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="estimated",
                policy_state_source="estimated",
                imu_source="synthetic",
                mission_seed=303,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=3,
                mission_num_segments_max=3,
                mission_allow_straight=True,
                mission_allow_turn=True,
                mission_allow_loiter=True,
                mission_allow_climb_on_straight=True,
            ),
        ]

    if eval_suite == "path_tracking_truth_primitives_nowind_v1":
        return [
            _case(
                "straight_primitive_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="truth",
                policy_state_source="truth",
                imu_source="synthetic",
                mission_seed=111,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=1,
                mission_num_segments_max=1,
                mission_allow_straight=True,
                mission_allow_turn=False,
                mission_allow_loiter=False,
                mission_allow_climb_on_straight=False,
            ),
            _case(
                "turn_primitive_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="truth",
                policy_state_source="truth",
                imu_source="synthetic",
                mission_seed=222,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=1,
                mission_num_segments_max=1,
                mission_allow_straight=False,
                mission_allow_turn=True,
                mission_allow_loiter=False,
                mission_allow_climb_on_straight=False,
            ),
            _case(
                "loiter_primitive_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="truth",
                policy_state_source="truth",
                imu_source="synthetic",
                mission_seed=333,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=1,
                mission_num_segments_max=1,
                mission_allow_straight=False,
                mission_allow_turn=False,
                mission_allow_loiter=True,
                mission_allow_climb_on_straight=False,
            ),
        ]

    if eval_suite == "path_tracking_estimated_primitives_nowind_v1":
        return [
            _case(
                "straight_primitive_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="estimated",
                policy_state_source="estimated",
                imu_source="synthetic",
                mission_seed=111,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=1,
                mission_num_segments_max=1,
                mission_allow_straight=True,
                mission_allow_turn=False,
                mission_allow_loiter=False,
                mission_allow_climb_on_straight=False,
            ),
            _case(
                "turn_primitive_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="estimated",
                policy_state_source="estimated",
                imu_source="synthetic",
                mission_seed=222,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=1,
                mission_num_segments_max=1,
                mission_allow_straight=False,
                mission_allow_turn=True,
                mission_allow_loiter=False,
                mission_allow_climb_on_straight=False,
            ),
            _case(
                "loiter_primitive_nowind",
                wind_enabled=False,
                wind_xy_mps=(0.0, 0.0),
                wind_ou_enabled=False,
                teacher_state_source="estimated",
                policy_state_source="estimated",
                imu_source="synthetic",
                mission_seed=333,
                mission_increment_seed_per_reset=False,
                mission_num_segments_min=1,
                mission_num_segments_max=1,
                mission_allow_straight=False,
                mission_allow_turn=False,
                mission_allow_loiter=True,
                mission_allow_climb_on_straight=False,
            ),
        ]

    return [
        _case("calm", wind_enabled=False, wind_xy_mps=(0.0, 0.0), wind_ou_enabled=False),
        _case("crosswind_steady", wind_enabled=True, wind_xy_mps=(0.0, 2.0), wind_ou_enabled=False),
        _case(
            "crosswind_ou",
            wind_enabled=True,
            wind_xy_mps=(0.0, 1.5),
            wind_ou_enabled=True,
            wind_ou_sigma_xy_mps=(0.0, 0.8),
        ),
    ]

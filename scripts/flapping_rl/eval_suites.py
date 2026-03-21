"""Shared evaluation-suite definitions for flapping RL checkpoints."""

from __future__ import annotations


EVAL_SUITE_CHOICES = (
    "straight_standard",
    "single",
    "path_tracking_standard",
    "path_tracking_truth_nowind_v1",
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
    ) -> dict:
        case = {
            "name": name,
            "wind_enabled": wind_enabled,
            "wind_xy_mps": wind_xy_mps,
            "wind_ou_enabled": wind_ou_enabled,
            "wind_ou_tau_s": wind_ou_tau_s,
            "wind_ou_sigma_xy_mps": wind_ou_sigma_xy_mps,
        }
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
        return case

    if eval_suite == "single":
        return [
            _case("single", wind_enabled=False, wind_xy_mps=(0.0, 0.0), wind_ou_enabled=False)
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

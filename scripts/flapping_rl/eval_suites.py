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
    def _case(name: str, *, wind_enabled: bool, wind_xy_mps: tuple[float, float], wind_ou_enabled: bool,
              wind_ou_tau_s: float = 2.0, wind_ou_sigma_xy_mps: tuple[float, float] = (0.0, 0.0)) -> dict:
        return {
            "name": name,
            "wind_enabled": wind_enabled,
            "wind_xy_mps": wind_xy_mps,
            "wind_ou_enabled": wind_ou_enabled,
            "wind_ou_tau_s": wind_ou_tau_s,
            "wind_ou_sigma_xy_mps": wind_ou_sigma_xy_mps,
        }

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
            _case("straight_nowind", wind_enabled=False, wind_xy_mps=(0.0, 0.0), wind_ou_enabled=False),
            _case("loiter_nowind", wind_enabled=False, wind_xy_mps=(0.0, 0.0), wind_ou_enabled=False),
            _case("random_mission_nowind", wind_enabled=False, wind_xy_mps=(0.0, 0.0), wind_ou_enabled=False),
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

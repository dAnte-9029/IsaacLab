"""Shared evaluation-suite definitions for flapping RL checkpoints."""

from __future__ import annotations


EVAL_SUITE_CHOICES = ("straight_standard", "single", "path_tracking_standard")


def get_eval_suite_choices() -> tuple[str, ...]:
    """Return the supported evaluation-suite names."""
    return EVAL_SUITE_CHOICES


def build_eval_cases(eval_suite: str) -> list[dict]:
    """Build the evaluation cases for the requested suite."""
    if eval_suite == "single":
        return [
            {
                "name": "single",
                "wind_enabled": False,
                "wind_xy_mps": (0.0, 0.0),
                "wind_ou_enabled": False,
                "wind_ou_tau_s": 2.0,
                "wind_ou_sigma_xy_mps": (0.0, 0.0),
            }
        ]

    if eval_suite == "path_tracking_standard":
        return [
            {
                "name": "path_calm",
                "wind_enabled": False,
                "wind_xy_mps": (0.0, 0.0),
                "wind_ou_enabled": False,
                "wind_ou_tau_s": 2.0,
                "wind_ou_sigma_xy_mps": (0.0, 0.0),
            },
            {
                "name": "path_crosswind_steady",
                "wind_enabled": True,
                "wind_xy_mps": (0.0, 2.0),
                "wind_ou_enabled": False,
                "wind_ou_tau_s": 2.0,
                "wind_ou_sigma_xy_mps": (0.0, 0.0),
            },
            {
                "name": "path_crosswind_ou",
                "wind_enabled": True,
                "wind_xy_mps": (0.0, 1.5),
                "wind_ou_enabled": True,
                "wind_ou_tau_s": 2.0,
                "wind_ou_sigma_xy_mps": (0.0, 0.8),
            },
        ]

    return [
        {
            "name": "calm",
            "wind_enabled": False,
            "wind_xy_mps": (0.0, 0.0),
            "wind_ou_enabled": False,
            "wind_ou_tau_s": 2.0,
            "wind_ou_sigma_xy_mps": (0.0, 0.0),
        },
        {
            "name": "crosswind_steady",
            "wind_enabled": True,
            "wind_xy_mps": (0.0, 2.0),
            "wind_ou_enabled": False,
            "wind_ou_tau_s": 2.0,
            "wind_ou_sigma_xy_mps": (0.0, 0.0),
        },
        {
            "name": "crosswind_ou",
            "wind_enabled": True,
            "wind_xy_mps": (0.0, 1.5),
            "wind_ou_enabled": True,
            "wind_ou_tau_s": 2.0,
            "wind_ou_sigma_xy_mps": (0.0, 0.8),
        },
    ]

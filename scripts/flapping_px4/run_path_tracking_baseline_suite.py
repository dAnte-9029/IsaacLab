"""Run the unified path-tracking baseline suite."""

from __future__ import annotations

import argparse


def build_phase_names() -> list[str]:
    """Return the primitive baseline phases for generic path tracking."""
    return [
        "level_straight",
        "climb_straight",
        "descent_straight",
        "level_turn",
        "level_loiter",
    ]


def build_suite_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the path-tracking baseline suite."""
    parser = argparse.ArgumentParser(description="Run the PX4-like baseline suite for path-tracking primitives.")
    parser.add_argument("--task", type=str, default="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0")
    parser.add_argument("--headless", action="store_true")
    return parser


def main() -> None:
    """Parse CLI arguments for the path-tracking baseline suite."""
    build_suite_parser().parse_args()


if __name__ == "__main__":
    main()

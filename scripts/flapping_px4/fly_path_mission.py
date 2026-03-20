"""Entry point for unified PX4-like path-mission rollouts."""

from __future__ import annotations

import argparse


def build_path_mission_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for path-mission baseline rollouts."""
    parser = argparse.ArgumentParser(description="Run a PX4-like baseline on a generic path mission.")
    parser.add_argument("--task", type=str, default="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0")
    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--headless", action="store_true")
    return parser


def main() -> None:
    """Parse CLI arguments for the path-mission baseline entrypoint."""
    build_path_mission_parser().parse_args()


if __name__ == "__main__":
    main()

"""Evaluate a path-tracking checkpoint."""

from __future__ import annotations

import argparse


def build_eval_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for path-tracking checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a generic path-tracking checkpoint.")
    parser.add_argument("--task", type=str, default="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    return parser


def main() -> None:
    """Parse CLI arguments for the path-tracking checkpoint evaluator."""
    build_eval_parser().parse_args()


if __name__ == "__main__":
    main()

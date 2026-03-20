"""Behavior-cloning warm start entrypoint for generic path tracking."""

from __future__ import annotations

import argparse


def build_bc_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for path-tracking BC pretraining."""
    parser = argparse.ArgumentParser(description="Pretrain a path-tracking policy from teacher demonstrations.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    return parser


def main() -> None:
    """Parse arguments for the BC warm-start entrypoint."""
    build_bc_parser().parse_args()


if __name__ == "__main__":
    main()

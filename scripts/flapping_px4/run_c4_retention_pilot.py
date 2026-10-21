"""Execute the authorized seed-0 A--D retention comparison, without holdout access."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/flapping_bot'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from flapping_bot.analysis.c4_retention_pilot import run_pilot
    run_pilot(ROOT, args.output.resolve(), args.weights.resolve(), args.memory.resolve(), args.development.resolve())


if __name__ == '__main__':
    main()

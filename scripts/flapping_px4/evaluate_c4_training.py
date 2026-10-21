"""Freeze independent C4 sets, or evaluate one checkpoint with five legacy suites."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/flapping_bot'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--output', type=Path, required=True)
    freeze.add_argument('--count', type=int, default=96)
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--checkpoint', type=Path, required=True)
    evaluate.add_argument('--curves', type=Path, required=True, help='Explicit development or final holdout replay file')
    evaluate.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from flapping_bot.analysis.c4_training_evaluation import freeze_c4_sets, evaluate_with_retention
    if args.command == 'freeze':
        freeze_c4_sets(args.output.resolve(), args.count)
    else:
        result = evaluate_with_retention(ROOT, args.checkpoint.resolve(), args.curves.resolve(), args.output.resolve())
        if not result['all_suites_passed']:
            raise SystemExit(2)
    print(args.output.resolve())


if __name__ == '__main__':
    main()

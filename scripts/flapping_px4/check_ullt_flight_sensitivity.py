"""Run bounded time/space/initial-wake sensitivity for an existing replay."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay-dir", "wing-csv", "wing-urdf", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/flapping_bot"))
    import torch
    from flapping_bot.analysis.ullt_flight_sensitivity import run_sensitivity

    torch.set_num_threads(1)
    run_sensitivity(args.replay_dir, args.wing_csv, args.wing_urdf, args.output_dir)


if __name__ == "__main__":
    main()

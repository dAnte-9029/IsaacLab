"""Bounded spatial-resolution check of section incidence/load attribution."""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-screen", type=Path, required=True)
    parser.add_argument("--wing-csv", type=Path, required=True)
    parser.add_argument("--wing-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/flapping_bot"))
    import torch
    from flapping_bot.analysis.ullt_section_sensitivity import check_section_resolution

    torch.set_num_threads(1)
    result = check_section_resolution(args.source_screen, args.wing_csv, args.wing_urdf, args.output_dir)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate one checkpoint on a replayable C4 diagnostic grid (no training)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/flapping_bot"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tail-checkpoint", type=Path, help="Use this actor for tail channels on the same current observation")
    parser.add_argument("--tail-channels", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3],
                        help="Replacement channels: 1 rudder, 2 left elevon, 3 right elevon")
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Only first six paths; incomplete diagnostic")
    parser.add_argument("--preview-time-scale", type=float, default=1.0,
                        help="Diagnostic only: 1 gives 0.12--0.60 s; 2 gives 0.24--1.20 s")
    args = parser.parse_args()
    if args.smoke_steps < 0:
        parser.error("--smoke-steps must be nonnegative")
    if not math.isfinite(args.preview_time_scale) or args.preview_time_scale <= 0:
        parser.error("--preview-time-scale must be finite and positive")
    checkpoint, curves, output = args.checkpoint.resolve(), args.curves.resolve(), args.output.resolve()
    if not checkpoint.is_file() or not curves.is_file():
        parser.error("Checkpoint and curve input must exist")
    tail_checkpoint = args.tail_checkpoint.resolve() if args.tail_checkpoint else None
    if tail_checkpoint is not None and not tail_checkpoint.is_file():
        parser.error("Tail checkpoint must exist")
    if len(set(args.tail_channels)) != len(args.tail_channels):
        parser.error("Tail channels must be unique")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "experiment": "c4_zero_shot", "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "command": sys.argv, "seed": 0, "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "replacement_tail_channels": args.tail_channels if tail_checkpoint else [],
        "tail_checkpoint": str(tail_checkpoint) if tail_checkpoint else None,
        "tail_checkpoint_sha256": hashlib.sha256(tail_checkpoint.read_bytes()).hexdigest() if tail_checkpoint else None,
        "curves": str(curves),
        "preview_time_scale": args.preview_time_scale,
    }
    (output / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    app = None
    try:
        (output / "status.json").write_text('{"status": "running"}\n')
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device="cpu", kit_args=(
            f"--ext-folder {ROOT / 'source/flapping_bot/native_extensions'} "
            "--enable omni.flapping_bot.holonomic_constraint --/apps/extensions/fsWatcherEnabled=false"
        )).app
        from flapping_bot.analysis.continuous_curve_evaluation import evaluate_checkpoint
        result = evaluate_checkpoint(checkpoint, curves, output, ROOT, args.smoke_steps, args.preview_time_scale, tail_checkpoint, tuple(args.tail_channels))
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (output / "status.json").write_text('{"status": "completed"}\n')
    except BaseException:
        (output / "status.json").write_text(json.dumps({"status": "failed", "error": traceback.format_exc()}) + "\n")
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()

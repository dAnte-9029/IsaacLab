"""Explicit six-task C4 launcher; no evaluation or promotion is implied by training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/flapping_bot'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--weights', type=Path, help='Original full-joint weights; fresh optimizer/iteration')
    source.add_argument('--resume', type=Path, help='C4 checkpoint; restore optimizer and adapter progress')
    parser.add_argument('--output', type=Path, required=True, help='New directory, never overwritten')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=201, help='Additional rollout iterations')
    parser.add_argument('--steps', type=int, default=48)
    parser.add_argument('--minibatches', type=int, default=16)
    parser.add_argument('--save-interval', type=int, default=25)
    parser.add_argument('--retention-variant', choices=('A', 'B', 'C', 'D'),
                        help='Opt-in fixed/adaptive sampling crossed with absent/present behavior memory')
    parser.add_argument('--memory', type=Path, help='Training-only successful behavior bank, required for C/D')
    parser.add_argument('--memory-coefficient', type=float, default=1.0)
    parser.add_argument('--collect-memory-steps', type=int, default=0,
                        help='Frozen deterministic teacher collection; no PPO updates')
    parser.add_argument('--smoke', action='store_true', help='96 envs, 16 steps, 4 minibatches; 5 iterations (resume: 1)')
    args = parser.parse_args()
    if args.smoke:
        args.num_envs, args.steps, args.minibatches = 96, 16, 4
        args.iterations, args.save_interval = (1 if args.resume else 5), 1
    if min(args.num_envs,args.iterations,args.steps,args.minibatches,args.save_interval) < 1:
        parser.error('Training sizes must be positive')
    checkpoint = (args.weights or args.resume).resolve()
    if not checkpoint.is_file():
        parser.error('Checkpoint does not exist')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'launch.json').write_text(json.dumps(dict(
        command=sys.argv, timestamp=datetime.now(timezone.utc).isoformat(), seed=args.seed,
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        smoke=args.smoke),indent=2))
    app = None
    try:
        (output/'status.json').write_text('{"status":"running"}\n')
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True,device='cpu',kit_args=(
            f'--ext-folder {ROOT / "source/flapping_bot/native_extensions"} '
            '--enable omni.flapping_bot.holonomic_constraint --/apps/extensions/fsWatcherEnabled=false')).app
        from flapping_bot.analysis.c4_training import train_c4
        result = train_c4(checkpoint=checkpoint,output=output,root=ROOT,resume=bool(args.resume),seed=args.seed,
                          num_envs=args.num_envs,iterations=args.iterations,steps=args.steps,
                          minibatches=args.minibatches,save_interval=args.save_interval,
                          retention_variant=args.retention_variant, memory_path=args.memory.resolve() if args.memory else None,
                          memory_coefficient=args.memory_coefficient, collect_memory_steps=args.collect_memory_steps)
        (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        (output/'status.json').write_text('{"status":"completed"}\n')
    except BaseException:
        (output/'status.json').write_text(json.dumps(dict(status='failed',error=traceback.format_exc())))
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()

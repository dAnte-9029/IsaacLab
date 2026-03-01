"""Launch training and a parallel checkpoint evaluator (watcher).

Motivation: for long unattended runs, it's useful to continuously evaluate new
`model_*.pt` checkpoints and append metrics to `eval/summary.csv` while training
keeps running.

Typical usage:
  # Train on GPU0, evaluate on GPU1
  ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
    --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 \
    --run-name sf_long \
    --train-device cuda:0 \
    --eval-device cuda:1 \
    --num-envs 512 \
    --max-iterations 2000 \
    --save-interval 100 \
    --episodes 5 --poll-s 120 \
    --headless
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train + watch/eval new checkpoints.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-device", type=str, default="cuda:0")
    parser.add_argument("--eval-device", type=str, default="cuda:1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--poll-s", type=float, default=120.0)
    # pass-through to AppLauncher (headless / livestream / etc.)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def _safe_terminate(p: subprocess.Popen, timeout_s: float = 10.0):
    if p.poll() is not None:
        return
    try:
        p.send_signal(signal.SIGINT)
        p.wait(timeout=timeout_s)
    except Exception:
        try:
            p.terminate()
            p.wait(timeout=timeout_s)
        except Exception:
            p.kill()


def main():
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    # Training command (single process on train-device).
    train_cmd = [
        "./isaaclab.sh",
        "-p",
        "scripts/reinforcement_learning/rsl_rl/train.py",
        "--task",
        args.task,
        "--device",
        args.train_device,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.max_iterations),
        "--seed",
        str(args.seed),
        f"agent.run_name={args.run_name}",
        f"agent.save_interval={args.save_interval}",
    ]
    if args.headless:
        train_cmd.append("--headless")

    # Start training and parse the run directory from stdout.
    print("[INFO] Launching training:")
    print(" ", " ".join(train_cmd), flush=True)

    train = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_root: Path | None = None
    timestamp: str | None = None
    try:
        assert train.stdout is not None
        for line in train.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

            if log_root is None:
                m = re.search(r"Logging experiment in directory: (.+)$", line.strip())
                if m:
                    log_root = Path(m.group(1)).expanduser().resolve()
            if timestamp is None:
                m = re.search("Exact experiment name requested from command line: (\\S+)$", line.strip())
                if m:
                    timestamp = m.group(1)
            if log_root is not None and timestamp is not None:
                break

        if log_root is None or timestamp is None:
            raise RuntimeError("Failed to parse log directory from training output.")

        run_dir = (log_root / f"{timestamp}_{args.run_name}").resolve()
        # Wait for the directory to appear on disk.
        t0 = time.time()
        while not run_dir.is_dir():
            if train.poll() is not None:
                raise RuntimeError("Training exited before creating a run directory.")
            if time.time() - t0 > 60.0:
                raise TimeoutError(f"Timed out waiting for run dir: {run_dir}")
            time.sleep(0.5)

        # Start watcher on eval-device.
        watch_cmd = [
            "./isaaclab.sh",
            "-p",
            "scripts/flapping_rl/watch_and_eval.py",
            "--task",
            args.task,
            "--log_dir",
            str(run_dir),
            "--device",
            args.eval_device,
            "--episodes",
            str(args.episodes),
            "--num_envs",
            "1",
            "--poll_s",
            str(args.poll_s),
        ]
        if args.headless:
            watch_cmd.append("--headless")

        print("[INFO] Launching watcher:")
        print(" ", " ".join(watch_cmd), flush=True)
        watcher = subprocess.Popen(watch_cmd)

        # Stream remaining training output.
        assert train.stdout is not None
        for line in train.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        rc = train.wait()
        print(f"[INFO] Training finished with return code: {rc}", flush=True)
    except KeyboardInterrupt:
        print("[WARN] KeyboardInterrupt: stopping processes...", flush=True)
        rc = 130
    finally:
        try:
            if "watcher" in locals():
                _safe_terminate(watcher)
        except Exception:
            pass
        _safe_terminate(train)

    raise SystemExit(rc)


if __name__ == "__main__":
    main()

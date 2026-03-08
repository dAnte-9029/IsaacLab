"""Launch training and a parallel checkpoint evaluator (watcher).

Motivation: for long unattended runs, it's useful to continuously evaluate new
`model_*.pt` checkpoints and append metrics to `eval/summary.csv` while training
keeps running.

Typical usage:
  # Train on GPU0, evaluate on GPU1
  ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-TeacherRL-Direct-v0 \
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
import csv
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from checkpoint_selection import refresh_best_checkpoint_artifacts, select_best_checkpoint_row


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
    parser.add_argument("--run-dir-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--eval-suite",
        type=str,
        default="straight_standard",
        choices=("straight_standard", "single"),
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from a previous run/checkpoint.")
    parser.add_argument("--load_run", type=str, default=None, help="Existing run directory name used for resume.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename or regex used for resume.")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def _extract_ckpt_index(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _load_completed_checkpoints(summary_csv: Path) -> set[str]:
    completed: set[str] = set()
    if not summary_csv.is_file():
        return completed

    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            checkpoint = row.get("checkpoint")
            case_name = row.get("case")
            if checkpoint and case_name == "suite":
                completed.add(str(Path(checkpoint).expanduser().resolve()))
    return completed


def _latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=_extract_ckpt_index)
    if not ckpts:
        return None
    return ckpts[-1].resolve()


def _wait_for_run_dir(run_dir: Path, train: subprocess.Popen, *, timeout_s: float, poll_s: float = 0.5) -> None:
    t0 = time.time()
    while not run_dir.is_dir():
        if train.poll() is not None:
            raise RuntimeError("Training exited before creating a run directory.")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for run dir: {run_dir}")
        time.sleep(poll_s)


def _needs_final_eval(run_dir: Path) -> bool:
    latest_ckpt = _latest_checkpoint(run_dir)
    if latest_ckpt is None:
        return False
    completed = _load_completed_checkpoints(run_dir / "eval" / "summary.csv")
    return str(latest_ckpt) not in completed


def _safe_terminate(p: subprocess.Popen, timeout_s: float = 10.0):
    if p.poll() is not None:
        return

    try:
        if os.name == "posix":
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
        else:
            p.send_signal(signal.SIGINT)
        p.wait(timeout=timeout_s)
        return
    except Exception:
        pass

    try:
        if os.name == "posix":
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        else:
            p.terminate()
        p.wait(timeout=timeout_s)
        return
    except Exception:
        pass

    try:
        if os.name == "posix":
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        else:
            p.kill()
        p.wait(timeout=timeout_s)
    except Exception:
        p.kill()


def _run_final_eval_once(watch_cmd: list[str]) -> int:
    final_cmd = list(watch_cmd)
    if "--once" not in final_cmd:
        final_cmd.append("--once")
    print("[INFO] Running final one-shot evaluation:", flush=True)
    print(" ", " ".join(final_cmd), flush=True)
    return subprocess.call(final_cmd)


def _build_train_cmd(args: argparse.Namespace) -> list[str]:
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
    if bool(args.resume):
        train_cmd.append("--resume")
    if args.load_run is not None:
        train_cmd.extend(["--load_run", str(args.load_run)])
    if args.checkpoint is not None:
        train_cmd.extend(["--checkpoint", str(args.checkpoint)])
    if args.headless:
        train_cmd.append("--headless")
    return train_cmd


def main():
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    train_cmd = _build_train_cmd(args)

    print("[INFO] Launching training:")
    print(" ", " ".join(train_cmd), flush=True)

    train = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    log_root: Path | None = None
    timestamp: str | None = None
    run_dir: Path | None = None
    watch_cmd: list[str] | None = None
    final_eval_needed = False
    rc = 1
    try:
        assert train.stdout is not None
        for line in train.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

            if log_root is None:
                match = re.search(r"Logging experiment in directory: (.+)$", line.strip())
                if match:
                    log_root = Path(match.group(1)).expanduser().resolve()
            if timestamp is None:
                match = re.search(r"Exact experiment name requested from command line: (\S+)$", line.strip())
                if match:
                    timestamp = match.group(1)
            if log_root is not None and timestamp is not None:
                break

        if log_root is None or timestamp is None:
            raise RuntimeError("Failed to parse log directory from training output.")

        run_dir = (log_root / f"{timestamp}_{args.run_name}").resolve()
        _wait_for_run_dir(run_dir, train, timeout_s=float(args.run_dir_timeout_s))

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
            "--eval_suite",
            str(args.eval_suite),
        ]
        if args.headless:
            watch_cmd.append("--headless")

        print("[INFO] Launching watcher:")
        print(" ", " ".join(watch_cmd), flush=True)
        watcher = subprocess.Popen(watch_cmd, start_new_session=True)

        assert train.stdout is not None
        for line in train.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        rc = train.wait()
        print(f"[INFO] Training finished with return code: {rc}", flush=True)
        final_eval_needed = rc == 0 and run_dir is not None and watch_cmd is not None
    except KeyboardInterrupt:
        print("[WARN] KeyboardInterrupt: stopping processes...", flush=True)
        rc = 130
    finally:
        try:
            if "watcher" in locals():
                _safe_terminate(watcher)
        except Exception:
            pass

        if final_eval_needed and watch_cmd is not None and run_dir is not None:
            if _needs_final_eval(run_dir):
                final_eval_rc = _run_final_eval_once(watch_cmd)
                if final_eval_rc != 0:
                    print(f"[WARN] Final one-shot evaluation exited with code: {final_eval_rc}", flush=True)
                    rc = final_eval_rc if rc == 0 else rc
            else:
                print("[INFO] Latest checkpoint already has a suite row; skipping final one-shot evaluation.", flush=True)

            best_row = refresh_best_checkpoint_artifacts(run_dir)
            if best_row is None:
                best_row = select_best_checkpoint_row(run_dir / "eval" / "summary.csv")
            if best_row is not None:
                print(
                    "[INFO] Current best checkpoint:",
                    {
                        "checkpoint": best_row["checkpoint"],
                        "score": best_row["score"],
                        "ckpt_index": best_row.get("ckpt_index"),
                    },
                    flush=True,
                )

        _safe_terminate(train)

    raise SystemExit(rc)


if __name__ == "__main__":
    main()

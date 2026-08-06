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

  # Canonical CPU-native measured-wing plant
  ./isaaclab.sh -p scripts/flapping_rl/train_and_watch.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0 \
    --run-name native_cpu_pure_rl \
    --native-cpu \
    --freeze-steps-after-reset 0 \
    --num-envs 64 \
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
from eval_suites import get_eval_suite_choices


_NATIVE_HOLONOMIC_EXTENSION_ID = "omni.flapping_bot.holonomic_constraint"


def _resolve_eval_suite(task: str, eval_suite: str) -> str:
    if eval_suite == "straight_standard" and "PathTracking" in str(task):
        if "Primitive" in str(task):
            return "path_tracking_estimated_primitives_nowind_v1"
        return "path_tracking_estimated_nowind_v1"
    return str(eval_suite)


def _should_apply_estimated_teacher_defaults(task: str) -> bool:
    task_name = str(task)
    return "FlappingBot" in task_name and "RL" in task_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train + watch/eval new checkpoints.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mass-kg-override",
        type=float,
        default=None,
        help="Optional total vehicle mass override passed as total_mass_kg_override=<value>.",
    )
    parser.add_argument("--train-device", type=str, default="cuda:0")
    parser.add_argument("--eval-device", type=str, default="cuda:1")
    parser.add_argument(
        "--native-cpu",
        action="store_true",
        help=(
            "Run the simulator and policy on CPU and load the canonical native holonomic constraint extension. "
            "The extension is incompatible with direct-GPU PhysX."
        ),
    )
    parser.add_argument(
        "--native-extension-parent",
        type=Path,
        default=None,
        help="Parent directory containing the omni.flapping_bot.holonomic_constraint extension.",
    )
    parser.add_argument(
        "--agent-device",
        type=str,
        default=None,
        help="Optional RSL-RL policy/optimizer device override; --native-cpu defaults this to cpu.",
    )
    parser.add_argument(
        "--freeze-steps-after-reset",
        type=int,
        default=None,
        help="Optional environment reset-freeze override; use 0 for the native CPU PureRL P0 gate.",
    )
    parser.add_argument("--eval-num-envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--poll-s", type=float, default=120.0)
    parser.add_argument("--run-dir-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--eval-suite",
        type=str,
        default="straight_standard",
        choices=get_eval_suite_choices(),
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from a previous run/checkpoint.")
    parser.add_argument(
        "--load_weights_only",
        action="store_true",
        help="Warm-start policy weights from a checkpoint without restoring optimizer or iteration state.",
    )
    parser.add_argument("--load_run", type=str, default=None, help="Existing run directory name used for resume.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename or regex used for resume.")
    parser.add_argument(
        "--portable-root-base",
        type=Path,
        default=None,
        help="Base portable root used to isolate IsaacSim cache/state for the train and watcher child processes.",
    )
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


def _run_final_eval_once(watch_cmd: list[str], *, child_env: dict[str, str] | None = None) -> int:
    final_cmd = list(watch_cmd)
    if "--once" not in final_cmd:
        final_cmd.append("--once")
    print("[INFO] Running final one-shot evaluation:", flush=True)
    print(" ", " ".join(final_cmd), flush=True)
    return subprocess.call(final_cmd, env=child_env)


def _resolve_portable_root_base(args: argparse.Namespace) -> Path:
    configured = getattr(args, "portable_root_base", None)
    if configured is not None:
        return Path(configured)
    return Path("logs/portable/train_and_watch") / f"{args.run_name}_seed{args.seed}_pid{os.getpid()}"


def _portable_root_for_role(args: argparse.Namespace, role: str) -> Path:
    return _resolve_portable_root_base(args) / role


def _portable_kit_args(args: argparse.Namespace, role: str) -> str:
    return f"--portable-root {_portable_root_for_role(args, role)}"


def _native_cpu_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "native_cpu", False))


def _native_extension_parent(args: argparse.Namespace) -> Path:
    configured = getattr(args, "native_extension_parent", None)
    if configured is not None:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "source/flapping_bot/native_extensions").resolve()


def _validate_native_extension(args: argparse.Namespace) -> None:
    if not _native_cpu_enabled(args):
        return
    extension_parent = _native_extension_parent(args)
    extension_binary = (
        extension_parent
        / _NATIVE_HOLONOMIC_EXTENSION_ID
        / "omni/flapping_bot/holonomic_constraint/_native.so"
    )
    if not extension_binary.is_file():
        raise FileNotFoundError(
            f"Native extension binary does not exist: {extension_binary}. "
            "Build it with scripts/flapping_px4/build_holonomic_constraint_extension.sh."
        )


def _child_entrypoint(args: argparse.Namespace, script: str) -> list[str]:
    if _native_cpu_enabled(args):
        return [sys.executable, script]
    return ["./isaaclab.sh", "-p", script]


def _native_asset_source() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
    ).resolve()


def _native_asset_cache(args: argparse.Namespace) -> Path:
    return (_portable_root_for_role(args, "train") / "generated_assets/flap_robot_552").resolve()


def _child_process_env(args: argparse.Namespace) -> dict[str, str]:
    child_env = os.environ.copy()
    if not _native_cpu_enabled(args):
        return child_env
    repo_root = Path(__file__).resolve().parents[2]
    local_sources = [
        str((repo_root / "source/flapping_bot").resolve()),
        str((repo_root / "source/isaaclab_assets").resolve()),
    ]
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    if existing_pythonpath:
        local_sources.append(existing_pythonpath)
    child_env["PYTHONPATH"] = os.pathsep.join(local_sources)
    return child_env


def _kit_args(args: argparse.Namespace, role: str) -> str:
    kit_args = _portable_kit_args(args, role)
    if _native_cpu_enabled(args):
        kit_args += (
            f" --ext-folder {_native_extension_parent(args)}"
            f" --enable {_NATIVE_HOLONOMIC_EXTENSION_ID}"
        )
    return kit_args


def _sim_device(args: argparse.Namespace, role: str) -> str:
    if _native_cpu_enabled(args):
        return "cpu"
    return str(getattr(args, f"{role}_device"))


def _build_train_cmd(args: argparse.Namespace) -> list[str]:
    if bool(args.resume) and bool(getattr(args, "load_weights_only", False)):
        raise ValueError("--resume and --load_weights_only cannot both be enabled.")

    train_cmd = [
        *_child_entrypoint(args, "scripts/reinforcement_learning/rsl_rl/train.py"),
        "--task",
        args.task,
        "--device",
        _sim_device(args, "train"),
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
    if bool(getattr(args, "load_weights_only", False)):
        train_cmd.append("--load_weights_only")
    if args.load_run is not None:
        train_cmd.extend(["--load_run", str(args.load_run)])
    if args.checkpoint is not None:
        train_cmd.extend(["--checkpoint", str(args.checkpoint)])
    train_cmd.extend(["--kit_args", _kit_args(args, "train")])
    agent_device = getattr(args, "agent_device", None)
    if _native_cpu_enabled(args):
        agent_device = agent_device or "cpu"
    if agent_device is not None:
        train_cmd.append(f"agent.device={agent_device}")
    freeze_steps_after_reset = getattr(args, "freeze_steps_after_reset", None)
    if freeze_steps_after_reset is not None:
        if int(freeze_steps_after_reset) < 0:
            raise ValueError("--freeze-steps-after-reset must be non-negative.")
        train_cmd.append(f"env.freeze_steps_after_reset={int(freeze_steps_after_reset)}")
    if _native_cpu_enabled(args):
        train_cmd.extend(
            [
                f"env.robot.spawn.asset_path={_native_asset_source()}",
                f"env.robot.spawn.usd_dir={_native_asset_cache(args)}",
            ]
        )
    if _should_apply_estimated_teacher_defaults(args.task):
        train_cmd.extend(
            [
                "env.teacher_state_source=estimated",
                "env.policy_state_source=estimated",
                "env.imu_source=synthetic",
            ]
        )
    mass_kg_override = getattr(args, "mass_kg_override", None)
    if mass_kg_override is not None:
        train_cmd.append(f"env.total_mass_kg_override={float(mass_kg_override)}")
    if args.headless:
        train_cmd.append("--headless")
    return train_cmd


def _build_watch_cmd(args: argparse.Namespace, run_dir: Path) -> list[str]:
    watch_cmd = [
        *_child_entrypoint(args, "scripts/flapping_rl/watch_and_eval.py"),
        "--task",
        args.task,
        "--log_dir",
        str(run_dir),
        "--device",
        _sim_device(args, "eval"),
        "--episodes",
        str(args.episodes),
        "--num_envs",
        str(args.eval_num_envs),
        "--poll_s",
        str(args.poll_s),
        "--eval_suite",
        _resolve_eval_suite(args.task, str(args.eval_suite)),
    ]
    watch_cmd.extend(["--kit_args", _kit_args(args, "watch")])
    if args.headless:
        watch_cmd.append("--headless")
    return watch_cmd


def main():
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    args.portable_root_base = _resolve_portable_root_base(args)
    _validate_native_extension(args)
    _portable_root_for_role(args, "train").mkdir(parents=True, exist_ok=True)
    _portable_root_for_role(args, "watch").mkdir(parents=True, exist_ok=True)
    if _native_cpu_enabled(args):
        if not _native_asset_source().is_file():
            raise FileNotFoundError(f"Native PureRL URDF does not exist: {_native_asset_source()}")
        _native_asset_cache(args).mkdir(parents=True, exist_ok=True)

    train_cmd = _build_train_cmd(args)
    child_env = _child_process_env(args)

    print("[INFO] Launching training:")
    print(" ", " ".join(train_cmd), flush=True)

    train = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=child_env,
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

        watch_cmd = _build_watch_cmd(args, run_dir)

        print("[INFO] Launching watcher:")
        print(" ", " ".join(watch_cmd), flush=True)
        watcher = subprocess.Popen(watch_cmd, start_new_session=True, env=child_env)

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
                final_eval_rc = _run_final_eval_once(watch_cmd, child_env=child_env)
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

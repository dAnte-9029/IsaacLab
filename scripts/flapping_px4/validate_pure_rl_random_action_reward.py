"""Run the PureRL bounded random-action reward gate in a fresh process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback

from isaaclab.app import AppLauncher


_EXTENSION_ID = "omni.flapping_bot.holonomic_constraint"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded free-root CPU-native random-action rollout and validate PureRL reward, "
            "termination, observation and repeated-reset wiring."
        )
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--duration-s", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--asset-path", type=Path)
    parser.add_argument("--asset-cache", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), check=True, text=True, capture_output=True).stdout.strip()


def _require_local_package(repo_root: Path) -> str:
    expected_root = (repo_root / "source/flapping_bot").resolve()
    spec = importlib.util.find_spec("flapping_bot")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot resolve flapping_bot.")
    module_path = Path(spec.origin).resolve()
    if not module_path.is_relative_to(expected_root):
        raise RuntimeError(
            f"flapping_bot resolves outside this checkout: {module_path}; "
            f"prepend {expected_root} to PYTHONPATH."
        )
    return str(module_path)


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    extension_parent = repo_root / "source/flapping_bot/native_extensions"
    required_kit_args = f"--ext-folder {extension_parent} --enable {_EXTENSION_ID}"
    existing_kit_args = str(getattr(args, "kit_args", "") or "").strip()
    args.kit_args = f"{existing_kit_args} {required_kit_args}".strip()
    args.device = "cpu"
    print("[pure-rl-random-reward] launching Isaac Sim on CPU", flush=True)
    simulation_app = AppLauncher(args).app
    try:
        package_path = _require_local_package(repo_root)
        from flapping_bot.analysis.pure_rl_random_action_reward_validation import (
            run_random_action_reward_job,
        )

        for output in (args.summary, args.traces):
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}.")
        asset_path = args.asset_path or (
            repo_root
            / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
        )
        asset_cache = args.asset_cache or (args.summary.parent / "generated_assets/flap_robot_552")
        if not asset_path.is_file():
            raise FileNotFoundError(f"URDF does not exist: {asset_path}")
        asset_cache.mkdir(parents=True, exist_ok=True)
        result = run_random_action_reward_job(
            num_envs=int(args.num_envs),
            duration_s=float(args.duration_s),
            seed=int(args.seed),
            asset_path=asset_path,
            usd_dir=asset_cache,
        )
        result.summary["provenance"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "flapping_bot_module_path": package_path,
            "asset_path": str(asset_path.resolve()),
            "asset_cache": str(asset_cache.resolve()),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.traces.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        import numpy as np

        np.savez_compressed(args.traces, **result.traces)
        accepted = bool(result.summary["all_cases_accepted"])
        print(
            f"[pure-rl-random-reward] wrote {args.summary} and {args.traces}; accepted={accepted}",
            flush=True,
        )
        return 0 if accepted else 2
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())

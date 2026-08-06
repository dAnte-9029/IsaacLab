"""Run one PureRL actor-observation calibration gate in a fresh process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback


_EXTENSION_ID = "omni.flapping_bot.holonomic_constraint"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect physical-unit PureRL samples and evaluate the frozen normalization before clipping. "
            "Runtime modes use the CPU-native measured-wing plant."
        )
    )
    parser.add_argument("--mode", choices=("synthetic", "reset", "scripted", "boundary"), required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8192)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--reset-batches", type=int, default=32)
    parser.add_argument("--duration-s", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--asset-path", type=Path)
    parser.add_argument("--asset-cache", type=Path)
    parser.add_argument("--headless", action="store_true", default=True)
    args, remaining = parser.parse_known_args()
    args.remaining = remaining
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


def _write_result(args: argparse.Namespace, result: object, *, package_path: str) -> int:
    for output in (args.summary, args.samples):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}.")
    result.summary["provenance"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "flapping_bot_module_path": package_path,
        "calibration_partition_seed": int(args.seed),
        "verification_partition_is_disjoint": True,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    import numpy as np

    np.savez_compressed(args.samples, **result.samples)
    accepted = bool(result.summary["all_cases_accepted"])
    print(
        f"[pure-rl-observation] wrote {args.summary} and {args.samples}; accepted={accepted}",
        flush=True,
    )
    return 0 if accepted else 2


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        package_path = _require_local_package(repo_root)
        if args.mode == "synthetic":
            from flapping_bot.analysis.pure_rl_observation_calibration import run_synthetic_envelope_gate

            result = run_synthetic_envelope_gate(sample_count=args.sample_count, seed=args.seed)
            return _write_result(args, result, package_path=package_path)

        from isaaclab.app import AppLauncher

        extension_parent = repo_root / "source/flapping_bot/native_extensions"
        required_kit_args = f"--ext-folder {extension_parent} --enable {_EXTENSION_ID}"
        launcher_parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(launcher_parser)
        launcher_args, _ = launcher_parser.parse_known_args(args.remaining)
        launcher_args.device = "cpu"
        launcher_args.headless = bool(args.headless)
        existing_kit_args = str(getattr(launcher_args, "kit_args", "") or "").strip()
        launcher_args.kit_args = f"{existing_kit_args} {required_kit_args}".strip()
        print(f"[pure-rl-observation] launching Isaac Sim mode={args.mode}", flush=True)
        simulation_app = AppLauncher(launcher_args).app
        try:
            from flapping_bot.analysis.pure_rl_observation_calibration import run_runtime_gate

            asset_path = args.asset_path or (
                repo_root
                / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
            )
            asset_cache = args.asset_cache or (args.summary.parent / "generated_assets/flap_robot_552")
            if not asset_path.is_file():
                raise FileNotFoundError(f"URDF does not exist: {asset_path}")
            asset_cache.mkdir(parents=True, exist_ok=True)
            result = run_runtime_gate(
                mode=args.mode,
                num_envs=args.num_envs,
                seed=args.seed,
                reset_batches=args.reset_batches,
                duration_s=args.duration_s,
                asset_path=asset_path,
                usd_dir=asset_cache,
            )
            result.summary["runtime"] = {
                "device": "cpu",
                "physics_hz": 480.0,
                "policy_hz": 60.0,
                "num_envs": int(args.num_envs),
                "asset_path": str(asset_path.resolve()),
                "asset_cache": str(asset_cache.resolve()),
            }
            return _write_result(args, result, package_path=package_path)
        finally:
            simulation_app.close()
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

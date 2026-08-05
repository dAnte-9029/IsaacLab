"""Run one native-holonomic continuous-frequency validation worker."""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate continuous-frequency native holonomic multibody diagnostics."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--dt-denominator", type=int, choices=(240, 480, 1000), required=True)
    parser.add_argument("--free-root", action="store_true")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--airspeed-mps", type=float, default=8.0)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def _require_local_package() -> str:
    repo_root = Path(__file__).resolve().parents[2]
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


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), check=True, text=True, capture_output=True).stdout.strip()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    extension_parent = repo_root / "source/flapping_bot/native_extensions"
    required_kit_args = (
        f"--ext-folder {extension_parent} "
        "--enable omni.flapping_bot.holonomic_constraint"
    )
    existing_kit_args = str(getattr(args, "kit_args", "") or "").strip()
    args.kit_args = f"{existing_kit_args} {required_kit_args}".strip()
    print("[native-transient] launching Isaac Sim", flush=True)
    simulation_app = AppLauncher(args).app
    try:
        package_path = _require_local_package()
        from flapping_bot.analysis.native_holonomic_transient_validation import (
            run_native_holonomic_transient_job,
        )

        for output in (args.summary, args.traces):
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}.")
        result = run_native_holonomic_transient_job(
            dt_s=1.0 / float(args.dt_denominator),
            fixed_root=not bool(args.free_root),
            duration_s=args.duration_s,
            airspeed_mps=float(args.airspeed_mps),
            device=str(args.device),
            progress_callback=lambda message: print(f"[native-transient] {message}", flush=True),
        )
        summary = result.summary
        summary["provenance"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "flapping_bot_module_path": package_path,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.traces.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        import numpy as np

        np.savez_compressed(args.traces, **result.traces)
        print(f"[native-transient] wrote {args.summary} and {args.traces}", flush=True)
        return 0 if bool(summary["all_cases_accepted"]) else 2
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())

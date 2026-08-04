"""Run one native-holonomic frequency/time-step validation worker."""

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
        description="Validate the native PhysX holonomic wing mechanism at one physics step."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dt-denominator",
        type=int,
        choices=(240, 480, 1000, 2000, 4000),
        default=480,
    )
    parser.add_argument(
        "--frequencies-hz",
        type=float,
        nargs="+",
        default=(2.0, 3.0, 4.0, 5.0),
    )
    parser.add_argument("--aerodynamics", action="store_true")
    parser.add_argument("--airspeed-mps", type=float, default=8.0)
    parser.add_argument("--total-cycles", type=int, default=4)
    parser.add_argument("--measurement-cycles", type=int, default=2)
    parser.add_argument("--free-root", action="store_true")
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument(
        "--wing-link-load-mode",
        choices=("full_wing_link_wrench", "wing_link_force_only", "wing_link_moment_only"),
        default="full_wing_link_wrench",
    )
    parser.add_argument("--solver-position-iterations", type=int)
    parser.add_argument("--solver-velocity-iterations", type=int)
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
    return subprocess.run(
        ("git", *args),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


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
    print("[native-holonomic] launching Isaac Sim", flush=True)
    simulation_app = AppLauncher(args).app
    try:
        package_path = _require_local_package()
        from flapping_bot.analysis.native_holonomic_validation import (
            run_native_holonomic_free_aero_job,
            run_native_holonomic_frequency_job,
        )

        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}.")
        progress = lambda message: print(
            f"[native-holonomic] {message}",
            flush=True,
        )
        if args.free_root:
            if not args.aerodynamics:
                raise ValueError("--free-root currently requires --aerodynamics.")
            if args.solver_position_iterations is not None or args.solver_velocity_iterations is not None:
                raise ValueError("Free-root jobs use the native asset solver settings.")
            result = run_native_holonomic_free_aero_job(
                frequencies_hz=tuple(args.frequencies_hz),
                dt_s=1.0 / float(args.dt_denominator),
                device=str(args.device),
                airspeed_mps=float(args.airspeed_mps),
                duration_s=float(args.duration_s),
                wing_link_load_mode=str(args.wing_link_load_mode),
                progress_callback=progress,
            )
        else:
            result = run_native_holonomic_frequency_job(
                frequencies_hz=tuple(args.frequencies_hz),
                dt_s=1.0 / float(args.dt_denominator),
                device=str(args.device),
                enable_aerodynamics=bool(args.aerodynamics),
                airspeed_mps=float(args.airspeed_mps),
                total_cycles=int(args.total_cycles),
                measurement_cycles=int(args.measurement_cycles),
                solver_position_iterations=args.solver_position_iterations,
                solver_velocity_iterations=args.solver_velocity_iterations,
                progress_callback=progress,
            )
        result["provenance"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "flapping_bot_module_path": package_path,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[native-holonomic] wrote {args.output}", flush=True)
        if not bool(result["all_cases_accepted"]):
            print("[native-holonomic] one or more cases failed acceptance gates", flush=True)
            return 2
        return 0
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())

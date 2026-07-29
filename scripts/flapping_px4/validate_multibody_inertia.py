"""Run the no-external-force PhysX/SFWM inertial validation matrix."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from isaaclab.app import AppLauncher


def _local_flapping_bot_source() -> Path:
    """Require this checkout's project package instead of another editable install."""

    repository_root = Path(__file__).resolve().parents[2]
    expected_package_root = (repository_root / "source" / "flapping_bot").resolve()
    spec = importlib.util.find_spec("flapping_bot")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot resolve the flapping_bot package.")
    resolved_module = Path(spec.origin).resolve()
    if not resolved_module.is_relative_to(expected_package_root):
        raise RuntimeError(
            "flapping_bot resolves outside this checkout: "
            f"{resolved_module}. Prepend {expected_package_root} to PYTHONPATH."
        )
    return resolved_module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate measured-wing PhysX inertia against Amini SFWM without gravity or aerodynamics."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New output directory. Default: logs/flapping_px4/multibody_inertia/<UTC timestamp>.",
    )
    parser.add_argument("--total-cycles", type=int, default=8)
    parser.add_argument("--measurement-cycles", type=int, default=4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only 2 Hz at 1/240 s with four total cycles.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _write_case_csv(path: Path, cases: list[dict[str, object]]) -> None:
    fieldnames = [
        "frequency_hz",
        "dt_s",
        "max_sync_error_deg",
        "max_tracking_error_deg",
        "max_velocity_tracking_error_rad_s",
        "max_acceleration_tracking_error_rad_s2",
        "max_driver_torque_nm",
        "relative_linear_momentum_residual",
        "relative_angular_momentum_residual",
        "physx_to_amini_vertical_amplitude_ratio",
        "physx_minus_amini_vertical_phase_deg",
        "physx_to_amini_pitch_amplitude_ratio",
        "physx_minus_amini_pitch_phase_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow({name: case[name] for name in fieldnames})


def main() -> int:
    args = _parse_args()
    flapping_bot_module_path = _local_flapping_bot_source()
    print("[inertial-validation] launching Isaac Sim", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        print("[inertial-validation] importing benchmark", flush=True)
        from flapping_bot.analysis.physx_multibody_inertial_benchmark import (
            run_physx_multibody_inertial_benchmark,
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = args.output_dir or Path("logs/flapping_px4/multibody_inertia") / timestamp
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
        output_dir.mkdir(parents=True)
        print(f"[inertial-validation] output: {output_dir}", flush=True)
        frequencies = (2.0,) if args.smoke else (2.0, 3.0, 4.0, 5.0)
        time_steps = (1.0 / 240.0,) if args.smoke else (1.0 / 240.0, 1.0 / 480.0, 1.0e-3)
        total_cycles = 4 if args.smoke else int(args.total_cycles)
        measurement_cycles = 2 if args.smoke else int(args.measurement_cycles)
        result = run_physx_multibody_inertial_benchmark(
            frequencies_hz=frequencies,
            time_steps_s=time_steps,
            total_cycles=total_cycles,
            measurement_cycles=measurement_cycles,
        )
        print("[inertial-validation] simulation matrix complete", flush=True)
        result["provenance"] = {
            "timestamp_utc": timestamp,
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "git_dirty": bool(_git_output("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "flapping_bot_module_path": str(flapping_bot_module_path),
        }
        json_path = output_dir / "summary.json"
        json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_case_csv(output_dir / "cases.csv", result["cases"])
        print(f"Wrote {json_path}")
        print(f"Wrote {output_dir / 'cases.csv'}")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())

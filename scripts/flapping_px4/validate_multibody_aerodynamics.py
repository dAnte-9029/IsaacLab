"""Run measured-wing multibody aerodynamic validation matrices."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback

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
        description=(
            "Validate measured-wing per-link aerodynamics using free-body "
            "time-step/impulse closure and fixed-body wind-tunnel matrices."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New output directory. Default: logs/flapping_px4/multibody_aero/<UTC timestamp>.",
    )
    parser.add_argument("--total-cycles", type=int, default=8)
    parser.add_argument("--measurement-cycles", type=int, default=4)
    parser.add_argument(
        "--job",
        choices=("free", "fixed"),
        required=True,
        help="Run exactly one Isaac environment per process.",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=4.0,
        help="Free-body job frequency; ignored by a non-smoke fixed job.",
    )
    parser.add_argument(
        "--closure-duration-s",
        type=float,
        default=0.1,
        help="Duration of the free-body impulse/momentum-closure window.",
    )
    parser.add_argument(
        "--dt-denominator",
        type=int,
        choices=(240, 480),
        default=480,
        help="Physics step is 1/value seconds.",
    )
    parser.add_argument(
        "--coupling-mode",
        choices=(
            "actual_per_wing_link",
            "actual_motion_base_equivalent",
            "commanded_base_equivalent",
            "prescribed_per_wing_link",
            "ideal_torque_per_wing_link",
            "sinusoidal_phase_per_wing_link",
            "ideal_inverse_dynamics_per_wing_link",
        ),
        default="actual_per_wing_link",
        help="Fixed-body job coupling mode.",
    )
    parser.add_argument(
        "--acceleration-source",
        choices=(
            "actual_joint_acceleration",
            "prescribed_acceleration",
            "zero_acceleration",
        ),
        default="actual_joint_acceleration",
        help="Acceleration input used by the actual-motion DeLaurier path.",
    )
    parser.add_argument(
        "--wing-link-load-mode",
        choices=(
            "full_wing_link_wrench",
            "wing_link_force_only",
            "wing_link_moment_only",
        ),
        default="full_wing_link_wrench",
        help="Per-wing force/moment ablation; relevant to actual_per_wing_link.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use four/two cycles and one wind-tunnel point for the selected job.",
    )
    parser.add_argument(
        "--single-point",
        action="store_true",
        help="Use only the requested frequency at 8 m/s and zero angle of attack.",
    )
    parser.add_argument(
        "--frequency-matrix",
        action="store_true",
        help="Use 2/3/4/5 Hz at 8 m/s and zero angle of attack.",
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


def _flatten_case(case: dict[str, object]) -> dict[str, object]:
    row = {
        key: value
        for key, value in case.items()
        if key not in {"means", "harmonics", "momentum_closure", "symmetry"}
    }
    for group_name in ("means", "momentum_closure", "symmetry"):
        group = case.get(group_name, {})
        if isinstance(group, dict):
            for key, value in group.items():
                row[f"{group_name}.{key}"] = value
    harmonics = case.get("harmonics", {})
    if isinstance(harmonics, dict):
        for signal_name, metrics in harmonics.items():
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                row[f"harmonics.{signal_name}.{key}"] = value
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    flapping_bot_module_path = _local_flapping_bot_source()
    print("[multibody-aero] launching Isaac Sim", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        from flapping_bot.analysis.physx_multibody_aero_benchmark import (
            run_fixed_multibody_aero_job,
            run_free_multibody_aero_job,
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = args.output_dir or Path(
            "logs/flapping_px4/multibody_aero"
        ) / timestamp
        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output directory: {output_dir}"
            )
        output_dir.mkdir(parents=True)
        dt_s = 1.0 / float(args.dt_denominator)
        total_cycles = 4 if args.smoke else int(args.total_cycles)
        measurement_cycles = 2 if args.smoke else int(args.measurement_cycles)
        single_point = bool(args.smoke or args.single_point)
        frequency_matrix = bool(args.frequency_matrix)
        progress_callback = lambda message: print(
            f"[multibody-aero] {message}",
            flush=True,
        )
        if args.job == "free":
            result = run_free_multibody_aero_job(
                frequency_hz=float(args.frequency_hz),
                dt_s=dt_s,
                duration_s=float(args.closure_duration_s),
                progress_callback=progress_callback,
            )
        else:
            result = run_fixed_multibody_aero_job(
                dt_s=dt_s,
                coupling_mode=str(args.coupling_mode),
                acceleration_source=str(args.acceleration_source),
                wing_link_load_mode=str(args.wing_link_load_mode),
                frequencies_hz=(float(args.frequency_hz),)
                if single_point and not frequency_matrix
                else (2.0, 3.0, 4.0, 5.0),
                airspeeds_mps=(8.0,)
                if single_point or frequency_matrix
                else (4.0, 6.0, 8.0),
                angles_of_attack_deg=(0.0,)
                if single_point or frequency_matrix
                else (-5.0, 0.0, 5.0),
                total_cycles=total_cycles,
                measurement_cycles=measurement_cycles,
                progress_callback=progress_callback,
            )
        result["provenance"] = {
            "timestamp_utc": timestamp,
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "git_dirty": bool(_git_output("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "flapping_bot_module_path": str(flapping_bot_module_path),
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_rows(
            output_dir / "free_cases.csv",
            [_flatten_case(case) for case in result["free_cases"]],
        )
        _write_rows(
            output_dir / "fixed_cases.csv",
            [_flatten_case(case) for case in result["fixed_cases"]],
        )
        print(f"[multibody-aero] wrote {summary_path}", flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())

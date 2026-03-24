"""Run the unified path-tracking teacher baseline suite."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
from typing import Sequence


def build_phase_names() -> list[str]:
    """Return the primitive baseline phases for generic path tracking."""
    return [
        "level_straight",
        "climb_straight",
        "descent_straight",
        "level_turn",
        "level_loiter",
        "multi_segment",
    ]


def build_suite_cases() -> list[dict[str, object]]:
    """Return canonical rollout cases for teacher baseline evaluation."""
    return [
        {"phase": "level_straight", "steps": 2200},
        {"phase": "climb_straight", "steps": 2400},
        {"phase": "descent_straight", "steps": 2400},
        {"phase": "level_turn", "steps": 2400},
        {"phase": "level_loiter", "steps": 2800},
        {"phase": "multi_segment", "steps": 3400},
    ]


def build_suite_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the path-tracking baseline suite."""
    parser = argparse.ArgumentParser(description="Run the PX4-like baseline suite for path-tracking primitives.")
    parser.add_argument("--task", type=str, default="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--height_sp", type=float, default=10.0)
    parser.add_argument("--metrics_warmup_s", type=float, default=3.0)
    parser.add_argument("--straight_length_m", type=float, default=60.0)
    parser.add_argument("--turn_radius_m", type=float, default=20.0)
    parser.add_argument("--loiter_radius_m", type=float, default=20.0)
    parser.add_argument("--turn_sweep_deg", type=float, default=90.0)
    parser.add_argument("--loiter_turns", type=float, default=1.0)
    parser.add_argument("--climb_delta_m", type=float, default=3.0)
    parser.add_argument("--path_manager_max_roll_deg", type=float, default=35.0)
    parser.add_argument("--path_manager_max_flight_path_angle_deg", type=float, default=10.0)
    parser.add_argument("--wind_x_mps", type=float, default=0.0)
    parser.add_argument("--wind_y_mps", type=float, default=0.0)
    parser.add_argument("--wind_ou", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wind_ou_tau_s", type=float, default=2.0)
    parser.add_argument("--wind_ou_sigma_x_mps", type=float, default=0.0)
    parser.add_argument("--wind_ou_sigma_y_mps", type=float, default=0.0)
    parser.add_argument("--wind_ou_clip_to_range", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot_dpi", type=int, default=160)
    parser.add_argument("--out_root", type=Path, default=Path("logs/flapping_px4/path_tracking_suite"))
    parser.add_argument("--print_every", type=int, default=250)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _run_cmd(cmd: Sequence[str], *, cwd: Path, dry_run: bool) -> int:
    print(f"\n[run] {shlex.join(cmd)}")
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(completed.returncode)


def _latest_subdir(path: Path) -> str | None:
    if not path.exists():
        return None
    candidates = [p for p in path.iterdir() if p.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main() -> None:
    args = build_suite_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "created_at": datetime.now().isoformat(),
        "task": args.task,
        "out_root": str(out_root),
        "plot_enabled": bool(args.plot),
        "cases": [],
    }

    for case in build_suite_cases():
        phase = str(case["phase"])
        phase_out = out_root / phase
        phase_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            "./isaaclab.sh",
            "-p",
            "scripts/flapping_px4/fly_path_mission.py",
            "--task",
            args.task,
            "--phase",
            phase,
            "--num_envs",
            str(args.num_envs),
            "--steps",
            str(int(case["steps"])),
            "--auto_extend_steps",
            "--height_sp",
            str(args.height_sp),
            "--metrics_warmup_s",
            str(args.metrics_warmup_s),
            "--straight_length_m",
            str(args.straight_length_m),
            "--turn_radius_m",
            str(args.turn_radius_m),
            "--loiter_radius_m",
            str(args.loiter_radius_m),
            "--turn_sweep_deg",
            str(args.turn_sweep_deg),
            "--loiter_turns",
            str(args.loiter_turns),
            "--climb_delta_m",
            str(args.climb_delta_m),
            "--path_manager_max_roll_deg",
            str(args.path_manager_max_roll_deg),
            "--path_manager_max_flight_path_angle_deg",
            str(args.path_manager_max_flight_path_angle_deg),
            "--wind_x_mps",
            str(args.wind_x_mps),
            "--wind_y_mps",
            str(args.wind_y_mps),
            "--wind_ou" if args.wind_ou else "--no-wind_ou",
            "--wind_ou_tau_s",
            str(args.wind_ou_tau_s),
            "--wind_ou_sigma_x_mps",
            str(args.wind_ou_sigma_x_mps),
            "--wind_ou_sigma_y_mps",
            str(args.wind_ou_sigma_y_mps),
            "--wind_ou_clip_to_range" if args.wind_ou_clip_to_range else "--no-wind_ou_clip_to_range",
            "--out_dir",
            str(out_root),
            "--print_every",
            str(args.print_every),
        ]
        if args.headless:
            cmd.append("--headless")

        return_code = _run_cmd(cmd, cwd=repo_root, dry_run=bool(args.dry_run))
        case_record: dict[str, object] = {
            "phase": phase,
            "steps": int(case["steps"]),
            "return_code": int(return_code),
            "phase_out": str(phase_out),
            "latest_run_dir": _latest_subdir(out_root / phase),
        }
        latest_run_dir = case_record["latest_run_dir"]

        if bool(args.plot):
            case_record["plot_requested"] = True
            if bool(args.dry_run):
                case_record["plot_planned"] = True
            elif return_code == 0 and isinstance(latest_run_dir, str):
                plot_cmd = [
                    "./isaaclab.sh",
                    "-p",
                    "scripts/flapping_px4/plot_path_mission.py",
                    "--run_dir",
                    latest_run_dir,
                    "--dpi",
                    str(args.plot_dpi),
                ]
                plot_return_code = _run_cmd(plot_cmd, cwd=repo_root, dry_run=False)
                case_record["plot_return_code"] = int(plot_return_code)
                case_record["plots_png"] = str(Path(latest_run_dir) / "plots.png")
                if plot_return_code != 0:
                    case_record["return_code"] = int(plot_return_code)
                    manifest["cases"].append(case_record)
                    break

        manifest["cases"].append(case_record)
        if return_code != 0:
            break

    manifest_path = out_root / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

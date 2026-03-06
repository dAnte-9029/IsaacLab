"""Run staged flapping PX4-like evaluations: straight -> loiter, no-wind -> wind.

Example:
  python scripts/flapping_px4/run_progressive_mission.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0 \
    --state_source estimated --headless
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
from typing import Sequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged straight/loiter evaluations with no-wind then wind.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0",
        help="Gym task id for both straight and loiter phases.",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps_straight", type=int, default=2500)
    parser.add_argument("--steps_loiter", type=int, default=3000)
    parser.add_argument("--line_length", type=float, default=120.0)
    parser.add_argument("--height_sp", type=float, default=10.0)
    parser.add_argument("--loiter_center_x", type=float, default=40.0)
    parser.add_argument("--loiter_center_y", type=float, default=0.0)
    parser.add_argument("--loiter_radius_m", type=float, default=20.0)
    parser.add_argument("--loiter_clockwise", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--state_source",
        type=str,
        choices=("truth", "estimated", "compare"),
        default="truth",
        help="State source for both controllers.",
    )
    parser.add_argument("--sensor_noise_scale", type=float, default=1.0)
    parser.add_argument("--sensor_bias_scale", type=float, default=1.0)
    parser.add_argument("--sensor_delay_scale", type=float, default=1.0)
    parser.add_argument("--estimator_attitude_gain", type=float, default=0.05)
    parser.add_argument("--estimator_attitude_gain_min", type=float, default=0.0)
    parser.add_argument("--estimator_accel_gate_sigma_mps2", type=float, default=1.25)
    parser.add_argument("--estimator_accel_gate_gyro_dps", type=float, default=90.0)
    parser.add_argument("--estimator_attitude_max_pitch_deg", type=float, default=85.0)
    parser.add_argument("--estimator_yaw_gain", type=float, default=0.08)
    parser.add_argument("--estimator_wind_tau_s", type=float, default=1.5)

    parser.add_argument(
        "--wind_model",
        type=str,
        choices=("ou", "constant"),
        default="ou",
        help="Wind model used for wind phases.",
    )
    parser.add_argument("--wind_x_mps", type=float, default=2.0)
    parser.add_argument("--wind_y_mps", type=float, default=1.0)
    parser.add_argument("--wind_ou_tau_s", type=float, default=2.0)
    parser.add_argument("--wind_ou_sigma_x_mps", type=float, default=0.8)
    parser.add_argument("--wind_ou_sigma_y_mps", type=float, default=0.8)
    parser.add_argument("--wind_ou_clip_to_range", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--out_root", type=Path, default=Path("logs/flapping_px4/progressive"))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="Generate plots after each phase.")
    parser.add_argument("--plot_dpi", type=int, default=160)
    parser.add_argument("--print_every", type=int, default=250)
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    return parser.parse_args()


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


def _common_args(args: argparse.Namespace) -> list[str]:
    return [
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--height_sp",
        str(args.height_sp),
        "--state_source",
        str(args.state_source),
        "--sensor_noise_scale",
        str(args.sensor_noise_scale),
        "--sensor_bias_scale",
        str(args.sensor_bias_scale),
        "--sensor_delay_scale",
        str(args.sensor_delay_scale),
        "--estimator_attitude_gain",
        str(args.estimator_attitude_gain),
        "--estimator_attitude_gain_min",
        str(args.estimator_attitude_gain_min),
        "--estimator_accel_gate_sigma_mps2",
        str(args.estimator_accel_gate_sigma_mps2),
        "--estimator_accel_gate_gyro_dps",
        str(args.estimator_accel_gate_gyro_dps),
        "--estimator_attitude_max_pitch_deg",
        str(args.estimator_attitude_max_pitch_deg),
        "--estimator_yaw_gain",
        str(args.estimator_yaw_gain),
        "--estimator_wind_tau_s",
        str(args.estimator_wind_tau_s),
        "--print_every",
        str(args.print_every),
    ]


def _wind_args(args: argparse.Namespace, *, with_wind: bool) -> list[str]:
    if not with_wind:
        return [
            "--wind_x_mps",
            "0.0",
            "--wind_y_mps",
            "0.0",
            "--no-wind_ou",
            "--no-random_wind",
        ]
    if args.wind_model == "constant":
        return [
            "--wind_x_mps",
            str(args.wind_x_mps),
            "--wind_y_mps",
            str(args.wind_y_mps),
            "--no-wind_ou",
            "--no-random_wind",
        ]
    return [
        "--wind_x_mps",
        str(args.wind_x_mps),
        "--wind_y_mps",
        str(args.wind_y_mps),
        "--wind_ou",
        "--wind_ou_tau_s",
        str(args.wind_ou_tau_s),
        "--wind_ou_sigma_x_mps",
        str(args.wind_ou_sigma_x_mps),
        "--wind_ou_sigma_y_mps",
        str(args.wind_ou_sigma_y_mps),
        "--wind_ou_clip_to_range" if args.wind_ou_clip_to_range else "--no-wind_ou_clip_to_range",
        "--no-random_wind",
    ]


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    phases = [
        ("01_straight_nowind", "scripts/flapping_px4/fly_straight_line.py", int(args.steps_straight), False),
        ("02_loiter_nowind", "scripts/flapping_px4/fly_loiter.py", int(args.steps_loiter), False),
        ("03_straight_wind", "scripts/flapping_px4/fly_straight_line.py", int(args.steps_straight), True),
        ("04_loiter_wind", "scripts/flapping_px4/fly_loiter.py", int(args.steps_loiter), True),
    ]

    manifest: dict[str, object] = {
        "created_at": datetime.now().isoformat(),
        "task": args.task,
        "state_source": args.state_source,
        "wind_model": args.wind_model,
        "plot_enabled": bool(args.plot),
        "out_root": str(out_root),
        "phases": [],
    }

    common = _common_args(args)
    for phase_name, script_path, steps, with_wind in phases:
        phase_out = out_root / phase_name
        phase_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            "./isaaclab.sh",
            "-p",
            script_path,
            "--steps",
            str(steps),
            "--out_dir",
            str(phase_out),
        ]
        cmd.extend(common)
        cmd.extend(_wind_args(args, with_wind=with_wind))
        if script_path.endswith("fly_straight_line.py"):
            cmd.extend(["--line_length", str(args.line_length)])
        else:
            cmd.extend(
                [
                    "--loiter_center_x",
                    str(args.loiter_center_x),
                    "--loiter_center_y",
                    str(args.loiter_center_y),
                    "--loiter_radius_m",
                    str(args.loiter_radius_m),
                    "--loiter_clockwise" if args.loiter_clockwise else "--no-loiter_clockwise",
                ]
            )
        if args.headless:
            cmd.append("--headless")

        return_code = _run_cmd(cmd, cwd=repo_root, dry_run=bool(args.dry_run))
        phase_record = {
            "name": phase_name,
            "script": script_path,
            "with_wind": bool(with_wind),
            "return_code": int(return_code),
            "out_dir": str(phase_out),
            "latest_run_dir": _latest_subdir(phase_out),
        }
        latest_run_dir = phase_record["latest_run_dir"]
        if bool(args.plot):
            phase_record["plot_requested"] = True
            if bool(args.dry_run):
                phase_record["plot_planned"] = True
            elif return_code == 0 and isinstance(latest_run_dir, str):
                if script_path.endswith("fly_straight_line.py"):
                    plot_script = "scripts/flapping_px4/plot_trajectory.py"
                else:
                    plot_script = "scripts/flapping_px4/plot_loiter.py"
                plot_cmd = [
                    "./isaaclab.sh",
                    "-p",
                    plot_script,
                    "--run_dir",
                    latest_run_dir,
                    "--dpi",
                    str(args.plot_dpi),
                ]
                plot_return_code = _run_cmd(plot_cmd, cwd=repo_root, dry_run=False)
                phase_record["plot_script"] = plot_script
                phase_record["plot_return_code"] = int(plot_return_code)
                phase_record["plots_png"] = str(Path(latest_run_dir) / "plots.png")
                if plot_return_code != 0:
                    phase_record["return_code"] = int(plot_return_code)
                    manifest["phases"].append(phase_record)
                    break
            else:
                phase_record["plot_planned"] = False
        manifest["phases"].append(phase_record)
        if return_code != 0:
            break

    manifest_path = out_root / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

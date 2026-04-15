"""Run the formal estimated-state complex path-tracking battery."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
from typing import Any, Sequence


def parse_seed_list(value: str) -> list[int]:
    """Parse a comma-separated seed list."""
    return [int(token.strip()) for token in str(value).split(",") if token.strip()]


def build_complex_suite_cases(
    *,
    random_seeds: Sequence[int] = (25, 30, 33, 36),
    include_fixed_multi_segment: bool = True,
    steps: int = 5200,
) -> list[dict[str, Any]]:
    """Build the deterministic formal case layout."""
    cases: list[dict[str, Any]] = []
    if include_fixed_multi_segment:
        cases.append(
            {
                "name": "multi_segment",
                "mission_mode": "fixed",
                "phase": "multi_segment",
                "steps": int(steps),
            }
        )
    for seed in random_seeds:
        case_index = len(cases) + 1
        cases.append(
            {
                "name": f"case{case_index:02d}_seed{int(seed):03d}",
                "mission_mode": "random",
                "mission_seed": int(seed),
                "mission_label": f"random_seed_{int(seed):06d}",
                "steps": int(steps),
            }
        )
    return cases


def build_case_command(
    args: argparse.Namespace,
    *,
    case: dict[str, Any],
    cases_root: Path,
    portable_root: Path,
) -> list[str]:
    """Build the rollout command for one formal complex-suite case."""
    cmd = [
        "./isaaclab.sh",
        "-p",
        "scripts/flapping_px4/fly_path_mission.py",
        "--task",
        args.task,
        "--mission_mode",
        str(case["mission_mode"]),
        "--num_envs",
        str(args.num_envs),
        "--teacher_state_source",
        str(args.teacher_state_source),
        "--policy_state_source",
        str(args.policy_state_source),
        "--imu_source",
        str(args.imu_source),
        "--controller_tuning_profile",
        str(args.controller_tuning_profile),
        "--total_mass_kg_override",
        str(args.total_mass_kg_override),
        "--steps",
        str(case.get("steps", args.steps)),
        "--seed",
        str(case.get("mission_seed", 0)),
        "--auto_extend_steps" if args.auto_extend_steps else "--no-auto_extend_steps",
        "--nominal_speed_mps",
        str(args.nominal_speed_mps),
        "--completion_margin_s",
        str(args.completion_margin_s),
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
        "--path_warmup_enabled" if args.path_warmup_enabled else "--no-path_warmup_enabled",
        "--path_warmup_straight_length_m",
        str(args.path_warmup_straight_length_m),
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
        str(cases_root),
        "--print_every",
        str(args.print_every),
        "--portable-root",
        str(portable_root),
        "--device",
        str(args.device),
    ]
    if case["mission_mode"] == "fixed":
        cmd.extend(["--phase", str(case["phase"])])
    else:
        cmd.extend(["--mission_seed", str(case["mission_seed"])])
        cmd.extend(["--mission_label", str(case.get("mission_label", case["name"]))])
    if args.teacher_roll_kd is not None:
        cmd.extend(["--teacher_roll_kd", str(args.teacher_roll_kd)])
    if args.teacher_max_roll_deg is not None:
        cmd.extend(["--teacher_max_roll_deg", str(args.teacher_max_roll_deg)])
    if args.teacher_inner_elevon_pitch_rate_limit_per_s is not None:
        cmd.extend(
            [
                "--teacher_inner_elevon_pitch_rate_limit_per_s",
                str(args.teacher_inner_elevon_pitch_rate_limit_per_s),
            ]
        )
    if args.teacher_inner_elevon_roll_rate_limit_per_s is not None:
        cmd.extend(
            [
                "--teacher_inner_elevon_roll_rate_limit_per_s",
                str(args.teacher_inner_elevon_roll_rate_limit_per_s),
            ]
        )
    if args.wind_x_range_min_mps is not None:
        cmd.extend(["--wind_x_range_min_mps", str(args.wind_x_range_min_mps)])
    if args.wind_x_range_max_mps is not None:
        cmd.extend(["--wind_x_range_max_mps", str(args.wind_x_range_max_mps)])
    if args.wind_y_range_min_mps is not None:
        cmd.extend(["--wind_y_range_min_mps", str(args.wind_y_range_min_mps)])
    if args.wind_y_range_max_mps is not None:
        cmd.extend(["--wind_y_range_max_mps", str(args.wind_y_range_max_mps)])
    if args.headless:
        cmd.append("--headless")
    return cmd


def aggregate_case_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate case-level summary metrics."""
    if not summaries:
        return {
            "cases": 0,
            "completion_rate": 0.0,
            "worst_mean_lateral_case": None,
            "worst_p95_lateral_case": None,
            "mean_case_lateral_error_m": float("nan"),
            "mean_case_height_error_m": float("nan"),
        }

    def _mean(key: str) -> float:
        values = [float(summary[key]) for summary in summaries if key in summary]
        if not values:
            return float("nan")
        return round(float(sum(values) / len(values)), 12)

    worst_mean_lateral = max(summaries, key=lambda summary: float(summary.get("mean_abs_lateral_error_m", float("-inf"))))
    worst_p95_lateral = max(summaries, key=lambda summary: float(summary.get("p95_abs_lateral_error_m", float("-inf"))))
    worst_mean_height = max(summaries, key=lambda summary: float(summary.get("mean_abs_height_error_m", float("-inf"))))
    return {
        "cases": len(summaries),
        "completed_cases": int(sum(bool(summary.get("completed_path", False)) for summary in summaries)),
        "completion_rate": float(
            sum(bool(summary.get("completed_path", False)) for summary in summaries) / len(summaries)
        ),
        "mean_case_lateral_error_m": _mean("mean_abs_lateral_error_m"),
        "mean_case_height_error_m": _mean("mean_abs_height_error_m"),
        "mean_case_p95_lateral_error_m": _mean("p95_abs_lateral_error_m"),
        "mean_case_p95_height_error_m": _mean("p95_abs_height_error_m"),
        "mean_case_final_progress_ratio": _mean("final_progress_ratio"),
        "worst_mean_lateral_case": worst_mean_lateral.get("case_name"),
        "worst_p95_lateral_case": worst_p95_lateral.get("case_name"),
        "worst_mean_height_case": worst_mean_height.get("case_name"),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the suite-runner CLI parser."""
    parser = argparse.ArgumentParser(description="Run the estimated-state complex path-tracking battery.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0",
    )
    parser.add_argument(
        "--scenario",
        choices=("baseline_nowind", "steady_crosswind", "ou_gust", "live_imu_spotcheck"),
        default="baseline_nowind",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--teacher_state_source", type=str, choices=("truth", "estimated"), default="estimated")
    parser.add_argument("--policy_state_source", type=str, choices=("truth", "estimated"), default="estimated")
    parser.add_argument("--imu_source", type=str, choices=("synthetic", "isaacsim"), default="synthetic")
    parser.add_argument(
        "--controller_tuning_profile",
        choices=("auto", "truth_baseline", "estimated_teacher"),
        default="estimated_teacher",
    )
    parser.add_argument("--total_mass_kg_override", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=5200)
    parser.add_argument("--auto_extend_steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nominal_speed_mps", type=float, default=7.0)
    parser.add_argument("--completion_margin_s", type=float, default=3.0)
    parser.add_argument("--height_sp", type=float, default=10.0)
    parser.add_argument("--metrics_warmup_s", type=float, default=3.0)
    parser.add_argument("--straight_length_m", type=float, default=80.0)
    parser.add_argument("--turn_radius_m", type=float, default=25.0)
    parser.add_argument("--loiter_radius_m", type=float, default=40.0)
    parser.add_argument("--turn_sweep_deg", type=float, default=90.0)
    parser.add_argument("--loiter_turns", type=float, default=1.5)
    parser.add_argument("--climb_delta_m", type=float, default=3.0)
    parser.add_argument("--path_manager_max_roll_deg", type=float, default=35.0)
    parser.add_argument("--path_manager_max_flight_path_angle_deg", type=float, default=10.0)
    parser.add_argument("--path_warmup_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--path_warmup_straight_length_m", type=float, default=25.0)
    parser.add_argument("--teacher_roll_kd", type=float, default=None)
    parser.add_argument("--teacher_max_roll_deg", type=float, default=None)
    parser.add_argument("--teacher_inner_elevon_pitch_rate_limit_per_s", type=float, default=None)
    parser.add_argument("--teacher_inner_elevon_roll_rate_limit_per_s", type=float, default=None)
    parser.add_argument("--seed_list", type=str, default="25,30,33,36")
    parser.add_argument("--include_fixed_multi_segment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steady_crosswind_x_mps", type=float, default=0.0)
    parser.add_argument("--steady_crosswind_y_mps", type=float, default=2.0)
    parser.add_argument("--ou_gust_sigma_x_mps", type=float, default=1.0)
    parser.add_argument("--ou_gust_sigma_y_mps", type=float, default=1.5)
    parser.add_argument("--ou_gust_clip_sigma", type=float, default=3.0)
    parser.add_argument("--wind_x_mps", type=float, default=0.0)
    parser.add_argument("--wind_y_mps", type=float, default=0.0)
    parser.add_argument("--wind_x_range_min_mps", type=float, default=None)
    parser.add_argument("--wind_x_range_max_mps", type=float, default=None)
    parser.add_argument("--wind_y_range_min_mps", type=float, default=None)
    parser.add_argument("--wind_y_range_max_mps", type=float, default=None)
    parser.add_argument("--wind_ou", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wind_ou_tau_s", type=float, default=2.0)
    parser.add_argument("--wind_ou_sigma_x_mps", type=float, default=0.0)
    parser.add_argument("--wind_ou_sigma_y_mps", type=float, default=0.0)
    parser.add_argument("--wind_ou_clip_to_range", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--episode_timeout_s", type=float, default=900.0)
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("logs/flapping_px4/estimated_path_tracking_complex_suite"),
    )
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _configure_scenario(args: argparse.Namespace) -> argparse.Namespace:
    if args.scenario == "baseline_nowind":
        args.wind_x_mps = 0.0
        args.wind_y_mps = 0.0
        args.wind_ou = False
        args.wind_ou_sigma_x_mps = 0.0
        args.wind_ou_sigma_y_mps = 0.0
        args.imu_source = "synthetic"
    elif args.scenario == "steady_crosswind":
        args.wind_x_mps = float(args.steady_crosswind_x_mps)
        args.wind_y_mps = float(args.steady_crosswind_y_mps)
        args.wind_ou = False
        args.wind_ou_sigma_x_mps = 0.0
        args.wind_ou_sigma_y_mps = 0.0
        args.imu_source = "synthetic"
    elif args.scenario == "ou_gust":
        args.wind_x_mps = 0.0
        args.wind_y_mps = 0.0
        args.wind_ou = True
        args.wind_ou_sigma_x_mps = float(args.ou_gust_sigma_x_mps)
        args.wind_ou_sigma_y_mps = float(args.ou_gust_sigma_y_mps)
        args.wind_ou_clip_to_range = True
        clip_sigma = max(float(args.ou_gust_clip_sigma), 0.0)
        if args.wind_x_range_min_mps is None:
            args.wind_x_range_min_mps = -clip_sigma * float(args.ou_gust_sigma_x_mps)
        if args.wind_x_range_max_mps is None:
            args.wind_x_range_max_mps = clip_sigma * float(args.ou_gust_sigma_x_mps)
        if args.wind_y_range_min_mps is None:
            args.wind_y_range_min_mps = -clip_sigma * float(args.ou_gust_sigma_y_mps)
        if args.wind_y_range_max_mps is None:
            args.wind_y_range_max_mps = clip_sigma * float(args.ou_gust_sigma_y_mps)
        args.imu_source = "synthetic"
    elif args.scenario == "live_imu_spotcheck":
        args.wind_x_mps = 0.0
        args.wind_y_mps = 0.0
        args.wind_ou = False
        args.wind_ou_sigma_x_mps = 0.0
        args.wind_ou_sigma_y_mps = 0.0
        args.imu_source = "isaacsim"
    return args


def _run_cmd(cmd: Sequence[str], *, cwd: Path, dry_run: bool, timeout_s: float | None) -> int:
    print(f"\n[run] {shlex.join(cmd)}")
    if dry_run:
        return 0
    process = subprocess.Popen(cmd, cwd=str(cwd), start_new_session=True)
    try:
        return int(process.wait(timeout=timeout_s))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 124


def _latest_subdir(path: Path) -> Path | None:
    if not path.exists():
        return None
    candidates = [candidate for candidate in path.iterdir() if candidate.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _configure_scenario(build_parser().parse_args())
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root / args.scenario / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    cases_root = out_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    portable_root_base = out_root / "portable"
    portable_root_base.mkdir(parents=True, exist_ok=True)
    timeout_s = None if float(args.episode_timeout_s) <= 0.0 else float(args.episode_timeout_s)

    cases = build_complex_suite_cases(
        random_seeds=parse_seed_list(args.seed_list),
        include_fixed_multi_segment=bool(args.include_fixed_multi_segment),
        steps=int(args.steps),
    )
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "scenario": args.scenario,
        "task": args.task,
        "out_root": str(out_root),
        "cases": cases,
    }
    case_records: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    for case in cases:
        portable_root = portable_root_base / str(case["name"])
        portable_root.mkdir(parents=True, exist_ok=True)
        cmd = build_case_command(args, case=case, cases_root=cases_root, portable_root=portable_root)
        return_code = _run_cmd(cmd, cwd=repo_root, dry_run=bool(args.dry_run), timeout_s=timeout_s)
        mission_dir = case.get("mission_label", case["name"])
        run_dir = _latest_subdir(cases_root / str(mission_dir))
        record: dict[str, Any] = {
            "case_name": str(case["name"]),
            "return_code": int(return_code),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "portable_root": str(portable_root),
            "timed_out": bool(return_code == 124),
        }
        if not args.dry_run and return_code == 0 and run_dir is not None:
            summary_path = run_dir / "summary.json"
            summary = _load_json(summary_path)
            summary["case_name"] = str(case["name"])
            case_summaries.append(summary)
            record.update(
                {
                    "summary_path": str(summary_path),
                    "completed_path": bool(summary.get("completed_path", False)),
                    "final_progress_ratio": float(summary.get("final_progress_ratio", 0.0)),
                    "mean_abs_lateral_error_m": float(summary.get("mean_abs_lateral_error_m", float("nan"))),
                    "p95_abs_lateral_error_m": float(summary.get("p95_abs_lateral_error_m", float("nan"))),
                    "mean_abs_height_error_m": float(summary.get("mean_abs_height_error_m", float("nan"))),
                    "p95_abs_height_error_m": float(summary.get("p95_abs_height_error_m", float("nan"))),
                }
            )
            plot_cmd = [
                "./isaaclab.sh",
                "-p",
                "scripts/flapping_px4/plot_path_mission.py",
                "--run_dir",
                str(run_dir),
            ]
            _run_cmd(plot_cmd, cwd=repo_root, dry_run=False, timeout_s=300.0)
        case_records.append(record)
        if return_code != 0:
            break

    aggregate = aggregate_case_summaries(case_summaries)
    aggregate.update(
        {
            "scenario": args.scenario,
            "task": args.task,
            "teacher_state_source": args.teacher_state_source,
            "policy_state_source": args.policy_state_source,
            "imu_source": args.imu_source,
            "controller_tuning_profile": args.controller_tuning_profile,
            "total_mass_kg_override": args.total_mass_kg_override,
            "wind_x_mps": args.wind_x_mps,
            "wind_y_mps": args.wind_y_mps,
            "wind_ou": bool(args.wind_ou),
            "wind_ou_tau_s": args.wind_ou_tau_s,
            "wind_ou_sigma_x_mps": args.wind_ou_sigma_x_mps,
            "wind_ou_sigma_y_mps": args.wind_ou_sigma_y_mps,
            "cases_requested": len(cases),
            "cases_attempted": len(case_records),
            "out_root": str(out_root),
        }
    )

    with (out_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    with (out_root / "summary.json").open("w") as f:
        json.dump(aggregate, f, indent=2)
    _write_csv(case_records, out_root / "cases.csv")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

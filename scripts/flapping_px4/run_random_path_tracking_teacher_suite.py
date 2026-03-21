"""Run a seeded truth-teacher benchmark over random path-tracking missions."""

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

_MAX_PROGRESS_JUMP_M = 5.0
_EARLY_PROGRESS_JUMP_TIME_S = 10.0


def build_episode_seeds(*, seed_start: int, episodes: int) -> list[int]:
    """Build the deterministic seed list for the benchmark."""
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    return [int(seed_start) + index for index in range(int(episodes))]


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(0.0, min(1.0, float(q))) * float(len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - float(lower)
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def aggregate_episode_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode teacher rollout summaries into benchmark metrics."""
    completed = [bool(summary.get("completed_path", False)) for summary in summaries]
    jump_free_completed = [
        bool(summary.get("completed_path", False)) and not bool(summary.get("has_progress_jump", False))
        for summary in summaries
    ]
    lateral = [float(summary["mean_abs_lateral_error_m"]) for summary in summaries if "mean_abs_lateral_error_m" in summary]
    height = [float(summary["mean_abs_height_error_m"]) for summary in summaries if "mean_abs_height_error_m" in summary]
    align = [float(summary["mean_abs_align_error_deg"]) for summary in summaries if "mean_abs_align_error_deg" in summary]
    progress = [float(summary["final_progress_ratio"]) for summary in summaries if "final_progress_ratio" in summary]
    jump_magnitudes = [
        float(summary["max_single_step_progress_jump_m"])
        for summary in summaries
        if "max_single_step_progress_jump_m" in summary
    ]
    suspicious_jump_seeds = [
        int(summary["mission_seed"])
        for summary in summaries
        if bool(summary.get("has_progress_jump", False)) and "mission_seed" in summary
    ]
    early_jump_seeds = [
        int(summary["mission_seed"])
        for summary in summaries
        if bool(summary.get("has_early_progress_jump", False)) and "mission_seed" in summary
    ]
    aggregate: dict[str, Any] = {
        "episodes": len(summaries),
        "completed_episodes": int(sum(completed)),
        "completion_rate": float(sum(completed) / len(summaries)) if summaries else 0.0,
        "jump_free_completed_episodes": int(sum(jump_free_completed)),
        "jump_free_completion_rate": float(sum(jump_free_completed) / len(summaries)) if summaries else 0.0,
        "mean_episode_lateral_error_m": _mean(lateral),
        "p95_episode_lateral_error_m": _quantile(lateral, 0.95),
        "mean_episode_height_error_m": _mean(height),
        "p95_episode_height_error_m": _quantile(height, 0.95),
        "mean_episode_align_error_deg": _mean(align),
        "p95_episode_align_error_deg": _quantile(align, 0.95),
        "mean_final_progress_ratio": _mean(progress),
        "p95_final_progress_ratio": _quantile(progress, 0.95),
        "episodes_with_progress_jump": len(suspicious_jump_seeds),
        "episodes_with_early_progress_jump": len(early_jump_seeds),
        "max_single_step_progress_jump_m": max(jump_magnitudes) if jump_magnitudes else 0.0,
        "suspicious_jump_seeds": suspicious_jump_seeds,
        "early_jump_seeds": early_jump_seeds,
        "worst_progress_seed": None,
    }
    if summaries and progress:
        worst_index = min(range(len(progress)), key=lambda index: progress[index])
        aggregate["worst_progress_seed"] = int(summaries[worst_index]["mission_seed"])
    return aggregate


def analyze_trajectory_progress(
    traj_path: Path,
    *,
    jump_threshold_m: float = _MAX_PROGRESS_JUMP_M,
    early_time_s: float = _EARLY_PROGRESS_JUMP_TIME_S,
) -> dict[str, Any]:
    """Analyze a rollout trajectory for impossible path-progress discontinuities."""
    if not traj_path.exists():
        return {
            "trajectory_available": False,
            "trajectory_samples": 0,
            "max_single_step_progress_jump_m": 0.0,
            "min_single_step_progress_delta_m": 0.0,
            "has_progress_jump": False,
            "has_early_progress_jump": False,
            "suspicious_jump_count": 0,
            "first_suspicious_jump_time_s": None,
            "first_suspicious_jump_m": None,
        }

    with traj_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) <= 1:
        return {
            "trajectory_available": True,
            "trajectory_samples": len(rows),
            "max_single_step_progress_jump_m": 0.0,
            "min_single_step_progress_delta_m": 0.0,
            "has_progress_jump": False,
            "has_early_progress_jump": False,
            "suspicious_jump_count": 0,
            "first_suspicious_jump_time_s": None,
            "first_suspicious_jump_m": None,
        }

    max_delta_s = float("-inf")
    min_delta_s = float("inf")
    suspicious_jumps: list[dict[str, float]] = []

    previous_time_s = float(rows[0]["t"])
    previous_progress_s = float(rows[0]["progress_s"])
    for row in rows[1:]:
        time_s = float(row["t"])
        progress_s = float(row["progress_s"])
        delta_s = progress_s - previous_progress_s
        max_delta_s = max(max_delta_s, delta_s)
        min_delta_s = min(min_delta_s, delta_s)
        if delta_s > float(jump_threshold_m):
            suspicious_jumps.append(
                {
                    "time_s": time_s,
                    "delta_s": delta_s,
                    "prev_time_s": previous_time_s,
                    "prev_progress_s": previous_progress_s,
                    "progress_s": progress_s,
                }
            )
        previous_time_s = time_s
        previous_progress_s = progress_s

    first_jump = suspicious_jumps[0] if suspicious_jumps else None
    return {
        "trajectory_available": True,
        "trajectory_samples": len(rows),
        "max_single_step_progress_jump_m": float(max_delta_s),
        "min_single_step_progress_delta_m": float(min_delta_s),
        "has_progress_jump": bool(suspicious_jumps),
        "has_early_progress_jump": any(jump["time_s"] <= float(early_time_s) for jump in suspicious_jumps),
        "suspicious_jump_count": len(suspicious_jumps),
        "first_suspicious_jump_time_s": None if first_jump is None else float(first_jump["time_s"]),
        "first_suspicious_jump_m": None if first_jump is None else float(first_jump["delta_s"]),
    }


def summarize_episode_records(records: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize subprocess-level episode outcomes, including launch failures and timeouts."""
    return {
        "episodes_attempted": len(records),
        "episodes_succeeded": sum(1 for record in records if int(record.get("return_code", 1)) == 0),
        "failed_launches": sum(1 for record in records if int(record.get("return_code", 1)) != 0),
        "timed_out_episodes": sum(1 for record in records if bool(record.get("timed_out", False))),
    }


def build_random_teacher_suite_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the random truth-teacher benchmark."""
    parser = argparse.ArgumentParser(description="Run the seeded truth-teacher benchmark for random path missions.")
    parser.add_argument("--task", type=str, default="Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3400)
    parser.add_argument("--auto_extend_steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nominal_speed_mps", type=float, default=7.0)
    parser.add_argument("--completion_margin_s", type=float, default=2.0)
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
    parser.add_argument("--mission_num_segments_min", type=int, default=2)
    parser.add_argument("--mission_num_segments_max", type=int, default=4)
    parser.add_argument("--mission_allow_straight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mission_allow_turn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mission_allow_loiter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mission_allow_climb_on_straight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wind_x_mps", type=float, default=0.0)
    parser.add_argument("--wind_y_mps", type=float, default=0.0)
    parser.add_argument("--wind_ou", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wind_ou_tau_s", type=float, default=2.0)
    parser.add_argument("--wind_ou_sigma_x_mps", type=float, default=0.0)
    parser.add_argument("--wind_ou_sigma_y_mps", type=float, default=0.0)
    parser.add_argument("--wind_ou_clip_to_range", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--episode_timeout_s",
        type=float,
        default=900.0,
        help="Kill a rollout subprocess if it exceeds this wall-clock time. Use <=0 to disable.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out_root", type=Path, default=Path("logs/flapping_px4/random_path_tracking_teacher_suite"))
    parser.add_argument("--print_every", type=int, default=250)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def build_episode_command(
    args: argparse.Namespace,
    *,
    mission_seed: int,
    mission_label: str,
    episodes_root: Path,
    portable_root: Path,
) -> list[str]:
    """Build the rollout command for one seeded teacher benchmark episode."""
    cmd = [
        "./isaaclab.sh",
        "-p",
        "scripts/flapping_px4/fly_path_mission.py",
        "--task",
        args.task,
        "--mission_mode",
        "random",
        "--mission_seed",
        str(mission_seed),
        "--mission_label",
        mission_label,
        "--num_envs",
        str(args.num_envs),
        "--steps",
        str(args.steps),
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
        "--mission_num_segments_min",
        str(args.mission_num_segments_min),
        "--mission_num_segments_max",
        str(args.mission_num_segments_max),
        "--mission_allow_straight" if args.mission_allow_straight else "--no-mission_allow_straight",
        "--mission_allow_turn" if args.mission_allow_turn else "--no-mission_allow_turn",
        "--mission_allow_loiter" if args.mission_allow_loiter else "--no-mission_allow_loiter",
        "--mission_allow_climb_on_straight"
        if args.mission_allow_climb_on_straight
        else "--no-mission_allow_climb_on_straight",
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
        str(episodes_root),
        "--print_every",
        str(args.print_every),
        "--portable-root",
        str(portable_root),
    ]
    if args.headless:
        cmd.append("--headless")
    return cmd


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


def _load_summary(summary_path: Path) -> dict[str, Any]:
    with summary_path.open() as f:
        return json.load(f)


def _write_episode_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_random_teacher_suite_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    episodes_root = out_root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    portable_root_base = out_root / "portable"
    portable_root_base.mkdir(parents=True, exist_ok=True)
    timeout_s = None if float(args.episode_timeout_s) <= 0.0 else float(args.episode_timeout_s)

    episode_records: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "task": args.task,
        "out_root": str(out_root),
        "episodes_requested": int(args.episodes),
        "seed_start": int(args.seed_start),
        "cases": episode_records,
    }

    for mission_seed in build_episode_seeds(seed_start=int(args.seed_start), episodes=int(args.episodes)):
        mission_label = f"random_seed_{mission_seed:06d}"
        portable_root = portable_root_base / mission_label
        portable_root.mkdir(parents=True, exist_ok=True)
        cmd = build_episode_command(
            args,
            mission_seed=int(mission_seed),
            mission_label=mission_label,
            episodes_root=episodes_root,
            portable_root=portable_root,
        )

        return_code = _run_cmd(cmd, cwd=repo_root, dry_run=bool(args.dry_run), timeout_s=timeout_s)
        run_dir = _latest_subdir(episodes_root / mission_label)
        episode_record: dict[str, Any] = {
            "mission_seed": int(mission_seed),
            "mission_label": mission_label,
            "return_code": int(return_code),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "portable_root": str(portable_root),
            "episode_timeout_s": timeout_s,
            "timed_out": bool(return_code == 124),
        }

        if not args.dry_run and return_code == 0 and run_dir is not None:
            summary_path = run_dir / "summary.json"
            traj_path = run_dir / "trajectory_env0.csv"
            episode_summary = _load_summary(summary_path)
            progress_check = analyze_trajectory_progress(traj_path)
            episode_summary.update(progress_check)
            episode_summaries.append(episode_summary)
            episode_record.update(
                {
                    "summary_path": str(summary_path),
                    "traj_path": str(traj_path),
                    "completed_path": bool(episode_summary.get("completed_path", False)),
                    "final_progress_ratio": float(episode_summary.get("final_progress_ratio", 0.0)),
                    "mean_abs_lateral_error_m": float(episode_summary.get("mean_abs_lateral_error_m", float("nan"))),
                    "mean_abs_height_error_m": float(episode_summary.get("mean_abs_height_error_m", float("nan"))),
                    "mean_abs_align_error_deg": float(episode_summary.get("mean_abs_align_error_deg", float("nan"))),
                    "jump_free_completed_path": bool(
                        episode_summary.get("completed_path", False) and not episode_summary.get("has_progress_jump", False)
                    ),
                    "has_progress_jump": bool(episode_summary.get("has_progress_jump", False)),
                    "has_early_progress_jump": bool(episode_summary.get("has_early_progress_jump", False)),
                    "max_single_step_progress_jump_m": float(
                        episode_summary.get("max_single_step_progress_jump_m", 0.0)
                    ),
                }
            )

        episode_records.append(episode_record)
        if return_code != 0:
            break

    aggregate = aggregate_episode_summaries(episode_summaries)
    aggregate.update(summarize_episode_records(episode_records))
    aggregate.update(
        {
            "task": args.task,
            "episodes_requested": int(args.episodes),
            "seed_start": int(args.seed_start),
            "steps": int(args.steps),
            "out_root": str(out_root),
        }
    )

    summary_path = out_root / "summary.json"
    manifest_path = out_root / "manifest.json"
    csv_path = out_root / "episodes.csv"
    postcheck_path = out_root / "postcheck.json"
    with summary_path.open("w") as f:
        json.dump(aggregate, f, indent=2)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    _write_episode_csv(episode_records, csv_path)
    with postcheck_path.open("w") as f:
        json.dump(
            {
                "count": len(episode_summaries),
                "completed_count": int(sum(bool(summary.get("completed_path", False)) for summary in episode_summaries)),
                "jump_free_completed_count": int(
                    sum(
                        bool(summary.get("completed_path", False)) and not bool(summary.get("has_progress_jump", False))
                        for summary in episode_summaries
                    )
                ),
                "max_single_step_progress_jump_m": aggregate["max_single_step_progress_jump_m"],
                "suspicious_jump_seeds": aggregate["suspicious_jump_seeds"],
                "early_jump_seeds": aggregate["early_jump_seeds"],
            },
            f,
            indent=2,
        )

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

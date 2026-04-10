"""Sweep reset trim parameters on the controller-only path-tracking baseline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Sequence


_DEFAULT_TASK = "Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0"


@dataclass(frozen=True)
class Candidate:
    """One reset-trim candidate."""

    reset_pitch_deg: float
    reset_flap_hz: float
    reset_elevon_pitch_deg: float

    @property
    def label(self) -> str:
        return (
            f"pitch_{self.reset_pitch_deg:.2f}".replace("-", "m").replace(".", "p")
            + "__"
            + f"flap_{self.reset_flap_hz:.2f}".replace("-", "m").replace(".", "p")
            + "__"
            + f"elevon_{self.reset_elevon_pitch_deg:.2f}".replace("-", "m").replace(".", "p")
        )


def build_sweep_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for reset-trim sweeps."""
    parser = argparse.ArgumentParser(description="Sweep reset trim parameters using controller-only level_straight rollouts.")
    parser.add_argument("--task", type=str, default=_DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2200)
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
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out_root", type=Path, default=Path("logs/flapping_px4/reset_trim_sweep"))
    parser.add_argument("--reset_pitch_deg_values", type=float, nargs="+", default=[4.0, 6.0, 8.0, 10.0, 12.0])
    parser.add_argument("--reset_flap_hz_values", type=float, nargs="+", default=[2.5, 2.8, 3.1, 3.4])
    parser.add_argument(
        "--reset_elevon_pitch_deg_values",
        type=float,
        nargs="+",
        default=[-16.0, -14.0, -12.0, -10.0, -8.0],
    )
    return parser


def build_rollout_command(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    reset_pitch_deg: float,
    reset_flap_hz: float,
    reset_elevon_pitch_deg: float,
) -> list[str]:
    """Build one controller-only level_straight rollout command."""
    cmd = [
        "./isaaclab.sh",
        "-p",
        "scripts/flapping_px4/fly_path_mission.py",
        "--task",
        args.task,
        "--phase",
        "level_straight",
        "--num_envs",
        str(args.num_envs),
        "--steps",
        str(int(args.steps)),
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
        "--reset_pitch_deg",
        str(float(reset_pitch_deg)),
        "--reset_flap_hz",
        str(float(reset_flap_hz)),
        "--reset_elevon_pitch_deg",
        str(float(reset_elevon_pitch_deg)),
        "--out_dir",
        str(out_dir),
        "--print_every",
        str(args.print_every),
    ]
    if args.headless:
        cmd.append("--headless")
    return cmd


def _run_cmd(cmd: Sequence[str], *, cwd: Path, dry_run: bool) -> int:
    print(f"[run] {shlex.join(cmd)}")
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(completed.returncode)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads(next(run_dir.rglob("summary.json")).read_text())


def _first3s_metrics(run_dir: Path) -> dict[str, float]:
    traj_path = next(run_dir.rglob("trajectory_env0.csv"))
    min_height_error_first3s = float("inf")
    max_abs_pitch_first3s = 0.0
    with traj_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if float(row["t"]) > 3.0:
                continue
            min_height_error_first3s = min(min_height_error_first3s, float(row["height_error_m"]))
            max_abs_pitch_first3s = max(max_abs_pitch_first3s, abs(float(row["pitch_deg"])))
    if min_height_error_first3s == float("inf"):
        min_height_error_first3s = float("nan")
    return {
        "min_height_error_first3s": min_height_error_first3s,
        "max_abs_pitch_first3s": max_abs_pitch_first3s,
    }


def _candidate_rows(args: argparse.Namespace) -> list[Candidate]:
    rows: list[Candidate] = []
    for reset_pitch_deg in args.reset_pitch_deg_values:
        for reset_flap_hz in args.reset_flap_hz_values:
            for reset_elevon_pitch_deg in args.reset_elevon_pitch_deg_values:
                rows.append(
                    Candidate(
                        reset_pitch_deg=float(reset_pitch_deg),
                        reset_flap_hz=float(reset_flap_hz),
                        reset_elevon_pitch_deg=float(reset_elevon_pitch_deg),
                    )
                )
    return rows


def _score_row(summary: dict[str, Any], first3s: dict[str, float]) -> tuple[int, float, float, float]:
    done_t_s = float(summary["steps_completed"]) * float(summary["env_dt_s"])
    survived_3s = 0 if done_t_s >= 3.0 else 1
    return (
        survived_3s,
        -done_t_s,
        abs(float(first3s["min_height_error_first3s"])),
        float(first3s["max_abs_pitch_first3s"]),
    )


def main() -> None:
    args = build_sweep_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.out_root / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for candidate in _candidate_rows(args):
        candidate_dir = run_root / candidate.label
        cmd = build_rollout_command(
            args,
            out_dir=candidate_dir,
            reset_pitch_deg=candidate.reset_pitch_deg,
            reset_flap_hz=candidate.reset_flap_hz,
            reset_elevon_pitch_deg=candidate.reset_elevon_pitch_deg,
        )
        return_code = _run_cmd(cmd, cwd=repo_root, dry_run=bool(args.dry_run))
        row: dict[str, Any] = {
            "candidate_label": candidate.label,
            "reset_pitch_deg": candidate.reset_pitch_deg,
            "reset_flap_hz": candidate.reset_flap_hz,
            "reset_elevon_pitch_deg": candidate.reset_elevon_pitch_deg,
            "return_code": int(return_code),
        }
        if return_code == 0 and not args.dry_run:
            summary = _load_summary(candidate_dir)
            first3s = _first3s_metrics(candidate_dir)
            row.update(summary)
            row.update(first3s)
            rank_key = _score_row(summary, first3s)
            row["rank_survived_3s"] = int(rank_key[0])
            row["rank_done_t_s"] = -float(rank_key[1])
            row["rank_abs_min_height_error_first3s"] = float(rank_key[2])
            row["rank_max_abs_pitch_first3s"] = float(rank_key[3])
        rows.append(row)

    leaderboard = sorted(
        rows,
        key=lambda row: (
            int(row.get("rank_survived_3s", 1)),
            -float(row.get("rank_done_t_s", 0.0)),
            float(row.get("rank_abs_min_height_error_first3s", float("inf"))),
            float(row.get("rank_max_abs_pitch_first3s", float("inf"))),
            str(row["candidate_label"]),
        ),
    )

    leaderboard_path = run_root / "leaderboard.csv"
    fieldnames: list[str] = []
    for row in leaderboard:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with leaderboard_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(leaderboard)

    print(json.dumps({"out_root": str(run_root), "leaderboard_csv": str(leaderboard_path)}, indent=2))


if __name__ == "__main__":
    main()

"""Run a controller-only grid search over tail pitch-authority balance."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Iterable, Sequence


_CANONICAL_CASES: tuple[tuple[str, int], ...] = (
    ("level_straight", 2200),
    ("level_turn", 2400),
    ("level_loiter", 2800),
)
_DEFAULT_STAGE1_FIXED_VALUES = (0.6, 0.7, 0.8, 0.9, 1.0)
_DEFAULT_STAGE1_ELEVON_VALUES = (1.0, 1.2, 1.4, 1.6, 1.8)
_DEFAULT_RANDOM_SEEDS = (11, 23, 37, 53, 71, 89)
_DEFAULT_HOLDOUT_RANDOM_SEEDS = (101, 113, 127, 149, 167, 191)
_DEFAULT_TASK = "Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0"


@dataclass(frozen=True)
class RolloutSpec:
    """Describe one rollout case for a candidate."""

    phase: str | None
    mission_mode: str
    mission_seed: int | None
    mission_label: str
    steps: int
    group: str


def build_search_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the tail-balance grid search."""
    parser = argparse.ArgumentParser(description="Search tail-balance parameters on controller-only path tracking.")
    parser.add_argument("--task", type=str, default=_DEFAULT_TASK)
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
    parser.add_argument("--nominal_speed_mps", type=float, default=7.0)
    parser.add_argument("--completion_margin_s", type=float, default=2.0)
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
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("logs/flapping_px4/tail_balance_grid_search"),
    )
    parser.add_argument("--baseline_fixed_horizontal_effectiveness", type=float, default=1.0)
    parser.add_argument("--baseline_elevon_effectiveness", type=float, default=1.0)
    parser.add_argument(
        "--stage1_fixed_values",
        type=float,
        nargs="+",
        default=list(_DEFAULT_STAGE1_FIXED_VALUES),
    )
    parser.add_argument(
        "--stage1_elevon_values",
        type=float,
        nargs="+",
        default=list(_DEFAULT_STAGE1_ELEVON_VALUES),
    )
    parser.add_argument("--top_k_refine", type=int, default=4)
    parser.add_argument("--refine_fixed_step", type=float, default=0.05)
    parser.add_argument("--refine_elevon_step", type=float, default=0.10)
    parser.add_argument("--refine_fixed_bounds", type=float, nargs=2, default=[0.55, 1.05])
    parser.add_argument("--refine_elevon_bounds", type=float, nargs=2, default=[0.90, 1.90])
    parser.add_argument("--random_steps", type=int, default=2600)
    parser.add_argument("--random_seeds", type=int, nargs="+", default=list(_DEFAULT_RANDOM_SEEDS))
    parser.add_argument("--holdout_top_k", type=int, default=3)
    parser.add_argument("--holdout_random_seeds", type=int, nargs="+", default=list(_DEFAULT_HOLDOUT_RANDOM_SEEDS))
    parser.add_argument("--straight_height_guard_scale", type=float, default=1.05)
    parser.add_argument("--random_completion_rate_drop_tol", type=float, default=0.05)
    parser.add_argument("--freq_sat_threshold_hz", type=float, default=4.9)
    parser.add_argument("--pitch_sat_threshold", type=float, default=-0.98)
    return parser


def build_stage1_grid(*, fixed_values: Sequence[float], elevon_values: Sequence[float]) -> list[tuple[float, float]]:
    """Return the ordered Cartesian product for the coarse grid."""
    pairs: list[tuple[float, float]] = []
    for fixed in fixed_values:
        for elevon in elevon_values:
            pairs.append((float(fixed), float(elevon)))
    return pairs


def build_refinement_grid(
    *,
    top_candidates: Sequence[tuple[float, float]],
    fixed_step: float,
    elevon_step: float,
    fixed_bounds: tuple[float, float],
    elevon_bounds: tuple[float, float],
    existing_pairs: set[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Build a local refinement grid around top candidates."""
    existing = set() if existing_pairs is None else {(_round_pair_key(a), _round_pair_key(b)) for a, b in existing_pairs}
    pairs: set[tuple[float, float]] = set()
    for fixed_center, elevon_center in top_candidates:
        fixed_values = {
            _clamp(float(fixed_center) - float(fixed_step), fixed_bounds),
            _clamp(float(fixed_center), fixed_bounds),
            _clamp(float(fixed_center) + float(fixed_step), fixed_bounds),
        }
        elevon_values = {
            _clamp(float(elevon_center) - float(elevon_step), elevon_bounds),
            _clamp(float(elevon_center), elevon_bounds),
            _clamp(float(elevon_center) + float(elevon_step), elevon_bounds),
        }
        for fixed in fixed_values:
            for elevon in elevon_values:
                key = (_round_pair_key(fixed), _round_pair_key(elevon))
                if key in existing:
                    continue
                pairs.add((key[0], key[1]))
    return sorted(pairs)


def evaluate_hard_constraints(
    candidate: dict[str, float | bool],
    baseline: dict[str, float | bool],
    *,
    straight_height_guard_scale: float = 1.05,
    random_completion_rate_drop_tol: float = 0.05,
) -> dict[str, Any]:
    """Evaluate the agreed hard constraints for a candidate."""
    failed_constraints: list[str] = []
    if bool(candidate.get("canonical_early_failure", False)):
        failed_constraints.append("canonical_early_failure_guard")

    baseline_straight_completed = bool(baseline.get("straight_completed", False))
    candidate_straight_completed = bool(candidate.get("straight_completed", False))
    if baseline_straight_completed and not candidate_straight_completed:
        failed_constraints.append("straight_completion_guard")

    baseline_straight_height = float(baseline.get("straight_height_error", float("inf")))
    candidate_straight_height = float(candidate.get("straight_height_error", float("inf")))
    if candidate_straight_height > float(straight_height_guard_scale) * baseline_straight_height:
        failed_constraints.append("straight_height_guard")

    baseline_random_completion = float(baseline.get("random_completion_rate", 0.0))
    candidate_random_completion = float(candidate.get("random_completion_rate", 0.0))
    if candidate_random_completion < baseline_random_completion - float(random_completion_rate_drop_tol):
        failed_constraints.append("random_completion_guard")

    return {
        "feasible": not failed_constraints,
        "failed_constraints": failed_constraints,
    }


def score_candidate(candidate: dict[str, float], baseline: dict[str, float]) -> float:
    """Score a feasible candidate against the agreed weighted objective."""
    turn_height_gain = _improvement_to_minimize(
        baseline=float(baseline["turn_height_error"]),
        candidate=float(candidate["turn_height_error"]),
    )
    loiter_height_gain = _improvement_to_minimize(
        baseline=float(baseline["loiter_height_error"]),
        candidate=float(candidate["loiter_height_error"]),
    )
    random_height_gain = _improvement_to_minimize(
        baseline=float(baseline["random_height_error"]),
        candidate=float(candidate["random_height_error"]),
    )
    random_progress_gain = _improvement_to_maximize(
        baseline=float(baseline["random_progress"]),
        candidate=float(candidate["random_progress"]),
    )
    early_height_gain = _improvement_to_minimize(
        baseline=float(baseline.get("canonical_early_height_drop_m", 0.0)),
        candidate=float(candidate.get("canonical_early_height_drop_m", 0.0)),
    )
    canonical_done_t_gain = _improvement_to_maximize(
        baseline=float(baseline.get("canonical_done_t_s", 0.0)),
        candidate=float(candidate.get("canonical_done_t_s", 0.0)),
    )
    straight_margin_den = max(1.05 * float(baseline["straight_height_error"]), 1.0e-6)
    straight_margin = (1.05 * float(baseline["straight_height_error"]) - float(candidate["straight_height_error"])) / (
        straight_margin_den
    )
    canonical_terminated_frac = float(candidate.get("canonical_terminated_frac", 0.0))
    saturation_penalty = 0.5 * float(candidate.get("freq_sat_frac", 0.0)) + 0.5 * float(
        candidate.get("pitch_sat_frac", 0.0)
    )
    return (
        0.35 * turn_height_gain
        + 0.35 * loiter_height_gain
        + 0.15 * random_height_gain
        + 0.10 * random_progress_gain
        + 0.05 * straight_margin
        + 0.05 * early_height_gain
        + 0.03 * canonical_done_t_gain
        - 0.03 * canonical_terminated_frac
        - 0.05 * saturation_penalty
    )


def compute_saturation_fractions(
    traj_path: Path,
    *,
    freq_sat_threshold_hz: float = 4.9,
    pitch_sat_threshold: float = -0.98,
) -> dict[str, float]:
    """Compute rollout saturation fractions from an existing trajectory CSV."""
    with traj_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"freq_sat_frac": 0.0, "pitch_sat_frac": 0.0}
    freq_hits = 0
    pitch_hits = 0
    for row in rows:
        if float(row["freq_hz"]) >= float(freq_sat_threshold_hz):
            freq_hits += 1
        if float(row["action_elevon_pitch"]) <= float(pitch_sat_threshold):
            pitch_hits += 1
    row_count = float(len(rows))
    return {
        "freq_sat_frac": float(freq_hits / row_count),
        "pitch_sat_frac": float(pitch_hits / row_count),
    }


def build_rollout_command(
    args: argparse.Namespace,
    *,
    phase: str | None,
    mission_mode: str,
    mission_seed: int | None,
    mission_label: str,
    steps: int,
    out_dir: Path,
    tail_fixed_horizontal_effectiveness: float,
    tail_elevon_effectiveness: float,
) -> list[str]:
    """Build a controller-only rollout command for one canonical or random case."""
    cmd = [
        "./isaaclab.sh",
        "-p",
        "scripts/flapping_px4/fly_path_mission.py",
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--steps",
        str(int(steps)),
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
        "--nominal_speed_mps",
        str(args.nominal_speed_mps),
        "--completion_margin_s",
        str(args.completion_margin_s),
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
        "--tail_fixed_horizontal_effectiveness",
        str(float(tail_fixed_horizontal_effectiveness)),
        "--tail_elevon_effectiveness",
        str(float(tail_elevon_effectiveness)),
        "--out_dir",
        str(out_dir),
        "--print_every",
        str(args.print_every),
        "--mission_label",
        mission_label,
    ]
    if mission_mode == "canonical":
        if phase is None:
            raise ValueError("canonical rollout requires a phase")
        cmd.extend(["--phase", phase])
    elif mission_mode == "random":
        if mission_seed is None:
            raise ValueError("random rollout requires a mission_seed")
        cmd.extend(
            [
                "--mission_mode",
                "random",
                "--mission_seed",
                str(int(mission_seed)),
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
            ]
        )
    else:
        raise ValueError(f"unsupported mission_mode: {mission_mode}")
    if args.headless:
        cmd.append("--headless")
    return cmd


def rank_candidate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidates with feasible rows first, then score descending."""
    return sorted(
        rows,
        key=lambda row: (
            0 if bool(row.get("feasible", False)) else 1,
            -float(row.get("score", float("-inf")) if row.get("score") is not None else float("-inf")),
            str(row.get("candidate_label", "")),
        ),
    )


def _canonical_specs() -> list[RolloutSpec]:
    return [
        RolloutSpec(phase=phase, mission_mode="canonical", mission_seed=None, mission_label=phase, steps=steps, group="canonical")
        for phase, steps in _CANONICAL_CASES
    ]


def _random_specs(*, seeds: Sequence[int], steps: int, group: str) -> list[RolloutSpec]:
    return [
        RolloutSpec(
            phase=None,
            mission_mode="random",
            mission_seed=int(seed),
            mission_label=f"{group}_seed_{int(seed):06d}",
            steps=int(steps),
            group=group,
        )
        for seed in seeds
    ]


def _candidate_label(fixed_horizontal_effectiveness: float, elevon_effectiveness: float) -> str:
    return (
        f"fixed_{fixed_horizontal_effectiveness:.2f}".replace(".", "p")
        + "__"
        + f"elevon_{elevon_effectiveness:.2f}".replace(".", "p")
    )


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float(min(max(value, lower), upper))


def _round_pair_key(value: float) -> float:
    return round(float(value), 6)


def _improvement_to_minimize(*, baseline: float, candidate: float) -> float:
    return (float(baseline) - float(candidate)) / max(abs(float(baseline)), 1.0e-6)


def _improvement_to_maximize(*, baseline: float, candidate: float) -> float:
    return (float(candidate) - float(baseline)) / max(abs(float(baseline)), 1.0e-6)


def _baseline_metrics_from_row(row: dict[str, Any]) -> dict[str, float | bool]:
    """Extract the baseline metrics required by constraints and scoring."""
    return {
        "straight_completed": bool(row.get("straight_completed", False)),
        "straight_height_error": float(row.get("straight_height_error", float("inf"))),
        "turn_height_error": float(row.get("turn_height_error", float("inf"))),
        "loiter_height_error": float(row.get("loiter_height_error", float("inf"))),
        "random_completion_rate": float(row.get("random_completion_rate", 0.0)),
        "random_height_error": float(row.get("random_height_error", float("inf"))),
        "random_progress": float(row.get("random_progress", 0.0)),
        "canonical_done_t_s": float(row.get("canonical_done_t_s", 0.0)),
        "canonical_early_height_drop_m": float(row.get("canonical_early_height_drop_m", 0.0)),
    }


def _run_cmd(cmd: Sequence[str], *, cwd: Path, dry_run: bool) -> int:
    print(f"\n[run] {shlex.join(cmd)}")
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(completed.returncode)


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


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_metric(summary: dict[str, Any], preferred_key: str, fallback_key: str) -> float:
    if preferred_key in summary:
        return float(summary[preferred_key])
    return float(summary[fallback_key])


def _aggregate_random(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return float("nan")
    return float(sum(items) / len(items))


def _build_candidate_rollout_specs(args: argparse.Namespace, *, holdout: bool = False) -> list[RolloutSpec]:
    specs = _canonical_specs()
    if holdout:
        specs.extend(_random_specs(seeds=args.holdout_random_seeds, steps=args.random_steps, group="holdout"))
    else:
        specs.extend(_random_specs(seeds=args.random_seeds, steps=args.random_steps, group="random"))
    return specs


def _collect_rollout_metrics(
    *,
    summary: dict[str, Any],
    traj_path: Path,
    freq_sat_threshold_hz: float,
    pitch_sat_threshold: float,
    metrics_warmup_s: float,
) -> dict[str, Any]:
    saturation = compute_saturation_fractions(
        traj_path,
        freq_sat_threshold_hz=freq_sat_threshold_hz,
        pitch_sat_threshold=pitch_sat_threshold,
    )
    with traj_path.open(newline="") as f:
        traj_rows = list(csv.DictReader(f))
    done_t_s = float(traj_rows[-1]["t"]) if traj_rows else float("nan")
    early_rows = [row for row in traj_rows if float(row["t"]) <= float(metrics_warmup_s)]
    min_height_error_first3s = (
        min(float(row["height_error_m"]) for row in early_rows) if early_rows else float("nan")
    )
    failure_kind = summary.get("failure_kind")
    row: dict[str, Any] = {
        "completed_path": bool(summary.get("completed_path", False)),
        "final_progress_ratio": float(summary.get("final_progress_ratio", 0.0)),
        "failure_kind": None if failure_kind is None else str(failure_kind),
        "steps_completed": int(summary.get("steps_completed", len(traj_rows))),
        "done_t_s": done_t_s,
        "min_height_error_first3s": min_height_error_first3s,
        "terminated_before_warmup": bool(
            failure_kind == "terminated" and done_t_s < float(metrics_warmup_s)
        ),
        "mean_abs_height_error_post_warmup_m": _summary_metric(
            summary,
            "mean_abs_height_error_post_warmup_m",
            "mean_abs_height_error_m",
        ),
        "mean_abs_lateral_error_post_warmup_m": _summary_metric(
            summary,
            "mean_abs_lateral_error_post_warmup_m",
            "mean_abs_lateral_error_m",
        ),
        "mean_abs_align_error_post_warmup_deg": _summary_metric(
            summary,
            "mean_abs_align_error_post_warmup_deg",
            "mean_abs_align_error_deg",
        ),
    }
    row.update(saturation)
    return row


def _run_candidate(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    search_root: Path,
    stage: str,
    fixed_horizontal_effectiveness: float,
    elevon_effectiveness: float,
    rollout_specs: Sequence[RolloutSpec],
    freq_sat_threshold_hz: float,
    pitch_sat_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_label = _candidate_label(fixed_horizontal_effectiveness, elevon_effectiveness)
    candidate_root = search_root / "runs" / stage / candidate_label
    candidate_root.mkdir(parents=True, exist_ok=True)

    rollout_rows: list[dict[str, Any]] = []
    canonical_metrics: dict[str, dict[str, Any]] = {}
    random_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    rollout_failure = False

    for spec in rollout_specs:
        out_dir = candidate_root / spec.group
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_rollout_command(
            args,
            phase=spec.phase,
            mission_mode=spec.mission_mode,
            mission_seed=spec.mission_seed,
            mission_label=spec.mission_label,
            steps=spec.steps,
            out_dir=out_dir,
            tail_fixed_horizontal_effectiveness=fixed_horizontal_effectiveness,
            tail_elevon_effectiveness=elevon_effectiveness,
        )
        return_code = _run_cmd(cmd, cwd=repo_root, dry_run=bool(args.dry_run))
        mission_root = out_dir / spec.mission_label
        run_dir = _latest_subdir(mission_root)
        row: dict[str, Any] = {
            "candidate_label": candidate_label,
            "stage": stage,
            "phase": spec.phase,
            "mission_mode": spec.mission_mode,
            "mission_group": spec.group,
            "mission_seed": spec.mission_seed,
            "mission_label": spec.mission_label,
            "steps": int(spec.steps),
            "tail_fixed_horizontal_effectiveness": float(fixed_horizontal_effectiveness),
            "tail_elevon_effectiveness": float(elevon_effectiveness),
            "return_code": int(return_code),
            "run_dir": None if run_dir is None else str(run_dir),
        }
        if not args.dry_run and return_code == 0 and run_dir is not None:
            summary = _load_json(run_dir / "summary.json")
            rollout_metrics = _collect_rollout_metrics(
                summary=summary,
                traj_path=run_dir / "trajectory_env0.csv",
                freq_sat_threshold_hz=freq_sat_threshold_hz,
                pitch_sat_threshold=pitch_sat_threshold,
                metrics_warmup_s=float(args.metrics_warmup_s),
            )
            row.update(summary)
            row.update(rollout_metrics)
            if spec.mission_mode == "canonical" and spec.phase is not None:
                canonical_metrics[spec.phase] = rollout_metrics
            elif spec.group == "random":
                random_rows.append(rollout_metrics)
            elif spec.group == "holdout":
                holdout_rows.append(rollout_metrics)
        elif not args.dry_run:
            rollout_failure = True
        rollout_rows.append(row)
        if return_code != 0 and not args.dry_run:
            rollout_failure = True
            break

    candidate_row: dict[str, Any] = {
        "candidate_label": candidate_label,
        "stage": stage,
        "tail_fixed_horizontal_effectiveness": float(fixed_horizontal_effectiveness),
        "tail_elevon_effectiveness": float(elevon_effectiveness),
        "rollout_failure": bool(rollout_failure),
        "rollout_count": len(rollout_rows),
    }
    if not args.dry_run and not rollout_failure and canonical_metrics:
        straight_metrics = canonical_metrics["level_straight"]
        turn_metrics = canonical_metrics["level_turn"]
        loiter_metrics = canonical_metrics["level_loiter"]
        candidate_row.update(
            {
                "straight_completed": bool(straight_metrics["completed_path"]),
                "straight_height_error": float(straight_metrics["mean_abs_height_error_post_warmup_m"]),
                "turn_height_error": float(turn_metrics["mean_abs_height_error_post_warmup_m"]),
                "loiter_height_error": float(loiter_metrics["mean_abs_height_error_post_warmup_m"]),
                "straight_failure_kind": straight_metrics.get("failure_kind"),
                "turn_failure_kind": turn_metrics.get("failure_kind"),
                "loiter_failure_kind": loiter_metrics.get("failure_kind"),
                "straight_done_t_s": float(straight_metrics["done_t_s"]),
                "turn_done_t_s": float(turn_metrics["done_t_s"]),
                "loiter_done_t_s": float(loiter_metrics["done_t_s"]),
                "straight_min_height_error_first3s": float(straight_metrics["min_height_error_first3s"]),
                "turn_min_height_error_first3s": float(turn_metrics["min_height_error_first3s"]),
                "loiter_min_height_error_first3s": float(loiter_metrics["min_height_error_first3s"]),
                "canonical_early_failure": any(
                    bool(metrics["terminated_before_warmup"]) for metrics in canonical_metrics.values()
                ),
                "canonical_done_t_s": _aggregate_random(
                    float(metrics["done_t_s"]) for metrics in canonical_metrics.values()
                ),
                "canonical_early_height_drop_m": _aggregate_random(
                    abs(float(metrics["min_height_error_first3s"])) for metrics in canonical_metrics.values()
                ),
                "canonical_terminated_frac": _aggregate_random(
                    1.0 if metrics.get("failure_kind") == "terminated" else 0.0
                    for metrics in canonical_metrics.values()
                ),
                "random_completion_rate": _aggregate_random(
                    1.0 if bool(row["completed_path"]) else 0.0 for row in random_rows
                ),
                "random_height_error": _aggregate_random(
                    row["mean_abs_height_error_post_warmup_m"] for row in random_rows
                ),
                "random_progress": _aggregate_random(row["final_progress_ratio"] for row in random_rows),
                "random_lateral_error": _aggregate_random(
                    row["mean_abs_lateral_error_post_warmup_m"] for row in random_rows
                ),
                "freq_sat_frac": _aggregate_random(row["freq_sat_frac"] for row in (random_rows + list(canonical_metrics.values()))),
                "pitch_sat_frac": _aggregate_random(
                    row["pitch_sat_frac"] for row in (random_rows + list(canonical_metrics.values()))
                ),
            }
        )
        if holdout_rows:
            candidate_row.update(
                {
                    "holdout_completion_rate": _aggregate_random(
                        1.0 if bool(row["completed_path"]) else 0.0 for row in holdout_rows
                    ),
                    "holdout_height_error": _aggregate_random(
                        row["mean_abs_height_error_post_warmup_m"] for row in holdout_rows
                    ),
                    "holdout_progress": _aggregate_random(row["final_progress_ratio"] for row in holdout_rows),
                }
            )
    return candidate_row, rollout_rows


def _count_constraint_failures(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for name in row.get("failed_constraints", []):
            counts[name] = counts.get(name, 0) + 1
    return counts


def _write_summary_md(
    *,
    path: Path,
    baseline_row: dict[str, Any],
    ranked_rows: Sequence[dict[str, Any]],
    holdout_rows: Sequence[dict[str, Any]],
    failed_constraint_counts: dict[str, int],
) -> None:
    best_non_baseline = next((row for row in ranked_rows if row["candidate_label"] != baseline_row["candidate_label"]), None)
    top_rows = [row for row in ranked_rows if row["candidate_label"] != baseline_row["candidate_label"]][:3]
    lines = [
        "# Tail Balance Grid Search Summary",
        "",
        "## Baseline",
        "",
        f"- candidate: `{baseline_row['candidate_label']}`",
        f"- straight height error: `{baseline_row.get('straight_height_error', 'n/a')}`",
        f"- turn height error: `{baseline_row.get('turn_height_error', 'n/a')}`",
        f"- loiter height error: `{baseline_row.get('loiter_height_error', 'n/a')}`",
        f"- random completion rate: `{baseline_row.get('random_completion_rate', 'n/a')}`",
        "",
        "## Best Feasible Candidate",
        "",
    ]
    if best_non_baseline is None:
        lines.append("- no non-baseline feasible candidate")
    else:
        lines.extend(
            [
                f"- candidate: `{best_non_baseline['candidate_label']}`",
                f"- score: `{best_non_baseline.get('score', 'n/a')}`",
                f"- fixed effectiveness: `{best_non_baseline['tail_fixed_horizontal_effectiveness']}`",
                f"- elevon effectiveness: `{best_non_baseline['tail_elevon_effectiveness']}`",
                f"- straight height error: `{best_non_baseline.get('straight_height_error', 'n/a')}`",
                f"- turn height error: `{best_non_baseline.get('turn_height_error', 'n/a')}`",
                f"- loiter height error: `{best_non_baseline.get('loiter_height_error', 'n/a')}`",
                f"- random completion rate: `{best_non_baseline.get('random_completion_rate', 'n/a')}`",
            ]
        )
    lines.extend(["", "## Top Candidates", ""])
    if not top_rows:
        lines.append("- none")
    for row in top_rows:
        lines.append(
            f"- `{row['candidate_label']}` score={row.get('score', 'n/a')} "
            f"straight={row.get('straight_height_error', 'n/a')} "
            f"turn={row.get('turn_height_error', 'n/a')} "
            f"loiter={row.get('loiter_height_error', 'n/a')}"
        )
    lines.extend(["", "## Constraint Failures", ""])
    if not failed_constraint_counts:
        lines.append("- none")
    else:
        for name, count in sorted(failed_constraint_counts.items()):
            lines.append(f"- `{name}`: {count}")
    if holdout_rows:
        lines.extend(["", "## Holdout", ""])
        for row in holdout_rows:
            lines.append(
                f"- `{row['candidate_label']}` holdout_completion={row.get('holdout_completion_rate', 'n/a')} "
                f"holdout_height={row.get('holdout_height_error', 'n/a')} "
                f"holdout_progress={row.get('holdout_progress', 'n/a')}"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_search_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_root = args.out_root / timestamp
    search_root.mkdir(parents=True, exist_ok=True)

    baseline_pair = (
        float(args.baseline_fixed_horizontal_effectiveness),
        float(args.baseline_elevon_effectiveness),
    )
    stage1_pairs = build_stage1_grid(
        fixed_values=args.stage1_fixed_values,
        elevon_values=args.stage1_elevon_values,
    )
    stage1_pairs = [pair for pair in stage1_pairs if pair != baseline_pair]

    per_run_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "task": args.task,
        "out_root": str(search_root),
        "baseline_pair": baseline_pair,
        "stage1_pairs": stage1_pairs,
        "random_seeds": list(args.random_seeds),
        "holdout_random_seeds": list(args.holdout_random_seeds),
    }

    baseline_row, baseline_runs = _run_candidate(
        args,
        repo_root=repo_root,
        search_root=search_root,
        stage="baseline",
        fixed_horizontal_effectiveness=baseline_pair[0],
        elevon_effectiveness=baseline_pair[1],
        rollout_specs=_build_candidate_rollout_specs(args, holdout=False),
        freq_sat_threshold_hz=float(args.freq_sat_threshold_hz),
        pitch_sat_threshold=float(args.pitch_sat_threshold),
    )
    per_run_rows.extend(baseline_runs)
    candidate_rows.append(baseline_row)

    baseline_metrics = _baseline_metrics_from_row(baseline_row)
    baseline_row["feasible"] = not bool(baseline_row.get("rollout_failure", False))
    baseline_row["failed_constraints"] = []
    baseline_row["score"] = 0.0 if baseline_row["feasible"] else float("-inf")

    for fixed_horizontal_effectiveness, elevon_effectiveness in stage1_pairs:
        candidate_row, rollout_rows = _run_candidate(
            args,
            repo_root=repo_root,
            search_root=search_root,
            stage="stage1",
            fixed_horizontal_effectiveness=fixed_horizontal_effectiveness,
            elevon_effectiveness=elevon_effectiveness,
            rollout_specs=_build_candidate_rollout_specs(args, holdout=False),
            freq_sat_threshold_hz=float(args.freq_sat_threshold_hz),
            pitch_sat_threshold=float(args.pitch_sat_threshold),
        )
        per_run_rows.extend(rollout_rows)
        if not args.dry_run and not bool(candidate_row.get("rollout_failure", False)):
            constraints = evaluate_hard_constraints(
                candidate_row,
                baseline_metrics,
                straight_height_guard_scale=float(args.straight_height_guard_scale),
                random_completion_rate_drop_tol=float(args.random_completion_rate_drop_tol),
            )
            candidate_row.update(constraints)
            candidate_row["score"] = score_candidate(candidate_row, baseline_metrics) if constraints["feasible"] else float("-inf")
        else:
            candidate_row["feasible"] = False
            candidate_row["failed_constraints"] = ["dry_run"] if args.dry_run else ["rollout_failure"]
            candidate_row["score"] = float("-inf")
        candidate_rows.append(candidate_row)

    feasible_stage1 = [
        row
        for row in candidate_rows
        if row["stage"] == "stage1" and bool(row.get("feasible", False))
    ]
    top_stage1 = rank_candidate_rows(feasible_stage1)[: max(int(args.top_k_refine), 0)]
    refinement_pairs = build_refinement_grid(
        top_candidates=[
            (
                float(row["tail_fixed_horizontal_effectiveness"]),
                float(row["tail_elevon_effectiveness"]),
            )
            for row in top_stage1
        ],
        fixed_step=float(args.refine_fixed_step),
        elevon_step=float(args.refine_elevon_step),
        fixed_bounds=(float(args.refine_fixed_bounds[0]), float(args.refine_fixed_bounds[1])),
        elevon_bounds=(float(args.refine_elevon_bounds[0]), float(args.refine_elevon_bounds[1])),
        existing_pairs={baseline_pair, *stage1_pairs},
    )
    manifest["refinement_pairs"] = refinement_pairs

    for fixed_horizontal_effectiveness, elevon_effectiveness in refinement_pairs:
        candidate_row, rollout_rows = _run_candidate(
            args,
            repo_root=repo_root,
            search_root=search_root,
            stage="stage2",
            fixed_horizontal_effectiveness=fixed_horizontal_effectiveness,
            elevon_effectiveness=elevon_effectiveness,
            rollout_specs=_build_candidate_rollout_specs(args, holdout=False),
            freq_sat_threshold_hz=float(args.freq_sat_threshold_hz),
            pitch_sat_threshold=float(args.pitch_sat_threshold),
        )
        per_run_rows.extend(rollout_rows)
        if not args.dry_run and not bool(candidate_row.get("rollout_failure", False)):
            constraints = evaluate_hard_constraints(
                candidate_row,
                baseline_metrics,
                straight_height_guard_scale=float(args.straight_height_guard_scale),
                random_completion_rate_drop_tol=float(args.random_completion_rate_drop_tol),
            )
            candidate_row.update(constraints)
            candidate_row["score"] = score_candidate(candidate_row, baseline_metrics) if constraints["feasible"] else float("-inf")
        else:
            candidate_row["feasible"] = False
            candidate_row["failed_constraints"] = ["dry_run"] if args.dry_run else ["rollout_failure"]
            candidate_row["score"] = float("-inf")
        candidate_rows.append(candidate_row)

    ranked_rows = rank_candidate_rows(candidate_rows)
    feasible_non_baseline = [
        row for row in ranked_rows if bool(row.get("feasible", False)) and row["candidate_label"] != baseline_row["candidate_label"]
    ]
    holdout_rows: list[dict[str, Any]] = []
    if not args.dry_run and int(args.holdout_top_k) > 0 and args.holdout_random_seeds:
        for row in feasible_non_baseline[: int(args.holdout_top_k)]:
            holdout_row, holdout_run_rows = _run_candidate(
                args,
                repo_root=repo_root,
                search_root=search_root,
                stage="holdout",
                fixed_horizontal_effectiveness=float(row["tail_fixed_horizontal_effectiveness"]),
                elevon_effectiveness=float(row["tail_elevon_effectiveness"]),
                rollout_specs=_build_candidate_rollout_specs(args, holdout=True),
                freq_sat_threshold_hz=float(args.freq_sat_threshold_hz),
                pitch_sat_threshold=float(args.pitch_sat_threshold),
            )
            per_run_rows.extend(holdout_run_rows)
            holdout_rows.append(holdout_row)

    manifest_path = search_root / "manifest.json"
    leaderboard_path = search_root / "leaderboard.csv"
    per_run_path = search_root / "per_run_metrics.csv"
    response_surface_path = search_root / "response_surface.csv"
    summary_json_path = search_root / "summary.json"
    summary_md_path = search_root / "summary.md"

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    _write_csv(candidate_rows, leaderboard_path)
    _write_csv(per_run_rows, per_run_path)
    response_rows = [
        {
            "candidate_label": row["candidate_label"],
            "stage": row["stage"],
            "tail_fixed_horizontal_effectiveness": row["tail_fixed_horizontal_effectiveness"],
            "tail_elevon_effectiveness": row["tail_elevon_effectiveness"],
            "straight_height_error": row.get("straight_height_error"),
            "turn_height_error": row.get("turn_height_error"),
            "loiter_height_error": row.get("loiter_height_error"),
            "random_height_error": row.get("random_height_error"),
            "random_completion_rate": row.get("random_completion_rate"),
            "score": row.get("score"),
            "feasible": row.get("feasible"),
        }
        for row in candidate_rows
    ]
    _write_csv(response_rows, response_surface_path)
    failed_constraint_counts = _count_constraint_failures(candidate_rows)
    summary = {
        "baseline_candidate": baseline_row["candidate_label"],
        "baseline_metrics": baseline_row,
        "ranked_candidates": ranked_rows,
        "top_feasible_non_baseline": feasible_non_baseline[:3],
        "constraint_failures": failed_constraint_counts,
        "holdout_candidates": holdout_rows,
        "search_root": str(search_root),
    }
    with summary_json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    _write_summary_md(
        path=summary_md_path,
        baseline_row=baseline_row,
        ranked_rows=ranked_rows,
        holdout_rows=holdout_rows,
        failed_constraint_counts=failed_constraint_counts,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

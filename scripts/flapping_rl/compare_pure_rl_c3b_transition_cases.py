"""Trace matched passing and failing C3b transition cases for one checkpoint.

The parent launches four fresh CPU-native Isaac processes.  For each of the
turn-then-climb and climb-then-turn templates it compares heading 0 (passing in
the frozen grid) with heading 180 (failing), while fixing every other case
parameter.  Outputs are diagnostic only and do not alter the promotion gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = Path(__file__).resolve()
_TASK = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3b-Direct-v0"
_NATIVE_EXTENSION_ID = "omni.flapping_bot.holonomic_constraint"
_ACTION_NAMES = ("flap_frequency", "rudder", "left_elevon", "right_elevon")
_CASE_SPECS = (
    ("turn_then_climb_control_h0", 5, 0.0, "control"),
    ("turn_then_climb_failure_h180", 5, 180.0, "failure"),
    ("climb_then_turn_control_h0", 7, 0.0, "control"),
    ("climb_then_turn_failure_h180", 7, 180.0, "failure"),
)


def _parse_parent_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slope-deg", type=float, default=12.0)
    parser.add_argument("--geometry-roll-deg", type=float, default=13.5)
    parser.add_argument("--turn-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--flap-phase-deg", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _parse_worker_args(argv: Sequence[str]) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Internal fixed C3b case trace worker.")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--expected-result", choices=("control", "failure"), required=True)
    parser.add_argument("--template-id", type=int, choices=(5, 7), required=True)
    parser.add_argument("--heading-deg", type=float, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--slope-deg", type=float, required=True)
    parser.add_argument("--geometry-roll-deg", type=float, required=True)
    parser.add_argument("--turn-sign", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--flap-phase-deg", type=float, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


def _validate_condition(args: argparse.Namespace) -> None:
    if not math.isfinite(float(args.slope_deg)) or not 0.0 < float(args.slope_deg) <= 15.0:
        raise ValueError("--slope-deg must be finite and within (0, 15].")
    if not math.isfinite(float(args.geometry_roll_deg)) or not 0.0 < float(args.geometry_roll_deg) <= 30.0:
        raise ValueError("--geometry-roll-deg must be finite and within (0, 30].")
    if float(args.duration_s) <= 0.0:
        raise ValueError("--duration-s must be positive.")


def _worker_command(
    *,
    checkpoint: Path,
    case_spec: tuple[str, int, float, str],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build one fresh-process command for an exact C3b case."""

    case_id, template_id, heading_deg, expected_result = case_spec
    return [
        sys.executable,
        str(_SCRIPT_PATH),
        "--worker",
        "--checkpoint",
        str(checkpoint),
        "--case-id",
        case_id,
        "--expected-result",
        expected_result,
        "--template-id",
        str(template_id),
        "--heading-deg",
        str(heading_deg),
        "--trace-output",
        str(output_dir / f"{case_id}_trace.csv"),
        "--summary-output",
        str(output_dir / f"{case_id}_summary.json"),
        "--slope-deg",
        str(float(args.slope_deg)),
        "--geometry-roll-deg",
        str(float(args.geometry_roll_deg)),
        "--turn-sign",
        str(int(args.turn_sign)),
        "--flap-phase-deg",
        str(float(args.flap_phase_deg)),
        "--duration-s",
        str(float(args.duration_s)),
        "--seed",
        str(int(args.seed)),
        "--device",
        "cpu",
        "--headless",
    ]


def _child_environment() -> dict[str, str]:
    child_env = os.environ.copy()
    local_sources = [
        str((_REPO_ROOT / "source/flapping_bot").resolve()),
        str((_REPO_ROOT / "source/isaaclab_assets").resolve()),
        str((_REPO_ROOT / "source/isaaclab_tasks").resolve()),
    ]
    if child_env.get("PYTHONPATH"):
        local_sources.append(child_env["PYTHONPATH"])
    child_env["PYTHONPATH"] = os.pathsep.join(local_sources)
    child_env["TERM"] = "xterm"
    return child_env


def _read_trace(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    if not raw_rows:
        raise ValueError(f"Trace is empty: {path}")
    text_columns = {"case_id", "expected_result", "phase"}
    return [
        {
            key: value if key in text_columns else float(value)
            for key, value in row.items()
        }
        for row in raw_rows
    ]


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _first_index(rows: Sequence[Mapping[str, float | str]], column: str, threshold: float) -> int | None:
    return next(
        (index for index, row in enumerate(rows) if abs(float(row[column])) > threshold),
        None,
    )


def _snapshot(rows: Sequence[Mapping[str, float | str]], index: int | None) -> dict[str, float] | None:
    if index is None:
        return None
    row = rows[index]
    columns = (
        "time_s",
        "progress_m",
        "cross_track_error_m",
        "height_error_m",
        "lateral_normal_velocity_mps",
        "vertical_normal_velocity_mps",
        "abs_roll_deg",
        "pitch_deg",
        "roll_rate_rad_s",
        "pitch_rate_rad_s",
        "yaw_rate_rad_s",
        "actual_flap_frequency_hz",
        "applied_flap_frequency_action",
        "applied_rudder_action",
        "applied_left_elevon_action",
        "applied_right_elevon_action",
    )
    return {column: float(row[column]) for column in columns}


def _summarize_trace(rows: Sequence[Mapping[str, float | str]]) -> dict[str, object]:
    """Summarize event timing, switch state, actions, and failure mode."""

    if not rows:
        raise ValueError("rows must not be empty.")
    times = [float(row["time_s"]) for row in rows]
    if any(next_time <= time for time, next_time in zip(times, times[1:], strict=False)):
        raise ValueError("Trace times must be strictly increasing.")
    turn_index = _first_index(rows, "active_curvature_rad_per_m", 1.0e-6)
    climb_index = _first_index(rows, "active_slope_deg", 1.0e-3)
    template_id = int(float(rows[0]["template_id"]))
    second_event_index = climb_index if template_id == 5 else turn_index
    rolls = [abs(float(row["abs_roll_deg"])) for row in rows]
    cross_errors = [abs(float(row["cross_track_error_m"])) for row in rows]
    height_errors = [abs(float(row["height_error_m"])) for row in rows]
    return {
        "case_id": str(rows[0]["case_id"]),
        "expected_result": str(rows[0]["expected_result"]),
        "template_id": template_id,
        "step_count": len(rows),
        "duration_s": times[-1],
        "turn_onset_time_s": None if turn_index is None else times[turn_index],
        "climb_onset_time_s": None if climb_index is None else times[climb_index],
        "second_event_state_action": _snapshot(rows, second_event_index),
        "reached_all_events": bool(int(float(rows[-1]["reached_all_events"]))),
        "terminated": bool(int(float(rows[-1]["terminated"]))),
        "roll_limit_termination": bool(int(float(rows[-1]["roll_limit_termination"]))),
        "ground_termination": bool(int(float(rows[-1]["ground_termination"]))),
        "tilt_termination": bool(int(float(rows[-1]["tilt_termination"]))),
        "cross_track_termination": bool(int(float(rows[-1]["cross_track_termination"]))),
        "height_termination": bool(int(float(rows[-1]["height_termination"]))),
        "max_abs_roll_deg": max(rolls),
        "p95_abs_roll_deg": _quantile(rolls, 0.95),
        "max_abs_cross_track_error_m": max(cross_errors),
        "max_abs_height_error_m": max(height_errors),
        "tail_limit_fraction": sum(float(row["tail_limit_active"]) for row in rows) / len(rows),
        "frequency_limit_fraction": sum(float(row["frequency_limit_active"]) for row in rows) / len(rows),
        "mean_normalized_action_delta": sum(float(row["normalized_action_delta"]) for row in rows) / len(rows),
    }


def _comparison_delta(control: Mapping[str, object], failure: Mapping[str, object]) -> dict[str, object]:
    control_snapshot = control.get("second_event_state_action")
    failure_snapshot = failure.get("second_event_state_action")
    deltas: dict[str, float] = {}
    if isinstance(control_snapshot, Mapping) and isinstance(failure_snapshot, Mapping):
        for key in control_snapshot:
            if key != "time_s" and key in failure_snapshot:
                deltas[key] = float(failure_snapshot[key]) - float(control_snapshot[key])
    return {
        "template_id": int(control["template_id"]),
        "control_case": str(control["case_id"]),
        "failure_case": str(failure["case_id"]),
        "failure_minus_control_at_second_event": deltas,
        "max_abs_roll_delta_deg": float(failure["max_abs_roll_deg"]) - float(control["max_abs_roll_deg"]),
        "p95_abs_roll_delta_deg": float(failure["p95_abs_roll_deg"]) - float(control["p95_abs_roll_deg"]),
        "control_reached_all_events": bool(control["reached_all_events"]),
        "failure_reached_all_events": bool(failure["reached_all_events"]),
        "control_roll_limit_termination": bool(control["roll_limit_termination"]),
        "failure_roll_limit_termination": bool(failure["roll_limit_termination"]),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_template_comparison(
    path: Path,
    control: Sequence[Mapping[str, float | str]],
    failure: Sequence[Mapping[str, float | str]],
    *,
    template_id: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True, constrained_layout=True)
    for rows, label, color in (
        (control, "heading 0: frozen-grid pass", "#0072B2"),
        (failure, "heading 180: frozen-grid fail", "#D55E00"),
    ):
        time_s = [float(row["time_s"]) for row in rows]
        axes[0].plot(time_s, [float(row["active_slope_deg"]) for row in rows], color=color, label=f"{label} slope")
        axes[0].plot(time_s, [100.0 * float(row["active_curvature_rad_per_m"]) for row in rows], color=color, linestyle=":", label=f"{label} curvature x100")
        axes[1].plot(time_s, [float(row["cross_track_error_m"]) for row in rows], color=color, label=label)
        axes[1].plot(time_s, [float(row["height_error_m"]) for row in rows], color=color, linestyle=":")
        axes[2].plot(time_s, [float(row["abs_roll_deg"]) for row in rows], color=color, label=label)
        axes[3].plot(time_s, [float(row["roll_rate_rad_s"]) for row in rows], color=color, label=label)
        axes[4].plot(time_s, [float(row["applied_flap_frequency_action"]) for row in rows], color=color, label=label)
        axes[5].plot(time_s, [float(row["applied_rudder_action"]) for row in rows], color=color, label=f"{label} rudder")
        axes[5].plot(time_s, [float(row["applied_left_elevon_action"]) - float(row["applied_right_elevon_action"]) for row in rows], color=color, linestyle=":", label=f"{label} elevon diff")
    labels = (
        "slope [deg] / curvature x100",
        "cross / height error [m]",
        "abs roll [deg]",
        "body roll rate [rad/s]",
        "frequency action",
        "tail action",
    )
    for axis, label in zip(axes, labels, strict=True):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"C3b template {template_id}: matched heading control")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _run_worker(argv: Sequence[str]) -> int:
    args = _parse_worker_args(argv)
    _validate_condition(args)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    trace_output = args.trace_output.expanduser().resolve()
    summary_output = args.summary_output.expanduser().resolve()
    if trace_output.exists() or summary_output.exists():
        raise FileExistsError("Worker outputs already exist; choose a new --output-dir.")
    trace_output.parent.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    extension_parent = (_REPO_ROOT / "source/flapping_bot/native_extensions").resolve()
    portable_root = (trace_output.parent / "portable" / args.case_id).resolve()
    required_kit_args = (
        f"--portable-root {portable_root}"
        " --/apps/extensions/fsWatcherEnabled=false"
        f" --ext-folder {(_REPO_ROOT / 'source').resolve()}"
        f" --ext-folder {extension_parent}"
        f" --enable {_NATIVE_EXTENSION_ID}"
    )
    args.device = "cpu"
    args.headless = True
    args.kit_args = " ".join(part for part in (getattr(args, "kit_args", None), required_kit_args) if part)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import torch
        from isaaclab.utils.math import euler_xyz_from_quat
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

        script_dir = str(_SCRIPT_PATH.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from pure_rl_eval_common import assert_pure_rl_spatial_reset_schedule, read_pure_rl_step_metrics

        heading_rad = math.radians(float(args.heading_deg))
        phase_rad = math.radians(float(args.flap_phase_deg))
        env_cfg = parse_env_cfg(_TASK, device="cpu", num_envs=1)
        env_cfg.seed = int(args.seed)
        env_cfg.episode_length_s = float(args.duration_s)
        env_cfg.randomize_commands = False
        env_cfg.randomize_straight_line_heading = False
        env_cfg.randomize_flap_phase_at_reset = False
        env_cfg.pure_rl_eval_heading_schedule_rad = (heading_rad,)
        env_cfg.pure_rl_eval_flap_phase_schedule_rad = (phase_rad,)
        env_cfg.pure_rl_eval_spatial_template_schedule = (int(args.template_id),)
        env_cfg.pure_rl_eval_spatial_geometry_roll_deg_schedule = (float(args.geometry_roll_deg),)
        env_cfg.pure_rl_eval_spatial_slope_deg_schedule = (float(args.slope_deg),)
        env_cfg.pure_rl_eval_spatial_turn_sign_schedule = (int(args.turn_sign),)
        env_cfg.wind_enabled = False
        env_cfg.wind_ou_enabled = False
        robot_asset = (_REPO_ROOT / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf").resolve()
        robot_usd_dir = (trace_output.parent / "generated_assets" / args.case_id / "flap_robot_552").resolve()
        robot_usd_dir.mkdir(parents=True, exist_ok=True)
        env_cfg.robot.spawn.asset_path = str(robot_asset)
        env_cfg.robot.spawn.usd_dir = str(robot_usd_dir)

        agent_cfg = load_cfg_from_registry(_TASK, "rsl_rl_cfg_entry_point")
        agent_cfg.device = "cpu"
        agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["device"] = "cpu"
        agent_cfg_dict["policy"]["class_name"] = "PureRLSplitActorCritic"
        base_env = gym.make(_TASK, cfg=env_cfg)
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg_dict.get("clip_actions"))
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device="cpu")
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        path = env.unwrapped._pure_rl_spatial_path
        assert path is not None
        assert_pure_rl_spatial_reset_schedule(
            actual_heading_rad=env.unwrapped._straight_line_heading_rad,
            actual_flap_phase_rad=env.unwrapped._phase,
            actual_template_id=path.template_id,
            actual_geometry_roll_rad=path.peak_geometry_roll_rad,
            actual_slope_rad=path.peak_slope_rad,
            actual_turn_sign=path.turn_sign,
            expected_heading_schedule_rad=(heading_rad,),
            expected_flap_phase_schedule_rad=(phase_rad,),
            expected_template_schedule=(int(args.template_id),),
            expected_geometry_roll_deg_schedule=(float(args.geometry_roll_deg),),
            expected_slope_deg_schedule=(float(args.slope_deg),),
            expected_turn_sign_schedule=(int(args.turn_sign),),
        )

        obs = env.get_observations()
        policy_nn.reset(torch.ones(1, dtype=torch.long, device=env.unwrapped.device))
        step_dt_s = float(env.unwrapped.step_dt)
        max_steps = int(round(float(args.duration_s) / step_dt_s))
        rows: list[dict[str, object]] = []
        for step in range(max_steps):
            with torch.inference_mode():
                requested_actions = policy(obs)
            obs, _reward, dones, _extras = env.step(requested_actions)
            metrics = read_pure_rl_step_metrics(env.unwrapped)
            done = bool(dones[0].item())
            terminated = bool(env.unwrapped.reset_terminated[0].item()) if done else False
            active_slope_deg = math.degrees(float(metrics["active_slope_rad"][0].item()))
            active_curvature = float(metrics["active_curvature_rad_per_m"][0].item())
            if abs(active_slope_deg) > 1.0e-3 and abs(active_curvature) > 1.0e-6:
                phase_name = "coupled"
            elif abs(active_curvature) > 1.0e-6:
                phase_name = "turn"
            elif abs(active_slope_deg) > 1.0e-3:
                phase_name = "climb"
            else:
                phase_name = "straight"
            if done:
                pitch_deg = yaw_deg = float("nan")
                rates = (float("nan"),) * 3
            else:
                _roll, pitch, yaw = euler_xyz_from_quat(env.unwrapped._robot.data.root_quat_w)
                pitch_deg = math.degrees(float(pitch[0].item()))
                yaw_deg = math.degrees(float(yaw[0].item()))
                rates = tuple(float(value) for value in env.unwrapped._robot.data.root_ang_vel_b[0].tolist())
            applied_actions = env.unwrapped._debug_last_exec_action[0].detach().cpu()
            row: dict[str, object] = {
                "case_id": args.case_id,
                "expected_result": args.expected_result,
                "template_id": int(args.template_id),
                "heading_deg": float(args.heading_deg),
                "step": step,
                "time_s": (step + 1) * step_dt_s,
                "phase": phase_name,
                "progress_m": float(metrics["along_track_progress_m"][0].item()),
                "cross_track_error_m": float(metrics["cross_track_error_m"][0].item()),
                "height_error_m": float(metrics["height_error_m"][0].item()),
                "along_track_velocity_mps": float(metrics["along_track_velocity_mps"][0].item()),
                "lateral_normal_velocity_mps": float(metrics["lateral_normal_velocity_mps"][0].item()),
                "vertical_normal_velocity_mps": float(metrics["vertical_normal_velocity_mps"][0].item()),
                "active_slope_deg": active_slope_deg,
                "active_curvature_rad_per_m": active_curvature,
                "turn_activity": float(metrics["turn_activity"][0].item()),
                "abs_roll_deg": math.degrees(float(metrics["abs_roll_rad"][0].item())),
                "pitch_deg": pitch_deg,
                "yaw_deg": yaw_deg,
                "roll_rate_rad_s": rates[0],
                "pitch_rate_rad_s": rates[1],
                "yaw_rate_rad_s": rates[2],
                "actual_flap_frequency_hz": float(metrics["actual_flap_frequency_hz"][0].item()),
                "frequency_limit_active": float(metrics["frequency_limit_active"][0].item()),
                "tail_limit_active": float(metrics["tail_limit_active"][0].item()),
                "normalized_action_delta": float(metrics["normalized_action_delta"][0].item()),
                "frequency_slew_hz_per_s": float(metrics["frequency_slew_hz_per_s"][0].item()),
                "frequency_governor_limited": float(metrics["frequency_governor_limited"][0].item()),
                "reached_all_events": int(bool(metrics["reached_all_events"][0].item())),
                "roll_limit_termination": int(bool(metrics["roll_limit_termination"][0].item())),
                "ground_termination": int(bool(metrics["ground_termination"][0].item())),
                "tilt_termination": int(bool(metrics["tilt_termination"][0].item())),
                "cross_track_termination": int(bool(metrics["cross_track_termination"][0].item())),
                "height_termination": int(bool(metrics["height_termination"][0].item())),
                "terminated": int(terminated),
                "done": int(done),
            }
            for action_index, action_name in enumerate(_ACTION_NAMES):
                row[f"requested_{action_name}_action"] = float(requested_actions[0, action_index].item())
                row[f"applied_{action_name}_action"] = float(applied_actions[action_index].item())
            row.update(
                {
                    "rudder_cmd_deg": math.degrees(float(env.unwrapped._debug_last_exec_rudder_rad[0].item())),
                    "left_elevon_cmd_deg": math.degrees(float(env.unwrapped._debug_last_exec_left_elevon_rad[0].item())),
                    "right_elevon_cmd_deg": math.degrees(float(env.unwrapped._debug_last_exec_right_elevon_rad[0].item())),
                }
            )
            rows.append(row)
            policy_nn.reset(dones)
            if done:
                break

        summary = _summarize_trace(rows)
        summary.update(
            {
                "checkpoint": str(checkpoint),
                "task": _TASK,
                "condition": {
                    "heading_deg": float(args.heading_deg),
                    "flap_phase_deg": float(args.flap_phase_deg),
                    "template_id": int(args.template_id),
                    "geometry_roll_deg": float(args.geometry_roll_deg),
                    "slope_deg": float(args.slope_deg),
                    "turn_sign": int(args.turn_sign),
                    "duration_s": float(args.duration_s),
                    "seed": int(args.seed),
                },
            }
        )
        _write_csv(trace_output, rows)
        summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] Wrote C3b trace: {trace_output}", flush=True)
        env.close()
        return 0
    finally:
        simulation_app.close()


def _run_parent(argv: Sequence[str] | None = None) -> int:
    args = _parse_parent_args(argv)
    _validate_condition(args)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    child_env = _child_environment()
    for case_spec in _CASE_SPECS:
        command = _worker_command(
            checkpoint=checkpoint,
            case_spec=case_spec,
            output_dir=output_dir,
            args=args,
        )
        print(f"[INFO] Tracing {case_spec[0]} in a fresh CPU-native process:", flush=True)
        print(" ", " ".join(command), flush=True)
        subprocess.run(command, env=child_env, check=True)

    traces = {
        case_id: _read_trace(output_dir / f"{case_id}_trace.csv")
        for case_id, _template_id, _heading_deg, _expected_result in _CASE_SPECS
    }
    summaries = {case_id: _summarize_trace(rows) for case_id, rows in traces.items()}
    comparisons = []
    for template_name, template_id in (("turn_then_climb", 5), ("climb_then_turn", 7)):
        control_id = f"{template_name}_control_h0"
        failure_id = f"{template_name}_failure_h180"
        comparisons.append(_comparison_delta(summaries[control_id], summaries[failure_id]))
        _plot_template_comparison(
            output_dir / f"template_{template_id}_comparison.png",
            traces[control_id],
            traces[failure_id],
            template_id=template_id,
        )
    report = {
        "checkpoint": str(checkpoint),
        "task": _TASK,
        "matched_condition": {
            "slope_deg": float(args.slope_deg),
            "geometry_roll_deg": float(args.geometry_roll_deg),
            "turn_sign": int(args.turn_sign),
            "flap_phase_deg": float(args.flap_phase_deg),
            "duration_s": float(args.duration_s),
            "seed": int(args.seed),
        },
        "case_summaries": summaries,
        "template_comparisons": comparisons,
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] C3b matched-case comparison complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    if "--worker" in sys.argv[1:]:
        raise SystemExit(_run_worker(sys.argv[1:]))
    raise SystemExit(_run_parent(sys.argv[1:]))

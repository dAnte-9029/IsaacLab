"""Compare two PureRL C2c policies on one identical fixed trajectory.

The parent process launches one fresh CPU-native Isaac process per checkpoint,
then aligns their 60 Hz traces and writes CSV, JSON, and PNG diagnostics.  The
frozen promotion evaluator is not modified or reused as a tuning loop.
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
_TASK = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C2c-Direct-v0"
_NATIVE_EXTENSION_ID = "omni.flapping_bot.holonomic_constraint"
_ACTION_NAMES = ("flap_frequency", "rudder", "left_elevon", "right_elevon")


def _parse_parent_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slope-deg", type=float, default=12.0)
    parser.add_argument("--heading-deg", type=float, default=0.0)
    parser.add_argument("--flap-phase-deg", type=float, default=0.0)
    parser.add_argument("--entry-length-m", type=float, default=17.5)
    parser.add_argument("--slope-length-m", type=float, default=25.0)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _parse_worker_args(argv: Sequence[str]) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Internal fixed-case trace worker.")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--slope-deg", type=float, required=True)
    parser.add_argument("--heading-deg", type=float, required=True)
    parser.add_argument("--flap-phase-deg", type=float, required=True)
    parser.add_argument("--entry-length-m", type=float, required=True)
    parser.add_argument("--slope-length-m", type=float, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


def _validate_condition(args: argparse.Namespace) -> None:
    if not math.isfinite(float(args.slope_deg)) or abs(float(args.slope_deg)) > 15.0:
        raise ValueError("--slope-deg must be finite and within [-15, 15].")
    if float(args.entry_length_m) <= 0.0 or float(args.slope_length_m) <= 0.0:
        raise ValueError("Path lengths must be positive.")
    if float(args.duration_s) <= 0.0:
        raise ValueError("--duration-s must be positive.")


def _worker_command(
    *,
    checkpoint: Path,
    label: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build one fresh-process fixed-case trace command."""

    return [
        sys.executable,
        str(_SCRIPT_PATH),
        "--worker",
        "--checkpoint",
        str(checkpoint),
        "--label",
        label,
        "--trace-output",
        str(output_dir / f"{label}_trace.csv"),
        "--summary-output",
        str(output_dir / f"{label}_summary.json"),
        "--slope-deg",
        str(float(args.slope_deg)),
        "--heading-deg",
        str(float(args.heading_deg)),
        "--flap-phase-deg",
        str(float(args.flap_phase_deg)),
        "--entry-length-m",
        str(float(args.entry_length_m)),
        "--slope-length-m",
        str(float(args.slope_length_m)),
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
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Trace is empty: {path}")
    numeric_columns = set(rows[0]).difference({"label", "phase"})
    converted: list[dict[str, float | str]] = []
    for row in rows:
        converted.append(
            {
                key: (value if key in {"label", "phase"} else float(value))
                for key, value in row.items()
                if key not in numeric_columns or value != ""
            }
        )
    return converted


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


def _first_sustained_response(
    rows: Sequence[Mapping[str, float | str]],
    *,
    start_index: int,
    fraction: float = 0.90,
    consecutive_steps: int = 5,
) -> float | None:
    for index in range(start_index, len(rows) - consecutive_steps + 1):
        window = rows[index : index + consecutive_steps]
        reached = []
        for row in window:
            desired = float(row["desired_vertical_velocity_mps"])
            actual = float(row["vertical_velocity_mps"])
            if abs(desired) < 1.0e-6:
                reached.append(False)
            elif desired > 0.0:
                reached.append(actual >= fraction * desired)
            else:
                reached.append(actual <= fraction * desired)
        if all(reached):
            return float(rows[index]["time_s"]) - float(rows[start_index]["time_s"])
    return None


def _phase_mean(rows: Sequence[Mapping[str, float | str]], phase: str, column: str) -> float:
    values = [float(row[column]) for row in rows if row["phase"] == phase]
    return sum(values) / len(values) if values else float("nan")


def _summarize_trace(rows: Sequence[Mapping[str, float | str]]) -> dict[str, object]:
    """Summarize path accuracy, response timing, and action usage for one trace."""

    if not rows:
        raise ValueError("rows must not be empty.")
    times = [float(row["time_s"]) for row in rows]
    if any(next_time <= time for time, next_time in zip(times, times[1:], strict=False)):
        raise ValueError("Trace times must be strictly increasing.")
    slope_indices = [index for index, row in enumerate(rows) if row["phase"] == "slope"]
    if not slope_indices:
        raise ValueError("Trace never reached the slope phase.")
    slope_start = slope_indices[0]
    slope_rows = [rows[index] for index in slope_indices]
    errors = [float(row["height_error_m"]) for row in rows]
    slope_errors = [float(row["height_error_m"]) for row in slope_rows]
    worst_slope_index = max(slope_indices, key=lambda index: abs(float(rows[index]["height_error_m"])))
    response_delay = _first_sustained_response(rows, start_index=slope_start)
    summary: dict[str, object] = {
        "label": str(rows[0]["label"]),
        "step_count": len(rows),
        "duration_s": times[-1],
        "terminated": bool(int(float(rows[-1]["terminated"]))),
        "slope_onset_time_s": float(rows[slope_start]["time_s"]),
        "vertical_velocity_90pct_response_delay_s": response_delay,
        "height_error_at_slope_onset_m": float(rows[slope_start]["height_error_m"]),
        "mean_abs_height_error_m": sum(abs(value) for value in errors) / len(errors),
        "p95_abs_height_error_m": _quantile([abs(value) for value in errors], 0.95),
        "slope_mean_height_error_m": sum(slope_errors) / len(slope_errors),
        "slope_mean_abs_height_error_m": sum(abs(value) for value in slope_errors) / len(slope_errors),
        "slope_worst_abs_height_error_m": abs(float(rows[worst_slope_index]["height_error_m"])),
        "slope_worst_error_time_after_onset_s": (
            float(rows[worst_slope_index]["time_s"]) - float(rows[slope_start]["time_s"])
        ),
        "entry_mean_height_error_m": _phase_mean(rows, "entry", "height_error_m"),
        "recovery_mean_height_error_m": _phase_mean(rows, "recovery", "height_error_m"),
        "slope_mean_vertical_velocity_mps": _phase_mean(rows, "slope", "vertical_velocity_mps"),
        "slope_mean_desired_vertical_velocity_mps": _phase_mean(
            rows, "slope", "desired_vertical_velocity_mps"
        ),
    }
    for action_name in _ACTION_NAMES:
        summary[f"entry_mean_applied_{action_name}_action"] = _phase_mean(
            rows, "entry", f"applied_{action_name}_action"
        )
        summary[f"slope_mean_applied_{action_name}_action"] = _phase_mean(
            rows, "slope", f"applied_{action_name}_action"
        )
    summary["entry_mean_pitch_deg"] = _phase_mean(rows, "entry", "pitch_deg")
    summary["slope_mean_pitch_deg"] = _phase_mean(rows, "slope", "pitch_deg")
    return summary


def _align_traces(
    baseline: Sequence[Mapping[str, float | str]],
    candidate: Sequence[Mapping[str, float | str]],
) -> list[dict[str, float | str]]:
    """Align traces by policy step and retain every comparable scalar."""

    count = min(len(baseline), len(candidate))
    aligned: list[dict[str, float | str]] = []
    for index in range(count):
        baseline_row = baseline[index]
        candidate_row = candidate[index]
        if not math.isclose(
            float(baseline_row["time_s"]),
            float(candidate_row["time_s"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(f"Trace time mismatch at step {index}.")
        row: dict[str, float | str] = {
            "step": float(baseline_row["step"]),
            "time_s": float(baseline_row["time_s"]),
            "phase": str(baseline_row["phase"]),
        }
        for key in baseline_row:
            if key not in {"label", "step", "time_s", "phase"} and key in candidate_row:
                row[f"baseline_{key}"] = baseline_row[key]
                row[f"candidate_{key}"] = candidate_row[key]
        aligned.append(row)
    return aligned


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(path: Path, aligned: Sequence[Mapping[str, float | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_s = [float(row["time_s"]) for row in aligned]
    fig, axes = plt.subplots(7, 1, figsize=(12, 20), sharex=True, constrained_layout=True)
    colors = {"baseline": "#0072B2", "candidate": "#D55E00"}
    for prefix, label in (("baseline", "promoted C2c model_550"), ("candidate", "joint model_175")):
        color = colors[prefix]
        axes[0].plot(time_s, [float(row[f"{prefix}_reference_altitude_m"]) for row in aligned],
                     color="black", linestyle="--", alpha=0.35 if prefix == "candidate" else 0.8,
                     label="reference" if prefix == "baseline" else None)
        axes[0].plot(time_s, [float(row[f"{prefix}_altitude_m"]) for row in aligned],
                     color=color, label=label)
        axes[1].plot(time_s, [float(row[f"{prefix}_height_error_m"]) for row in aligned],
                     color=color, label=label)
        axes[2].plot(time_s, [float(row[f"{prefix}_vertical_velocity_mps"]) for row in aligned],
                     color=color, label=label)
        axes[3].plot(time_s, [float(row[f"{prefix}_actual_flap_frequency_hz"]) for row in aligned],
                     color=color, label=label)
        requested_frequency = [
            max(-1.0, min(1.0, float(row[f"{prefix}_requested_flap_frequency_action"])))
            for row in aligned
        ]
        axes[4].plot(time_s, requested_frequency, color=color, linestyle=":", alpha=0.8,
                     label=f"{label} requested (clamped)")
        axes[4].plot(time_s, [float(row[f"{prefix}_applied_flap_frequency_action"]) for row in aligned],
                     color=color, label=f"{label} applied")
        axes[5].plot(time_s, [float(row[f"{prefix}_pitch_deg"]) for row in aligned],
                     color=color, label=label)
        axes[6].plot(time_s, [float(row[f"{prefix}_applied_left_elevon_action"]) for row in aligned],
                     color=color, label=f"{label} left")
        axes[6].plot(time_s, [float(row[f"{prefix}_applied_right_elevon_action"]) for row in aligned],
                     color=color, linestyle=":", label=f"{label} right")
    axes[2].plot(
        time_s,
        [float(row["baseline_desired_vertical_velocity_mps"]) for row in aligned],
        color="black",
        linestyle="--",
        label="path-matched vertical velocity",
    )
    axes[0].set_ylabel("altitude [m]")
    axes[1].set_ylabel("height error [m]")
    axes[2].set_ylabel("vertical velocity [m/s]")
    axes[3].set_ylabel("flap frequency [Hz]")
    axes[4].set_ylabel("frequency action")
    axes[5].set_ylabel("pitch [deg]")
    axes[6].set_ylabel("applied elevon action")
    axes[6].set_xlabel("time [s]")
    for axis in axes:
        axis.axhline(0.0, color="0.5", linewidth=0.7, alpha=0.5)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    fig.suptitle("Fixed C2c trajectory: promoted policy vs joint low-precision policy")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    portable_root = (trace_output.parent / "portable" / args.label).resolve()
    required_kit_args = (
        f"--portable-root {portable_root}"
        " --/apps/extensions/fsWatcherEnabled=false"
        f" --ext-folder {(_REPO_ROOT / 'source').resolve()}"
        f" --ext-folder {extension_parent}"
        f" --enable {_NATIVE_EXTENSION_ID}"
    )
    args.device = "cpu"
    args.headless = True
    args.kit_args = " ".join(
        part for part in (getattr(args, "kit_args", None), required_kit_args) if part
    )
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
        from pure_rl_eval_common import assert_pure_rl_longitudinal_reset_schedule, read_pure_rl_step_metrics

        env_cfg = parse_env_cfg(_TASK, device="cpu", num_envs=1)
        heading_rad = math.radians(float(args.heading_deg))
        phase_rad = math.radians(float(args.flap_phase_deg))
        task_id = 0 if float(args.slope_deg) == 0.0 else (1 if float(args.slope_deg) > 0.0 else 2)
        env_cfg.seed = int(args.seed)
        env_cfg.episode_length_s = float(args.duration_s)
        env_cfg.randomize_commands = False
        env_cfg.randomize_straight_line_heading = False
        env_cfg.randomize_flap_phase_at_reset = False
        env_cfg.pure_rl_eval_heading_schedule_rad = (heading_rad,)
        env_cfg.pure_rl_eval_flap_phase_schedule_rad = (phase_rad,)
        env_cfg.pure_rl_eval_longitudinal_task_schedule = (task_id,)
        env_cfg.pure_rl_eval_longitudinal_slope_deg_schedule = (float(args.slope_deg),)
        env_cfg.pure_rl_eval_entry_length_m_schedule = (float(args.entry_length_m),)
        env_cfg.pure_rl_eval_slope_length_m_schedule = (float(args.slope_length_m),)
        env_cfg.wind_enabled = False
        env_cfg.wind_ou_enabled = False
        robot_asset = (
            _REPO_ROOT
            / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
        ).resolve()
        robot_usd_dir = (trace_output.parent / "generated_assets" / args.label / "flap_robot_552").resolve()
        robot_usd_dir.mkdir(parents=True, exist_ok=True)
        env_cfg.robot.spawn.asset_path = str(robot_asset)
        env_cfg.robot.spawn.usd_dir = str(robot_usd_dir)

        agent_cfg = load_cfg_from_registry(_TASK, "rsl_rl_cfg_entry_point")
        agent_cfg.device = "cpu"
        agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["device"] = "cpu"
        base_env = gym.make(_TASK, cfg=env_cfg)
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg_dict.get("clip_actions"))
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device="cpu")
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        path = env.unwrapped._pure_rl_longitudinal_path
        assert path is not None
        assert_pure_rl_longitudinal_reset_schedule(
            actual_heading_rad=env.unwrapped._straight_line_heading_rad,
            actual_flap_phase_rad=env.unwrapped._phase,
            actual_task_id=path.task_id,
            actual_signed_slope_rad=path.signed_slope_rad,
            expected_heading_schedule_rad=(heading_rad,),
            expected_flap_phase_schedule_rad=(phase_rad,),
            expected_task_schedule=(task_id,),
            expected_signed_slope_deg_schedule=(float(args.slope_deg),),
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
            progress_m = float(metrics["along_track_progress_m"][0].item())
            active_slope_rad = float(metrics["active_slope_rad"][0].item())
            tangent_velocity = float(metrics["along_track_velocity_mps"][0].item())
            vertical_normal_velocity = float(metrics["vertical_normal_velocity_mps"][0].item())
            horizontal_velocity = (
                math.cos(active_slope_rad) * tangent_velocity
                - math.sin(active_slope_rad) * vertical_normal_velocity
            )
            vertical_velocity = (
                math.sin(active_slope_rad) * tangent_velocity
                + math.cos(active_slope_rad) * vertical_normal_velocity
            )
            desired_vertical_velocity = horizontal_velocity * math.tan(active_slope_rad)
            slope_progress_m = min(
                max(progress_m - float(args.entry_length_m), 0.0),
                float(args.slope_length_m),
            )
            reference_altitude = float(path.initial_altitude_m[0].item()) + math.tan(
                math.radians(float(args.slope_deg))
            ) * slope_progress_m
            height_error = float(metrics["height_error_m"][0].item())
            terminated = bool(env.unwrapped.reset_terminated[0].item()) if bool(dones[0].item()) else False
            if progress_m < float(args.entry_length_m):
                phase_name = "entry"
            elif progress_m < float(args.entry_length_m) + float(args.slope_length_m):
                phase_name = "slope"
            else:
                phase_name = "recovery"

            if bool(dones[0].item()):
                roll_deg = pitch_deg = yaw_deg = float("nan")
            else:
                roll, pitch, yaw = euler_xyz_from_quat(env.unwrapped._robot.data.root_quat_w)
                roll_deg = math.degrees(float(roll[0].item()))
                pitch_deg = math.degrees(float(pitch[0].item()))
                yaw_deg = math.degrees(float(yaw[0].item()))
            applied_actions = env.unwrapped._debug_last_exec_action[0].detach().cpu()
            row: dict[str, object] = {
                "label": args.label,
                "step": step,
                "time_s": (step + 1) * step_dt_s,
                "phase": phase_name,
                "progress_m": progress_m,
                "reference_altitude_m": reference_altitude,
                "altitude_m": reference_altitude + height_error,
                "height_error_m": height_error,
                "cross_track_error_m": float(metrics["cross_track_error_m"][0].item()),
                "active_slope_deg": math.degrees(active_slope_rad),
                "horizontal_velocity_mps": horizontal_velocity,
                "vertical_velocity_mps": vertical_velocity,
                "desired_vertical_velocity_mps": desired_vertical_velocity,
                "vertical_normal_velocity_mps": vertical_normal_velocity,
                "pitch_deg": pitch_deg,
                "roll_deg": roll_deg,
                "yaw_deg": yaw_deg,
                "pitch_rate_rad_s": float(env.unwrapped._robot.data.root_ang_vel_b[0, 1].item())
                if not bool(dones[0].item())
                else float("nan"),
                "actual_flap_frequency_hz": float(metrics["actual_flap_frequency_hz"][0].item()),
                "reached_recovery": int(bool(metrics["reached_recovery"][0].item())),
                "terminated": int(terminated),
                "done": int(bool(dones[0].item())),
            }
            for action_index, action_name in enumerate(_ACTION_NAMES):
                row[f"requested_{action_name}_action"] = float(requested_actions[0, action_index].item())
                row[f"applied_{action_name}_action"] = float(applied_actions[action_index].item())
            row.update(
                {
                    "rudder_cmd_deg": math.degrees(float(env.unwrapped._debug_last_exec_rudder_rad[0].item())),
                    "left_elevon_cmd_deg": math.degrees(
                        float(env.unwrapped._debug_last_exec_left_elevon_rad[0].item())
                    ),
                    "right_elevon_cmd_deg": math.degrees(
                        float(env.unwrapped._debug_last_exec_right_elevon_rad[0].item())
                    ),
                }
            )
            rows.append(row)
            policy_nn.reset(dones)
            if bool(dones[0].item()):
                break

        summary = _summarize_trace(rows)
        summary.update(
            {
                "checkpoint": str(checkpoint),
                "task": _TASK,
                "condition": {
                    "slope_deg": float(args.slope_deg),
                    "heading_deg": float(args.heading_deg),
                    "flap_phase_deg": float(args.flap_phase_deg),
                    "entry_length_m": float(args.entry_length_m),
                    "slope_length_m": float(args.slope_length_m),
                    "duration_s": float(args.duration_s),
                    "seed": int(args.seed),
                },
            }
        )
        _write_csv(trace_output, rows)
        summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] Wrote {args.label} trace: {trace_output}", flush=True)
        env.close()
        return 0
    finally:
        simulation_app.close()


def _run_parent(argv: Sequence[str] | None = None) -> int:
    args = _parse_parent_args(argv)
    _validate_condition(args)
    baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    candidate_checkpoint = args.candidate_checkpoint.expanduser().resolve()
    for checkpoint in (baseline_checkpoint, candidate_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    commands = {
        "baseline": _worker_command(
            checkpoint=baseline_checkpoint,
            label="baseline",
            output_dir=output_dir,
            args=args,
        ),
        "candidate": _worker_command(
            checkpoint=candidate_checkpoint,
            label="candidate",
            output_dir=output_dir,
            args=args,
        ),
    }
    child_env = _child_environment()
    for label, command in commands.items():
        print(f"[INFO] Tracing {label} checkpoint in a fresh CPU-native process:", flush=True)
        print(" ", " ".join(command), flush=True)
        subprocess.run(command, env=child_env, check=True)

    baseline_rows = _read_trace(output_dir / "baseline_trace.csv")
    candidate_rows = _read_trace(output_dir / "candidate_trace.csv")
    baseline_summary = _summarize_trace(baseline_rows)
    candidate_summary = _summarize_trace(candidate_rows)
    aligned = _align_traces(baseline_rows, candidate_rows)
    comparison = {
        "condition": {
            "slope_deg": float(args.slope_deg),
            "heading_deg": float(args.heading_deg),
            "flap_phase_deg": float(args.flap_phase_deg),
            "entry_length_m": float(args.entry_length_m),
            "slope_length_m": float(args.slope_length_m),
            "duration_s": float(args.duration_s),
            "seed": int(args.seed),
        },
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint),
        "aligned_step_count": len(aligned),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
    }
    _write_csv(output_dir / "aligned_trace.csv", aligned)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_comparison(output_dir / "comparison.png", aligned)
    print(f"[OK] Comparison complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    if "--worker" in sys.argv[1:]:
        raise SystemExit(_run_worker(sys.argv[1:]))
    raise SystemExit(_run_parent(sys.argv[1:]))

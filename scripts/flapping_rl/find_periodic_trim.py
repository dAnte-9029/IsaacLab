"""Find a fixed-input periodic trim for the native CPU PureRL plant.

The search has two deliberately separate stages:

1. clamp the root state and rank a batched longitudinal grid by cycle-mean
   force and moment residuals;
2. release the top candidates under the same fixed action and record their
   short-horizon open-loop drift.

This is an experiment and diagnostic entry point. It does not run a controller,
teacher, optimizer or policy.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Iterable


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASK_ID = "Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0"
_EXTENSION_ID = "omni.flapping_bot.holonomic_constraint"
_DEFAULT_EXTENSION_PARENT = _REPO_ROOT / "source/flapping_bot/native_extensions"
_DEFAULT_ASSET_PATH = (
    _REPO_ROOT
    / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the periodic-trim search command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("logs/flapping_rl/periodic_trim"),
    )
    parser.add_argument("--pitch-deg-values", type=float, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument(
        "--frequency-hz-values",
        type=float,
        nargs="+",
        default=[3.0, 3.5, 4.0, 4.5, 5.0],
    )
    parser.add_argument(
        "--elevon-pitch-deg-values",
        type=float,
        nargs="+",
        default=[-30, -24, -18, -12, -6],
    )
    parser.add_argument("--rudder-deg-values", type=float, nargs="+", default=[0.0])
    parser.add_argument("--elevon-roll-deg-values", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--score-mode",
        choices=("longitudinal", "full_wrench"),
        default="longitudinal",
        help="Score Fx/Fz/My only or all six force and moment residuals.",
    )
    parser.add_argument("--airspeed-mps", type=float, default=7.0)
    parser.add_argument("--height-m", type=float, default=10.0)
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--measure-cycles", type=float, default=8.0)
    parser.add_argument("--release-settle-s", type=float, default=1.0)
    parser.add_argument("--free-flight-s", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument(
        "--moment-reference-length-m",
        type=float,
        default=0.5,
        help="Length used only to nondimensionalize the moment residual score.",
    )
    parser.add_argument("--force-gate-weight-fraction", type=float, default=0.05)
    parser.add_argument("--moment-gate-weight-length-fraction", type=float, default=0.02)
    parser.add_argument("--frequency-tolerance-hz", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=120)
    parser.add_argument("--extension-parent", type=Path, default=_DEFAULT_EXTENSION_PARENT)
    parser.add_argument("--asset-path", type=Path, default=_DEFAULT_ASSET_PATH)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (Path(args.out_root) / timestamp).expanduser().resolve()


def _validate_args(args: argparse.Namespace, candidate_count: int) -> None:
    positive_values = {
        "airspeed_mps": args.airspeed_mps,
        "height_m": args.height_m,
        "settle_s": args.settle_s,
        "measure_cycles": args.measure_cycles,
        "release_settle_s": args.release_settle_s,
        "moment_reference_length_m": args.moment_reference_length_m,
        "force_gate_weight_fraction": args.force_gate_weight_fraction,
        "moment_gate_weight_length_fraction": args.moment_gate_weight_length_fraction,
        "frequency_tolerance_hz": args.frequency_tolerance_hz,
    }
    for name, value in positive_values.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    if not math.isfinite(float(args.free_flight_s)) or float(args.free_flight_s) < 0.0:
        raise ValueError("--free-flight-s must be non-negative and finite.")
    if int(args.top_k) <= 0:
        raise ValueError("--top-k must be positive.")
    if candidate_count > int(args.max_candidates):
        raise ValueError(
            f"Candidate grid has {candidate_count} entries, exceeding --max-candidates={args.max_candidates}."
        )
    if any(float(value) > 5.0 for value in args.frequency_hz_values):
        raise ValueError("--frequency-hz-values must not exceed the PureRL 5 Hz maximum.")
    if any(abs(float(value)) > 41.0 for value in args.elevon_pitch_deg_values):
        raise ValueError("--elevon-pitch-deg-values must remain within the 41 degree command limit.")
    if any(abs(float(value)) > 25.0 for value in args.rudder_deg_values):
        raise ValueError("--rudder-deg-values must remain within the 25 degree command limit.")
    if any(abs(float(value)) > 41.0 for value in args.elevon_roll_deg_values):
        raise ValueError("--elevon-roll-deg-values must remain within the 41 degree command limit.")


def _prepend_worktree_sources() -> None:
    for path in reversed(
        (
            _REPO_ROOT / "source/flapping_bot",
            _REPO_ROOT / "source/isaaclab_assets",
        )
    ):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _linear_slope(time_s: list[float], values: list[float]) -> float:
    if len(time_s) < 2 or len(values) != len(time_s):
        return float("nan")
    t_mean = sum(time_s) / len(time_s)
    v_mean = sum(values) / len(values)
    denominator = sum((time - t_mean) ** 2 for time in time_s)
    if denominator <= 0.0:
        return float("nan")
    return sum((time - t_mean) * (value - v_mean) for time, value in zip(time_s, values)) / denominator


def _render_report(
    *,
    ranked_rows: list[dict[str, Any]],
    best_trim: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    top_rows = ranked_rows[: min(10, len(ranked_rows))]
    lines = [
        "# Native CPU PureRL Periodic Trim Search",
        "",
        f"Date: {manifest['created_at']}",
        "",
        "## Result",
        "",
        f"- Selected candidate: `{best_trim['candidate_id']}`",
        f"- Fixed-root trim gate passed: `{str(best_trim['gate_passed']).lower()}`",
        f"- Pitch: `{best_trim['state']['pitch_nose_up_deg']:.6g} deg` nose-up",
        f"- Frequency: `{best_trim['action']['frequency_hz']:.6g} Hz`",
        f"- Elevon pitch: `{best_trim['action']['elevon_pitch_deg']:.6g} deg`",
        f"- Total normalized residual score: `{best_trim['scores']['total']:.8g}`",
        "",
        (
            "The fixed-root gate tests cycle-mean force and moment balance. It is a necessary balance "
            "condition, not a Poincare periodic-orbit solve. The free-flight section is an open-loop drift "
            "diagnostic, not a stability-controller result."
        ),
        "",
        "## Top candidates",
        "",
        f"- Score mode: `{manifest['search']['score_mode']}`",
        "",
        (
            "| rank | id | pitch deg | frequency Hz | elevon pitch deg | rudder deg | elevon roll deg | "
            "force score | moment score | total | gate |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in top_rows:
        lines.append(
            "| {rank} | {candidate_id} | {pitch_nose_up_deg:.4g} | {frequency_target_hz:.4g} | "
            "{elevon_pitch_deg:.4g} | {rudder_deg:.4g} | {elevon_roll_deg:.4g} | "
            "{force_score:.5g} | {moment_score:.5g} | {total_score:.5g} | {gate} |".format(
                gate="pass" if bool(row["gate_passed"]) else "fail",
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Selected mean residuals",
            "",
            f"- Force residual in body FLU: `{best_trim['mean_force_residual_b_n']} N`",
            f"- Moment about system COM in body FLU: `{best_trim['mean_moment_system_com_b_nm']} N m`",
            f"- Mean actual frequency: `{best_trim['frequency_actual_mean_hz']:.8g} Hz`",
            "",
            "## Open-loop release",
            "",
        ]
    )
    release = best_trim.get("free_flight", {})
    if release:
        lines.extend(
            [
                f"- Height slope: `{release['height_slope_mps']:.8g} m/s`",
                f"- Body-forward speed slope: `{release['body_vx_slope_mps2']:.8g} m/s^2`",
                f"- Nose-up pitch slope: `{release['pitch_slope_deg_s']:.8g} deg/s`",
                f"- Final body angular velocity: `{release['final_body_angular_velocity_rad_s']} rad/s`",
            ]
        )
    else:
        lines.append("Free-flight validation was disabled with `--free-flight-s 0`.")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- A longitudinal score uses Fx/Fz/My only; a full-wrench score uses all six residual components.",
            (
                "- This workflow does not solve the phase-conditioned free-flight Poincare state; "
                "`periodic_orbit_validated` is therefore false."
            ),
            "- No controller, teacher, policy, reward or PPO update was active.",
            "- Passing this numerical gate does not establish aerodynamic-model or real-flight validity.",
            "",
        ]
    )
    return "\n".join(lines)


def _system_com_offset_b(env: Any, body_masses: Any, quat_apply_inverse: Any) -> Any:
    body_com_w = env._robot.data.body_com_pos_w
    total_mass = body_masses.sum(dim=1)
    system_com_w = (body_masses.unsqueeze(-1) * body_com_w).sum(dim=1) / total_mass.unsqueeze(-1)
    base_com_w = body_com_w[:, int(env._base_body_ids[0]), :]
    return quat_apply_inverse(env._robot.data.root_quat_w, system_com_w - base_com_w)


def _write_candidate_root_states(
    env: Any,
    *,
    env_ids: Any,
    pitch_nose_up_deg: Any,
    airspeed_mps: float,
    height_m: float,
    quat_from_euler_xyz: Any,
) -> None:
    import torch

    count = int(env_ids.numel())
    positions = env.scene.env_origins[env_ids].clone()
    positions[:, 2] += float(height_m)
    pitch_isaac_rad = -torch.deg2rad(pitch_nose_up_deg)
    zeros = torch.zeros_like(pitch_isaac_rad)
    orientation = quat_from_euler_xyz(zeros, pitch_isaac_rad, zeros)
    linear_velocity = torch.zeros((count, 3), device=env.device)
    linear_velocity[:, 0] = float(airspeed_mps)
    angular_velocity = torch.zeros_like(linear_velocity)
    root_state = torch.cat((positions, orientation, linear_velocity, angular_velocity), dim=1)
    env._robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    env._spawn_root_state[env_ids] = root_state


def _prime_action_history(env: Any, *, env_ids: Any, actions: Any) -> None:
    env._actions[env_ids] = actions[env_ids]
    env._act_lpf[env_ids] = actions[env_ids]
    env._act_cmd[env_ids] = actions[env_ids]


def _collect_fixed_root_measurements(
    env: Any,
    *,
    actions: Any,
    candidates: list[Any],
    settle_s: float,
    measure_cycles: float,
    score_mode: str,
    moment_reference_length_m: float,
    force_gate_weight_fraction: float,
    moment_gate_weight_length_fraction: float,
    frequency_tolerance_hz: float,
    print_every: int,
    quat_apply_inverse: Any,
) -> list[dict[str, Any]]:
    import torch

    from flapping_bot.direct.flapping_bot.periodic_trim import (
        compute_full_wrench_trim_scores,
        compute_trim_scores,
        rank_trim_rows,
        transfer_moment_to_system_com,
    )

    count = len(candidates)
    settle_steps = max(1, math.ceil(float(settle_s) / float(env.step_dt)))
    for step in range(settle_steps):
        env.step(actions)
        if print_every > 0 and (step + 1) % print_every == 0:
            print(f"[trim] fixed-root settle {step + 1}/{settle_steps}", flush=True)

    phase_start = env._ideal_inverse_phase_state.phase_rad.clone()
    phase_end = phase_start + 2.0 * math.pi * float(measure_cycles)
    weight_sum = torch.zeros(count, device=env.device)
    force_aero_sum = torch.zeros((count, 3), device=env.device)
    force_residual_sum = torch.zeros_like(force_aero_sum)
    force_residual_sq_sum = torch.zeros_like(force_aero_sum)
    moment_sum = torch.zeros_like(force_aero_sum)
    moment_sq_sum = torch.zeros_like(force_aero_sum)
    actual_frequency_sum = torch.zeros(count, device=env.device)
    body_masses = env._robot.root_physx_view.get_masses().to(device=env.device)
    total_mass = body_masses.sum(dim=1)
    gravity_magnitude = math.sqrt(sum(float(value) ** 2 for value in env.cfg.sim.gravity))
    weight_n = total_mass * gravity_magnitude
    max_measure_s = float(measure_cycles) / min(candidate.frequency_hz for candidate in candidates) + 1.0
    max_steps = math.ceil(max_measure_s / float(env.step_dt))

    for step in range(max_steps):
        phase_previous = env._ideal_inverse_phase_state.phase_rad.clone()
        env.step(actions)
        phase_current = env._ideal_inverse_phase_state.phase_rad
        phase_delta = torch.clamp(phase_current - phase_previous, min=1.0e-12)
        active = phase_previous < phase_end
        fraction = torch.clamp((phase_end - phase_previous) / phase_delta, min=0.0, max=1.0)
        weights = active.to(dtype=phase_delta.dtype) * fraction * float(env.step_dt)
        if not bool(torch.any(weights > 0.0)):
            break

        force_aero_b = env._debug_last_force_b
        gravity_force_b = env._robot.data.projected_gravity_b * weight_n.unsqueeze(1)
        force_residual_b = force_aero_b + gravity_force_b
        system_com_from_base_b = _system_com_offset_b(env, body_masses, quat_apply_inverse)
        moment_system_com_b = transfer_moment_to_system_com(
            force_b_n=force_aero_b,
            moment_about_base_com_b_nm=env._debug_last_torque_b,
            system_com_from_base_com_b_m=system_com_from_base_b,
        )
        weight_column = weights.unsqueeze(1)
        weight_sum += weights
        force_aero_sum += weight_column * force_aero_b
        force_residual_sum += weight_column * force_residual_b
        force_residual_sq_sum += weight_column * force_residual_b.square()
        moment_sum += weight_column * moment_system_com_b
        moment_sq_sum += weight_column * moment_system_com_b.square()
        actual_frequency_sum += weights * env._freq
        if print_every > 0 and (step + 1) % print_every == 0:
            finished = int(torch.count_nonzero(phase_current >= phase_end).item())
            print(f"[trim] measured candidates {finished}/{count}", flush=True)

    if bool(torch.any(weight_sum <= 0.0)) or bool(torch.any(env._ideal_inverse_phase_state.phase_rad < phase_end)):
        raise RuntimeError("Not all candidates completed the requested measurement cycles.")
    divisor = weight_sum.unsqueeze(1)
    mean_force_aero = force_aero_sum / divisor
    mean_force_residual = force_residual_sum / divisor
    mean_moment = moment_sum / divisor
    force_std = torch.sqrt(torch.clamp(force_residual_sq_sum / divisor - mean_force_residual.square(), min=0.0))
    moment_std = torch.sqrt(torch.clamp(moment_sq_sum / divisor - mean_moment.square(), min=0.0))
    mean_actual_frequency = actual_frequency_sum / weight_sum
    score_function = compute_trim_scores
    if score_mode == "full_wrench":
        score_function = compute_full_wrench_trim_scores
    scores = score_function(
        mean_force_residual_b_n=mean_force_residual,
        mean_moment_system_com_b_nm=mean_moment,
        weight_n=weight_n,
        moment_reference_length_m=moment_reference_length_m,
    )

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        finite = all(
            math.isfinite(value)
            for value in (
                *mean_force_aero[index].tolist(),
                *mean_force_residual[index].tolist(),
                *mean_moment[index].tolist(),
                float(scores.total_score[index].item()),
            )
        )
        frequency_error = abs(float(mean_actual_frequency[index].item()) - candidate.frequency_hz)
        valid = finite and frequency_error <= float(frequency_tolerance_hz)
        force_score = float(scores.force_score[index].item())
        moment_score = float(scores.moment_score[index].item())
        row = {
            "candidate_id": candidate.candidate_id,
            "pitch_nose_up_deg": candidate.pitch_nose_up_deg,
            "frequency_target_hz": candidate.frequency_hz,
            "frequency_actual_mean_hz": float(mean_actual_frequency[index].item()),
            "frequency_error_hz": frequency_error,
            "elevon_pitch_deg": candidate.elevon_pitch_deg,
            "rudder_deg": candidate.rudder_deg,
            "elevon_roll_deg": candidate.elevon_roll_deg,
            "action_frequency_normalized": float(actions[index, 0].item()),
            "action_rudder_normalized": float(actions[index, 1].item()),
            "action_elevon_pitch_normalized": float(actions[index, 2].item()),
            "action_elevon_roll_normalized": float(actions[index, 3].item()),
            "mean_aero_force_b_x_n": float(mean_force_aero[index, 0].item()),
            "mean_aero_force_b_y_n": float(mean_force_aero[index, 1].item()),
            "mean_aero_force_b_z_n": float(mean_force_aero[index, 2].item()),
            "mean_force_residual_b_x_n": float(mean_force_residual[index, 0].item()),
            "mean_force_residual_b_y_n": float(mean_force_residual[index, 1].item()),
            "mean_force_residual_b_z_n": float(mean_force_residual[index, 2].item()),
            "std_force_residual_b_x_n": float(force_std[index, 0].item()),
            "std_force_residual_b_y_n": float(force_std[index, 1].item()),
            "std_force_residual_b_z_n": float(force_std[index, 2].item()),
            "mean_moment_system_com_b_x_nm": float(mean_moment[index, 0].item()),
            "mean_moment_system_com_b_y_nm": float(mean_moment[index, 1].item()),
            "mean_moment_system_com_b_z_nm": float(mean_moment[index, 2].item()),
            "std_moment_system_com_b_x_nm": float(moment_std[index, 0].item()),
            "std_moment_system_com_b_y_nm": float(moment_std[index, 1].item()),
            "std_moment_system_com_b_z_nm": float(moment_std[index, 2].item()),
            "weight_n": float(weight_n[index].item()),
            "measured_duration_s": float(weight_sum[index].item()),
            "measured_cycles": float(measure_cycles),
            "force_score": force_score,
            "moment_score": moment_score,
            "total_score": float(scores.total_score[index].item()),
            "valid": valid,
            "gate_passed": (
                valid
                and force_score <= float(force_gate_weight_fraction)
                and moment_score <= float(moment_gate_weight_length_fraction)
            ),
        }
        rows.append(row)
    return rank_trim_rows(rows)


def _collect_free_flight(
    env: Any,
    *,
    actions: Any,
    ranked_rows: list[dict[str, Any]],
    pitch_by_candidate: Any,
    airspeed_mps: float,
    height_m: float,
    release_settle_s: float,
    free_flight_s: float,
    top_k: int,
    timeseries_dir: Path,
    quat_from_euler_xyz: Any,
    euler_xyz_from_quat: Any,
) -> dict[int, dict[str, Any]]:
    import torch

    if free_flight_s <= 0.0:
        return {}
    selected_rows = [row for row in ranked_rows if bool(row["valid"])][: int(top_k)]
    if not selected_rows:
        raise RuntimeError("No valid trim candidate is available for free-flight validation.")
    selected_ids = torch.tensor(
        [int(row["candidate_id"]) for row in selected_rows],
        device=env.device,
        dtype=torch.int64,
    )
    env._reset_idx(selected_ids)
    _write_candidate_root_states(
        env,
        env_ids=selected_ids,
        pitch_nose_up_deg=pitch_by_candidate[selected_ids],
        airspeed_mps=airspeed_mps,
        height_m=height_m,
        quat_from_euler_xyz=quat_from_euler_xyz,
    )
    _prime_action_history(env, env_ids=selected_ids, actions=actions)
    env._freeze_steps[:] = 1_000_000
    settle_steps = max(1, math.ceil(float(release_settle_s) / float(env.step_dt)))
    for _ in range(settle_steps):
        env.step(actions)
    env._freeze_steps[selected_ids] = 0

    series: dict[int, list[dict[str, float]]] = {int(index): [] for index in selected_ids.tolist()}
    free_steps = max(1, math.ceil(float(free_flight_s) / float(env.step_dt)))
    for step in range(free_steps):
        _, _, terminated, truncated, _ = env.step(actions)
        if bool(torch.any(terminated[selected_ids] | truncated[selected_ids])):
            raise RuntimeError("A selected free-flight candidate terminated despite permissive gates.")
        roll, pitch_isaac, yaw = euler_xyz_from_quat(env._robot.data.root_quat_w[selected_ids])
        local_position = env._robot.data.root_pos_w[selected_ids] - env.scene.env_origins[selected_ids]
        velocity_b = env._robot.data.root_lin_vel_b[selected_ids]
        angular_velocity_b = env._robot.data.root_ang_vel_b[selected_ids]
        for local_index, candidate_id in enumerate(selected_ids.tolist()):
            series[int(candidate_id)].append(
                {
                    "time_s": float((step + 1) * env.step_dt),
                    "x_m": float(local_position[local_index, 0].item()),
                    "y_m": float(local_position[local_index, 1].item()),
                    "height_m": float(local_position[local_index, 2].item()),
                    "body_vx_mps": float(velocity_b[local_index, 0].item()),
                    "body_vy_mps": float(velocity_b[local_index, 1].item()),
                    "body_vz_mps": float(velocity_b[local_index, 2].item()),
                    "roll_deg": math.degrees(float(roll[local_index].item())),
                    "pitch_nose_up_deg": -math.degrees(float(pitch_isaac[local_index].item())),
                    "yaw_deg": math.degrees(float(yaw[local_index].item())),
                    "body_p_rad_s": float(angular_velocity_b[local_index, 0].item()),
                    "body_q_rad_s": float(angular_velocity_b[local_index, 1].item()),
                    "body_r_rad_s": float(angular_velocity_b[local_index, 2].item()),
                    "frequency_actual_hz": float(env._freq[candidate_id].item()),
                }
            )

    timeseries_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[int, dict[str, Any]] = {}
    for candidate_id, rows in series.items():
        _write_csv(timeseries_dir / f"candidate_{candidate_id:04d}.csv", rows)
        time_s = [row["time_s"] for row in rows]
        final = rows[-1]
        summaries[candidate_id] = {
            "duration_s": time_s[-1],
            "height_slope_mps": _linear_slope(time_s, [row["height_m"] for row in rows]),
            "body_vx_slope_mps2": _linear_slope(time_s, [row["body_vx_mps"] for row in rows]),
            "pitch_slope_deg_s": _linear_slope(time_s, [row["pitch_nose_up_deg"] for row in rows]),
            "final_position_m": [final["x_m"], final["y_m"], final["height_m"]],
            "final_body_velocity_mps": [final["body_vx_mps"], final["body_vy_mps"], final["body_vz_mps"]],
            "final_attitude_deg": [final["roll_deg"], final["pitch_nose_up_deg"], final["yaw_deg"]],
            "final_body_angular_velocity_rad_s": [
                final["body_p_rad_s"],
                final["body_q_rad_s"],
                final["body_r_rad_s"],
            ],
        }
    return summaries


def _build_best_trim(
    *,
    best_row: dict[str, Any],
    free_flight: dict[str, Any] | None,
    airspeed_mps: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "result_type": "fixed_root_cycle_mean_wrench_balance_candidate",
        "periodic_orbit_validated": False,
        "task": _TASK_ID,
        "candidate_id": int(best_row["candidate_id"]),
        "condition": {
            "airspeed_mps": float(airspeed_mps),
            "wind_mps": [0.0, 0.0, 0.0],
            "flight_path_angle_deg": 0.0,
        },
        "state": {
            "pitch_nose_up_deg": float(best_row["pitch_nose_up_deg"]),
            "roll_deg": 0.0,
            "yaw_deg": 0.0,
        },
        "action": {
            "frequency_hz": float(best_row["frequency_target_hz"]),
            "rudder_deg": float(best_row["rudder_deg"]),
            "elevon_pitch_deg": float(best_row["elevon_pitch_deg"]),
            "elevon_roll_deg": float(best_row["elevon_roll_deg"]),
        },
        "normalized_action": [
            float(best_row["action_frequency_normalized"]),
            float(best_row["action_rudder_normalized"]),
            float(best_row["action_elevon_pitch_normalized"]),
            float(best_row["action_elevon_roll_normalized"]),
        ],
        "frequency_actual_mean_hz": float(best_row["frequency_actual_mean_hz"]),
        "mean_force_residual_b_n": [
            float(best_row[f"mean_force_residual_b_{axis}_n"]) for axis in "xyz"
        ],
        "mean_moment_system_com_b_nm": [
            float(best_row[f"mean_moment_system_com_b_{axis}_nm"]) for axis in "xyz"
        ],
        "scores": {
            "force": float(best_row["force_score"]),
            "moment": float(best_row["moment_score"]),
            "total": float(best_row["total_score"]),
        },
        "valid": bool(best_row["valid"]),
        "gate_passed": bool(best_row["gate_passed"]),
        "fixed_root_wrench_gate_passed": bool(best_row["gate_passed"]),
        "free_flight": free_flight or {},
    }


def main() -> None:
    args = build_parser().parse_args()
    run_dir = _resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=False)
    _prepend_worktree_sources()

    from flapping_bot.direct.flapping_bot.periodic_trim import build_periodic_trim_grid

    candidates = build_periodic_trim_grid(
        pitch_nose_up_deg_values=args.pitch_deg_values,
        frequency_hz_values=args.frequency_hz_values,
        elevon_pitch_deg_values=args.elevon_pitch_deg_values,
        rudder_deg_values=args.rudder_deg_values,
        elevon_roll_deg_values=args.elevon_roll_deg_values,
    )
    _validate_args(args, len(candidates))
    extension_parent = Path(args.extension_parent).expanduser().resolve()
    extension_binary = extension_parent / _EXTENSION_ID / "omni/flapping_bot/holonomic_constraint/_native.so"
    asset_path = Path(args.asset_path).expanduser().resolve()
    if not extension_binary.is_file():
        raise FileNotFoundError(f"Native extension binary does not exist: {extension_binary}")
    if not asset_path.is_file():
        raise FileNotFoundError(f"Flapping robot URDF does not exist: {asset_path}")

    from isaaclab.app import AppLauncher

    kit_args = (
        f"--portable-root {run_dir / 'kit'} "
        f"--ext-folder {extension_parent} "
        f"--enable {_EXTENSION_ID}"
    )
    simulation_app = AppLauncher(headless=bool(args.headless), device="cpu", kit_args=kit_args).app

    import omni.physx
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz

    from flapping_bot.direct.flapping_bot.action_contract import (
        COMMAND_TAIL_AERO_DEFLECTION,
        MIXED_ELEVON_ACTION,
        normalized_action_to_frequency_hz,
    )
    from flapping_bot.direct.flapping_bot.straight_flight_env import (
        FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
        FlappingBotStraightFlightEnv,
    )

    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg()
    # Preserve the parameterization and artifact meaning of this completed
    # fixed-root pitch/roll trim diagnostic after the training task moves to
    # direct left/right surface actions. New policies do not inherit this
    # compatibility override.
    cfg.action_interface = MIXED_ELEVON_ACTION
    cfg.tail_aero_deflection_source = COMMAND_TAIL_AERO_DEFLECTION
    cfg.scene.num_envs = len(candidates)
    cfg.scene.env_spacing = 5.0
    cfg.sim.device = "cpu"
    cfg.seed = int(args.seed)
    cfg.freeze_steps_after_reset = 1_000_000
    cfg.episode_length_s = max(
        100.0,
        float(args.settle_s)
        + float(args.measure_cycles) / min(candidate.frequency_hz for candidate in candidates)
        + float(args.release_settle_s)
        + float(args.free_flight_s)
        + 10.0,
    )
    cfg.terminate_ground_height = -1.0e6
    # The environment computes tilt as sqrt(gx^2 + gy^2) and compares it with
    # sin(threshold). Exactly 90 degrees disables only this automatic reset;
    # open-loop tumbling remains visible in the recorded trajectory.
    cfg.terminate_tilt_deg = 90.0
    cfg.terminate_abs_y = 1.0e6
    cfg.randomize_commands = False
    cfg.wind_enabled = False
    cfg.randomize_wind = False
    cfg.robot = cfg.robot.replace(
        spawn=cfg.robot.spawn.replace(
            asset_path=str(asset_path),
            usd_dir=str(run_dir / "generated_assets/flap_robot_552"),
        )
    )

    env = None
    try:
        env = FlappingBotStraightFlightEnv(cfg)
        env.reset()
        env_ids = torch.arange(len(candidates), device=env.device, dtype=torch.int64)
        pitch_by_candidate = torch.tensor(
            [candidate.pitch_nose_up_deg for candidate in candidates],
            device=env.device,
        )
        frequency_by_candidate = torch.tensor(
            [candidate.frequency_hz for candidate in candidates],
            device=env.device,
        )
        elevon_by_candidate = torch.tensor(
            [candidate.elevon_pitch_deg for candidate in candidates],
            device=env.device,
        )
        rudder_by_candidate = torch.tensor(
            [candidate.rudder_deg for candidate in candidates],
            device=env.device,
        )
        roll_by_candidate = torch.tensor(
            [candidate.elevon_roll_deg for candidate in candidates],
            device=env.device,
        )
        actions = torch.zeros((len(candidates), cfg.action_space), device=env.device)
        actions[:, 0] = 2.0 * (
            frequency_by_candidate - float(cfg.min_flap_hz)
        ) / (float(cfg.max_flap_hz) - float(cfg.min_flap_hz)) - 1.0
        assert torch.allclose(
            normalized_action_to_frequency_hz(
                actions[:, 0],
                minimum_frequency_hz=float(cfg.min_flap_hz),
                maximum_frequency_hz=float(cfg.max_flap_hz),
            ),
            frequency_by_candidate,
        )
        actions[:, 1] = rudder_by_candidate / float(cfg.rudder_max_deg)
        actions[:, 2] = elevon_by_candidate / float(cfg.elevon_max_deg)
        actions[:, 3] = roll_by_candidate / float(cfg.elevon_max_deg)
        _write_candidate_root_states(
            env,
            env_ids=env_ids,
            pitch_nose_up_deg=pitch_by_candidate,
            airspeed_mps=float(args.airspeed_mps),
            height_m=float(args.height_m),
            quat_from_euler_xyz=quat_from_euler_xyz,
        )
        _prime_action_history(env, env_ids=env_ids, actions=actions)
        ranked_rows = _collect_fixed_root_measurements(
            env,
            actions=actions,
            candidates=candidates,
            settle_s=float(args.settle_s),
            measure_cycles=float(args.measure_cycles),
            score_mode=str(args.score_mode),
            moment_reference_length_m=float(args.moment_reference_length_m),
            force_gate_weight_fraction=float(args.force_gate_weight_fraction),
            moment_gate_weight_length_fraction=float(args.moment_gate_weight_length_fraction),
            frequency_tolerance_hz=float(args.frequency_tolerance_hz),
            print_every=int(args.print_every),
            quat_apply_inverse=quat_apply_inverse,
        )
        _write_csv(run_dir / "trim_candidates.csv", ranked_rows)
        (run_dir / "fixed_root_complete.json").write_text(
            json.dumps(
                {
                    "candidate_count": len(ranked_rows),
                    "valid_candidate_count": sum(bool(row["valid"]) for row in ranked_rows),
                    "gate_pass_count": sum(bool(row["gate_passed"]) for row in ranked_rows),
                },
                indent=2,
            )
            + "\n"
        )
        free_flight = _collect_free_flight(
            env,
            actions=actions,
            ranked_rows=ranked_rows,
            pitch_by_candidate=pitch_by_candidate,
            airspeed_mps=float(args.airspeed_mps),
            height_m=float(args.height_m),
            release_settle_s=float(args.release_settle_s),
            free_flight_s=float(args.free_flight_s),
            top_k=int(args.top_k),
            timeseries_dir=run_dir / "top_candidates_timeseries",
            quat_from_euler_xyz=quat_from_euler_xyz,
            euler_xyz_from_quat=euler_xyz_from_quat,
        )
        for row in ranked_rows:
            candidate_id = int(row["candidate_id"])
            if candidate_id in free_flight:
                summary = free_flight[candidate_id]
                row.update({f"free_{key}": value for key, value in summary.items() if not isinstance(value, list)})
        valid_rows = [row for row in ranked_rows if bool(row["valid"])]
        if not valid_rows:
            raise RuntimeError("Periodic trim search produced no valid candidates.")
        best_row = valid_rows[0]
        best_trim = _build_best_trim(
            best_row=best_row,
            free_flight=free_flight.get(int(best_row["candidate_id"])),
            airspeed_mps=float(args.airspeed_mps),
        )
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "task": _TASK_ID,
            "branch": _git_value("branch", "--show-current"),
            "commit": _git_value("rev-parse", "HEAD"),
            "git_status_porcelain": _git_value("status", "--short"),
            "python": sys.executable,
            "candidate_count": len(candidates),
            "search": {
                "pitch_nose_up_deg_values": [float(value) for value in args.pitch_deg_values],
                "frequency_hz_values": [float(value) for value in args.frequency_hz_values],
                "elevon_pitch_deg_values": [float(value) for value in args.elevon_pitch_deg_values],
                "rudder_deg_values": [float(value) for value in args.rudder_deg_values],
                "elevon_roll_deg_values": [float(value) for value in args.elevon_roll_deg_values],
                "score_mode": str(args.score_mode),
                "airspeed_mps": float(args.airspeed_mps),
                "settle_s": float(args.settle_s),
                "measure_cycles": float(args.measure_cycles),
                "release_settle_s": float(args.release_settle_s),
                "free_flight_s": float(args.free_flight_s),
                "top_k": int(args.top_k),
            },
            "gates": {
                "force_weight_fraction": float(args.force_gate_weight_fraction),
                "moment_weight_length_fraction": float(args.moment_gate_weight_length_fraction),
                "moment_reference_length_m": float(args.moment_reference_length_m),
                "frequency_tolerance_hz": float(args.frequency_tolerance_hz),
            },
            "plant": {
                "sim_device": str(cfg.sim.device),
                "physics_dt_s": float(cfg.sim.dt),
                "decimation": int(cfg.decimation),
                "min_flap_hz": float(cfg.min_flap_hz),
                "max_flap_hz": float(cfg.max_flap_hz),
                "asset_path": str(asset_path),
                "generated_asset_dir": str(run_dir / "generated_assets/flap_robot_552"),
                "native_extension_binary": str(extension_binary),
            },
            "candidate_grid": [asdict(candidate) for candidate in candidates],
        }
        _write_csv(run_dir / "trim_candidates.csv", ranked_rows)
        (run_dir / "best_trim.json").write_text(json.dumps(best_trim, indent=2) + "\n")
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (run_dir / "report.md").write_text(
            _render_report(ranked_rows=ranked_rows, best_trim=best_trim, manifest=manifest)
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "candidate_count": len(candidates),
                    "best_candidate_id": best_trim["candidate_id"],
                    "gate_passed": best_trim["gate_passed"],
                    "total_score": best_trim["scores"]["total"],
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        if env is not None:
            omni.physx.get_physx_simulation_interface().detach_stage()
            env.close()
        # Isaac shutdown may replace a pending Python exception with exit code
        # zero. Exit explicitly after flushing so automation fails closed.
        os._exit(1)
    else:
        omni.physx.get_physx_simulation_interface().detach_stage()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

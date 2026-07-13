"""Diagnose a straight-flight policy checkpoint with rollout tables and plots.

The script is intentionally focused on failure analysis rather than benchmark scoring.
It records per-step state/action/reward terms, per-episode termination summaries,
and static PNG plots for quick inspection.

Example:
  ./isaaclab.sh -p scripts/flapping_rl/diagnose_straight_flight_checkpoint.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0 \
    --checkpoint logs/rsl_rl/flapping_bot_straight_flight/<RUN>/model_400.pt \
    --episodes 32 --num_envs 32 --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path
from typing import Any


ACTION_COLUMNS = ("cmd_a0", "cmd_a1", "cmd_a2", "cmd_a3")
RAW_ACTION_COLUMNS = ("raw_a0", "raw_a1", "raw_a2", "raw_a3")
TERMINATION_REASONS = ("timeout", "ground", "tilt", "lateral", "terminated")


def infer_termination_reason(
    timed_out: bool,
    terminated: bool,
    height_m: float,
    tilt_deg: float,
    abs_y_m: float,
    ground_height_m: float,
    terminate_tilt_deg: float,
    terminate_abs_y_m: float,
) -> str:
    """Infer a human-readable termination reason from the last pre-reset state."""
    if timed_out:
        return "timeout"
    if not terminated:
        return "running"
    if height_m <= ground_height_m:
        return "ground"
    if tilt_deg > terminate_tilt_deg:
        return "tilt"
    if abs_y_m > terminate_abs_y_m:
        return "lateral"
    return "terminated"


def refine_terminated_reason_from_episode_rows(
    reason: str,
    episode_step_rows: list[dict[str, Any]],
    *,
    ground_height_m: float,
    terminate_tilt_deg: float,
    terminate_abs_y_m: float,
    ground_margin_m: float = 0.25,
    tilt_margin_deg: float = 2.0,
    lateral_margin_m: float = 1.0,
) -> str:
    """Refine ambiguous terminal failures using near-threshold episode history."""
    if reason != "terminated" or not episode_step_rows:
        return reason
    min_height = min(float(row["height_m"]) for row in episode_step_rows if "height_m" in row)
    max_tilt = max(float(row["tilt_deg"]) for row in episode_step_rows if "tilt_deg" in row)
    max_abs_y = max(abs(float(row["y_m"])) for row in episode_step_rows if "y_m" in row)
    if min_height <= float(ground_height_m) + float(ground_margin_m):
        return "ground"
    if max_tilt >= float(terminate_tilt_deg) - float(tilt_margin_deg):
        return "tilt"
    if max_abs_y >= float(terminate_abs_y_m) - float(lateral_margin_m):
        return "lateral"
    return reason


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def summarize_rollout_records(
    step_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    *,
    action_saturation_threshold: float = 0.95,
) -> dict[str, float | int]:
    """Summarize rollout records into scalar diagnostics."""
    summary: dict[str, float | int] = {
        "episodes": len(episode_rows),
        "steps": len(step_rows),
        "mean_episode_steps": _mean([float(r["steps"]) for r in episode_rows]),
        "mean_abs_z_err_m": _mean([abs(float(r["z_err_m"])) for r in step_rows]),
        "mean_abs_vx_err_mps": _mean([abs(float(r["vx_err_mps"])) for r in step_rows]),
        "mean_abs_y_m": _mean([abs(float(r["y_m"])) for r in step_rows if "y_m" in r]),
        "mean_abs_roll_deg": _mean([abs(float(r["roll_deg"])) for r in step_rows if "roll_deg" in r]),
        "mean_abs_pitch_deg": _mean([abs(float(r["pitch_deg"])) for r in step_rows if "pitch_deg" in r]),
        "max_tilt_deg": max((float(r["tilt_deg"]) for r in step_rows), default=float("nan")),
    }

    reason_counts = Counter(str(r["termination_reason"]) for r in episode_rows)
    denom = max(len(episode_rows), 1)
    for reason in TERMINATION_REASONS:
        summary[f"{reason}_count"] = int(reason_counts.get(reason, 0))
        summary[f"{reason}_rate"] = float(reason_counts.get(reason, 0) / denom)

    sat_threshold = abs(float(action_saturation_threshold))
    any_sat = 0
    for action_col in ACTION_COLUMNS:
        values = [abs(float(r[action_col])) for r in step_rows if action_col in r]
        sat_count = sum(1 for v in values if v >= sat_threshold)
        summary[f"{action_col}_saturation_rate"] = float(sat_count / len(values)) if values else float("nan")
    for row in step_rows:
        if any(abs(float(row[col])) >= sat_threshold for col in ACTION_COLUMNS if col in row):
            any_sat += 1
    summary["any_action_saturation_rate"] = float(any_sat / len(step_rows)) if step_rows else float("nan")
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_yaml_compat(path: Path) -> dict:
    import yaml

    class _Loader(yaml.FullLoader):
        pass

    def _construct_posixpath(loader: yaml.FullLoader, node: yaml.Node):
        parts = loader.construct_sequence(node)
        return Path(*parts)

    _Loader.add_constructor("tag:yaml.org,2002:python/object/apply:pathlib.PosixPath", _construct_posixpath)
    with path.open("r") as f:
        return yaml.load(f, Loader=_Loader)


def _update_class_from_dict_allow_none(obj, data: dict, _ns: str = "") -> None:
    from isaaclab.utils.string import string_to_callable

    for key, value in data.items():
        key_ns = _ns + "/" + str(key)
        if hasattr(obj, key) or (isinstance(obj, dict) and key in obj):
            obj_mem = obj[key] if isinstance(obj, dict) else getattr(obj, key)

            if isinstance(value, Mapping):
                if obj_mem is None:
                    if isinstance(obj, dict):
                        obj[key] = value
                    else:
                        setattr(obj, key, value)
                    continue
                _update_class_from_dict_allow_none(obj_mem, value, _ns=key_ns)
                continue

            if isinstance(value, Iterable) and not isinstance(value, str):
                if all(not isinstance(el, Mapping) for el in value):
                    value = tuple(value) if isinstance(obj_mem, tuple) else value
                elif obj_mem is None:
                    pass
                elif isinstance(obj_mem, Sized) and isinstance(value, Sized) and len(obj_mem) != len(value):
                    value = tuple(value) if isinstance(obj_mem, tuple) else value
                else:
                    set_obj = True
                    for i in range(len(obj_mem)):
                        if isinstance(value[i], Mapping):
                            _update_class_from_dict_allow_none(obj_mem[i], value[i], _ns=key_ns)
                            set_obj = False
                    if not set_obj:
                        continue
            elif callable(obj_mem):
                value = string_to_callable(value)

            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)


def _parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Diagnose a FlappingBot straight-flight checkpoint.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps_per_episode", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--plot_episodes", type=int, default=4)
    parser.add_argument("--action_saturation_threshold", type=float, default=0.95)
    parser.add_argument("--no_saved_cfg", action="store_true")
    parser.add_argument("--vx_cmd", type=float, default=None)
    parser.add_argument("--height_cmd", type=float, default=None)
    parser.add_argument("--no_fixed_commands", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def _tensor_to_list(tensor, env_id: int, width: int) -> list[float]:
    return [float(tensor[env_id, i].item()) for i in range(width)]


def _euler_deg_from_env(env):
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat

    roll, pitch, yaw = euler_xyz_from_quat(env.unwrapped._robot.data.root_quat_w)
    return torch.rad2deg(roll), torch.rad2deg(pitch), torch.rad2deg(yaw)


def _reward_terms_from_env(env) -> dict[str, Any]:
    import torch

    u = env.unwrapped
    pos_w = u._robot.data.root_pos_w - u.scene.env_origins
    height = pos_w[:, 2]
    height_error = height - u._height_cmd
    r_height = 1.0 - torch.tanh(torch.abs(height_error) / 0.75)

    vx = u._robot.data.root_lin_vel_b[:, 0]
    vx_error = vx - u._vx_cmd
    r_vx = 1.0 - torch.tanh(torch.abs(vx_error) / 0.75)

    vy = u._robot.data.root_lin_vel_b[:, 1]
    p_vy = torch.tanh(torch.abs(vy) / 1.0)
    y = pos_w[:, 1]
    p_y = torch.tanh(torch.abs(y) / float(u.cfg.terminate_abs_y))

    roll_deg, pitch_deg, _yaw_deg = _euler_deg_from_env(env)
    pitch_cmd = -float(u.cfg.pitch_cmd_deg)
    roll_cmd = float(u.cfg.roll_cmd_deg)
    pitch_scale = max(float(u.cfg.att_pitch_err_deg), 1.0e-6)
    roll_scale = max(float(u.cfg.att_roll_err_deg), 1.0e-6)
    r_pitch = 1.0 - torch.tanh(torch.abs(pitch_deg - pitch_cmd) / pitch_scale)
    r_roll = 1.0 - torch.tanh(torch.abs(roll_deg - roll_cmd) / roll_scale)
    r_att = 0.5 * (r_pitch + r_roll)

    ang = torch.linalg.norm(u._robot.data.root_ang_vel_b, dim=1)
    p_ang = torch.tanh(ang)
    p_act = torch.sum(u._act_cmd**2, dim=1)

    weighted = (
        float(u.cfg.w_height) * r_height
        + float(u.cfg.w_vx) * r_vx
        + float(u.cfg.w_att) * r_att
        - float(u.cfg.w_vy) * p_vy
        - float(u.cfg.w_y) * p_y
        - float(u.cfg.w_act) * p_act
        - float(u.cfg.w_ang) * p_ang
    )
    return {
        "r_height": r_height,
        "r_vx": r_vx,
        "r_att": r_att,
        "p_vy": p_vy,
        "p_y": p_y,
        "p_ang": p_ang,
        "p_act": p_act,
        "reward_formula": weighted,
    }


def _snapshot_state(env) -> dict[str, Any]:
    import torch

    u = env.unwrapped
    pos_local = u._robot.data.root_pos_w - u.scene.env_origins
    lin_vel_b = u._robot.data.root_lin_vel_b
    ang_vel_b = u._robot.data.root_ang_vel_b
    g_b = u._robot.data.projected_gravity_b
    tilt_deg = torch.rad2deg(torch.asin(torch.clamp(torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2), 0.0, 1.0)))
    roll_deg, pitch_deg, yaw_deg = _euler_deg_from_env(env)
    reward_terms = _reward_terms_from_env(env)
    return {
        "pos_local": pos_local.detach().clone(),
        "lin_vel_b": lin_vel_b.detach().clone(),
        "ang_vel_b": ang_vel_b.detach().clone(),
        "tilt_deg": tilt_deg.detach().clone(),
        "roll_deg": roll_deg.detach().clone(),
        "pitch_deg": pitch_deg.detach().clone(),
        "yaw_deg": yaw_deg.detach().clone(),
        "vx_cmd": u._vx_cmd.detach().clone(),
        "height_cmd": u._height_cmd.detach().clone(),
        "freq_hz": u._freq.detach().clone(),
        "act_cmd": u._act_cmd.detach().clone(),
        "reward_terms": {key: value.detach().clone() for key, value in reward_terms.items()},
    }


def _make_step_row(
    *,
    episode_id: int,
    env_id: int,
    episode_step: int,
    sim_dt_s: float,
    state: dict[str, Any],
    raw_actions,
    reward,
) -> dict[str, Any]:
    pos = state["pos_local"]
    lin = state["lin_vel_b"]
    ang = state["ang_vel_b"]
    act_cmd = state["act_cmd"]
    terms = state["reward_terms"]
    row: dict[str, Any] = {
        "episode": episode_id,
        "env_id": env_id,
        "step": episode_step,
        "time_s": episode_step * sim_dt_s,
        "height_m": float(pos[env_id, 2].item()),
        "height_cmd_m": float(state["height_cmd"][env_id].item()),
        "z_err_m": float((pos[env_id, 2] - state["height_cmd"][env_id]).item()),
        "x_m": float(pos[env_id, 0].item()),
        "y_m": float(pos[env_id, 1].item()),
        "vx_mps": float(lin[env_id, 0].item()),
        "vx_cmd_mps": float(state["vx_cmd"][env_id].item()),
        "vx_err_mps": float((lin[env_id, 0] - state["vx_cmd"][env_id]).item()),
        "vy_mps": float(lin[env_id, 1].item()),
        "vz_mps": float(lin[env_id, 2].item()),
        "p_radps": float(ang[env_id, 0].item()),
        "q_radps": float(ang[env_id, 1].item()),
        "r_radps": float(ang[env_id, 2].item()),
        "roll_deg": float(state["roll_deg"][env_id].item()),
        "pitch_deg": float(state["pitch_deg"][env_id].item()),
        "yaw_deg": float(state["yaw_deg"][env_id].item()),
        "tilt_deg": float(state["tilt_deg"][env_id].item()),
        "freq_hz": float(state["freq_hz"][env_id].item()),
        "reward": float(reward[env_id].item()) if reward is not None else float("nan"),
    }
    raw = _tensor_to_list(raw_actions, env_id, 4)
    cmd = _tensor_to_list(act_cmd, env_id, 4)
    row.update({key: raw[i] for i, key in enumerate(RAW_ACTION_COLUMNS)})
    row.update({key: cmd[i] for i, key in enumerate(ACTION_COLUMNS)})
    for key, tensor in terms.items():
        row[key] = float(tensor[env_id].item())
    return row


def _plot_diagnostics(step_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path, plot_episodes: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_eps = [int(row["episode"]) for row in episode_rows[: max(0, int(plot_episodes))]]
    if not selected_eps:
        return
    selected = [row for row in step_rows if int(row["episode"]) in selected_eps]
    by_ep: dict[int, list[dict[str, Any]]] = {}
    for row in selected:
        by_ep.setdefault(int(row["episode"]), []).append(row)

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=False)
    for ep, rows in by_ep.items():
        rows = sorted(rows, key=lambda r: int(r["step"]))
        t = [float(r["time_s"]) for r in rows]
        axes[0].plot(t, [float(r["height_m"]) for r in rows], label=f"ep {ep}")
        axes[1].plot(t, [float(r["vx_mps"]) for r in rows], label=f"ep {ep}")
        axes[2].plot(t, [float(r["pitch_deg"]) for r in rows], label=f"pitch ep {ep}")
        axes[2].plot(t, [float(r["roll_deg"]) for r in rows], linestyle="--", label=f"roll ep {ep}")
        axes[3].plot(t, [float(r["freq_hz"]) for r in rows], label=f"freq ep {ep}")
    axes[0].set_ylabel("height [m]")
    axes[1].set_ylabel("vx body [m/s]")
    axes[2].set_ylabel("attitude [deg]")
    axes[3].set_ylabel("flap freq [Hz]")
    axes[3].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Straight-flight rollout state diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / "state_timeseries.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    for ep, rows in by_ep.items():
        rows = sorted(rows, key=lambda r: int(r["step"]))
        t = [float(r["time_s"]) for r in rows]
        axes[0].plot(t, [float(r["cmd_a0"]) for r in rows], label=f"freq action ep {ep}")
        axes[1].plot(t, [float(r["cmd_a1"]) for r in rows], label=f"rudder ep {ep}")
        axes[1].plot(t, [float(r["cmd_a2"]) for r in rows], linestyle="--", label=f"elev pitch ep {ep}")
        axes[1].plot(t, [float(r["cmd_a3"]) for r in rows], linestyle=":", label=f"elev roll ep {ep}")
        axes[2].plot(t, [float(r["reward"]) for r in rows], label=f"reward ep {ep}")
    axes[0].set_ylabel("cmd a0")
    axes[1].set_ylabel("cmd a1-a3")
    axes[2].set_ylabel("reward")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.axhline(0.95, color="tab:red", linewidth=0.8, alpha=0.4)
        ax.axhline(-0.95, color="tab:red", linewidth=0.8, alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Straight-flight action and reward diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / "action_reward_timeseries.png", dpi=180)
    plt.close(fig)

    reasons = [reason for reason in TERMINATION_REASONS if f"{reason}_count" in summary]
    counts = [float(summary[f"{reason}_count"]) for reason in reasons]
    sat_keys = [key for key in summary if key.endswith("_saturation_rate")]
    sat_vals = [float(summary[key]) for key in sat_keys]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(reasons, counts)
    axes[0].set_ylabel("episodes")
    axes[0].set_title("Termination reasons")
    axes[1].bar([key.replace("_saturation_rate", "") for key in sat_keys], sat_vals)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("rate")
    axes[1].set_title("Action saturation")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_bars.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    run_dir = checkpoint_path.parent
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "diagnostics" / checkpoint_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if not bool(args.no_saved_cfg):
        saved_env = run_dir / "params" / "env.yaml"
        if saved_env.is_file():
            _update_class_from_dict_allow_none(env_cfg, _load_yaml_compat(saved_env))
            env_cfg.sim.device = args.device
            env_cfg.scene.num_envs = int(args.num_envs)
    if not args.no_fixed_commands:
        env_cfg.randomize_commands = False
        if args.vx_cmd is not None:
            env_cfg.vx_cmd = float(args.vx_cmd)
        if args.height_cmd is not None:
            env_cfg.height_cmd = float(args.height_cmd)

    agent_cfg_dict = None
    if not bool(args.no_saved_cfg):
        saved_agent = run_dir / "params" / "agent.yaml"
        if saved_agent.is_file():
            agent_cfg_dict = _load_yaml_compat(saved_agent)
    if agent_cfg_dict is None:
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args.device if args.device is not None else agent_cfg.device
        agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict["device"] = args.device if args.device is not None else agent_cfg_dict.get("device", "cuda:0")

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_dict.get("clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg_dict["device"])
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    obs = env.get_observations()
    n_env = int(env.unwrapped.num_envs)
    max_len = int(env.unwrapped.max_episode_length)
    steps_per_ep = int(args.steps_per_episode) if args.steps_per_episode is not None else max_len
    steps_per_ep = max(1, min(steps_per_ep, max_len))
    sim_dt_s = float(env.unwrapped.step_dt)

    episode_ids = list(range(n_env))
    next_episode_id = n_env
    episode_steps = [0 for _ in range(n_env)]
    episode_step_rows: list[list[dict[str, Any]]] = [[] for _ in range(n_env)]
    completed = 0
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    while completed < int(args.episodes):
        pre_state = _snapshot_state(env)
        with torch.inference_mode():
            actions = policy(obs)
        obs, rewards, dones, _info = env.step(actions)
        policy_nn.reset(dones)

        done_ids = set(torch.nonzero(dones > 0, as_tuple=False).squeeze(-1).tolist())
        for env_id in range(n_env):
            if completed >= int(args.episodes):
                break
            row = _make_step_row(
                episode_id=episode_ids[env_id],
                env_id=env_id,
                episode_step=episode_steps[env_id],
                sim_dt_s=sim_dt_s,
                state=pre_state,
                raw_actions=actions,
                reward=rewards,
            )
            step_rows.append(row)
            episode_step_rows[env_id].append(row)
            episode_steps[env_id] += 1

            forced_timeout = episode_steps[env_id] >= steps_per_ep
            if env_id in done_ids or forced_timeout:
                terminated = bool(env.unwrapped.reset_terminated[env_id].item()) if env_id in done_ids else False
                timed_out = bool(env.unwrapped.reset_time_outs[env_id].item()) if env_id in done_ids else True
                reason = infer_termination_reason(
                    timed_out=timed_out or forced_timeout,
                    terminated=terminated,
                    height_m=float(pre_state["pos_local"][env_id, 2].item()),
                    tilt_deg=float(pre_state["tilt_deg"][env_id].item()),
                    abs_y_m=abs(float(pre_state["pos_local"][env_id, 1].item())),
                    ground_height_m=float(env.unwrapped.cfg.terminate_ground_height),
                    terminate_tilt_deg=float(env.unwrapped.cfg.terminate_tilt_deg),
                    terminate_abs_y_m=float(env.unwrapped.cfg.terminate_abs_y),
                )
                reason = refine_terminated_reason_from_episode_rows(
                    reason,
                    episode_step_rows[env_id],
                    ground_height_m=float(env.unwrapped.cfg.terminate_ground_height),
                    terminate_tilt_deg=float(env.unwrapped.cfg.terminate_tilt_deg),
                    terminate_abs_y_m=float(env.unwrapped.cfg.terminate_abs_y),
                )
                episode_rows.append(
                    {
                        "episode": episode_ids[env_id],
                        "env_id": env_id,
                        "steps": episode_steps[env_id],
                        "duration_s": episode_steps[env_id] * sim_dt_s,
                        "termination_reason": reason,
                        "terminated": int(terminated),
                        "time_out": int(timed_out or forced_timeout),
                    }
                )
                completed += 1
                episode_ids[env_id] = next_episode_id
                next_episode_id += 1
                episode_steps[env_id] = 0
                episode_step_rows[env_id] = []
                if forced_timeout and env_id not in done_ids:
                    env.reset()
                    obs = env.get_observations()

    summary = summarize_rollout_records(
        step_rows,
        episode_rows,
        action_saturation_threshold=float(args.action_saturation_threshold),
    )
    summary.update(
        {
            "task": args.task,
            "checkpoint": str(checkpoint_path),
            "num_envs": int(args.num_envs),
            "action_saturation_threshold": float(args.action_saturation_threshold),
        }
    )

    _write_csv(out_dir / "rollout_steps.csv", step_rows)
    _write_csv(out_dir / "episodes.csv", episode_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_diagnostics(step_rows, episode_rows, summary, out_dir, int(args.plot_episodes))

    print("[OK] Diagnostics complete.")
    print(json.dumps(summary, indent=2))
    print(f"[OK] Wrote diagnostics to: {out_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

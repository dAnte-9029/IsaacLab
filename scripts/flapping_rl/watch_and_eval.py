"""Watch a training run directory and evaluate new checkpoints.

This script is intended for long unattended runs:
- Training writes `model_*.pt` checkpoints into a run directory.
- This watcher evaluates each new checkpoint and appends metrics to `eval/summary.csv`.

Example:
  ./isaaclab.sh -p scripts/flapping_rl/watch_and_eval.py \
    --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 \
    --log_dir logs/rsl_rl/flapping_bot_straight_flight/<RUN> \
    --episodes 5 --num_envs 1 --poll_s 60 --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from collections.abc import Iterable, Mapping, Sized

from isaaclab.app import AppLauncher

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from checkpoint_selection import refresh_best_checkpoint_artifacts
from eval_suites import build_eval_cases, get_eval_suite_choices
from path_tracking_eval_common import (
    aggregate_case_row,
    aggregate_suite_row,
    apply_eval_case_to_cfg as _apply_eval_case_to_cfg_common,
    compute_path_tracking_finish_masks,
    compute_path_tracking_score,
    is_path_tracking_task,
    read_path_tracking_step_metrics,
    reset_path_tracking_eval_envs,
    row_meets_path_tracking_success_gate,
    resolve_eval_suite as _resolve_eval_suite_common,
    summarize_path_tracking_episode,
)
from pure_rl_eval_common import (
    PURE_RL_CURRICULUM1_EVAL_CONTRACT,
    PURE_RL_CURRICULUM1_EVAL_SUITE,
    aggregate_pure_rl_case_row,
    aggregate_pure_rl_suite_row,
    allocate_episode_quotas,
    assert_pure_rl_reset_schedule,
    compute_pure_rl_score,
    is_measured_pure_rl_task,
    read_pure_rl_step_metrics,
    summarize_pure_rl_episode,
)


def _resolve_eval_suite(task: str, eval_suite: str) -> str:
    resolved = _resolve_eval_suite_common(task, eval_suite)
    if resolved == "straight_standard" and is_measured_pure_rl_task(task):
        return PURE_RL_CURRICULUM1_EVAL_SUITE
    return resolved


def _apply_eval_case_to_cfg(case: dict, cfg, *, vx_cmd: float | None, height_cmd: float | None) -> None:
    _apply_eval_case_to_cfg_common(case, cfg, vx_cmd=vx_cmd, height_cmd=height_cmd)
    if "straight_line_heading_schedule_rad" in case:
        cfg.randomize_straight_line_heading = False
        cfg.randomize_flap_phase_at_reset = False
        cfg.pure_rl_eval_heading_schedule_rad = tuple(case["straight_line_heading_schedule_rad"])
        cfg.pure_rl_eval_flap_phase_schedule_rad = tuple(case["flap_phase_schedule_rad"])


def _resolve_eval_shape(
    task: str,
    eval_suite: str,
    *,
    num_envs: int | None,
    episodes: int | None,
) -> tuple[int, int]:
    """Resolve task-aware evaluation defaults while preserving explicit overrides."""

    pure_rl_grid = is_measured_pure_rl_task(task) and eval_suite == PURE_RL_CURRICULUM1_EVAL_SUITE
    resolved_num_envs = 16 if num_envs is None and pure_rl_grid else (1 if num_envs is None else int(num_envs))
    resolved_episodes = 16 if episodes is None and pure_rl_grid else (5 if episodes is None else int(episodes))
    if resolved_num_envs <= 0 or resolved_episodes <= 0:
        raise ValueError("Evaluation environment and episode counts must be positive.")
    return resolved_num_envs, resolved_episodes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a run directory and evaluate new checkpoints.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True, help="Run directory that contains model_*.pt files.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--poll_s", type=float, default=60.0)
    parser.add_argument("--once", action="store_true", help="Evaluate current checkpoints once and exit.")
    parser.add_argument(
        "--no_saved_cfg",
        action="store_true",
        help="Do not load env/agent config from <log_dir>/params/{env,agent}.yaml.",
    )
    # fixed-command evaluation
    parser.add_argument("--vx_cmd", type=float, default=None)
    parser.add_argument("--height_cmd", type=float, default=None)
    parser.add_argument(
        "--eval_suite",
        type=str,
        default="straight_standard",
        choices=get_eval_suite_choices(),
        help=(
            "Evaluation suite. `straight_standard` runs calm / steady-crosswind / OU-crosswind cases; "
            "`path_tracking_estimated_nowind_v1` freezes the first no-wind straight / loiter / random-mission target set."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def _extract_ckpt_index(p: Path) -> int:
    m = re.match(r"model_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else -1


def _score_row(row: dict) -> float:
    if row.get("evaluation_contract") == PURE_RL_CURRICULUM1_EVAL_CONTRACT:
        return compute_pure_rl_score(row)
    if "completion_rate" in row and "mean_abs_lateral_error_m" in row:
        return compute_path_tracking_score(
            completion_rate=float(row["completion_rate"]),
            mean_final_progress_ratio=float(row.get("mean_final_progress_ratio", row["completion_rate"])),
            mean_abs_lateral_error_m=float(row["mean_abs_lateral_error_m"]),
            mean_abs_height_error_m=float(row["mean_abs_height_error_m"]),
            mean_abs_align_error_deg=float(row["mean_abs_align_error_deg"]),
            termination_rate=float(row["termination_rate"]),
        )
    cost = (
        float(row["mean_abs_vx_err"])
        + float(row["mean_abs_z_err"])
        + 0.5 * float(row["mean_max_abs_y"])
        + 0.05 * float(row["mean_max_tilt_deg"])
        + 5.0 * float(row["termination_rate"])
    )
    return 100.0 / (1.0 + cost)


def _append_summary_row(summary_csv: Path, row: Mapping[str, object]) -> None:
    row_dict = dict(row)
    if not summary_csv.exists():
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
            writer.writeheader()
            writer.writerow(row_dict)
        return

    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        existing_fieldnames = list(reader.fieldnames or [])

    if existing_fieldnames == list(row_dict.keys()):
        with summary_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=existing_fieldnames)
            writer.writerow(row_dict)
        return

    merged_fieldnames = existing_fieldnames.copy()
    for key in row_dict.keys():
        if key not in merged_fieldnames:
            merged_fieldnames.append(key)

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow({key: existing_row.get(key, "") for key in merged_fieldnames})
        writer.writerow({key: row_dict.get(key, "") for key in merged_fieldnames})


def main():
    args = _parse_args()
    eval_suite = _resolve_eval_suite(args.task, args.eval_suite)
    args.num_envs, args.episodes = _resolve_eval_shape(
        args.task,
        eval_suite,
        num_envs=args.num_envs,
        episodes=args.episodes,
    )
    evaluation_contract = (
        PURE_RL_CURRICULUM1_EVAL_CONTRACT
        if is_measured_pure_rl_task(args.task)
        else None
    )

    def _update_class_from_dict_allow_none(obj, data: dict, _ns: str = "") -> None:
        """Update class from dict, but allow `None` defaults to be overwritten."""
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
                        out_val = tuple(value) if isinstance(obj_mem, tuple) else value
                        if isinstance(obj, dict):
                            obj[key] = out_val
                        else:
                            setattr(obj, key, out_val)
                        continue

                    if obj_mem is None:
                        if isinstance(obj, dict):
                            obj[key] = value
                        else:
                            setattr(obj, key, value)
                        continue

                    if isinstance(obj_mem, Sized) and isinstance(value, Sized) and len(obj_mem) != len(value):
                        out_val = tuple(value) if isinstance(obj_mem, tuple) else value
                        if isinstance(obj, dict):
                            obj[key] = out_val
                        else:
                            setattr(obj, key, out_val)
                        continue

                    if isinstance(obj_mem, tuple):
                        value = tuple(value)
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

                elif obj_mem is None or value is None or isinstance(value, type(obj_mem)):
                    pass
                else:
                    pass

                if isinstance(obj, dict):
                    obj[key] = value
                else:
                    setattr(obj, key, value)
            else:
                continue

    def _load_yaml_compat(path: Path) -> dict:
        # IsaacLab's dump_yaml can emit python-specific tags for pathlib.Path. Handle that safely.
        import yaml

        class _Loader(yaml.FullLoader):
            pass

        def _construct_posixpath(loader: yaml.FullLoader, node: yaml.Node):
            parts = loader.construct_sequence(node)
            return Path(*parts)

        _Loader.add_constructor("tag:yaml.org,2002:python/object/apply:pathlib.PosixPath", _construct_posixpath)

        with path.open("r") as f:
            return yaml.load(f, Loader=_Loader)

    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.is_dir():
        raise NotADirectoryError(log_dir)

    eval_dir = log_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = eval_dir / "summary.csv"

    # launch Isaac Sim once and reuse env/runner across checkpoints
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    use_saved_cfg = not bool(args.no_saved_cfg)
    if use_saved_cfg:
        saved_env = log_dir / "params" / "env.yaml"
        if saved_env.is_file():
            saved_dict = _load_yaml_compat(saved_env)
            _update_class_from_dict_allow_none(env_cfg, saved_dict)
            # Override runtime knobs after loading the saved cfg.
            env_cfg.sim.device = args.device
            env_cfg.scene.num_envs = int(args.num_envs)

    env_cfg.randomize_commands = False
    if args.vx_cmd is not None:
        env_cfg.vx_cmd = float(args.vx_cmd)
    if args.height_cmd is not None:
        env_cfg.height_cmd = float(args.height_cmd)

    agent_cfg_dict = None
    if use_saved_cfg:
        saved_agent = log_dir / "params" / "agent.yaml"
        if saved_agent.is_file():
            agent_cfg_dict = _load_yaml_compat(saved_agent)
    if agent_cfg_dict is None:
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args.device if args.device is not None else agent_cfg.device
        agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict["device"] = args.device if args.device is not None else agent_cfg_dict.get("device", "cuda:0")

    eval_cases = build_eval_cases(eval_suite)
    _apply_eval_case_to_cfg(eval_cases[0], env_cfg, vx_cmd=args.vx_cmd, height_cmd=args.height_cmd)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_dict.get("clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg_dict["device"])
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # track evaluated checkpoints
    evaluated: set[str] = set()
    if summary_csv.exists():
        with summary_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_name = row.get("case")
                row_contract = row.get("evaluation_contract") or None
                if (
                    "checkpoint" in row
                    and case_name in (None, "suite")
                    and row_contract == evaluation_contract
                ):
                    evaluated.add(row["checkpoint"])

    def _eval_case(ckpt: Path, case: dict) -> dict:
        # load weights
        runner.load(str(ckpt))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        _apply_eval_case_to_cfg(case, env.unwrapped.cfg, vx_cmd=args.vx_cmd, height_cmd=args.height_cmd)
        obs, _ = env.reset()
        policy_nn.reset(torch.ones(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device))

        if is_measured_pure_rl_task(args.task):
            if "straight_line_heading_schedule_rad" in case:
                assert_pure_rl_reset_schedule(
                    actual_heading_rad=env.unwrapped._straight_line_heading_rad,
                    actual_flap_phase_rad=env.unwrapped._phase,
                    expected_heading_schedule_rad=case["straight_line_heading_schedule_rad"],
                    expected_flap_phase_schedule_rad=case["flap_phase_schedule_rad"],
                )
            n_env = int(env.unwrapped.num_envs)
            target_episodes = int(args.episodes)
            quotas = allocate_episode_quotas(target_episodes, n_env)
            completed = [0 for _ in range(n_env)]
            metric_names = (
                "cross_track_error_m",
                "height_error_m",
                "along_track_progress_m",
                "along_track_velocity_mps",
                "tilt_rad",
                "angular_rate_rad_s",
                "actual_flap_frequency_hz",
                "frequency_limit_active",
                "tail_limit_active",
                "normalized_action_delta",
                "frequency_slew_hz_per_s",
                "frequency_governor_limited",
            )
            episode_metrics = [
                {name: [] for name in metric_names}
                for _ in range(n_env)
            ]
            episode_rows: list[dict[str, float | int]] = []

            while sum(completed) < target_episodes:
                with torch.inference_mode():
                    actions = policy(obs)
                obs, _rew, dones, _info = env.step(actions)
                step_metrics = read_pure_rl_step_metrics(env.unwrapped)

                for env_id in range(n_env):
                    if completed[env_id] >= quotas[env_id]:
                        continue
                    for name in metric_names:
                        episode_metrics[env_id][name].append(float(step_metrics[name][env_id].item()))

                done_ids = torch.nonzero(dones > 0, as_tuple=False).squeeze(-1)
                for env_id in done_ids.tolist():
                    if completed[env_id] < quotas[env_id] and episode_metrics[env_id]["tilt_rad"]:
                        episode_rows.append(
                            summarize_pure_rl_episode(
                                step_metrics=episode_metrics[env_id],
                                step_dt_s=float(env.unwrapped.step_dt),
                                terminated=bool(env.unwrapped.reset_terminated[env_id].item()),
                                time_out=bool(env.unwrapped.reset_time_outs[env_id].item()),
                                termination_causes={
                                    "ground": bool(step_metrics["ground_termination"][env_id].item()),
                                    "tilt": bool(step_metrics["tilt_termination"][env_id].item()),
                                    "cross_track": bool(step_metrics["cross_track_termination"][env_id].item()),
                                    "height": bool(step_metrics["height_termination"][env_id].item()),
                                },
                            )
                        )
                        completed[env_id] += 1
                    for values in episode_metrics[env_id].values():
                        values.clear()
                policy_nn.reset(dones)

            return aggregate_pure_rl_case_row(
                checkpoint=ckpt,
                ckpt_index=_extract_ckpt_index(ckpt),
                case=case,
                episode_rows=episode_rows,
            )

        if is_path_tracking_task(args.task):
            target_episodes = int(args.episodes)
            ep_done = 0
            step_abs_lateral = [[] for _ in range(env.unwrapped.num_envs)]
            step_abs_height = [[] for _ in range(env.unwrapped.num_envs)]
            step_abs_align_deg = [[] for _ in range(env.unwrapped.num_envs)]
            step_loiter_radial = [[] for _ in range(env.unwrapped.num_envs)]
            step_loiter_progress = [[] for _ in range(env.unwrapped.num_envs)]
            loiter_quarter_turn_complete = [False for _ in range(env.unwrapped.num_envs)]
            final_progress_ratio = [0.0 for _ in range(env.unwrapped.num_envs)]
            ep_rows: list[dict[str, float | int]] = []

            while ep_done < target_episodes:
                with torch.inference_mode():
                    actions = policy(obs)
                obs, _rew, dones, _info = env.step(actions)

                step_metrics = read_path_tracking_step_metrics(env.unwrapped)
                lateral_error = step_metrics["abs_lateral_error_m"]
                height_error = step_metrics["abs_height_error_m"]
                align_error_deg = step_metrics["abs_align_error_deg"]
                progress_ratio = step_metrics["progress_ratio"]
                loiter_radial_error = step_metrics["loiter_radial_error_m"]
                loiter_progress_ratio = step_metrics["loiter_progress_ratio"]
                loiter_quarter_turn = step_metrics["loiter_quarter_turn_complete"]

                for env_id in range(env.unwrapped.num_envs):
                    step_abs_lateral[env_id].append(float(lateral_error[env_id].item()))
                    step_abs_height[env_id].append(float(height_error[env_id].item()))
                    step_abs_align_deg[env_id].append(float(align_error_deg[env_id].item()))
                    if torch.isfinite(loiter_radial_error[env_id]):
                        step_loiter_radial[env_id].append(float(loiter_radial_error[env_id].item()))
                    if torch.isfinite(loiter_progress_ratio[env_id]):
                        step_loiter_progress[env_id].append(float(loiter_progress_ratio[env_id].item()))
                    loiter_quarter_turn_complete[env_id] = loiter_quarter_turn_complete[env_id] or bool(
                        loiter_quarter_turn[env_id].item()
                    )
                    final_progress_ratio[env_id] = float(progress_ratio[env_id].item())

                finish_mask, early_success_mask, _completed_mask = compute_path_tracking_finish_masks(
                    progress_ratio=progress_ratio,
                    dones=dones,
                    completion_ratio=0.98,
                )
                finish_ids = torch.nonzero(finish_mask, as_tuple=False).squeeze(-1)

                for env_id in finish_ids.tolist():
                    if ep_done >= target_episodes:
                        break
                    ep_rows.append(
                        summarize_path_tracking_episode(
                            step_abs_lateral=step_abs_lateral[env_id],
                            step_abs_height=step_abs_height[env_id],
                            step_abs_align_deg=step_abs_align_deg[env_id],
                            step_loiter_radial_error=step_loiter_radial[env_id],
                            step_loiter_progress_ratio=step_loiter_progress[env_id],
                            loiter_quarter_turn_complete=loiter_quarter_turn_complete[env_id],
                            final_progress_ratio=float(final_progress_ratio[env_id]),
                            completion_ratio=0.98,
                            terminated=bool(env.unwrapped.reset_terminated[env_id].item()),
                            time_out=bool(env.unwrapped.reset_time_outs[env_id].item()),
                            stalled=bool(
                                getattr(env.unwrapped, "reset_stalled", env.unwrapped.reset_terminated * 0)[env_id].item()
                            ),
                        )
                    )
                    ep_done += 1
                    step_abs_lateral[env_id].clear()
                    step_abs_height[env_id].clear()
                    step_abs_align_deg[env_id].clear()
                    step_loiter_radial[env_id].clear()
                    step_loiter_progress[env_id].clear()
                    loiter_quarter_turn_complete[env_id] = False
                    final_progress_ratio[env_id] = 0.0

                early_success_ids = torch.nonzero(early_success_mask, as_tuple=False).squeeze(-1)
                if early_success_ids.numel() > 0:
                    obs = reset_path_tracking_eval_envs(env, early_success_ids)

                reset_flags = dones.clone()
                if early_success_ids.numel() > 0:
                    reset_flags[early_success_ids] = 1
                policy_nn.reset(reset_flags)

            return aggregate_case_row(
                checkpoint=ckpt,
                ckpt_index=_extract_ckpt_index(ckpt),
                case=case,
                episode_rows=ep_rows,
            )

        # per-episode aggregation for straight-flight metrics
        n_env = int(env.unwrapped.num_envs)
        target_episodes = int(args.episodes)

        sum_abs_vx = torch.zeros(n_env, device=env.unwrapped.device)
        sum_abs_z = torch.zeros(n_env, device=env.unwrapped.device)
        max_abs_y = torch.zeros(n_env, device=env.unwrapped.device)
        max_tilt = torch.zeros(n_env, device=env.unwrapped.device)
        steps = torch.zeros(n_env, dtype=torch.long, device=env.unwrapped.device)

        ep_rows = []
        ep_done = 0
        while ep_done < target_episodes:
            with torch.inference_mode():
                actions = policy(obs)
            obs, _rew, dones, _info = env.step(actions)
            policy_nn.reset(dones)

            pos_local = env.unwrapped._robot.data.root_pos_w - env.unwrapped.scene.env_origins
            vx = env.unwrapped._robot.data.root_lin_vel_b[:, 0]
            z = pos_local[:, 2]
            y = pos_local[:, 1].abs()
            g_b = env.unwrapped._robot.data.projected_gravity_b
            tilt = torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2)
            vx_cmd = env.unwrapped._vx_cmd
            z_cmd = env.unwrapped._height_cmd

            sum_abs_vx += (vx - vx_cmd).abs()
            sum_abs_z += (z - z_cmd).abs()
            max_abs_y = torch.maximum(max_abs_y, y)
            max_tilt = torch.maximum(max_tilt, tilt)
            steps += 1

            done_ids = torch.nonzero(dones > 0, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                for i in done_ids.tolist():
                    if ep_done >= target_episodes:
                        break
                    n_steps = int(steps[i].item())
                    if n_steps <= 0:
                        continue
                    ep_rows.append(
                        {
                            "mean_abs_vx_err": float((sum_abs_vx[i] / n_steps).item()),
                            "mean_abs_z_err": float((sum_abs_z[i] / n_steps).item()),
                            "max_abs_y": float(max_abs_y[i].item()),
                            "max_tilt_deg": float(torch.rad2deg(torch.asin(max_tilt[i].clamp(0.0, 1.0))).item()),
                            "terminated": int(env.unwrapped.reset_terminated[i].item()),
                            "time_out": int(env.unwrapped.reset_time_outs[i].item()),
                        }
                    )
                    ep_done += 1
                    sum_abs_vx[i] = 0.0
                    sum_abs_z[i] = 0.0
                    max_abs_y[i] = 0.0
                    max_tilt[i] = 0.0
                    steps[i] = 0

        mean_abs_vx = sum(r["mean_abs_vx_err"] for r in ep_rows) / len(ep_rows)
        mean_abs_z = sum(r["mean_abs_z_err"] for r in ep_rows) / len(ep_rows)
        mean_max_y = sum(r["max_abs_y"] for r in ep_rows) / len(ep_rows)
        mean_max_tilt = sum(r["max_tilt_deg"] for r in ep_rows) / len(ep_rows)
        term_rate = sum(r["terminated"] for r in ep_rows) / len(ep_rows)
        timeout_rate = sum(r["time_out"] for r in ep_rows) / len(ep_rows)

        row = {
            "checkpoint": str(ckpt),
            "case": str(case["name"]),
            "ckpt_index": _extract_ckpt_index(ckpt),
            "episodes": target_episodes,
            "mean_abs_vx_err": mean_abs_vx,
            "mean_abs_z_err": mean_abs_z,
            "mean_max_abs_y": mean_max_y,
            "mean_max_tilt_deg": mean_max_tilt,
            "termination_rate": term_rate,
            "timeout_rate": timeout_rate,
            "wind_enabled": int(bool(case["wind_enabled"])),
            "wind_x_mps": float(case["wind_xy_mps"][0]),
            "wind_y_mps": float(case["wind_xy_mps"][1]),
            "wind_ou_enabled": int(bool(case["wind_ou_enabled"])),
            "wind_ou_sigma_x_mps": float(case["wind_ou_sigma_xy_mps"][0]),
            "wind_ou_sigma_y_mps": float(case["wind_ou_sigma_xy_mps"][1]),
        }
        row["score"] = _score_row(row)
        return row

    def _eval_checkpoint(ckpt: Path) -> list[dict]:
        case_rows = [_eval_case(ckpt, case) for case in eval_cases]
        if is_measured_pure_rl_task(args.task):
            suite_row = aggregate_pure_rl_suite_row(case_rows)
            return case_rows + [suite_row]
        if is_path_tracking_task(args.task):
            suite_row = aggregate_suite_row(case_rows)
            suite_row["score"] = _score_row(suite_row)
            return case_rows + [suite_row]
        suite_row = {
            "checkpoint": str(ckpt),
            "case": "suite",
            "ckpt_index": _extract_ckpt_index(ckpt),
            "episodes": sum(int(r["episodes"]) for r in case_rows),
            "mean_abs_vx_err": sum(float(r["mean_abs_vx_err"]) for r in case_rows) / len(case_rows),
            "mean_abs_z_err": sum(float(r["mean_abs_z_err"]) for r in case_rows) / len(case_rows),
            "mean_max_abs_y": sum(float(r["mean_max_abs_y"]) for r in case_rows) / len(case_rows),
            "mean_max_tilt_deg": sum(float(r["mean_max_tilt_deg"]) for r in case_rows) / len(case_rows),
            "termination_rate": sum(float(r["termination_rate"]) for r in case_rows) / len(case_rows),
            "timeout_rate": sum(float(r["timeout_rate"]) for r in case_rows) / len(case_rows),
            "wind_enabled": int(any(bool(r["wind_enabled"]) for r in case_rows)),
            "wind_x_mps": float("nan"),
            "wind_y_mps": float("nan"),
            "wind_ou_enabled": int(any(bool(r["wind_ou_enabled"]) for r in case_rows)),
            "wind_ou_sigma_x_mps": float("nan"),
            "wind_ou_sigma_y_mps": float("nan"),
        }
        suite_row["score"] = sum(float(r["score"]) for r in case_rows) / len(case_rows)
        return case_rows + [suite_row]

    # main watch loop
    try:
        while True:
            ckpts = sorted(log_dir.glob("model_*.pt"), key=_extract_ckpt_index)
            new_ckpts = [p for p in ckpts if str(p) not in evaluated]

            for ckpt in new_ckpts:
                rows = _eval_checkpoint(ckpt)
                for row in rows:
                    _append_summary_row(summary_csv, row)
                (eval_dir / f"{Path(ckpt).stem}.json").write_text(json.dumps(rows, indent=2))
                best_row = refresh_best_checkpoint_artifacts(
                    log_dir,
                    summary_csv=summary_csv,
                    evaluation_contract=evaluation_contract,
                )
                evaluated.add(str(ckpt))
                suite_row = next(row for row in rows if row["case"] == "suite")
                print(
                    "[OK] Evaluated:",
                    ckpt.name,
                    {
                        "suite_score": suite_row["score"],
                        "suite_success_gate_passed": suite_row.get("success_gate_passed"),
                        "best_checkpoint": None if best_row is None else best_row["checkpoint"],
                        "best_score": None if best_row is None else best_row["score"],
                        "best_success_gate_passed": None if best_row is None else best_row.get("success_gate_passed"),
                    },
                )

            if args.once:
                break

            time.sleep(float(args.poll_s))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

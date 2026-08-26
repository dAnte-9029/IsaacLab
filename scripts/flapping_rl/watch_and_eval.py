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
import math
import os
import re
import sys
import time
import traceback
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
    assert_pure_rl_longitudinal_reset_schedule,
    assert_pure_rl_reset_schedule,
    assert_pure_rl_spatial_reset_schedule,
    compute_pure_rl_score,
    is_measured_pure_rl_task,
    longitudinal_stage_for_task,
    read_pure_rl_step_metrics,
    spatial_stage_for_task,
    summarize_pure_rl_episode,
)
from pure_rl_longitudinal_eval import (
    LONGITUDINAL_EVAL_CONTRACTS,
    build_longitudinal_diagnostic_grid,
    build_longitudinal_evaluation_grid,
    row_meets_longitudinal_promotion_gate,
    summarize_longitudinal_evaluation,
)
from pure_rl_spatial_eval import (
    SPATIAL_EVAL_CONTRACTS,
    build_spatial_evaluation_grid,
    row_meets_spatial_promotion_gate,
    summarize_spatial_evaluation,
)


def _resolve_eval_suite(task: str, eval_suite: str) -> str:
    resolved = _resolve_eval_suite_common(task, eval_suite)
    if resolved == "straight_standard" and is_measured_pure_rl_task(task):
        stage_id = longitudinal_stage_for_task(task)
        if stage_id is not None:
            return LONGITUDINAL_EVAL_CONTRACTS[stage_id]
        spatial_stage_id = spatial_stage_for_task(task)
        if spatial_stage_id is not None:
            return SPATIAL_EVAL_CONTRACTS[spatial_stage_id]
        return PURE_RL_CURRICULUM1_EVAL_SUITE
    return resolved


def _apply_eval_case_to_cfg(case: dict, cfg, *, vx_cmd: float | None, height_cmd: float | None) -> None:
    _apply_eval_case_to_cfg_common(case, cfg, vx_cmd=vx_cmd, height_cmd=height_cmd)
    if "straight_line_heading_schedule_rad" in case:
        cfg.randomize_straight_line_heading = False
        cfg.randomize_flap_phase_at_reset = False
        cfg.pure_rl_eval_heading_schedule_rad = tuple(case["straight_line_heading_schedule_rad"])
        cfg.pure_rl_eval_flap_phase_schedule_rad = tuple(case["flap_phase_schedule_rad"])
    if "longitudinal_task_schedule" in case:
        cfg.pure_rl_eval_longitudinal_task_schedule = tuple(case["longitudinal_task_schedule"])
        cfg.pure_rl_eval_longitudinal_slope_deg_schedule = tuple(
            case["longitudinal_slope_deg_schedule"]
        )
        cfg.pure_rl_eval_entry_length_m_schedule = tuple(
            case["longitudinal_entry_length_m_schedule"]
        )
        cfg.pure_rl_eval_slope_length_m_schedule = tuple(
            case["longitudinal_slope_length_m_schedule"]
        )
    if "spatial_template_schedule" in case:
        cfg.pure_rl_eval_spatial_template_schedule = tuple(case["spatial_template_schedule"])
        cfg.pure_rl_eval_spatial_geometry_roll_deg_schedule = tuple(
            case["spatial_geometry_roll_deg_schedule"]
        )
        cfg.pure_rl_eval_spatial_slope_deg_schedule = tuple(case["spatial_slope_deg_schedule"])
        cfg.pure_rl_eval_spatial_turn_sign_schedule = tuple(case["spatial_turn_sign_schedule"])


def _resolve_eval_shape(
    task: str,
    eval_suite: str,
    *,
    num_envs: int | None,
    episodes: int | None,
) -> tuple[int, int]:
    """Resolve task-aware evaluation defaults while preserving explicit overrides."""

    stage_id = longitudinal_stage_for_task(task)
    spatial_stage_id = spatial_stage_for_task(task)
    if stage_id is not None and eval_suite == LONGITUDINAL_EVAL_CONTRACTS[stage_id]:
        default_count = len(build_longitudinal_evaluation_grid(stage_id))
    elif spatial_stage_id is not None and eval_suite == SPATIAL_EVAL_CONTRACTS[spatial_stage_id]:
        default_count = len(build_spatial_evaluation_grid(spatial_stage_id))
    elif is_measured_pure_rl_task(task) and eval_suite == PURE_RL_CURRICULUM1_EVAL_SUITE:
        default_count = 16
    else:
        default_count = None
    resolved_num_envs = default_count if num_envs is None and default_count is not None else (
        1 if num_envs is None else int(num_envs)
    )
    resolved_episodes = default_count if episodes is None and default_count is not None else (
        5 if episodes is None else int(episodes)
    )
    if resolved_num_envs <= 0 or resolved_episodes <= 0:
        raise ValueError("Evaluation environment and episode counts must be positive.")
    return resolved_num_envs, resolved_episodes


def _apply_robot_asset_overrides(
    env_cfg,
    *,
    asset_path: str | None,
    usd_dir: str | None,
) -> None:
    """Apply an explicit source asset and writable conversion cache as one contract."""

    if (asset_path is None) != (usd_dir is None):
        raise ValueError("--robot-asset-path and --robot-usd-dir must be provided together.")
    if asset_path is None:
        return

    resolved_asset_path = Path(asset_path).expanduser().resolve()
    if not resolved_asset_path.is_file():
        raise FileNotFoundError(f"Robot source asset does not exist: {resolved_asset_path}")
    resolved_usd_dir = Path(usd_dir).expanduser().resolve()
    resolved_usd_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.robot.spawn.asset_path = str(resolved_asset_path)
    env_cfg.robot.spawn.usd_dir = str(resolved_usd_dir)


def _resolve_robot_asset_overrides(
    task: str,
    log_dir: Path,
    *,
    asset_path: str | None,
    usd_dir: str | None,
) -> tuple[str | None, str | None]:
    """Default measured PureRL conversion to an evaluation-owned writable directory."""

    if asset_path is not None or usd_dir is not None or not is_measured_pure_rl_task(task):
        return asset_path, usd_dir
    repo_root = Path(__file__).resolve().parents[2]
    default_asset = (
        repo_root
        / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
    ).resolve()
    default_usd_dir = (log_dir / "eval/generated_assets/flap_robot_552").resolve()
    return str(default_asset), str(default_usd_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a run directory and evaluate new checkpoints.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True, help="Run directory that contains model_*.pt files.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--poll_s", type=float, default=60.0)
    parser.add_argument("--once", action="store_true", help="Evaluate current checkpoints once and exit.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Evaluate this exact checkpoint instead of scanning <log_dir>/model_*.pt; requires --once.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Evaluation output directory. Defaults to <log_dir>/eval.",
    )
    parser.add_argument(
        "--no-best-artifacts",
        action="store_true",
        help="Do not update best_checkpoint files or symlinks in the source run directory.",
    )
    parser.add_argument(
        "--no_saved_cfg",
        action="store_true",
        help="Do not load env/agent config from <log_dir>/params/{env,agent}.yaml.",
    )
    parser.add_argument(
        "--actor-hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Explicit actor hidden dimensions for checkpoints whose network differs from the registered default.",
    )
    parser.add_argument(
        "--robot-asset-path",
        type=str,
        default=None,
        help="Optional robot source asset override; requires --robot-usd-dir.",
    )
    parser.add_argument(
        "--robot-usd-dir",
        type=str,
        default=None,
        help="Writable robot conversion directory; requires --robot-asset-path.",
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


def _resolve_checkpoint_candidates(log_dir: Path, checkpoint: Path | None) -> list[Path]:
    """Resolve either one explicit checkpoint or the run-directory checkpoint scan."""

    if checkpoint is None:
        return sorted(
            (path.resolve() for path in log_dir.glob("model_*.pt")),
            key=_extract_ckpt_index,
        )
    resolved = checkpoint.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved}")
    if _extract_ckpt_index(resolved) < 0:
        raise ValueError("Explicit checkpoint must be named model_<iteration>.pt.")
    return [resolved]


def _score_row(row: dict) -> float:
    if row.get("evaluation_contract") in SPATIAL_EVAL_CONTRACTS.values():
        path_error = float(row["mean_abs_horizontal_error_m"]) + float(
            row["mean_abs_vertical_error_m"]
        )
        return 100.0 * float(row["overall_success_rate"]) / (1.0 + path_error)
    if row.get("evaluation_contract") in LONGITUDINAL_EVAL_CONTRACTS.values():
        survival = float(row["overall_survival_rate"])
        path_error = float(row["mean_abs_cross_track_error_m"]) + float(
            row["mean_abs_height_error_m"]
        )
        return 100.0 * survival / (1.0 + path_error)
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


def _report_evaluation_failure(error: BaseException) -> None:
    """Flush an evaluation traceback before Isaac Sim starts fast shutdown."""

    print("[ERROR] Checkpoint evaluation failed before Isaac shutdown.", file=sys.stderr, flush=True)
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    sys.stderr.flush()


def main():
    args = _parse_args()
    if args.checkpoint is not None and not bool(args.once):
        raise ValueError("--checkpoint requires --once so direct evaluation cannot become a watcher.")
    eval_suite = _resolve_eval_suite(args.task, args.eval_suite)
    args.num_envs, args.episodes = _resolve_eval_shape(
        args.task,
        eval_suite,
        num_envs=args.num_envs,
        episodes=args.episodes,
    )
    longitudinal_stage_id = longitudinal_stage_for_task(args.task)
    spatial_stage_id = spatial_stage_for_task(args.task)
    evaluation_contract = (
        LONGITUDINAL_EVAL_CONTRACTS[longitudinal_stage_id]
        if longitudinal_stage_id is not None
        else (
            SPATIAL_EVAL_CONTRACTS[spatial_stage_id]
            if spatial_stage_id is not None
            else (PURE_RL_CURRICULUM1_EVAL_CONTRACT if is_measured_pure_rl_task(args.task) else None)
        )
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
    args.robot_asset_path, args.robot_usd_dir = _resolve_robot_asset_overrides(
        args.task,
        log_dir,
        asset_path=args.robot_asset_path,
        usd_dir=args.robot_usd_dir,
    )

    eval_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else log_dir / "eval"
    )
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
    _apply_robot_asset_overrides(
        env_cfg,
        asset_path=args.robot_asset_path,
        usd_dir=args.robot_usd_dir,
    )
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
    if args.actor_hidden_dims is not None:
        actor_hidden_dims = [int(value) for value in args.actor_hidden_dims]
        if not actor_hidden_dims or any(value <= 0 for value in actor_hidden_dims):
            raise ValueError("--actor-hidden-dims values must be positive integers.")
        policy_cfg = agent_cfg_dict.get("policy")
        if not isinstance(policy_cfg, dict):
            raise ValueError("Agent configuration does not contain a policy dictionary.")
        policy_cfg["actor_hidden_dims"] = actor_hidden_dims
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
            if spatial_stage_id is not None:
                path = env.unwrapped._pure_rl_spatial_path
                if path is None:
                    raise RuntimeError("C3 evaluation task did not allocate spatial path state.")
                assert_pure_rl_spatial_reset_schedule(
                    actual_heading_rad=env.unwrapped._straight_line_heading_rad,
                    actual_flap_phase_rad=env.unwrapped._phase,
                    actual_template_id=path.template_id,
                    actual_geometry_roll_rad=path.peak_geometry_roll_rad,
                    actual_slope_rad=path.peak_slope_rad,
                    actual_turn_sign=path.turn_sign,
                    expected_heading_schedule_rad=case["straight_line_heading_schedule_rad"],
                    expected_flap_phase_schedule_rad=case["flap_phase_schedule_rad"],
                    expected_template_schedule=case["spatial_template_schedule"],
                    expected_geometry_roll_deg_schedule=case[
                        "spatial_geometry_roll_deg_schedule"
                    ],
                    expected_slope_deg_schedule=case["spatial_slope_deg_schedule"],
                    expected_turn_sign_schedule=case["spatial_turn_sign_schedule"],
                )
                registered_cases = build_spatial_evaluation_grid(spatial_stage_id)
                if tuple(item.case_id for item in registered_cases) != tuple(case["spatial_case_ids"]):
                    raise RuntimeError("Evaluation case schedule does not match its registered C3 grid.")
                n_env = int(env.unwrapped.num_envs)
                target_episodes = len(registered_cases)
                if n_env < target_episodes:
                    raise ValueError(
                        "C3 fixed-grid evaluation requires at least one environment per registered case."
                    )
                completed = [False for _ in range(n_env)]
                metric_names = (
                    "cross_track_error_m",
                    "height_error_m",
                    "along_track_velocity_mps",
                    "abs_roll_rad",
                    "reached_all_events",
                    "roll_limit_termination",
                )
                episode_metrics = [{name: [] for name in metric_names} for _ in range(n_env)]
                episode_rows: list[dict[str, object]] = []

                while len(episode_rows) < target_episodes:
                    with torch.inference_mode():
                        actions = policy(obs)
                    obs, _rew, dones, _info = env.step(actions)
                    step_metrics = read_pure_rl_step_metrics(env.unwrapped)
                    for env_id in range(target_episodes):
                        if completed[env_id]:
                            continue
                        for name in metric_names:
                            episode_metrics[env_id][name].append(
                                float(step_metrics[name][env_id].item())
                            )

                    done_ids = torch.nonzero(dones > 0, as_tuple=False).squeeze(-1)
                    for env_id in done_ids.tolist():
                        if env_id >= target_episodes or completed[env_id]:
                            continue
                        trace = episode_metrics[env_id]
                        if not trace["height_error_m"]:
                            continue
                        registered_case = registered_cases[env_id]
                        terminated = bool(env.unwrapped.reset_terminated[env_id].item())
                        events_reached = any(bool(value) for value in trace["reached_all_events"])
                        numeric_values = (
                            trace["cross_track_error_m"]
                            + trace["height_error_m"]
                            + trace["along_track_velocity_mps"]
                            + trace["abs_roll_rad"]
                        )
                        episode_rows.append(
                            {
                                "case_id": registered_case.case_id,
                                "stage_id": registered_case.stage_id,
                                "template_id": registered_case.template_id,
                                "turn_sign": registered_case.turn_sign,
                                "vertical_sign": registered_case.vertical_sign,
                                "severity_id": registered_case.severity_id,
                                "terminated": terminated,
                                "events_reached": events_reached,
                                "success": (not terminated) and events_reached,
                                "horizontal_normal_error_m": trace["cross_track_error_m"],
                                "vertical_normal_error_m": trace["height_error_m"],
                                "tangent_velocity_mps": trace["along_track_velocity_mps"],
                                "roll_rad": trace["abs_roll_rad"],
                                "roll_limit_termination": any(
                                    bool(value) for value in trace["roll_limit_termination"]
                                ),
                                "finite_metrics": all(math.isfinite(value) for value in numeric_values),
                            }
                        )
                        completed[env_id] = True
                    policy_nn.reset(dones)

                row = summarize_spatial_evaluation(
                    episode_rows,
                    expected_cases=registered_cases,
                    checkpoint=str(ckpt),
                    ppo_iteration=_extract_ckpt_index(ckpt),
                )
                row["case"] = str(case["name"])
                row["episodes"] = target_episodes
                row["termination_rate"] = 1.0 - float(row["overall_survival_rate"])
                row["timeout_rate"] = float(row["overall_survival_rate"])
                row["promotion_gate_passed"] = row_meets_spatial_promotion_gate(row)
                row["success_gate_passed"] = int(row["promotion_gate_passed"])
                row["c1_retention_passed"] = 0
                row["score"] = _score_row(row)
                return row

            if longitudinal_stage_id is not None:
                path = env.unwrapped._pure_rl_longitudinal_path
                if path is None:
                    raise RuntimeError("C2 evaluation task did not allocate longitudinal path state.")
                assert_pure_rl_longitudinal_reset_schedule(
                    actual_heading_rad=env.unwrapped._straight_line_heading_rad,
                    actual_flap_phase_rad=env.unwrapped._phase,
                    actual_task_id=path.task_id,
                    actual_signed_slope_rad=path.signed_slope_rad,
                    expected_heading_schedule_rad=case["straight_line_heading_schedule_rad"],
                    expected_flap_phase_schedule_rad=case["flap_phase_schedule_rad"],
                    expected_task_schedule=case["longitudinal_task_schedule"],
                    expected_signed_slope_deg_schedule=case["longitudinal_slope_deg_schedule"],
                )
                registered_cases = (
                    build_longitudinal_evaluation_grid(longitudinal_stage_id)
                    if bool(case["promotion_eligible"])
                    else build_longitudinal_diagnostic_grid(longitudinal_stage_id)
                )
                expected_case_ids = tuple(item.case_id for item in registered_cases)
                if expected_case_ids != tuple(case["longitudinal_case_ids"]):
                    raise RuntimeError("Evaluation case schedule does not match its registered C2 grid.")
                n_env = int(env.unwrapped.num_envs)
                target_episodes = len(registered_cases)
                if n_env < target_episodes:
                    raise ValueError(
                        "C2 fixed-grid evaluation requires at least one environment per registered case."
                    )
                completed = [False for _ in range(n_env)]
                metric_names = (
                    "cross_track_error_m",
                    "height_error_m",
                    "along_track_velocity_mps",
                    "reached_recovery",
                )
                episode_metrics = [{name: [] for name in metric_names} for _ in range(n_env)]
                episode_rows: list[dict[str, object]] = []

                while len(episode_rows) < target_episodes:
                    with torch.inference_mode():
                        actions = policy(obs)
                    obs, _rew, dones, _info = env.step(actions)
                    step_metrics = read_pure_rl_step_metrics(env.unwrapped)
                    for env_id in range(target_episodes):
                        if completed[env_id]:
                            continue
                        for name in metric_names:
                            episode_metrics[env_id][name].append(float(step_metrics[name][env_id].item()))

                    done_ids = torch.nonzero(dones > 0, as_tuple=False).squeeze(-1)
                    for env_id in done_ids.tolist():
                        if env_id >= target_episodes or completed[env_id]:
                            continue
                        trace = episode_metrics[env_id]
                        if not trace["height_error_m"]:
                            continue
                        registered_case = registered_cases[env_id]
                        terminated = bool(env.unwrapped.reset_terminated[env_id].item())
                        recovery_reached = any(bool(value) for value in trace["reached_recovery"])
                        numeric_values = (
                            trace["cross_track_error_m"]
                            + trace["height_error_m"]
                            + trace["along_track_velocity_mps"]
                        )
                        episode_rows.append(
                            {
                                "case_id": registered_case.case_id,
                                "stage_id": registered_case.stage_id,
                                "task": registered_case.task,
                                "promotion_eligible": registered_case.promotion_eligible,
                                "terminated": terminated,
                                "success": (not terminated) and recovery_reached,
                                "recovery_reached": recovery_reached,
                                "termination_causes": {
                                    "ground": bool(step_metrics["ground_termination"][env_id].item()),
                                    "tilt": bool(step_metrics["tilt_termination"][env_id].item()),
                                    "cross_track": bool(
                                        step_metrics["cross_track_termination"][env_id].item()
                                    ),
                                    "height_error": bool(
                                        step_metrics["height_termination"][env_id].item()
                                    ),
                                },
                                "cross_track_error_m": trace["cross_track_error_m"],
                                "height_error_m": trace["height_error_m"],
                                "tangent_velocity_mps": trace["along_track_velocity_mps"],
                                "finite_metrics": all(math.isfinite(value) for value in numeric_values),
                            }
                        )
                        completed[env_id] = True
                    policy_nn.reset(dones)

                row = summarize_longitudinal_evaluation(
                    episode_rows,
                    expected_cases=registered_cases,
                    checkpoint=str(ckpt),
                    ppo_iteration=_extract_ckpt_index(ckpt),
                )
                row["case"] = str(case["name"])
                row["episodes"] = target_episodes
                row["termination_rate"] = 1.0 - float(row["overall_survival_rate"])
                row["timeout_rate"] = float(row["overall_survival_rate"])
                row["promotion_gate_passed"] = bool(
                    bool(case["promotion_eligible"])
                    and row_meets_longitudinal_promotion_gate(row)
                )
                row["success_gate_passed"] = int(row["promotion_gate_passed"])
                row["c1_retention_passed"] = 0
                row["score"] = _score_row(row)
                return row

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
        if longitudinal_stage_id is not None or spatial_stage_id is not None:
            promotion_row = next(row for row in case_rows if "promotion_grid" in str(row["case"]))
            suite_row = dict(promotion_row)
            suite_row["case"] = "suite"
            return case_rows + [suite_row]
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
            ckpts = _resolve_checkpoint_candidates(log_dir, args.checkpoint)
            new_ckpts = [p for p in ckpts if str(p) not in evaluated]

            for ckpt in new_ckpts:
                rows = _eval_checkpoint(ckpt)
                for row in rows:
                    _append_summary_row(summary_csv, row)
                (eval_dir / f"{Path(ckpt).stem}.json").write_text(json.dumps(rows, indent=2))
                best_row = None
                if not bool(args.no_best_artifacts):
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
    except Exception as error:
        _report_evaluation_failure(error)
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

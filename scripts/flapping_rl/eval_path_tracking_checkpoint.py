"""Evaluate a trained generic path-tracking policy checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from collections.abc import Iterable, Mapping, Sized

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_suites import build_eval_cases, get_eval_suite_choices
from path_tracking_eval_common import aggregate_case_row, aggregate_suite_row, apply_eval_case_to_cfg, resolve_eval_suite


def _add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--headless", action="store_true")
    else:
        AppLauncher.add_app_launcher_args(parser)


def build_eval_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for path-tracking checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a generic path-tracking checkpoint.")
    parser.add_argument("--task", type=str, required=True, help="Gym task id.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint file.")
    parser.add_argument("--eval_suite", type=str, default="straight_standard", choices=get_eval_suite_choices())
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--completion_ratio", type=float, default=0.98)
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <run_dir>/eval_path_tracking).")
    parser.add_argument(
        "--no_saved_cfg",
        action="store_true",
        help="Do not load env/agent config from <run_dir>/params/{env,agent}.yaml.",
    )
    parser.add_argument("--vx_cmd", type=float, default=None)
    parser.add_argument("--height_cmd", type=float, default=None)
    _add_app_launcher_args(parser)
    return parser


def _parse_args() -> argparse.Namespace:
    parser = build_eval_parser()
    args, _ = parser.parse_known_args()
    return args


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values = sorted(float(v) for v in values)
    idx = (len(values) - 1) * max(0.0, min(1.0, q))
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return values[lower]
    alpha = idx - lower
    return values[lower] + (values[upper] - values[lower]) * alpha


def main() -> None:
    args = _parse_args()

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:
        raise RuntimeError("IsaacLab runtime is required to evaluate checkpoints.") from exc

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

                if isinstance(obj, dict):
                    obj[key] = value
                else:
                    setattr(obj, key, value)

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

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    run_dir = checkpoint_path.parent
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (run_dir / "eval_path_tracking")
    out_dir.mkdir(parents=True, exist_ok=True)

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
        saved_env = run_dir / "params" / "env.yaml"
        if saved_env.is_file():
            saved_dict = _load_yaml_compat(saved_env)
            _update_class_from_dict_allow_none(env_cfg, saved_dict)
            env_cfg.sim.device = args.device
            env_cfg.scene.num_envs = int(args.num_envs)

    agent_cfg_dict = None
    if use_saved_cfg:
        saved_agent = run_dir / "params" / "agent.yaml"
        if saved_agent.is_file():
            agent_cfg_dict = _load_yaml_compat(saved_agent)
    if agent_cfg_dict is None:
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args.device if args.device is not None else agent_cfg.device
        agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict["device"] = args.device if args.device is not None else agent_cfg_dict.get("device", "cuda:0")

    eval_suite = resolve_eval_suite(args.task, args.eval_suite)
    eval_cases = build_eval_cases(eval_suite)
    apply_eval_case_to_cfg(eval_cases[0], env_cfg, vx_cmd=args.vx_cmd, height_cmd=args.height_cmd)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_dict.get("clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg_dict["device"])
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    checkpoint_name = checkpoint_path.name
    ckpt_index = int(checkpoint_name.split("_")[1].split(".")[0]) if checkpoint_name.startswith("model_") else -1
    case_rows: list[dict[str, float | int | str]] = []

    for case in eval_cases:
        apply_eval_case_to_cfg(case, env.unwrapped.cfg, vx_cmd=args.vx_cmd, height_cmd=args.height_cmd)
        obs, _ = env.reset()
        policy_nn.reset(torch.ones(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device))

        ep_done = 0
        step_abs_lateral = [[] for _ in range(env.unwrapped.num_envs)]
        step_abs_height = [[] for _ in range(env.unwrapped.num_envs)]
        step_abs_align_deg = [[] for _ in range(env.unwrapped.num_envs)]
        final_progress_ratio = [0.0 for _ in range(env.unwrapped.num_envs)]
        episode_rows: list[dict[str, float | int]] = []

        while ep_done < int(args.episodes):
            with torch.inference_mode():
                actions = policy(obs)
            obs, _rew, dones, _info = env.step(actions)
            policy_nn.reset(dones)

            env.unwrapped._refresh_path_state()
            lateral_error = torch.abs(env.unwrapped._path_lateral_error_m).detach().cpu()
            height_error = torch.abs(env.unwrapped._path_height_error_m).detach().cpu()
            align_error_deg = torch.rad2deg(torch.abs(env.unwrapped._path_align_error_rad)).detach().cpu()
            progress_s = env.unwrapped._path_progress_s.detach().cpu()

            for env_id in range(env.unwrapped.num_envs):
                step_abs_lateral[env_id].append(float(lateral_error[env_id].item()))
                step_abs_height[env_id].append(float(height_error[env_id].item()))
                step_abs_align_deg[env_id].append(float(align_error_deg[env_id].item()))
                manager = env.unwrapped._path_managers[env_id]
                total_length = float(manager.total_length_m) if manager is not None else 1.0
                final_progress_ratio[env_id] = min(max(float(progress_s[env_id].item()) / max(total_length, 1.0e-6), 0.0), 1.0)

            done_ids = torch.nonzero(dones > 0, as_tuple=False).squeeze(-1)
            if done_ids.numel() == 0:
                continue

            for env_id in done_ids.tolist():
                if ep_done >= int(args.episodes):
                    break
                episode_rows.append(
                    {
                        "mean_abs_lateral_error_m": float(sum(step_abs_lateral[env_id]) / max(len(step_abs_lateral[env_id]), 1)),
                        "mean_abs_height_error_m": float(sum(step_abs_height[env_id]) / max(len(step_abs_height[env_id]), 1)),
                        "mean_abs_align_error_deg": float(sum(step_abs_align_deg[env_id]) / max(len(step_abs_align_deg[env_id]), 1)),
                        "p95_abs_lateral_error_m": _percentile(step_abs_lateral[env_id], 0.95),
                        "p95_abs_height_error_m": _percentile(step_abs_height[env_id], 0.95),
                        "p95_abs_align_error_deg": _percentile(step_abs_align_deg[env_id], 0.95),
                        "final_progress_ratio": float(final_progress_ratio[env_id]),
                        "completed": int(final_progress_ratio[env_id] >= float(args.completion_ratio)),
                        "terminated": int(env.unwrapped.reset_terminated[env_id].item()),
                        "time_out": int(env.unwrapped.reset_time_outs[env_id].item()),
                    }
                )
                ep_done += 1
                step_abs_lateral[env_id].clear()
                step_abs_height[env_id].clear()
                step_abs_align_deg[env_id].clear()
                final_progress_ratio[env_id] = 0.0

        case_rows.append(
            aggregate_case_row(
                checkpoint=checkpoint_path,
                ckpt_index=ckpt_index,
                case=case,
                episode_rows=episode_rows,
            )
        )

    suite_row = aggregate_suite_row(case_rows)
    rows = case_rows + [suite_row]

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(suite_row, indent=2))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

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
import time
from pathlib import Path
from collections.abc import Iterable, Mapping, Sized

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a run directory and evaluate new checkpoints.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True, help="Run directory that contains model_*.pt files.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num_envs", type=int, default=1)
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
        choices=("straight_standard", "single"),
        help="Evaluation suite. `straight_standard` runs calm / steady-crosswind / OU-crosswind cases.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def _extract_ckpt_index(p: Path) -> int:
    m = re.match(r"model_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else -1


def _default_eval_cases(eval_suite: str) -> list[dict]:
    if eval_suite == "single":
        return [
            {
                "name": "single",
                "wind_enabled": False,
                "wind_xy_mps": (0.0, 0.0),
                "wind_ou_enabled": False,
                "wind_ou_tau_s": 2.0,
                "wind_ou_sigma_xy_mps": (0.0, 0.0),
            }
        ]

    return [
        {
            "name": "calm",
            "wind_enabled": False,
            "wind_xy_mps": (0.0, 0.0),
            "wind_ou_enabled": False,
            "wind_ou_tau_s": 2.0,
            "wind_ou_sigma_xy_mps": (0.0, 0.0),
        },
        {
            "name": "crosswind_steady",
            "wind_enabled": True,
            "wind_xy_mps": (0.0, 2.0),
            "wind_ou_enabled": False,
            "wind_ou_tau_s": 2.0,
            "wind_ou_sigma_xy_mps": (0.0, 0.0),
        },
        {
            "name": "crosswind_ou",
            "wind_enabled": True,
            "wind_xy_mps": (0.0, 1.5),
            "wind_ou_enabled": True,
            "wind_ou_tau_s": 2.0,
            "wind_ou_sigma_xy_mps": (0.0, 0.8),
        },
    ]


def _score_row(row: dict) -> float:
    cost = (
        float(row["mean_abs_vx_err"])
        + float(row["mean_abs_z_err"])
        + 0.5 * float(row["mean_max_abs_y"])
        + 0.05 * float(row["mean_max_tilt_deg"])
        + 5.0 * float(row["termination_rate"])
    )
    return 100.0 / (1.0 + cost)


def main():
    args = _parse_args()

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

    def _apply_eval_case(case: dict, cfg) -> None:
        cfg.randomize_commands = False
        if args.vx_cmd is not None:
            cfg.vx_cmd = float(args.vx_cmd)
        if args.height_cmd is not None:
            cfg.height_cmd = float(args.height_cmd)

        if hasattr(cfg, "teacher_guidance_enabled"):
            cfg.teacher_guidance_enabled = False
        if hasattr(cfg, "wind_curriculum_enabled"):
            cfg.wind_curriculum_enabled = False

        cfg.wind_enabled = bool(case["wind_enabled"])
        cfg.randomize_wind = False
        cfg.wind_xy_mps = tuple(float(v) for v in case["wind_xy_mps"])
        cfg.wind_x_range_mps = (float(case["wind_xy_mps"][0]), float(case["wind_xy_mps"][0]))
        cfg.wind_y_range_mps = (float(case["wind_xy_mps"][1]), float(case["wind_xy_mps"][1]))
        cfg.wind_ou_enabled = bool(case["wind_ou_enabled"])
        cfg.wind_ou_tau_s = float(case["wind_ou_tau_s"])
        cfg.wind_ou_sigma_xy_mps = tuple(float(v) for v in case["wind_ou_sigma_xy_mps"])
        cfg.wind_ou_clip_to_range = False

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

    eval_cases = _default_eval_cases(args.eval_suite)
    _apply_eval_case(eval_cases[0], env_cfg)

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
                if "checkpoint" in row and (case_name in (None, "suite")):
                    evaluated.add(row["checkpoint"])

    def _append_row(row: dict):
        write_header = not summary_csv.exists()
        with summary_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _eval_case(ckpt: Path, case: dict) -> dict:
        # load weights
        runner.load(str(ckpt))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        _apply_eval_case(case, env.unwrapped.cfg)
        obs, _ = env.reset()
        policy_nn.reset(torch.ones(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device))

        # per-episode aggregation (similar to eval_straight_flight_checkpoint.py but kept minimal)
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
                    _append_row(row)
                (eval_dir / f"{Path(ckpt).stem}.json").write_text(json.dumps(rows, indent=2))
                evaluated.add(str(ckpt))
                suite_row = next(row for row in rows if row["case"] == "suite")
                print("[OK] Evaluated:", ckpt.name, {"suite_score": suite_row["score"], "rows": rows})

            if args.once:
                break

            time.sleep(float(args.poll_s))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

"""Evaluate a trained straight-flight policy checkpoint.

This script runs a fixed number of episodes and reports tracking / stability metrics.

Example:
  ./isaaclab.sh -p scripts/flapping_rl/eval_straight_flight_checkpoint.py \
    --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 \
    --checkpoint logs/rsl_rl/flapping_bot_straight_flight/<RUN>/model_200.pt \
    --episodes 10 --num_envs 1 --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from collections.abc import Iterable, Mapping, Sized

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a FlappingBot straight-flight checkpoint.")
    parser.add_argument("--task", type=str, required=True, help="Gym task id.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint file.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps_per_episode", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <run_dir>/eval).")
    parser.add_argument(
        "--no_saved_cfg",
        action="store_true",
        help="Do not load env/agent config from <run_dir>/params/{env,agent}.yaml.",
    )
    # fixed-command evaluation (recommended for early milestones)
    parser.add_argument("--vx_cmd", type=float, default=None)
    parser.add_argument("--height_cmd", type=float, default=None)
    parser.add_argument("--no_fixed_commands", action="store_true", help="Do not override commands to fixed values.")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def _tilt_deg_from_projected_gravity(g_bx: float, g_by: float) -> float:
    # projected_gravity_b has magnitude ~1; sqrt(gbx^2+gby^2) ~= sin(tilt)
    s = max(0.0, min(1.0, math.sqrt(g_bx * g_bx + g_by * g_by)))
    return math.degrees(math.asin(s))


def main():
    args = _parse_args()

    def _update_class_from_dict_allow_none(obj, data: dict, _ns: str = "") -> None:
        """Update class from dict, but allow `None` defaults to be overwritten.

        IsaacLab's strict `update_class_from_dict` rejects updates like `seed: None -> 42`.
        For evaluation we want to replay the saved `params/env.yaml` and `params/agent.yaml`,
        so we relax only this specific constraint while keeping callable resolution.
        """
        from isaaclab.utils.string import string_to_callable

        for key, value in data.items():
            key_ns = _ns + "/" + str(key)

            if hasattr(obj, key) or (isinstance(obj, dict) and key in obj):
                obj_mem = obj[key] if isinstance(obj, dict) else getattr(obj, key)

                # 1) nested mapping
                if isinstance(value, Mapping):
                    if obj_mem is None:
                        # best-effort: set directly (rare in our configs)
                        if isinstance(obj, dict):
                            obj[key] = value
                        else:
                            setattr(obj, key, value)
                        continue
                    _update_class_from_dict_allow_none(obj_mem, value, _ns=key_ns)
                    continue

                # 2) iterable (list/tuple/...)
                if isinstance(value, Iterable) and not isinstance(value, str):
                    # 2a) flat iterable -> replace wholesale
                    if all(not isinstance(el, Mapping) for el in value):
                        out_val = tuple(value) if isinstance(obj_mem, tuple) else value
                        if isinstance(obj, dict):
                            obj[key] = out_val
                        else:
                            setattr(obj, key, out_val)
                        continue

                    # 2b) existing value is None -> allow replacing
                    if obj_mem is None:
                        if isinstance(obj, dict):
                            obj[key] = value
                        else:
                            setattr(obj, key, value)
                        continue

                    # 2c) length mismatch -> fall back to overwrite
                    if isinstance(obj_mem, Sized) and isinstance(value, Sized) and len(obj_mem) != len(value):
                        out_val = tuple(value) if isinstance(obj_mem, tuple) else value
                        if isinstance(obj, dict):
                            obj[key] = out_val
                        else:
                            setattr(obj, key, out_val)
                        continue

                    # 2d) recurse elementwise for list-of-mappings (do not overwrite cfg objects)
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

                # 3) callable attribute -> resolve string
                elif callable(obj_mem):
                    value = string_to_callable(value)

                # 4) scalar / explicit None / allow None->value
                elif obj_mem is None or value is None or isinstance(value, type(obj_mem)):
                    pass

                # 5) type mismatch -> fall back to direct assignment (best-effort)
                else:
                    pass

                # 6) final assignment
                if isinstance(obj, dict):
                    obj[key] = value
                else:
                    setattr(obj, key, value)
            else:
                # Ignore unknown keys for forward-compatibility.
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

    # launch Isaac Sim
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
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (run_dir / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # env cfg
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    use_saved_cfg = not bool(args.no_saved_cfg)
    if use_saved_cfg:
        saved_env = run_dir / "params" / "env.yaml"
        if saved_env.is_file():
            saved_dict = _load_yaml_compat(saved_env)
            _update_class_from_dict_allow_none(env_cfg, saved_dict)
            # Override evaluation runtime knobs after loading the saved cfg.
            env_cfg.sim.device = args.device
            env_cfg.scene.num_envs = int(args.num_envs)

    if not args.no_fixed_commands:
        env_cfg.randomize_commands = False
        if args.vx_cmd is not None:
            env_cfg.vx_cmd = float(args.vx_cmd)
        if args.height_cmd is not None:
            env_cfg.height_cmd = float(args.height_cmd)

    # agent cfg
    agent_cfg_dict = None
    if use_saved_cfg:
        saved_agent = run_dir / "params" / "agent.yaml"
        if saved_agent.is_file():
            agent_cfg_dict = _load_yaml_compat(saved_agent)
    if agent_cfg_dict is None:
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args.device if args.device is not None else agent_cfg.device
        agent_cfg_dict = agent_cfg.to_dict()
    # Force the runner device to the evaluation device.
    agent_cfg_dict["device"] = args.device if args.device is not None else agent_cfg_dict.get("device", "cuda:0")

    # create env + wrapper
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_dict.get("clip_actions", None))

    # runner + policy
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg_dict["device"])
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # evaluation loop
    obs = env.get_observations()
    n_env = int(env.unwrapped.num_envs)
    max_len = int(env.unwrapped.max_episode_length)
    steps_per_ep = int(args.steps_per_episode) if args.steps_per_episode is not None else max_len
    steps_per_ep = max(1, min(steps_per_ep, max_len))

    ep_done = 0
    ep_idx = 0
    # per-env accumulators
    sum_abs_vx = torch.zeros(n_env, device=env.unwrapped.device)
    sum_abs_z = torch.zeros(n_env, device=env.unwrapped.device)
    max_abs_y = torch.zeros(n_env, device=env.unwrapped.device)
    max_tilt = torch.zeros(n_env, device=env.unwrapped.device)
    steps = torch.zeros(n_env, dtype=torch.long, device=env.unwrapped.device)

    rows: list[dict] = []

    while ep_done < int(args.episodes):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _rew, dones, _info = env.step(actions)
        policy_nn.reset(dones)

        # metrics from env state
        pos_local = env.unwrapped._robot.data.root_pos_w - env.unwrapped.scene.env_origins  # (N,3)
        vx = env.unwrapped._robot.data.root_lin_vel_b[:, 0]
        z = pos_local[:, 2]
        y = pos_local[:, 1].abs()
        g_b = env.unwrapped._robot.data.projected_gravity_b
        tilt = torch.sqrt(g_b[:, 0] ** 2 + g_b[:, 1] ** 2)  # ~= sin(tilt)

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
                if ep_done >= int(args.episodes):
                    break
                n_steps = int(steps[i].item())
                if n_steps <= 0:
                    continue
                mean_abs_vx = float((sum_abs_vx[i] / n_steps).item())
                mean_abs_z = float((sum_abs_z[i] / n_steps).item())
                max_y = float(max_abs_y[i].item())
                tilt_deg = float(math.degrees(math.asin(min(1.0, float(max_tilt[i].item())))))
                terminated = bool(env.unwrapped.reset_terminated[i].item())
                time_out = bool(env.unwrapped.reset_time_outs[i].item())
                rows.append(
                    {
                        "episode": ep_idx,
                        "env_id": i,
                        "steps": n_steps,
                        "mean_abs_vx_err": mean_abs_vx,
                        "mean_abs_z_err": mean_abs_z,
                        "max_abs_y": max_y,
                        "max_tilt_deg": tilt_deg,
                        "terminated": int(terminated),
                        "time_out": int(time_out),
                    }
                )
                ep_idx += 1
                ep_done += 1
                # reset accumulators for that env
                sum_abs_vx[i] = 0.0
                sum_abs_z[i] = 0.0
                max_abs_y[i] = 0.0
                max_tilt[i] = 0.0
                steps[i] = 0

        # hard cap episode length for evaluation consistency (even if env doesn't time out yet)
        if steps_per_ep < max_len:
            over = torch.nonzero(steps >= steps_per_ep, as_tuple=False).squeeze(-1)
            if over.numel() > 0:
                # force reset by calling env.reset(); direct env resets all at once, so keep it simple
                env.reset()
                obs = env.get_observations()

    # write outputs
    per_ep_csv = out_dir / f"eval_{checkpoint_path.stem}.csv"
    with per_ep_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["episode"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    if rows:
        mean_abs_vx = sum(r["mean_abs_vx_err"] for r in rows) / len(rows)
        mean_abs_z = sum(r["mean_abs_z_err"] for r in rows) / len(rows)
        mean_max_y = sum(r["max_abs_y"] for r in rows) / len(rows)
        mean_max_tilt = sum(r["max_tilt_deg"] for r in rows) / len(rows)
        term_rate = sum(r["terminated"] for r in rows) / len(rows)
        timeout_rate = sum(r["time_out"] for r in rows) / len(rows)
    else:
        mean_abs_vx = mean_abs_z = mean_max_y = mean_max_tilt = float("nan")
        term_rate = timeout_rate = float("nan")

    summary = {
        "task": args.task,
        "checkpoint": str(checkpoint_path),
        "episodes": int(args.episodes),
        "num_envs": int(args.num_envs),
        "mean_abs_vx_err": mean_abs_vx,
        "mean_abs_z_err": mean_abs_z,
        "mean_max_abs_y": mean_max_y,
        "mean_max_tilt_deg": mean_max_tilt,
        "termination_rate": term_rate,
        "timeout_rate": timeout_rate,
    }
    summary_json = out_dir / f"eval_{checkpoint_path.stem}.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    print("[OK] Evaluation complete.")
    print(json.dumps(summary, indent=2))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

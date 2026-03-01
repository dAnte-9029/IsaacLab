"""Open-loop rollout helper for FlappingBot straight-flight env.

This is for debugging dynamics and sign conventions without training.

Examples:
  ./isaaclab.sh -p scripts/flapping_rl/open_loop_straight_flight.py --task Isaac-FlappingBot-StraightFlight-Simple-Direct-v0 --headless --steps 2000 --f-hz 3.8
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-loop rollout for FlappingBot straight-flight env.")
    parser.add_argument("--task", type=str, required=True, help="Gym task id.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--f-hz", type=float, default=None, help="Hold flapping frequency (Hz). If omitted, uses cfg.")
    parser.add_argument("--elevator-deg", type=float, default=0.0, help="Hold elevator deflection (deg).")
    parser.add_argument("--rudder-deg", type=float, default=0.0, help="Hold rudder deflection (deg).")
    parser.add_argument("--vx-cmd", type=float, default=None, help="Override commanded forward speed (m/s).")
    parser.add_argument("--height-cmd", type=float, default=None, help="Override commanded height (m).")
    parser.add_argument("--reset-pitch-deg", type=float, default=None, help="Override reset pitch (deg, nose-up).")
    parser.add_argument("--reset-flap-hz", type=float, default=None, help="Override reset flap frequency (Hz).")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def _map_f_to_action(f_hz: float, f_min: float, f_max: float) -> float:
    f_hz = max(f_min, min(f_max, f_hz))
    return 2.0 * (f_hz - f_min) / (f_max - f_min) - 1.0


def _map_defl_to_action(defl_rad: float, joint_lower: float, joint_upper: float) -> float:
    # inverse of: cmd = mid + half * a
    mid = 0.5 * (joint_lower + joint_upper)
    half = 0.5 * (joint_upper - joint_lower)
    if half < 1e-9:
        return 0.0
    return float((defl_rad - mid) / half)


def main():
    args = _parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

    # Apply simple overrides for convenience.
    if args.vx_cmd is not None:
        env_cfg.vx_cmd = float(args.vx_cmd)
    if args.height_cmd is not None:
        env_cfg.height_cmd = float(args.height_cmd)
    if args.reset_pitch_deg is not None:
        env_cfg.reset_pitch_deg = float(args.reset_pitch_deg)
    if args.reset_flap_hz is not None:
        env_cfg.reset_flap_hz = float(args.reset_flap_hz)

    env = gym.make(args.task, cfg=env_cfg)
    device = env.unwrapped.device

    obs, _ = env.reset()
    assert "policy" in obs

    # Build constant actions.
    actions = torch.zeros(args.num_envs, env.unwrapped.cfg.action_space, device=device)

    # a0: frequency
    if args.f_hz is not None:
        actions[:, 0] = _map_f_to_action(float(args.f_hz), env.unwrapped.cfg.min_flap_hz, env.unwrapped.cfg.max_flap_hz)

    # a1/a2: tail deflections mapped through joint limit intersection inside env, so we invert using the resolved limits.
    ele = math.radians(float(args.elevator_deg))
    rud = math.radians(float(args.rudder_deg))
    jl = env.unwrapped._joint_lower_limits  # noqa: SLF001
    ju = env.unwrapped._joint_upper_limits  # noqa: SLF001
    idx_ele = env.unwrapped._IDX_LEFT_TAIL  # noqa: SLF001
    idx_rud = env.unwrapped._IDX_RIGHT_TAIL  # noqa: SLF001
    actions[:, 1] = _map_defl_to_action(ele, float(jl[idx_ele].item()), float(ju[idx_ele].item()))
    actions[:, 2] = _map_defl_to_action(rud, float(jl[idx_rud].item()), float(ju[idx_rud].item()))
    actions = torch.clamp(actions, -1.0, 1.0)

    # Rollout stats (env0 only).
    z_hist = []
    vx_hist = []
    pitch_hist = []
    roll_hist = []
    done_count = 0

    for _ in range(int(args.steps)):
        obs, rew, term, trunc, info = env.step(actions)
        if (term | trunc).any():
            done_count += int((term | trunc).sum().item())
            env.reset()

        # env-local pose and body velocities for env0.
        pos_w = env.unwrapped._robot.data.root_pos_w[0] - env.unwrapped.scene.env_origins[0]  # noqa: SLF001
        lin_b = env.unwrapped._robot.data.root_lin_vel_b[0]  # noqa: SLF001
        quat_w = env.unwrapped._robot.data.root_quat_w[0]  # noqa: SLF001
        from isaaclab.utils.math import euler_xyz_from_quat  # local import (jit scripted)

        roll, pitch, _yaw = euler_xyz_from_quat(quat_w.unsqueeze(0))
        roll_hist.append(float(roll[0].item()))
        pitch_hist.append(float(pitch[0].item()))
        z_hist.append(float(pos_w[2].item()))
        vx_hist.append(float(lin_b[0].item()))

    # Summaries.
    import numpy as np

    def _mean(x):
        return float(np.mean(np.asarray(x))) if len(x) else float("nan")

    # Flush explicitly: Isaac Sim teardown may terminate the process without flushing stdio buffers.
    print("[OK] Open-loop rollout completed.", flush=True)
    print(f"  done_count={done_count}", flush=True)
    print(f"  mean z={_mean(z_hist):.3f} m, mean vx={_mean(vx_hist):.3f} m/s", flush=True)
    print(
        f"  mean pitch={math.degrees(_mean(pitch_hist)):.2f} deg, mean roll={math.degrees(_mean(roll_hist)):.2f} deg",
        flush=True,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

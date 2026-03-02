"""Run PX4-like straight-line control on the flapping DeLaurier environment.

Example:
  ./isaaclab.sh -p scripts/flapping_px4/fly_straight_line.py \
    --task Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0 \
    --num_envs 1 --steps 3000 --headless
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PX4-like straight-line control for FlappingBot.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0",
        help="Gym task id.",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--line_length", type=float, default=120.0)
    parser.add_argument("--height_sp", type=float, default=10.0)
    parser.add_argument("--pitch_trim_deg", type=float, default=13.0)
    parser.add_argument("--freq_trim_hz", type=float, default=3.8)
    parser.add_argument("--enable_speed_hold", action="store_true")
    parser.add_argument("--speed_sp", type=float, default=7.0)
    parser.add_argument("--out_dir", type=Path, default=Path("logs/flapping_px4/straight_line"))
    parser.add_argument("--print_every", type=int, default=250)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = _parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.math import euler_xyz_from_quat
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    try:
        from flapping_bot.px4_like import PX4LikeStraightLineController, PX4LikeStraightLineControllerCfg
    except ModuleNotFoundError:
        from flapping_bot.flapping_bot.px4_like import PX4LikeStraightLineController, PX4LikeStraightLineControllerCfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.randomize_commands = False
    env_cfg.height_cmd = float(args.height_sp)
    env_cfg.action_space = 4
    if args.enable_speed_hold:
        env_cfg.vx_cmd = float(args.speed_sp)
    env = gym.make(args.task, cfg=env_cfg)

    env.reset()
    controller_cfg = PX4LikeStraightLineControllerCfg(
        line_start_xy=(0.0, 0.0),
        line_end_xy=(float(args.line_length), 0.0),
        height_sp_m=float(args.height_sp),
        pitch_trim_deg=float(args.pitch_trim_deg),
        freq_trim_hz=float(args.freq_trim_hz),
        min_flap_hz=float(env.unwrapped.cfg.min_flap_hz),
        max_flap_hz=float(env.unwrapped.cfg.max_flap_hz),
        enable_speed_hold=bool(args.enable_speed_hold),
        speed_sp_mps=float(args.speed_sp),
    )
    controller = PX4LikeStraightLineController(controller_cfg, device=env.unwrapped.device)

    resets = 0
    traj_rows: list[dict[str, float]] = []
    abs_track_err_hist: list[float] = []
    abs_course_err_deg_hist: list[float] = []

    for step in range(int(args.steps)):
        pos_local = env.unwrapped._robot.data.root_pos_w - env.unwrapped.scene.env_origins
        vel_local = env.unwrapped._robot.data.root_lin_vel_w
        roll, pitch, yaw = euler_xyz_from_quat(env.unwrapped._robot.data.root_quat_w)
        ang_vel_b = env.unwrapped._robot.data.root_ang_vel_b

        actions, diag = controller.compute_actions(
            pos_local=pos_local,
            ground_vel_local=vel_local,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            ang_vel_body=ang_vel_b,
        )

        _, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        if done.any():
            resets += int(done.sum().item())
            env.reset()

        # Log env-0 trajectory for quick diagnostics.
        idx = 0
        course_err_deg = float(torch.rad2deg(torch.abs(diag["course_err"][idx])).item())
        track_err_abs = float(torch.abs(diag["signed_track_error"][idx]).item())
        abs_course_err_deg_hist.append(course_err_deg)
        abs_track_err_hist.append(track_err_abs)
        traj_rows.append(
            {
                "step": float(step),
                "x": float(pos_local[idx, 0].item()),
                "y": float(pos_local[idx, 1].item()),
                "z": float(pos_local[idx, 2].item()),
                "vx": float(vel_local[idx, 0].item()),
                "vy": float(vel_local[idx, 1].item()),
                "vz": float(vel_local[idx, 2].item()),
                "roll_deg": float(torch.rad2deg(roll[idx]).item()),
                "pitch_deg": float(torch.rad2deg(pitch[idx]).item()),
                "yaw_deg": float(torch.rad2deg(yaw[idx]).item()),
                "course_sp_deg": float(torch.rad2deg(diag["course_sp"][idx]).item()),
                "roll_sp_deg": float(torch.rad2deg(diag["roll_sp"][idx]).item()),
                "track_error_m": float(diag["signed_track_error"][idx].item()),
                "course_error_deg": course_err_deg,
                "freq_hz": float(diag["freq_hz"][idx].item()),
                "reward": float(rewards[idx].item()),
                "done": float(done[idx].item()),
            }
        )

        if args.print_every > 0 and (step % int(args.print_every) == 0):
            print(
                f"[step {step:05d}] y={traj_rows[-1]['y']:.3f} m, "
                f"course_err={course_err_deg:.2f} deg, reward={traj_rows[-1]['reward']:.3f}"
            )

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_path = run_dir / "trajectory_env0.csv"
    summary_path = run_dir / "summary.json"

    with traj_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        writer.writeheader()
        writer.writerows(traj_rows)

    if abs_track_err_hist:
        track_tensor = torch.tensor(abs_track_err_hist)
        course_tensor = torch.tensor(abs_course_err_deg_hist)
        summary = {
            "task": args.task,
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "resets": int(resets),
            "mean_abs_track_error_m": float(torch.mean(track_tensor).item()),
            "p95_abs_track_error_m": float(torch.quantile(track_tensor, 0.95).item()),
            "mean_abs_course_error_deg": float(torch.mean(course_tensor).item()),
            "p95_abs_course_error_deg": float(torch.quantile(course_tensor, 0.95).item()),
            "output_dir": str(run_dir),
        }
    else:
        summary = {
            "task": args.task,
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "resets": int(resets),
            "output_dir": str(run_dir),
        }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    simulation_app.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

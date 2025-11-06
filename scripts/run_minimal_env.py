"""Minimal smoke test for the flapping bot environment."""

from __future__ import annotations

import argparse
import sys
import torch

from isaaclab.app import AppLauncher

# Ensure the custom extension is importable even if not installed via pip -e
from pathlib import Path
_EXT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(_EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal flapping bot simulation.")
    parser.add_argument("--steps", type=int, default=256, help="Number of environment steps to simulate.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the application running after the smoke test (ignored when headless).",
    )
    # Debug: print tail targets (deg) and actual joint positions (deg)
    parser.add_argument(
        "--debug-tail",
        action="store_true",
        help="Print left/right tail targets and positions during stepping.",
    )
    parser.add_argument(
        "--debug-interval",
        type=int,
        default=60,
        help="Print interval in steps when --debug-tail is set (default: 60).",
    )
    # Scripted tail demo: segments of pitch / roll / mixed
    parser.add_argument(
        "--tail-demo",
        action="store_true",
        help="Run scripted tail actions: pitch -> roll -> mixed, ignoring random actions.",
    )
    parser.add_argument(
        "--seg-dur",
        type=float,
        default=3.0,
        help="Segment duration in seconds for each tail action (pitch/roll/mixed).",
    )
    parser.add_argument(
        "--tail-freq",
        type=float,
        default=0.5,
        help="Tail demo signal frequency in Hz (continuous variation).",
    )
    parser.add_argument(
        "--tail-amp",
        type=float,
        default=0.6,
        help="Tail demo signal amplitude scale in [-1, 1] (before clamping).",
    )
    parser.add_argument(
        "--mid-demo",
        action="store_true",
        help="Continuously actuate mid-tail with a sine during demo.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interactive and args.headless:
        print("[WARN] --interactive is ignored when running headless.")
        args.interactive = False

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from flapping_bot import FlappingBotEnv, FlappingBotEnvCfg
    except ImportError as exc:  # pragma: no cover - import guard
        simulation_app.close()
        raise RuntimeError("Cannot import flapping_bot extension. Did you install the extension path?") from exc

    cfg = FlappingBotEnvCfg()
    cfg.scene.num_envs = args.num_envs

    render_mode = "headless" if args.headless else None
    env = FlappingBotEnv(cfg, render_mode=render_mode)

    obs, _ = env.reset()

    # Prepare debug helpers (use controlled-joint indices to avoid index-space mismatch)
    debug_limits_once = False
    def debug_print(step_i: int):
        nonlocal debug_limits_once
        if not args.debug_tail:
            return
        if step_i % max(1, args.debug_interval) != 0:
            return
        import math
        idxL, idxR = env._IDX_LEFT_TAIL, env._IDX_RIGHT_TAIL
        idxM = getattr(env, "_IDX_MID_TAIL", None)
        # Targets come from controlled joint target buffer (same order as controlled_joints)
        if idxM is None:
            tgt = env._joint_targets[0, [idxL, idxR]].detach().cpu().numpy()
        else:
            tgt = env._joint_targets[0, [idxL, idxR, idxM]].detach().cpu().numpy()
        # Positions: first gather controlled joints view, then pick L/R
        pos_ctrl = env._robot.data.joint_pos[0, env._joint_ids].detach().cpu().numpy()
        if idxM is None:
            pos = pos_ctrl[[idxL, idxR]]
        else:
            pos = pos_ctrl[[idxL, idxR, idxM]]
        tgt_deg = [round(math.degrees(v), 1) for v in tgt]
        pos_deg = [round(math.degrees(v), 1) for v in pos]
        if idxM is None:
            print(f"[DEBUG] step={step_i} tail target(deg) L/R={tgt_deg} pos(deg) L/R={pos_deg}")
        else:
            print(f"[DEBUG] step={step_i} tail target(deg) L/R/M={tgt_deg} pos(deg) L/R/M={pos_deg}")

        # One-time print of raw vs softened limits for tails (to explain ~25-26° vs ±30°)
        if not debug_limits_once:
            raw_limits = env._robot.data.joint_pos_limits[0, env._joint_ids].detach().cpu().numpy()
            raw_L = raw_limits[idxL]
            raw_R = raw_limits[idxR]
            soft_L = env._joint_lower_limits[idxL].item(), env._joint_upper_limits[idxL].item()
            soft_R = env._joint_lower_limits[idxR].item(), env._joint_upper_limits[idxR].item()
            def rad2deg_pair(p):
                return (round(math.degrees(p[0]), 1), round(math.degrees(p[1]), 1))
            print(
                "[DEBUG] tail raw limits deg L/R:", rad2deg_pair(raw_L), rad2deg_pair(raw_R),
                " softened (after joint_limit_softness) L/R:", rad2deg_pair(soft_L), rad2deg_pair(soft_R)
            )
            debug_limits_once = True

    if args.tail_demo:
        # Scripted tail actions over segments with continuous variation:
        # 1) Pitch (same-sign); 2) Roll (opposite-sign); 3) Mixed; 4) Optional mid-tail sine
        # Action layout in env: [freq, tail_pitch, tail_roll, mid_tail]
        action_dim = env.single_action_space.shape[0]
        pitch_idx = getattr(env, "_ACT_IDX_TAIL_PITCH", 1)
        roll_idx = getattr(env, "_ACT_IDX_TAIL_ROLL", 2)
        mid_idx = getattr(env, "_ACT_IDX_MID_TAIL", None)
        # Local time for tail signals
        t = 0.0
        dt = float(env.step_dt)
        two_pi = 6.283185307179586
        amp = float(max(-1.0, min(1.0, args.tail_amp)))
        freq = max(0.0, args.tail_freq)

        def run_segment(kind: str, seg_steps: int, start_step: int) -> tuple[int, float]:
            nonlocal t
            step = start_step
            for _ in range(seg_steps):
                # Generate continuous signals per kind
                phase = two_pi * freq * t
                if kind == "pitch":
                    pitch_val = amp * torch.sin(torch.tensor(phase, device=env.device, dtype=torch.float32))
                    roll_val = torch.tensor(0.0, device=env.device, dtype=torch.float32)
                elif kind == "roll":
                    pitch_val = torch.tensor(0.0, device=env.device, dtype=torch.float32)
                    roll_val = amp * torch.sin(torch.tensor(phase, device=env.device, dtype=torch.float32))
                else:  # mixed
                    # Slight phase offset on roll to make the combination clearer
                    pitch_val = amp * torch.sin(torch.tensor(phase, device=env.device, dtype=torch.float32))
                    roll_val = 0.5 * amp * torch.sin(torch.tensor(phase + 1.0471975512, device=env.device, dtype=torch.float32))  # +60°

                a = torch.zeros((env.num_envs, action_dim), device=env.device, dtype=torch.float32)
                a[:, pitch_idx] = pitch_val
                a[:, roll_idx] = roll_val
                if args.mid_demo and mid_idx is not None:
                    a[:, mid_idx] = 0.7 * amp * torch.sin(torch.tensor(phase + 0.5235987756, device=env.device, dtype=torch.float32))  # +30°

                ret = env.step(a)
                if len(ret) == 5:
                    obs, rew, terminated, truncated, info = ret
                    done = terminated | truncated
                else:
                    obs, rew, done, info = ret
                debug_print(step)
                step += 1
                t += dt
                if done.any():
                    env.reset()
            return step, t

        seg_steps = max(1, int(args.seg_dur / env.step_dt))
        # 1) Pitch (same-direction) varying
        s, t = run_segment("pitch", seg_steps, 0)
        # 2) Roll (opposite-direction) varying
        s, t = run_segment("roll", seg_steps, s)
        # 3) Mixed pitch+roll varying
        s, t = run_segment("mixed", seg_steps, s)
        # 4) Optional mid-tail only segment
        if args.mid_demo and mid_idx is not None:
            def run_mid(seg_steps: int, start_step: int):
                nonlocal t
                step = start_step
                for _ in range(seg_steps):
                    phase = two_pi * freq * t
                    a = torch.zeros((env.num_envs, action_dim), device=env.device, dtype=torch.float32)
                    a[:, mid_idx] = amp * torch.sin(torch.tensor(phase, device=env.device, dtype=torch.float32))
                    ret = env.step(a)
                    if len(ret) == 5:
                        obs, rew, terminated, truncated, info = ret
                        done = terminated | truncated
                    else:
                        obs, rew, done, info = ret
                    debug_print(step)
                    step += 1
                    t += dt
                    if done.any():
                        env.reset()
                return step, t
            _, t = run_mid(seg_steps, s)
    else:
        # Random actions (fallback)
        for i in range(args.steps):
            actions = torch.as_tensor(env.action_space.sample(), device=env.device, dtype=torch.float32)
            ret = env.step(actions)
            # Support 4- or 5-tuples depending on wrappers
            if len(ret) == 5:
                obs, rew, terminated, truncated, info = ret
                done = terminated | truncated
            else:
                obs, rew, done, info = ret
            debug_print(i)
            if done.any():
                env.reset()

    if args.interactive and not args.headless:
        print("[INFO] Interactive mode enabled. Close the Isaac Lab window to exit.", flush=True)
        try:
            while simulation_app.is_running():
                simulation_app.update()
        finally:
            env.close()
            simulation_app.close()
            print(f"[OK] Completed {args.steps} steps across {args.num_envs} environments.")
        return

    env.close()
    simulation_app.close()
    print(f"[OK] Completed {args.steps} steps across {args.num_envs} environments.")


if __name__ == "__main__":
    sys.exit(main())



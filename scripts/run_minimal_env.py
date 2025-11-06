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

    if args.tail-demo:
        # Build scripted tail actions over three segments: pitch -> roll -> mixed
        # Action layout in env: [freq_L, freq_R, tail_pitch, tail_roll]
        action_dim = env.single_action_space.shape[0]
        pitch_idx, roll_idx = 2, 3
        # frequency components are ignored by env when use_action_frequency=False
        def run_segment(pitch_val: float, roll_val: float, seg_steps: int):
            for _ in range(seg_steps):
                a = torch.zeros((env.num_envs, action_dim), device=env.device, dtype=torch.float32)
                a[:, pitch_idx] = pitch_val
                a[:, roll_idx] = roll_val
                ret = env.step(a)
                if len(ret) == 5:
                    obs, rew, terminated, truncated, info = ret
                    done = terminated | truncated
                else:
                    obs, rew, done, info = ret
                if done.any():
                    env.reset()

        seg_steps = max(1, int(args.seg-dur / env.step_dt))
        # 1) Pitch up (same-direction) ~ 60% amplitude
        run_segment(pitch_val=0.6, roll_val=0.0, seg_steps=seg_steps)
        # 2) Roll right (opposite-direction) ~ 60% amplitude
        run_segment(pitch_val=0.0, roll_val=0.6, seg_steps=seg_steps)
        # 3) Mixed pitch+roll
        run_segment(pitch_val=0.5, roll_val=-0.4, seg_steps=seg_steps)
    else:
        # Random actions (fallback)
        for _ in range(args.steps):
            actions = torch.as_tensor(env.action_space.sample(), device=env.device, dtype=torch.float32)
            ret = env.step(actions)
            # Support 4- or 5-tuples depending on wrappers
            if len(ret) == 5:
                obs, rew, terminated, truncated, info = ret
                done = terminated | truncated
            else:
                obs, rew, done, info = ret
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



"""Run the flapping bot with baked mass (robot_mass6.urdf -> flapping_bot_mass6.usd)."""

from __future__ import annotations

import argparse
import sys
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

# Ensure extension import path
EXT_ROOT = Path(__file__).resolve().parents[1]
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flapping bot (mass6 baked)")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--debug-tail", action="store_true")
    parser.add_argument("--debug-phys", action="store_true")
    parser.add_argument("--debug-interval", type=int, default=60)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interactive and args.headless:
        args.interactive = False

    app = AppLauncher(args).app
    try:
        from flapping_bot import FlappingBotEnv, FlappingBotEnvCfg
        from flapping_bot.assets import FlappingBotCfg
    except Exception:
        app.close()
        raise

    # Point to baked URDF/USD
    SRC_ROOT = Path(__file__).resolve().parents[2]
    robots_dir = SRC_ROOT / "isaaclab_assets" / "data" / "flapping_bot" / "robots"
    spawn = FlappingBotCfg.spawn.replace(
        asset_path=str(robots_dir / "robot_mass6.urdf"),
        usd_dir=str(robots_dir),
        usd_file_name="flapping_bot_mass6.usd",
    )
    robot_cfg = FlappingBotCfg.replace(spawn=spawn).replace(prim_path="/World/envs/env_.*/Robot")

    cfg = FlappingBotEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.robot = robot_cfg

    env = FlappingBotEnv(cfg, render_mode=("headless" if args.headless else None))
    obs, _ = env.reset()

    # Optional mass debug
    if args.debug_phys:
        try:
            import isaaclab.sim as sim_utils
            import omni.usd
            from pxr import UsdPhysics
            stage = omni.usd.get_context().get_stage()
            robot_prim = sim_utils.find_first_matching_prim(env._robot.cfg.prim_path)
            base = robot_prim.GetPath().pathString
            names = [n for n in getattr(env._robot, "body_names", []) if ("body" in n or "wing" in n or "tail" in n)]
            print("[PHYS] Mass properties (kg, m, kg·m^2):")
            for n in names:
                prim = stage.GetPrimAtPath(f"{base}/{n}")
                api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
                m = api.GetMassAttr().Get() if api.GetMassAttr().HasAuthoredValueOpinion() else None
                com = api.GetCenterOfMassAttr().Get() if api.GetCenterOfMassAttr().HasAuthoredValueOpinion() else None
                I = api.GetDiagonalInertiaAttr().Get() if api.GetDiagonalInertiaAttr().HasAuthoredValueOpinion() else None
                print(f"  {n}: mass={m} com={tuple(com) if com else None} inertia={tuple(I) if I else None}")
        except Exception as e:
            print("[WARN] debug-phys failed:", repr(e))

    for i in range(args.steps):
        actions = torch.as_tensor(env.action_space.sample(), device=env.device, dtype=torch.float32)
        ret = env.step(actions)
        if len(ret) == 5:
            obs, rew, terminated, truncated, info = ret
            done = terminated | truncated
        else:
            obs, rew, done, info = ret
        if done.any():
            env.reset()

    if args.interactive and not args.headless:
        print("[INFO] Interactive mode: close window to exit")
        try:
            while app.is_running():
                app.update()
        finally:
            env.close(); app.close()
        return 0

    env.close(); app.close()
    print(f"[OK] Completed {args.steps} steps across {args.num_envs} env(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

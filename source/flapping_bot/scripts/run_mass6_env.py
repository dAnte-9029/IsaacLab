"""Run the flapping bot using the current v50 URDF configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

# Ensure extension import path
EXT_ROOT = Path(__file__).resolve().parents[1]
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flapping bot (mass6 baked)")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--freq-hz", type=float, default=5.0, help="Fixed flapping frequency (Hz).")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--debug-tail", action="store_true")
    parser.add_argument("--debug-phys", action="store_true")
    parser.add_argument("--debug-interval", type=int, default=60)
    parser.add_argument(
        "--measure-wing-hz",
        action="store_true",
        help="Estimate actual wing flapping frequency from joint motion (env 0).",
    )
    parser.add_argument(
        "--log-qsm-forces",
        action="store_true",
        help="Log QSM total force in body frame and save lift/thrust curves.",
    )
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

    # Use the default flapping-bot articulation (currently v50 URDF) and just
    # adapt the prim path for this standalone test.
    robot_cfg = FlappingBotCfg.replace(prim_path="/World/envs/env_.*/Robot")

    cfg = FlappingBotEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.robot = robot_cfg
    # Use a fixed flapping frequency instead of action-controlled frequency.
    cfg.use_action_frequency = False
    cfg.flapping_freq_hz = float(args.freq_hz)
    # For open-loop forward-flight test: unlock freeze and keep tails at zero.
    cfg.freeze_steps_after_reset = 0
    cfg.lock_tail_at_zero = True
    # Make episodes long and relax tilt termination (but keep ground contact).
    cfg.episode_length_s = 1000.0
    cfg.terminate_ground_height = 0.05   # keep default ground termination
    cfg.terminate_tilt_deg = 89.0        # effectively "never" tilt-terminate

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

    # Optional wing frequency measurement (env 0, left wing if present)
    measure = args.measure_wing_hz
    wing_joint_id = None
    prev_sign = None
    zero_cross_count = 0
    baseline = None

    if measure:
        # Map resolved joint names to tensor indices
        name_to_idx = {n: i for i, n in enumerate(env._resolved_joint_names)}
        # Prefer left_wing if present, otherwise right_wing
        if "left_wing" in name_to_idx:
            wing_joint_id = name_to_idx["left_wing"]
        elif "right_wing" in name_to_idx:
            wing_joint_id = name_to_idx["right_wing"]
        if wing_joint_id is None:
            print("[WARN] measure-wing-hz requested but no wing joint name found.")
            measure = False

    # If steps <= 0, keep the app running so the user can drive the sim from the UI.
    if args.steps <= 0:
        print(
            "[INFO] steps<=0: keeping Isaac app running. "
            "Use the GUI (Play/Stop) to control the simulation and close the window to exit."
        )
        try:
            while app.is_running():
                app.update()
        finally:
            env.close(); app.close()
        return 0

    # Optional QSM force logging (body-frame total on base)
    qsm_ok = (
        hasattr(env, "_qsm_model")
        and env._qsm_model is not None
        and hasattr(env, "_qsm_joint_tensor_idx")
        and env._qsm_joint_tensor_idx is not None
    )
    log_qsm = args.log_qsm_forces and qsm_ok
    if args.log_qsm_forces and not qsm_ok:
        print("[QSM] WARNING: QSM model or joint mapping not available; no force samples will be collected.")
    times: list[float] = []
    thrust_bx: list[float] = []
    side_by: list[float] = []
    lift_bz: list[float] = []
    roll_tx: list[float] = []
    pitch_ty: list[float] = []
    yaw_tz: list[float] = []

    for i in range(args.steps):
        actions = torch.as_tensor(env.action_space.sample(), device=env.device, dtype=torch.float32)
        ret = env.step(actions)
        if len(ret) == 5:
            obs, rew, terminated, truncated, info = ret
            done = terminated | truncated
        else:
            obs, rew, done, info = ret
        if measure and wing_joint_id is not None:
            # joint_pos shape: [num_envs, num_joints]
            q = env._robot.data.joint_pos[0, env._joint_ids[wing_joint_id]].item()
            # Use the initial position as a simple baseline so we detect
            # oscillations around the starting offset rather than around 0.
            if baseline is None:
                baseline = q
            dq = q - baseline
            # Track zero crossings of (q - baseline)
            sign = 1 if dq >= 0.0 else -1
            if prev_sign is not None and sign != prev_sign:
                zero_cross_count += 1
            prev_sign = sign

            # Periodically log an online estimate of the wing frequency.
            if args.debug_interval > 0 and (i + 1) % args.debug_interval == 0:
                sim_time = (i + 1) * env.step_dt
                if sim_time > 0.0 and zero_cross_count > 0:
                    cycles = zero_cross_count / 2.0
                    freq_est = cycles / sim_time
                    print(
                        f"[MEASURE] step={i+1} sim_time={sim_time:.3f}s "
                        f"zero_cross={zero_cross_count} est_freq={freq_est:.2f} Hz "
                        f"(target={cfg.flapping_freq_hz:.2f} Hz)"
                    )

        # QSM force logging: recompute total body-frame force on base from QSM model.
        if (
            log_qsm
            and hasattr(env, "_qsm_model")
            and env._qsm_model is not None
            and env._qsm_joint_tensor_idx is not None
        ):
            # Gather joint states for QSM joints
            jpos_all = env._robot.data.joint_pos[:, env._joint_ids]
            jvel_all = env._robot.data.joint_vel[:, env._joint_ids]
            jpos = jpos_all[:, env._qsm_joint_tensor_idx]
            jvel = jvel_all[:, env._qsm_joint_tensor_idx]
            v_b = env._robot.data.root_lin_vel_b
            w_b = env._robot.data.root_ang_vel_b
            f_b, tau_b, _ = env._qsm_model.compute_forces(jpos, jvel, v_b, w_b)
            f_sum = torch.sum(f_b, dim=1)      # [N,3]
            tau_sum = torch.sum(tau_b, dim=1)  # [N,3]
            times.append((i + 1) * env.step_dt)
            thrust_bx.append(float(f_sum[0, 0].item()))
            side_by.append(float(f_sum[0, 1].item()))
            lift_bz.append(float(f_sum[0, 2].item()))
            roll_tx.append(float(tau_sum[0, 0].item()))
            pitch_ty.append(float(tau_sum[0, 1].item()))
            yaw_tz.append(float(tau_sum[0, 2].item()))

        # Log base pose / velocity periodically to inspect forward flight.
        if args.debug_interval > 0 and (i + 1) % args.debug_interval == 0:
            pos = env._robot.data.root_pos_w[0].tolist()
            vel = env._robot.data.root_lin_vel_w[0].tolist()
            sim_time = (i + 1) * env.step_dt
            print(
                f"[STATE] step={i+1} sim_time={sim_time:.3f}s "
                f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                f"vel=({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f})"
            )

    if args.interactive and not args.headless:
        print("[INFO] Interactive mode: close window to exit")
        try:
            while app.is_running():
                app.update()
        finally:
            env.close()
            app.close()
        return 0

    # Capture commanded frequency before any shutdown
    cmd_freq = None
    if measure and hasattr(env, "_freq_left"):
        try:
            cmd_freq = float(env._freq_left[0].item())
        except Exception:
            cmd_freq = None

    # Optionally save QSM lift/thrust samples to CSV
    if log_qsm:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "flapping_qsm_forces.csv"
        if times:
            try:
                with out_path.open("w", encoding="utf-8") as f:
                    f.write("t,thrust_bx,side_by,lift_bz,roll_tx,pitch_ty,yaw_tz\n")
                    for idx in range(len(times)):
                        f.write(
                            f"{times[idx]:.6f},"
                            f"{thrust_bx[idx]:.6f},"
                            f"{side_by[idx]:.6f},"
                            f"{lift_bz[idx]:.6f},"
                            f"{roll_tx[idx]:.6f},"
                            f"{pitch_ty[idx]:.6f},"
                            f"{yaw_tz[idx]:.6f}\n"
                        )
                print(f"[QSM] Saved lift/thrust samples to {out_path}")
            except Exception as e:
                print("[QSM] Failed to save lift/thrust CSV:", repr(e))
        else:
            print("[QSM] No QSM force samples collected; CSV not written.")

    if measure:
        if cmd_freq is not None:
            print(
                f"[MEASURE] Commanded flapping frequency (env 0): {cmd_freq:.2f} Hz "
                f"(cfg.flapping_freq_hz={cfg.flapping_freq_hz:.2f} Hz)"
            )
        if wing_joint_id is not None and zero_cross_count > 0:
            # Each full cycle has two zero-crossings; use env.step_dt as time base.
            sim_duration = args.steps * env.step_dt
            cycles = zero_cross_count / 2.0
            freq_meas = cycles / sim_duration if sim_duration > 0 else 0.0
            print(
                f"[MEASURE] Wing joint estimated frequency: {freq_meas:.2f} Hz "
                f"(target cfg.flapping_freq_hz={cfg.flapping_freq_hz:.2f} Hz)"
            )
        elif wing_joint_id is not None and zero_cross_count == 0:
            print(
                "[MEASURE] No zero-crossings detected on wing joint; "
                "joint likely oscillates around a bias with small amplitude."
            )

    print(f"[OK] Completed {args.steps} steps across {args.num_envs} env(s)")
    # Now tear down the env and app
    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

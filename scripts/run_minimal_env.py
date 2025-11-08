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
    # Tail step demo (square waves) and independent mid-tail step frequency
    parser.add_argument(
        "--tail-step",
        action="store_true",
        help="Use square/step signals for left/right tails instead of sine in demo.",
    )
    parser.add_argument(
        "--step-freq",
        type=float,
        default=1.0,
        help="Square-wave frequency (Hz) for --tail-step (left/right tails).",
    )
    parser.add_argument(
        "--mid-step-freq",
        type=float,
        default=None,
        help="Square-wave frequency (Hz) for mid-tail when --mid-demo is set. If omitted, mid-tail keeps sine.",
    )
    parser.add_argument(
        "--track-log",
        type=str,
        default="",
        help="CSV path to log tracking (time,freq, wing L/R tgt/pos, tail L/R tgt/pos, mid tgt/pos).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot tracking curves after run (requires matplotlib).",
    )
    parser.add_argument(
        "--debug-phys",
        action="store_true",
        help="Print mass/COM/inertia for body and wings/tails (from MassAPI).",
    )
    # Mid-tail command shaping knobs (optional)
    parser.add_argument(
        "--mid-rate",
        type=float,
        default=None,
        help="Mid-tail max rate (deg/s). 0 disables. If set, config is applied to env.",
    )
    parser.add_argument(
        "--mid-tau",
        type=float,
        default=None,
        help="First-order smoothing time constant (s) for mid-tail command. 0 disables.",
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
    # Apply optional mid-tail shaping
    if args.mid_rate is not None:
        cfg.mid_tail_rate_limit_deg_s = max(0.0, float(args.mid_rate))
    if args.mid_tau is not None:
        cfg.mid_tail_cmd_tau = max(0.0, float(args.mid_tau))

    render_mode = "headless" if args.headless else None
    env = FlappingBotEnv(cfg, render_mode=render_mode)

    obs, _ = env.reset()

    # Tracking buffers (optional)
    do_log = bool(getattr(args, "track_log", ""))
    log_rows = []
    t_sim = 0.0

    # Optional: print mass properties after any runtime overrides
    if args.debug_phys:
        try:
            import json
            import isaaclab.sim as sim_utils
            import omni.usd
            from pxr import UsdPhysics

            cfg_path = _EXT_ROOT / "config" / "mass_props.json"
            names = None
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    names = list(json.load(f).keys())
            if not names:
                names = [n for n in getattr(env._robot, "body_names", []) if ("body" in n or "wing" in n or "tail" in n)]

            stage = omni.usd.get_context().get_stage()
            robot_prim = sim_utils.find_first_matching_prim(env._robot.cfg.prim_path)
            base = robot_prim.GetPath().pathString
            print("[PHYS] Mass properties (kg, m, kg·m^2):")
            for n in names:
                p = f"{base}/{n}"
                prim = stage.GetPrimAtPath(p)
                if not prim or not prim.IsValid():
                    print(f"  {n}: <missing prim>")
                    continue
                api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
                m = api.GetMassAttr().Get() if api.GetMassAttr().HasAuthoredValueOpinion() else None
                com_attr = api.GetCenterOfMassAttr()
                I_attr = api.GetDiagonalInertiaAttr()
                com = tuple(com_attr.Get()) if com_attr and com_attr.HasAuthoredValueOpinion() else None
                I = tuple(I_attr.Get()) if I_attr and I_attr.HasAuthoredValueOpinion() else None
                print(f"  {n}: mass={m} com={com} inertia={I}")
        except Exception as e:
            print("[WARN] debug-phys failed:", repr(e))

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
            nonlocal t, t_sim
            step = start_step
            for _ in range(seg_steps):
                # Generate continuous signals per kind
                phase = two_pi * freq * t
                if args.tail_step:
                    # Drive L/R tails with a square wave at --step-freq
                    phase_step = two_pi * args.step_freq * t
                    sq = torch.sign(torch.sin(torch.tensor(phase_step, device=env.device, dtype=torch.float32)))
                    if kind == "pitch":
                        pitch_val = amp * sq
                        roll_val = torch.tensor(0.0, device=env.device, dtype=torch.float32)
                    elif kind == "roll":
                        pitch_val = torch.tensor(0.0, device=env.device, dtype=torch.float32)
                        roll_val = amp * sq
                    else:  # mixed
                        pitch_val = amp * sq
                        # roll uses same square with small phase offset for visual distinction
                        sq2 = torch.sign(torch.sin(torch.tensor(phase_step + 1.0471975512, device=env.device, dtype=torch.float32)))
                        roll_val = 0.5 * amp * sq2
                else:
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
                    if args.mid_step_freq is not None:
                        # Independent mid-tail square wave at --mid-step-freq
                        mid_phase = two_pi * float(args.mid_step_freq) * t
                        mid_sq = torch.sign(torch.sin(torch.tensor(mid_phase, device=env.device, dtype=torch.float32)))
                        a[:, mid_idx] = 0.7 * amp * mid_sq
                    else:
                        # Default mid-tail: sine (original behavior)
                        a[:, mid_idx] = 0.7 * amp * torch.sin(torch.tensor(phase + 0.5235987756, device=env.device, dtype=torch.float32))  # +30°

                ret = env.step(a)
                if len(ret) == 5:
                    obs, rew, terminated, truncated, info = ret
                    done = terminated | truncated
                else:
                    obs, rew, done, info = ret
                debug_print(step)
                if do_log:
                    idxLW, idxRW = env._IDX_LEFT_WING, env._IDX_RIGHT_WING
                    idxLT, idxRT = env._IDX_LEFT_TAIL, env._IDX_RIGHT_TAIL
                    idxMT = getattr(env, "_IDX_MID_TAIL", None)
                    tgt = env._joint_targets[0].detach().cpu().numpy()
                    pos = env._robot.data.joint_pos[0, env._joint_ids].detach().cpu().numpy()
                    row = [t_sim, float(env._freq_left[0].item())]
                    row += [tgt[idxLW], tgt[idxRW], pos[idxLW], pos[idxRW]]
                    row += [tgt[idxLT], tgt[idxRT], pos[idxLT], pos[idxRT]]
                    if idxMT is not None:
                        row += [tgt[idxMT], pos[idxMT]]
                    log_rows.append(row)
                step += 1
                t += dt
                t_sim += env.step_dt
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
                nonlocal t, t_sim
                step = start_step
                for _ in range(seg_steps):
                    # Mid-only segment: independent square if --mid-step-freq is given
                    if args.mid_step_freq is not None:
                        phase = two_pi * float(args.mid_step_freq) * t
                        mid_val = amp * torch.sign(torch.sin(torch.tensor(phase, device=env.device, dtype=torch.float32)))
                    else:
                        phase = two_pi * freq * t
                        mid_val = amp * torch.sin(torch.tensor(phase, device=env.device, dtype=torch.float32))
                    a = torch.zeros((env.num_envs, action_dim), device=env.device, dtype=torch.float32)
                    a[:, mid_idx] = mid_val
                    ret = env.step(a)
                    if len(ret) == 5:
                        obs, rew, terminated, truncated, info = ret
                        done = terminated | truncated
                    else:
                        obs, rew, done, info = ret
                    debug_print(step)
                    if do_log:
                        idxLW, idxRW = env._IDX_LEFT_WING, env._IDX_RIGHT_WING
                        idxLT, idxRT = env._IDX_LEFT_TAIL, env._IDX_RIGHT_TAIL
                        idxMT = getattr(env, "_IDX_MID_TAIL", None)
                        tgt = env._joint_targets[0].detach().cpu().numpy()
                        pos = env._robot.data.joint_pos[0, env._joint_ids].detach().cpu().numpy()
                        row = [t_sim, float(env._freq_left[0].item())]
                        row += [tgt[idxLW], tgt[idxRW], pos[idxLW], pos[idxRW]]
                        row += [tgt[idxLT], tgt[idxRT], pos[idxLT], pos[idxRT]]
                        if idxMT is not None:
                            row += [tgt[idxMT], pos[idxMT]]
                        log_rows.append(row)
                    t_sim += env.step_dt
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

    # Dump tracking log if requested
    if do_log and log_rows:
        import csv, os
        out = args.track_log
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'time','freq_hz',
                'tgt_wing_L','tgt_wing_R','pos_wing_L','pos_wing_R',
                'tgt_tail_L','tgt_tail_R','pos_tail_L','pos_tail_R',
                'tgt_mid','pos_mid'
            ]
            if not hasattr(env, '_IDX_MID_TAIL'):
                header = header[:-2]
            writer.writerow(header)
            for r in log_rows:
                writer.writerow([float(x) for x in r])
        print(f"[TRACK] wrote {len(log_rows)} rows to {out}")
        if getattr(args, 'plot', False):
            try:
                import numpy as np, matplotlib.pyplot as plt
                data = np.loadtxt(out, delimiter=',', skiprows=1)
                t = data[:,0]; freq = data[:,1]
                i=2
                tgt_wL,tgt_wR,pos_wL,pos_wR = data[:,i],data[:,i+1],data[:,i+2],data[:,i+3]; i+=4
                tgt_tL,tgt_tR,pos_tL,pos_tR = data[:,i],data[:,i+1],data[:,i+2],data[:,i+3]; i+=4
                has_mid = data.shape[1] > i
                if has_mid:
                    tgt_m,pos_m = data[:,i],data[:,i+1]
                fig,axs=plt.subplots(3 if has_mid else 2,1,figsize=(10,8),sharex=True)
                axs = np.atleast_1d(axs)
                axs[0].plot(t, np.degrees(tgt_wL)); axs[0].plot(t, np.degrees(pos_wL))
                axs[0].plot(t, np.degrees(tgt_wR)); axs[0].plot(t, np.degrees(pos_wR))
                ax2=axs[0].twinx(); ax2.plot(t, freq, color='gray', alpha=.3)
                axs[1].plot(t, np.degrees(tgt_tL)); axs[1].plot(t, np.degrees(pos_tL))
                axs[1].plot(t, np.degrees(tgt_tR)); axs[1].plot(t, np.degrees(pos_tR))
                if has_mid:
                    axs[2].plot(t, np.degrees(tgt_m)); axs[2].plot(t, np.degrees(pos_m))
                plt.tight_layout(); plt.show()
            except Exception as e:
                print('[WARN] plotting failed:', repr(e))

    env.close()
    simulation_app.close()
    print(f"[OK] Completed {args.steps} steps across {args.num_envs} environments.")


if __name__ == "__main__":
    sys.exit(main())





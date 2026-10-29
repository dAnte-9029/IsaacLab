"""Real CPU-native PhysX 480/60 Hz smoke and matched end-to-end timing; no PPO."""
import argparse
import json
from pathlib import Path
import time
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["DeLaurier", "L3D", "I3D"], required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--num-envs", type=int, default=4)
p.add_argument("--steps", type=int, default=60)
p.add_argument("--twist", type=float, default=35.0)
p.add_argument("--wind", type=float, default=0.0)
p.add_argument("--trim", type=Path)
p.add_argument("--trim-model", choices=["L3D", "I3D"], default="I3D")
p.add_argument("--wind-z", type=float, default=0.0)
p.add_argument("--ou", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.device = "cpu"
root = Path(__file__).resolve().parents[2]
args.kit_args = f'{args.kit_args or ""} --ext-folder {root}/source/flapping_bot/native_extensions --enable omni.flapping_bot.holonomic_constraint'
app = AppLauncher(args).app
try:
    import torch
    from flapping_bot.direct.flapping_bot.straight_flight_env import (
        FlappingBotStraightFlightEnv,
        FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
    )
    from flapping_bot.physics.control_aero_3d import ControlAero3DConfig
    from flapping_bot.analysis.pure_rl_fixed_action_reward_validation import _configure_robot_asset

    torch.set_num_threads(1)
    cfg = FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg()
    cfg.seed = 29
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cpu"
    cfg.scene.env_spacing = 5.0
    cfg.randomize_straight_line_heading = False
    cfg.randomize_flap_phase_at_reset = False
    cfg.wind_enabled = args.wind != 0 or args.wind_z != 0 or args.ou
    cfg.wind_z_mps = args.wind_z
    cfg.wind_ou_enabled = args.ou
    cfg.wind_ou_sigma_xy_mps = (0.1, 0.2) if args.ou else (0.0, 0.0)
    cfg.wind_ou_sigma_z_mps = 0.1 if args.ou else 0.0
    cfg.wind_ou_clip_to_range = False
    cfg.wind_xy_mps = (0.0, args.wind)
    cfg.delaurier_wing_geom_csv = root / "outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv"
    if args.backend != "DeLaurier":
        cfg.control_aero_3d = ControlAero3DConfig(backend=args.backend, twist_amplitude_deg=args.twist)
    _configure_robot_asset(
        cfg,
        asset_path=root / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf",
        usd_dir=root / "logs/control_aero_3d/assets",
    )
    if args.trim:
        trim = json.loads(args.trim.read_text())[args.trim_model]  # shared operating point for all backends
        cfg.reset_pitch_deg = trim["alpha_deg"]
        cfg.reset_flap_hz = trim["frequency_hz"]
        cfg.reset_elevon_pitch_deg = trim["symmetric_tail_deg"]
        cfg.vx_cmd = 8.0
    env = FlappingBotStraightFlightEnv(cfg)
    env.reset()
    action = env._actions.clone()
    action[:, 0] = 2 * cfg.reset_flap_hz / 5 - 1
    action[:, 1] = 0.0
    action[:, 2:4] = cfg.reset_elevon_pitch_deg / cfg.elevon_max_deg
    action[1::2, 1] = 0.15
    action[1::2, 2] += 0.1
    action[1::2, 3] -= 0.1
    trace = []
    terminated = 0
    # Warm-up excluded from timing; every backend uses identical steps and action.
    for _ in range(3):
        env.step(action)
    start = time.perf_counter()
    for i in range(args.steps):
        _, _, done, timeout, _ = env.step(action)
        terminated += int(done.sum())
        state = env._robot.data.root_state_w.clone()
        if not bool(torch.isfinite(state).all()):
            raise RuntimeError("Nonfinite PhysX state")
        trace.append(
            dict(
                step=i,
                position=state[:, :3].tolist(),
                velocity=state[:, 7:10].tolist(),
                omega=state[:, 10:13].tolist(),
                force=env._debug_last_force_b.tolist(),
                moment=env._debug_last_torque_b.tolist(),
                tail=env._debug_last_actual_rudder_rad.tolist(),
                wind=env._wind_w.tolist(),
            )
        )
    elapsed = time.perf_counter() - start
    from isaaclab.utils.math import quat_apply

    data = env._robot.data
    ids = env._wing_body_ids
    if args.backend != "DeLaurier":
        fresh = env._compute_control_3d_wrench()
        fw = quat_apply(data.body_link_quat_w[:, ids], fresh.force_link_n)
        mw = quat_apply(data.body_link_quat_w[:, ids], fresh.moment_link_about_com_nm)
        base_com = data.body_com_pos_w[:, int(env._base_body_ids[0])]
        mw += torch.linalg.cross(data.body_com_pos_w[:, ids] - base_com[:, None], fw)
        closure_error = max(
            float((fw.sum(1) - quat_apply(data.root_quat_w, fresh.net_force_b_n)).abs().max()),
            float((mw.sum(1) - quat_apply(data.root_quat_w, fresh.net_moment_b_about_base_com_nm)).abs().max()),
        )
        assert closure_error < 5e-5
    else:
        closure_error = None
    assert all(torch.isfinite(torch.tensor(row["force"])).all() for row in trace)

    # Actual reset subset through public environment internals; other envs remain live.
    before = env._robot.data.root_state_w[1:].clone()
    env._reset_idx(torch.tensor([0], device=env.device))
    after = env._robot.data.root_state_w[1:].clone()
    reset_error = float((before - after).abs().max()) if args.num_envs > 1 else 0.0
    result = dict(
        backend=args.backend,
        wrench_closure_error=closure_error,
        ou=args.ou,
        wind_z=args.wind_z,
        num_envs=args.num_envs,
        physics_steps=args.steps * cfg.decimation,
        dt=cfg.sim.dt,
        decimation=cfg.decimation,
        seconds=elapsed,
        env_physics_steps_per_second=args.num_envs * args.steps * cfg.decimation / elapsed,
        terminations=terminated,
        subset_reset_other_state_error=reset_error,
        trace=trace,
        twist=args.twist,
        wind=args.wind,
    )
    if args.backend != "DeLaurier":
        result["last_condition"] = env._debug_control_aero_loads.condition.tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k != "trace"}, flush=True)
    env.close()
finally:
    app.close()

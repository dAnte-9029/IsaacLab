# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab wind-tunnel rig for the FlappingBot v50 model with Wang2016 QSM.

This script uses your flapping aircraft URDF (v50) and applies the Wang et al. (JFM 2016)
quasi-steady aerodynamic wrench to the left/right wing links while the base is fixed
(wind-tunnel "stand").

Scenario:
  - Base pose fixed at a specified body pitch angle (default: +15 deg about +Y).
  - Left/right wings flap synchronously (same joint command).
  - A constant wind-tunnel flow (default: headwind 5 m/s along world -X) is converted into an
    equivalent forward-flight velocity and added to translational velocity per the paper note
    under eq. (2.5).

.. code-block:: bash

    ./isaaclab.sh -p scripts/isaac_wind_tunnel_flappingbot_v50.py --headless --steps 2400 --csv wang2016_v50_windtunnel.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
import importlib.util

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wang2016 QSM wind-tunnel: FlappingBot v50 fixed-base + forward flight.")
parser.add_argument("--steps", type=int, default=2400, help="Number of physics steps.")
parser.add_argument("--csv", type=Path, default=Path("wang2016_v50_windtunnel.csv"), help="Output CSV path.")

parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Physics step (s).")
parser.add_argument("--rho", type=float, default=1.225, help="Air density (kg/m^3).")
parser.add_argument("--N", type=int, default=60, help="Spanwise strip count for BEM discretization.")

parser.add_argument("--f_hz", type=float, default=2.0, help="Flapping frequency (Hz).")
parser.add_argument("--wing_amp_deg", type=float, default=30.0, help="Wing joint amplitude (deg).")
parser.add_argument("--body_pitch_deg", type=float, default=15.0, help="Fixed body pitch angle about +Y (deg).")
parser.add_argument("--airspeed", type=float, default=5.0, help="Wind-tunnel airspeed magnitude (m/s).")
parser.add_argument(
    "--flow_dir_world",
    type=float,
    nargs=3,
    default=(-1.0, 0.0, 0.0),
    metavar=("DX", "DY", "DZ"),
    help="Wind direction in world (unitless). This is the direction the air flows *towards* (default: headwind along -X).",
)
parser.add_argument(
    "--auto-geom-from-urdf",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Estimate wing planform area from the v50 URDF+STL and derive consistent (R,c) from area and --aspect_ratio.",
)
parser.add_argument("--mesh-scale", type=float, default=1.0, help="Mesh coordinate scale to meters for area estimation.")
parser.add_argument("--print-every", type=int, default=60, help="Print a summary every N steps (0 disables).")
parser.add_argument("--draw-forces", action="store_true", help="Draw force/torque arrows in the viewport (if available).")
parser.add_argument("--force-scale", type=float, default=0.02, help="Scale factor for drawing force vectors (m/N).")
parser.add_argument("--torque-scale", type=float, default=0.05, help="Scale factor for drawing torque vectors (m/(N·m)).")

parser.add_argument(
    "--wang-axes",
    type=str,
    nargs=3,
    default=("y", "z", "x"),
    metavar=("WANG_X", "WANG_Y", "WANG_Z"),
    help="Co-rotating Wang frame axes expressed in the wing link frame (e.g. 'y z x'). Supports +/- prefixes.",
)

parser.add_argument("--R", type=float, default=0.10, help="Span R (m).")
parser.add_argument("--c", type=float, default=0.05, help="Chord c (m).")
parser.add_argument("--dhat", type=float, default=0.25, help="Pitch axis offset d̂ (LE->axis)/c.")
parser.add_argument("--aspect_ratio", type=float, default=2.0, help="Effective aspect ratio for eq. (2.8)/(2.18).")

parser.add_argument("--disable-translation", action="store_true", help="Disable translation-induced load.")
parser.add_argument("--disable-rotation", action="store_true", help="Disable rotation-induced load.")
parser.add_argument("--disable-coupling", action="store_true", help="Disable coupling load.")
parser.add_argument("--disable-added-mass", action="store_true", help="Disable added-mass load.")
parser.add_argument("--include-wagner", action="store_true", help="Enable Wagner multiplier (requires t*; default=1).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Ensure repository `source/` is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from flapping_bot.flapping_bot.assets import FlappingBotCfg
from flapping_bot.flapping_bot.physics.qsm_wang2016 import WingGeometry, compute_aero_wrench_from_omega_alpha


def _estimate_wing_area_from_urdf(urdf_path: Path, *, mesh_scale: float) -> tuple[float | None, float | None]:
    """Estimate planform areas for left/right wings from the v50 URDF+STL.

    Returns:
        (area_left, area_right) in m^2. Values may be None if estimation fails.
    """
    try:
        helper_path = (Path(__file__).resolve().parents[1] / "source" / "flapping_bot" / "scripts" / "compute_wing_areas.py").resolve()
        spec = importlib.util.spec_from_file_location("_compute_wing_areas", helper_path)
        if spec is None or spec.loader is None:
            return None, None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        root = mod._parse_urdf(urdf_path)
        urdf_dir = urdf_path.parent

        def _area_for(link_name: str) -> float | None:
            mesh_rel = mod._find_link_mesh(root, link_name)
            if mesh_rel is None:
                return None
            axis = mod._find_joint_axis_for_child(root, link_name)
            if axis is None:
                axis = (1.0, 0.0, 0.0)
            mesh_path = (urdf_dir / mesh_rel).resolve()
            if not mesh_path.exists():
                return None
            tris = mod._stl_load(mesh_path)
            return float(mod._projected_area(tris, axis, scale=float(mesh_scale)))

        return _area_for("left_wing"), _area_for("right_wing")
    except Exception:
        return None, None


def _axis_to_vec(name: str) -> torch.Tensor:
    name = name.strip().lower()
    sign = 1.0
    if name.startswith("+"):
        name = name[1:]
    elif name.startswith("-"):
        sign = -1.0
        name = name[1:]
    if name == "x":
        v = torch.tensor([1.0, 0.0, 0.0])
    elif name == "y":
        v = torch.tensor([0.0, 1.0, 0.0])
    elif name == "z":
        v = torch.tensor([0.0, 0.0, 1.0])
    else:
        raise ValueError(f"Invalid axis '{name}'. Use x/y/z with optional +/- prefix.")
    return sign * v


def _link_to_wang_matrix(wang_axes: tuple[str, str, str], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Rows are Wang basis vectors expressed in link coordinates: v_wang = A @ v_link.
    ex = _axis_to_vec(wang_axes[0]).to(device=device, dtype=dtype)
    ey = _axis_to_vec(wang_axes[1]).to(device=device, dtype=dtype)
    ez = _axis_to_vec(wang_axes[2]).to(device=device, dtype=dtype)
    A = torch.stack((ex, ey, ez), dim=0)  # (3,3)
    # Ensure orthonormal (permutation/sign matrix).
    if not torch.allclose(A @ A.T, torch.eye(3, device=device, dtype=dtype), atol=1e-5):
        raise ValueError("--wang-axes must define an orthonormal axis mapping (pure permutation with sign flips).")
    return A


def main():
    sim_cfg = SimulationCfg(dt=args_cli.dt)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, -1.0, 0.8], target=[0.0, 0.0, 0.2])

    # Stage: ground + light
    cfg = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(8.0, 8.0))
    cfg.func("/World/ground", cfg)
    cfg = sim_utils.DistantLightCfg(intensity=1200.0, color=(0.85, 0.85, 0.85))
    cfg.func("/World/light", cfg)

    # Spawn the v50 flapping aircraft as a fixed-base rig.
    # Isaac uses a right-handed convention: positive rotation about +Y pitches the nose *down*.
    # Users typically specify +pitch as "nose up", so we negate it here.
    pitch = -math.radians(args_cli.body_pitch_deg)
    q_body = (math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0)  # (w,x,y,z)
    robot_cfg = FlappingBotCfg.replace(
        prim_path="/World/FlappingBotRig",
        spawn=FlappingBotCfg.spawn.replace(fix_base=True, force_usd_conversion=True),
        init_state=FlappingBotCfg.init_state.replace(
            pos=(0.0, 0.0, 0.6),
            rot=q_body,
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            joint_pos={
                "left_wing": 0.0,
                "right_wing": 0.0,
                "left_tail": 0.0,
                "right_tail": 0.0,
            },
        ),
    )
    robot = Articulation(robot_cfg)

    sim.reset()
    sim_dt = sim.get_physics_dt()
    robot.update(sim_dt)

    # Optional debug draw (works when a viewport is active; may be unavailable in strict headless runs).
    draw_interface = None
    if args_cli.draw_forces:
        try:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        except Exception as exc:
            print(f"[WARN] Debug draw unavailable: {exc}")
            draw_interface = None

    # Resolve joints and wing bodies.
    joint_ids, _ = robot.find_joints(["left_wing", "right_wing", "left_tail", "right_tail"], preserve_order=True)
    wing_body_ids, _ = robot.find_bodies(["left_wing", "right_wing"], preserve_order=True)
    base_body_ids, _ = robot.find_bodies(["base_link"], preserve_order=True)

    # QSM geometry used for both wings.
    # Default behavior is to estimate wing planform area from the v50 URDF+STL and derive (R,c) from area & aspect ratio.
    R = float(args_cli.R)
    c = float(args_cli.c)
    if args_cli.auto_geom_from_urdf:
        urdf_path = Path(robot_cfg.spawn.asset_path).resolve()
        aL, aR = _estimate_wing_area_from_urdf(urdf_path, mesh_scale=float(args_cli.mesh_scale))
        areas = [a for a in (aL, aR) if (a is not None and a > 0.0)]
        if areas:
            area_mean = float(sum(areas) / len(areas))
            AR = float(args_cli.aspect_ratio)
            if AR > 0.0:
                # Use S = R * c_bar and AR = R / c_bar => R = sqrt(S*AR), c_bar = sqrt(S/AR).
                R = math.sqrt(area_mean * AR)
                c = math.sqrt(area_mean / AR)
                print(f"[INFO] Auto-geom from URDF: wing_area≈{area_mean:.6f} m^2 -> R≈{R:.4f} m, c≈{c:.4f} m (AR={AR:g})")
        else:
            print("[WARN] Auto-geom requested but failed to estimate wing area; falling back to --R/--c.")

    wing_geom = WingGeometry.from_input(
        {
            "R": R,
            "N": args_cli.N,
            "c": c,
            "dhat": args_cli.dhat,
            "aspect_ratio": args_cli.aspect_ratio,
            "enable_translation": not args_cli.disable_translation,
            "enable_rotation": not args_cli.disable_rotation,
            "enable_coupling": not args_cli.disable_coupling,
            "enable_added_mass": not args_cli.disable_added_mass,
        },
        device=sim.device,
        dtype=torch.float32,
    )

    # Frame mapping (wing link -> Wang co-rotating frame).
    A_l2w = _link_to_wang_matrix(tuple(args_cli.wang_axes), device=sim.device, dtype=torch.float32)  # (3,3)
    A_w2l = A_l2w.T

    # Wing joint profile.
    w = 2.0 * math.pi * args_cli.f_hz
    amp = math.radians(args_cli.wing_amp_deg)

    # Wind-tunnel air flow in world frame (fixed, not body-aligned).
    # The paper's note under eq. (2.5) adds the wing's forward-flight velocity to v_c.
    # In a wind tunnel the wing is fixed while air moves, so we use: v_forward = -v_air.
    flow_dir = torch.tensor(args_cli.flow_dir_world, device=sim.device, dtype=torch.float32)
    flow_dir = flow_dir / torch.linalg.norm(flow_dir).clamp(min=1e-9)
    v_air_w = float(args_cli.airspeed) * flow_dir  # (3,)
    v_forward_w = -v_air_w  # (3,)

    args_cli.csv.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "t",
                "wing_q",
                "wing_qd",
                "wing_qdd",
                "airspeed",
                "v_air_wx",
                "v_air_wy",
                "v_air_wz",
                "body_pitch_deg",
                "F_wang_x_L",
                "F_wang_y_L",
                "F_wang_z_L",
                "tau_wang_x_L",
                "tau_wang_y_L",
                "tau_wang_z_L",
                "F_link_x_L",
                "F_link_y_L",
                "F_link_z_L",
                "tau_link_x_L",
                "tau_link_y_L",
                "tau_link_z_L",
                "F_world_x_L",
                "F_world_y_L",
                "F_world_z_L",
                "tau_world_x_L",
                "tau_world_y_L",
                "tau_world_z_L",
                "F_wang_x_R",
                "F_wang_y_R",
                "F_wang_z_R",
                "tau_wang_x_R",
                "tau_wang_y_R",
                "tau_wang_z_R",
                "F_link_x_R",
                "F_link_y_R",
                "F_link_z_R",
                "tau_link_x_R",
                "tau_link_y_R",
                "tau_link_z_R",
                "F_world_x_R",
                "F_world_y_R",
                "F_world_z_R",
                "tau_world_x_R",
                "tau_world_y_R",
                "tau_world_z_R",
            ]
        )

        for step in range(args_cli.steps):
            t = step * sim_dt

            # Current wing orientation (from sim state at the start of this step).
            q_w_link = robot.data.body_quat_w[0, wing_body_ids, :]  # (2,4), wxyz

            # Angular velocity/acceleration of wings in world (sim-provided).
            omega_w = robot.data.body_ang_vel_w[0, wing_body_ids, :]  # (2,3)
            alpha_w = robot.data.body_ang_acc_w[0, wing_body_ids, :]  # (2,3)

            # Express ω, α in wing link frame, then map to Wang co-rotating frame.
            omega_l = quat_apply_inverse(q_w_link, omega_w)
            alpha_l = quat_apply_inverse(q_w_link, alpha_w)
            omega_c = (A_l2w @ omega_l.unsqueeze(-1)).squeeze(-1)  # (2,3)
            alpha_c = (A_l2w @ alpha_l.unsqueeze(-1)).squeeze(-1)  # (2,3)

            # Forward-flight velocity expressed in Wang co-rotating frame.
            v_forward_l = quat_apply_inverse(q_w_link, v_forward_w.view(1, 3).expand(2, 3))
            v_forward_c = (A_l2w @ v_forward_l.unsqueeze(-1)).squeeze(-1)  # (2,3)

            # Compute aerodynamic wrench in Wang co-rotating frame and map back to link frame for application.
            F_c, tau_c = compute_aero_wrench_from_omega_alpha(
                omega_c,
                alpha_c,
                wing_geom,
                rho=args_cli.rho,
                include_wagner=bool(args_cli.include_wagner),
                v_forward_c=v_forward_c,
            )  # (2,3), (2,3)
            F_l = (A_w2l @ F_c.unsqueeze(-1)).squeeze(-1)
            tau_l = (A_w2l @ tau_c.unsqueeze(-1)).squeeze(-1)
            F_w = quat_apply(q_w_link, F_l)
            tau_w = quat_apply(q_w_link, tau_l)

            # Apply to wing links in link frame (local).
            robot.set_external_force_and_torque(
                forces=F_l.view(1, 2, 3), torques=tau_l.view(1, 2, 3), body_ids=wing_body_ids, is_global=False
            )

            # Prescribed joint targets (sync flapping). Tails locked at 0.
            q = amp * math.sin(w * t)
            qd = amp * w * math.cos(w * t)
            qdd = -amp * (w**2) * math.sin(w * t)
            joint_pos = torch.tensor([[q, q, 0.0, 0.0]], device=sim.device, dtype=torch.float32)
            robot.set_joint_position_target(joint_pos, joint_ids=joint_ids)

            # Push buffers and step.
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)

            if draw_interface is not None:
                try:
                    draw_interface.clear_lines()
                    p = robot.data.body_pos_w[0, wing_body_ids, :]  # (2,3)
                    pF = p + float(args_cli.force_scale) * F_w
                    pT = p + float(args_cli.torque_scale) * tau_w
                    colors_F = [[0.1, 0.8, 1.0, 1.0]] * 2
                    colors_T = [[1.0, 0.6, 0.1, 1.0]] * 2
                    thickness = [3.0] * 2
                    draw_interface.draw_lines(p.tolist(), pF.tolist(), colors_F, thickness)
                    draw_interface.draw_lines(p.tolist(), pT.tolist(), colors_T, thickness)
                except Exception:
                    # Keep simulation running even if drawing fails intermittently.
                    pass

            if args_cli.print_every > 0 and (step % int(args_cli.print_every) == 0):
                print(
                    f"[step {step:05d}] t={t:7.3f} q={q:+.3f} rad qd={qd:+.3f} rad/s | "
                    f"L Fy_wang={float(F_c[0,1]):+.3f} N, tau_x_wang={float(tau_c[0,0]):+.3f} N·m, tau_z_wang={float(tau_c[0,2]):+.3f} N·m | "
                    f"R Fy_wang={float(F_c[1,1]):+.3f} N, tau_x_wang={float(tau_c[1,0]):+.3f} N·m, tau_z_wang={float(tau_c[1,2]):+.3f} N·m"
                )

            writer.writerow(
                [
                    step,
                    float(t),
                    float(q),
                    float(qd),
                    float(qdd),
                    float(args_cli.airspeed),
                    float(v_air_w[0].item()),
                    float(v_air_w[1].item()),
                    float(v_air_w[2].item()),
                    float(args_cli.body_pitch_deg),
                    float(F_c[0, 0].item()),
                    float(F_c[0, 1].item()),
                    float(F_c[0, 2].item()),
                    float(tau_c[0, 0].item()),
                    float(tau_c[0, 1].item()),
                    float(tau_c[0, 2].item()),
                    float(F_l[0, 0].item()),
                    float(F_l[0, 1].item()),
                    float(F_l[0, 2].item()),
                    float(tau_l[0, 0].item()),
                    float(tau_l[0, 1].item()),
                    float(tau_l[0, 2].item()),
                    float(F_w[0, 0].item()),
                    float(F_w[0, 1].item()),
                    float(F_w[0, 2].item()),
                    float(tau_w[0, 0].item()),
                    float(tau_w[0, 1].item()),
                    float(tau_w[0, 2].item()),
                    float(F_c[1, 0].item()),
                    float(F_c[1, 1].item()),
                    float(F_c[1, 2].item()),
                    float(tau_c[1, 0].item()),
                    float(tau_c[1, 1].item()),
                    float(tau_c[1, 2].item()),
                    float(F_l[1, 0].item()),
                    float(F_l[1, 1].item()),
                    float(F_l[1, 2].item()),
                    float(tau_l[1, 0].item()),
                    float(tau_l[1, 1].item()),
                    float(tau_l[1, 2].item()),
                    float(F_w[1, 0].item()),
                    float(F_w[1, 1].item()),
                    float(F_w[1, 2].item()),
                    float(tau_w[1, 0].item()),
                    float(tau_w[1, 1].item()),
                    float(tau_w[1, 2].item()),
                ]
            )

    print(f"[OK] CSV written to: {args_cli.csv}")
    if base_body_ids:
        print(f"[INFO] base_link body id: {base_body_ids[0]} (fixed-base rig)")


if __name__ == "__main__":
    main()
    simulation_app.close()

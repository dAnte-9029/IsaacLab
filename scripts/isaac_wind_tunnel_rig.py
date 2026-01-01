# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab "wind-tunnel rig" demo for Wang2016 QSM with forward-flight velocity.

Scenario:
  - Fixed base (rig) with two identical wings
  - Both wings flap in sync (sweep φ(t))
  - Wing pitch is held constant at η0 (interpretable as an AoA setting in this rigid-wing QSM)
  - A constant forward-flight velocity v_forward_i is transformed into each wing's co-rotating frame
    and added to the translational velocity (paper note under eq. (2.5))
  - QSM wrench is applied as external wrench on each wing link

.. code-block:: bash

    ./isaaclab.sh -p scripts/isaac_wind_tunnel_rig.py --headless --steps 2400 --csv wang2016_wind_tunnel.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wang2016 wind-tunnel rig: dual wings + forward-flight velocity.")
parser.add_argument("--steps", type=int, default=2400, help="Number of physics steps.")
parser.add_argument("--csv", type=Path, default=Path("wang2016_wind_tunnel.csv"), help="Output CSV path.")

parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Physics step (s).")
parser.add_argument("--rho", type=float, default=1.225, help="Air density (kg/m^3).")
parser.add_argument("--N", type=int, default=60, help="Spanwise strip count for BEM discretization.")

parser.add_argument("--f_hz", type=float, default=2.0, help="Flapping frequency (Hz).")
parser.add_argument("--phi_amp_deg", type=float, default=30.0, help="Sweep amplitude φ (deg).")
parser.add_argument("--eta0_deg", type=float, default=15.0, help="Constant pitch η0 (deg).")

parser.add_argument(
    "--v-forward",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 5.0),
    metavar=("VX", "VY", "VZ"),
    help="Forward-flight velocity vector in inertial frame (m/s), added to translational velocity.",
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
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from flapping_bot.flapping_bot.physics.qsm_wang2016 import WingGeometry, compute_aero_wrench, rotation_matrix_i_from_c


def main():
    sim_cfg = SimulationCfg(dt=args_cli.dt)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.7, -1.0, 0.5], target=[0.1, 0.0, 0.0])

    # Stage: ground + light
    cfg = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(5.0, 5.0))
    cfg.func("/World/ground", cfg)
    cfg = sim_utils.DistantLightCfg(intensity=1200.0, color=(0.85, 0.85, 0.85))
    cfg.func("/World/light", cfg)

    urdf_path = (Path(__file__).parent / "assets" / "wang2016_dual_wing_rig.urdf").resolve()
    robot_cfg = ArticulationCfg(
        prim_path="/World/Wang2016Rig",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            fix_base=True,
            force_usd_conversion=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1500.0, damping=80.0)
            ),
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None),
        },
    )
    robot = Articulation(robot_cfg)

    sim.reset()
    sim_dt = sim.get_physics_dt()

    # Resolve joints and wing bodies.
    joint_names = ["left_sweep", "left_heave", "left_pitch", "right_sweep", "right_heave", "right_pitch"]
    joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
    wing_body_ids, _ = robot.find_bodies(["left_wing", "right_wing"], preserve_order=True)

    # Wing geometry for QSM.
    wing_geom = WingGeometry.from_input(
        {
            "R": args_cli.R,
            "N": args_cli.N,
            "c": args_cli.c,
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

    # Motion params.
    w = 2.0 * math.pi * args_cli.f_hz
    phi_amp = math.radians(args_cli.phi_amp_deg)
    eta0 = math.radians(args_cli.eta0_deg)
    v_forward_i = torch.tensor(args_cli.v_forward, device=sim.device, dtype=torch.float32)

    args_cli.csv.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "t",
                "phi",
                "eta0",
                "vfx_i",
                "vfy_i",
                "vfz_i",
                "Fy_c",
                "tau_x_c",
                "tau_z_c",
            ]
        )

        for step in range(args_cli.steps):
            t = step * sim_dt

            # Prescribed wing kinematics (Wang2016 Euler angles).
            phi = phi_amp * math.sin(w * t)
            theta = 0.0
            eta = eta0

            phid = phi_amp * w * math.cos(w * t)
            thetad = 0.0
            etad = 0.0
            phidd = -phi_amp * (w**2) * math.sin(w * t)
            thetadd = 0.0
            etadd = 0.0

            # Drive both wings identically.
            joint_pos = torch.tensor([[phi, theta, eta, phi, theta, eta]], device=sim.device, dtype=torch.float32)
            robot.set_joint_position_target(joint_pos, joint_ids=joint_ids)

            # Transform forward-flight velocity to the co-rotating frame for this (phi,theta,eta).
            phi_t = torch.tensor(phi, device=sim.device, dtype=torch.float32)
            theta_t = torch.tensor(theta, device=sim.device, dtype=torch.float32)
            eta_t = torch.tensor(eta, device=sim.device, dtype=torch.float32)
            R_i_c = rotation_matrix_i_from_c(phi_t, theta_t, eta_t)  # (3,3)
            v_forward_c = (R_i_c.transpose(-1, -2) @ v_forward_i.view(3, 1)).view(3)

            # QSM wrench in wing link frame (= co-rotating frame).
            F_c, tau_c = compute_aero_wrench(
                phi_t,
                theta_t,
                eta_t,
                torch.tensor(phid, device=sim.device, dtype=torch.float32),
                torch.tensor(thetad, device=sim.device, dtype=torch.float32),
                torch.tensor(etad, device=sim.device, dtype=torch.float32),
                torch.tensor(phidd, device=sim.device, dtype=torch.float32),
                torch.tensor(thetadd, device=sim.device, dtype=torch.float32),
                torch.tensor(etadd, device=sim.device, dtype=torch.float32),
                wing_geom,
                rho=args_cli.rho,
                include_wagner=bool(args_cli.include_wagner),
                v_forward_c=v_forward_c,
            )

            forces = torch.stack((F_c, F_c), dim=0).view(1, 2, 3)
            torques = torch.stack((tau_c, tau_c), dim=0).view(1, 2, 3)
            robot.set_external_force_and_torque(
                forces=forces, torques=torques, body_ids=wing_body_ids, is_global=False
            )

            # Push buffers and step.
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)

            writer.writerow(
                [
                    step,
                    float(t),
                    float(phi),
                    float(eta0),
                    float(v_forward_i[0].item()),
                    float(v_forward_i[1].item()),
                    float(v_forward_i[2].item()),
                    float(F_c[1].item()),
                    float(tau_c[0].item()),
                    float(tau_c[2].item()),
                ]
            )

    print(f"[OK] CSV written to: {args_cli.csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()

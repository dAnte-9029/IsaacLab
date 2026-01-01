# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab demo: apply Wang2016 quasi-steady aerodynamic wrench to a single wing link.

This is a minimal visualization/IO demo (not an RL environment):
  - Spawns a simple 3-DoF wing articulation (`scripts/assets/wang2016_single_wing.urdf`)
  - Drives joints with sinusoidal sweep/heave/pitch angles
  - Computes quasi-steady aero wrench in the wing's co-rotating frame (link frame)
  - Applies it as an external wrench to the wing rigid body
  - Logs forces/torques to CSV

.. code-block:: bash

    ./isaaclab.sh -p scripts/isaac_demo_task.py --headless --steps 1200 --csv /tmp/wang2016_demo.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wang2016 QSM demo: external wrench on a single wing link.")
parser.add_argument("--steps", type=int, default=2000, help="Number of physics steps.")
parser.add_argument("--csv", type=Path, default=Path("wang2016_isaac_demo.csv"), help="Output CSV path.")

parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Physics step (s).")
parser.add_argument("--rho", type=float, default=1.225, help="Air density (kg/m^3).")
parser.add_argument("--N", type=int, default=40, help="Spanwise strip count for BEM discretization.")

parser.add_argument("--f_hz", type=float, default=2.0, help="Motion frequency (Hz).")
parser.add_argument("--phi_amp_deg", type=float, default=30.0, help="Sweep amplitude (deg).")
parser.add_argument("--theta_amp_deg", type=float, default=0.0, help="Heave amplitude (deg).")
parser.add_argument("--eta_amp_deg", type=float, default=20.0, help="Pitch amplitude (deg).")
parser.add_argument("--eta_phase_deg", type=float, default=90.0, help="Pitch phase lead relative to sweep (deg).")

parser.add_argument("--R", type=float, default=0.10, help="Span R (m).")
parser.add_argument("--c", type=float, default=0.05, help="Chord c (m).")
parser.add_argument("--dhat", type=float, default=0.25, help="Pitch axis offset d̂ (LE->axis)/c.")
parser.add_argument("--aspect_ratio", type=float, default=2.0, help="Effective aspect ratio for eq. (2.8)/(2.18).")

parser.add_argument("--disable-translation", action="store_true", help="Disable translation-induced load.")
parser.add_argument("--disable-rotation", action="store_true", help="Disable rotation-induced load.")
parser.add_argument("--disable-coupling", action="store_true", help="Disable coupling load.")
parser.add_argument("--disable-added-mass", action="store_true", help="Disable added-mass load.")

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

from flapping_bot.flapping_bot.physics.qsm_wang2016 import WingGeometry, compute_aero_wrench


def main():
    sim_cfg = SimulationCfg(dt=args_cli.dt)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.6, -0.6, 0.4], target=[0.0, 0.0, 0.0])

    # Stage: ground + light
    cfg = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(5.0, 5.0))
    cfg.func("/World/ground", cfg)
    cfg = sim_utils.DistantLightCfg(intensity=1200.0, color=(0.85, 0.85, 0.85))
    cfg.func("/World/light", cfg)

    urdf_path = (Path(__file__).parent / "assets" / "wang2016_single_wing.urdf").resolve()
    robot_cfg = ArticulationCfg(
        prim_path="/World/WingRobot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            fix_base=True,
            force_usd_conversion=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1000.0, damping=50.0)
            ),
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None),
        },
    )
    robot = Articulation(robot_cfg)

    sim.reset()
    sim_dt = sim.get_physics_dt()

    # Resolve joints and wing body.
    joint_ids, _ = robot.find_joints(["sweep", "heave", "pitch"], preserve_order=True)
    wing_body_ids, _ = robot.find_bodies(["wing"], preserve_order=True)

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
    theta_amp = math.radians(args_cli.theta_amp_deg)
    eta_amp = math.radians(args_cli.eta_amp_deg)
    eta_phase = math.radians(args_cli.eta_phase_deg)

    args_cli.csv.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "t", "phi", "theta", "eta", "Fy_c", "tau_x_c", "tau_z_c"])

        for step in range(args_cli.steps):
            t = step * sim_dt
            # Prescribed joint angles (Wang2016 Euler angles).
            phi = phi_amp * math.sin(w * t)
            theta = theta_amp * math.sin(w * t)
            eta = eta_amp * math.sin(w * t + eta_phase)

            # Derivatives for the aerodynamic model.
            phid = phi_amp * w * math.cos(w * t)
            thetad = theta_amp * w * math.cos(w * t)
            etad = eta_amp * w * math.cos(w * t + eta_phase)
            phidd = -phi_amp * (w**2) * math.sin(w * t)
            thetadd = -theta_amp * (w**2) * math.sin(w * t)
            etadd = -eta_amp * (w**2) * math.sin(w * t + eta_phase)

            # Drive joints.
            joint_pos = torch.tensor([[phi, theta, eta]], device=sim.device, dtype=torch.float32)
            robot.set_joint_position_target(joint_pos, joint_ids=joint_ids)

            # QSM wrench in wing link frame (= co-rotating frame).
            # Use device tensors so the computation stays on the simulation device.
            phi_t = torch.tensor(phi, device=sim.device, dtype=torch.float32)
            theta_t = torch.tensor(theta, device=sim.device, dtype=torch.float32)
            eta_t = torch.tensor(eta, device=sim.device, dtype=torch.float32)
            phid_t = torch.tensor(phid, device=sim.device, dtype=torch.float32)
            thetad_t = torch.tensor(thetad, device=sim.device, dtype=torch.float32)
            etad_t = torch.tensor(etad, device=sim.device, dtype=torch.float32)
            phidd_t = torch.tensor(phidd, device=sim.device, dtype=torch.float32)
            thetadd_t = torch.tensor(thetadd, device=sim.device, dtype=torch.float32)
            etadd_t = torch.tensor(etadd, device=sim.device, dtype=torch.float32)
            F_c, tau_c = compute_aero_wrench(
                phi_t,
                theta_t,
                eta_t,
                phid_t,
                thetad_t,
                etad_t,
                phidd_t,
                thetadd_t,
                etadd_t,
                wing_geom,
                rho=args_cli.rho,
                include_wagner=False,
            )
            forces = F_c.view(1, 1, 3)
            torques = tau_c.view(1, 1, 3)
            robot.set_external_force_and_torque(forces=forces, torques=torques, body_ids=wing_body_ids, is_global=False)

            # Push buffers and step.
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)

            writer.writerow([step, t, phi, theta, eta, float(F_c[1]), float(tau_c[0]), float(tau_c[2])])

    print(f"[OK] CSV written to: {args_cli.csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()

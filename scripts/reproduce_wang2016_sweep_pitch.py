# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproduce (approximately) Wang2016 §3.1 sweeping–pitching plate.

This script is IsaacSim-independent. It generates a CSV time-series using the
Wang et al. (JFM 2016) quasi-steady model implemented in
`flapping_bot.physics.qsm_wang2016`.

.. code-block:: bash

    # CPU (no Isaac Sim needed)
    python scripts/reproduce_wang2016_sweep_pitch.py --case I --out /tmp/wang2016_caseI.csv

Notes / TODOs:
    - The exact pitching acceleration profile at start/stop is not provided in the paper.
      By default, this script disables the added-mass term (as in the discussion under Fig. 11-12).
      Use `--include-added-mass` to enable it with a piecewise-linear pitch (α spikes ignored).
    - The paper uses an effective aspect ratio A_eff=2.83 for this setup. We expose it as an override.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from flapping_bot.flapping_bot.physics.qsm_wang2016 import WingGeometry, compute_aero_wrench, rotation_matrix_i_from_c


def _sweep_profile(t: torch.Tensor, omega_target: float, t_accel: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sweep yaw φ(t): accelerate from rest to omega_target over t_accel, then constant ω."""
    # Constant acceleration phase
    a = omega_target / t_accel
    t1 = t_accel
    phidd = torch.where(t <= t1, torch.full_like(t, a), torch.zeros_like(t))
    phid = torch.where(t <= t1, a * t, torch.full_like(t, omega_target))
    phi = torch.where(t <= t1, 0.5 * a * t * t, 0.5 * a * t1 * t1 + omega_target * (t - t1))
    return phi, phid, phidd


def _pitch_profile_linear(
    t: torch.Tensor, *, eta_final: float, t_start: float, t_ramp: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pitch η(t): 0 until t_start, then linear ramp to eta_final over t_ramp, then constant."""
    t_end = t_start + t_ramp
    etad = torch.zeros_like(t)
    eta = torch.zeros_like(t)
    # Ramp region
    ramp_mask = (t >= t_start) & (t <= t_end)
    after_mask = t > t_end
    etad_val = eta_final / t_ramp
    eta = torch.where(ramp_mask, etad_val * (t - t_start), eta)
    eta = torch.where(after_mask, torch.full_like(t, eta_final), eta)
    etad = torch.where(ramp_mask, torch.full_like(t, etad_val), etad)
    etadd = torch.zeros_like(t)  # TODO: unknown accelerations at boundaries
    return eta, etad, etadd


def main():
    parser = argparse.ArgumentParser(description="Approx reproduction of Wang2016 §3.1 sweeping–pitching plate.")
    parser.add_argument("--case", choices=["I", "II"], default="I", help="Pitching case (I: 0.25s ramp, II: 0.5s).")
    parser.add_argument("--dt", type=float, default=0.002, help="Time step (s).")
    parser.add_argument("--t_end", type=float, default=2.0, help="End time (s).")
    parser.add_argument("--out", type=Path, default=Path("wang2016_sweep_pitch.csv"), help="Output CSV path.")
    parser.add_argument("--rho", type=float, default=1.225, help="Air density (kg/m^3).")
    parser.add_argument("--N", type=int, default=200, help="Spanwise strips for BEM discretization.")

    parser.add_argument("--include-added-mass", action="store_true", help="Enable added-mass term (see script notes).")
    parser.add_argument("--include-wagner", action="store_true", help="Enable Wagner multiplier (requires t*; default=1).")

    # Geometry defaults from §3.1 description / Fig. 9
    parser.add_argument("--R", type=float, default=0.1, help="Span R (m).")
    parser.add_argument("--c", type=float, default=0.05, help="Chord c (m).")
    parser.add_argument("--dhat", type=float, default=0.05, help="Pitch axis offset d̂ (LE->axis)/c.")
    parser.add_argument("--aspect-ratio", type=float, default=2.83, help="Effective aspect ratio A_eff used in paper.")

    args = parser.parse_args()

    t = torch.arange(0.0, args.t_end + args.dt * 0.5, args.dt, dtype=torch.float64)

    omega_sweep = math.radians(103.0)  # 103 deg/s
    t_accel = 0.5
    phi, phid, phidd = _sweep_profile(t, omega_sweep, t_accel)

    # No heave (θ=0)
    theta = torch.zeros_like(t)
    thetad = torch.zeros_like(t)
    thetadd = torch.zeros_like(t)

    # Pitch starts at 0.5s, reaches 45deg.
    eta_final = math.radians(45.0)
    t_ramp = 0.25 if args.case == "I" else 0.5
    eta, etad, etadd = _pitch_profile_linear(t, eta_final=eta_final, t_start=0.5, t_ramp=t_ramp)

    wing_geom = WingGeometry.from_input(
        {
            "R": args.R,
            "N": args.N,
            "c": args.c,
            "dhat": args.dhat,
            "aspect_ratio": args.aspect_ratio,
            "enable_added_mass": bool(args.include_added_mass),
        },
        dtype=t.dtype,
    )

    F_c, tau_c = compute_aero_wrench(
        phi,
        theta,
        eta,
        phid,
        thetad,
        etad,
        phidd,
        thetadd,
        etadd,
        wing_geom,
        rho=args.rho,
        include_wagner=bool(args.include_wagner),
    )

    # Transform to inertial frame for reporting.
    R_i_c = rotation_matrix_i_from_c(phi, theta, eta)  # (T,3,3)
    F_i = (R_i_c @ F_c.unsqueeze(-1)).squeeze(-1)
    tau_i = (R_i_c @ tau_c.unsqueeze(-1)).squeeze(-1)

    # Drag defined opposite to translational velocity direction (paper note under Fig. 11-12).
    x_ref = wing_geom.gyradius_rhat2 * wing_geom.R
    omega = torch.stack(
        (
            etad - phid * torch.sin(theta),
            thetad * torch.cos(eta) + phid * torch.cos(theta) * torch.sin(eta),
            phid * torch.cos(eta) * torch.cos(theta) - thetad * torch.sin(eta),
        ),
        dim=-1,
    )
    v_c_ref = x_ref * torch.stack((torch.zeros_like(t), omega[:, 2], -omega[:, 1]), dim=-1)
    v_i_ref = (R_i_c @ v_c_ref.unsqueeze(-1)).squeeze(-1)
    v_norm = torch.linalg.norm(v_i_ref, dim=-1).clamp(min=1e-9)
    v_hat = v_i_ref / v_norm.unsqueeze(-1)
    drag = -torch.sum(F_i * v_hat, dim=-1)
    lift_z = F_i[:, 2]  # vertical component (not identical to aerodynamic lift definition in paper)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t",
                "phi_rad",
                "eta_rad",
                "phid",
                "etad",
                "Fy_c",
                "tau_x_c",
                "tau_z_c",
                "Fx_i",
                "Fy_i",
                "Fz_i",
                "drag_i",
                "lift_z_i",
                "tau_x_i",
                "tau_z_i",
            ]
        )
        for i in range(t.numel()):
            w.writerow(
                [
                    float(t[i]),
                    float(phi[i]),
                    float(eta[i]),
                    float(phid[i]),
                    float(etad[i]),
                    float(F_c[i, 1]),
                    float(tau_c[i, 0]),
                    float(tau_c[i, 2]),
                    float(F_i[i, 0]),
                    float(F_i[i, 1]),
                    float(F_i[i, 2]),
                    float(drag[i]),
                    float(lift_z[i]),
                    float(tau_i[i, 0]),
                    float(tau_i[i, 2]),
                ]
            )

    print(f"[OK] Wrote {t.numel()} samples to: {args.out}")


if __name__ == "__main__":
    main()

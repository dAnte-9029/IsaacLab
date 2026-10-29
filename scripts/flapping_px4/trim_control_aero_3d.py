"""Solve one mean level-flight candidate; fit operating point, never aero parameters."""
from pathlib import Path
import argparse
import json, math
import numpy as np
import torch
from scipy.optimize import least_squares
from flapping_bot.analysis.crosswind_aero import CSV, URDF, CG, TAIL
from flapping_bot.physics.control_aero_kinematics import LinkKinematics3D
from flapping_bot.physics.control_aero_3d import *
from flapping_bot.physics.tail_aero import TailAeroModel
from flapping_bot.physics.measured_mass_properties import (
    measured_three_link_total_mass_kg,
    BODY_LINK_MASS_PROPERTIES,
    RIGHT_WING_LINK_MASS_PROPERTIES,
    LEFT_WING_LINK_MASS_PROPERTIES,
)
from flapping_bot.physics.ullt_geometry import rotation_xyz

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("logs/control_aero_3d/trim_system.json"))
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
torch.set_num_threads(1)
steps = 128
phase = torch.arange(steps, dtype=torch.float64) * 2 * math.pi / steps
weight = measured_three_link_total_mass_kg() * 9.81
out = {}
for model in ("L3D", "I3D"):
    cfg = ControlAero3DConfig(backend=model, twist_amplitude_deg=35)
    kin = LinkKinematics3D(CSV, URDF, cfg, dtype=torch.float64)
    backend = ControlAero3D(cfg)
    angles = phase.new_zeros((steps, 2, 3))
    angles[..., 0] = 0.5233 * phase.sin()[:, None] * phase.new_tensor([-1, 1])
    rotations = rotation_xyz(angles) @ torch.stack(kin.neutral_rot)[None]
    wing_com = phase.new_tensor([RIGHT_WING_LINK_MASS_PROPERTIES.com_m, LEFT_WING_LINK_MASS_PROPERTIES.com_m])
    com_positions = torch.stack(kin.origins)[None] + torch.einsum("bwij,wj->bwi", rotations, wing_com)
    system_com = (
        CG[None] * BODY_LINK_MASS_PROPERTIES.mass_kg + RIGHT_WING_LINK_MASS_PROPERTIES.mass_kg * com_positions.sum(1)
    ) / measured_three_link_total_mass_kg()

    def evaluate(x):
        alpha, freq, ele = x
        alpha = math.radians(alpha)
        v = torch.tensor([[8 * math.cos(alpha), 0.0, -8 * math.sin(alpha)]], dtype=torch.float64).expand(steps, -1)
        omega = v * 0
        ref = CG[None].expand(steps, -1)
        g, m, rel = kin.prescribed(v, omega, ref, phase, freq)
        loads = backend.evaluate(g, m, ref, rel)
        delta = phase * 0 + math.radians(ele)
        ft, mt = TailAeroModel(TAIL, "cpu").compute_wrench(
            root_lin_vel_b=v,
            root_ang_vel_b=omega,
            left_elevon_rad=delta,
            right_elevon_rad=delta,
            rudder_rad=delta * 0,
            base_com_pos_b=ref,
        )
        fb, mb = distributed_body_drag(v, omega, ref)
        force_phase = loads.force.sum(1) + ft + fb
        force = force_phase.mean(0)
        moment = (loads.moment.sum(1) + mt + mb - torch.linalg.cross(system_com - ref, force_phase)).mean(0)
        return np.array(
            [float(force[0]) - weight * math.sin(alpha), float(force[2]) - weight * math.cos(alpha), float(moment[1])]
        )

    fit = least_squares(
        evaluate,
        [8.0, 3.0, -4.0],
        bounds=([0.0, 1.0, -25.0], [20.0, 5.0, 25.0]),
        diff_step=1e-4,
        xtol=1e-7,
        ftol=1e-7,
        gtol=1e-7,
        max_nfev=60,
    )
    out[model] = dict(
        moment_reference="instantaneous measured three-link system COM",
        alpha_deg=fit.x[0],
        frequency_hz=fit.x[1],
        symmetric_tail_deg=fit.x[2],
        residual_N_N_Nm=evaluate(fit.x).tolist(),
        success=bool(fit.success),
        evaluations=fit.nfev,
    )
    print(model, out[model], flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, indent=2))

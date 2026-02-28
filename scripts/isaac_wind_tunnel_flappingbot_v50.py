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

    ./isaaclab.sh -p scripts/isaac_wind_tunnel_flappingbot_v50.py --headless --steps 600 --run-root outputs_DeLaurier/runs
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
parser.add_argument("--steps", type=int, default=600, help="Number of physics steps.")
parser.add_argument("--csv", type=Path, default=None, help="Output CSV path. If omitted, auto-create a run folder.")
parser.add_argument(
    "--run-root",
    type=Path,
    default=Path("outputs_DeLaurier/runs"),
    help="Root folder for per-run outputs when --csv is omitted.",
)
parser.add_argument(
    "--run-id",
    type=int,
    default=None,
    help="Optional run index (e.g. 1 => run_0001). If omitted, auto-increment.",
)

parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Physics step (s).")
parser.add_argument("--rho", type=float, default=1.225, help="Air density (kg/m^3).")
parser.add_argument("--N", type=int, default=60, help="Spanwise strip count for BEM discretization.")

parser.add_argument("--f_hz", type=float, default=2.0, help="Flapping frequency (Hz).")
parser.add_argument(
    "--aero-model",
    type=str,
    choices=("wang2016", "delaurier1993"),
    default="wang2016",
    help="Aerodynamic model to use for the wind-tunnel rig.",
)
parser.add_argument(
    "--enable-modulation",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable non-uniform flapping (downstroke faster/slower than upstroke).",
)
parser.add_argument(
    "--modulation-profile",
    type=str,
    choices=("phase_warp", "phase_ff"),
    default="phase_warp",
    help="Modulation profile: time-warped phase ('phase_warp') or phase feedforward ('phase_ff').",
)
parser.add_argument(
    "--downstroke-ratio",
    type=float,
    default=0.5,
    help="Downstroke duration ratio delta in (0,1). Downstroke lasts delta*T.",
)
parser.add_argument(
    "--modulation-mode",
    type=str,
    choices=("fixed_f", "fixed_peak_qd"),
    default="fixed_f",
    help=(
        "Phase-warp constraint: "
        "'fixed_f' keeps --f_hz as the period, "
        "'fixed_peak_qd' sets f so downstroke peak |qdot| matches a sine at --modulation-f-ref-hz."
    ),
)
parser.add_argument(
    "--modulation-smoothness",
    type=float,
    default=0.0,
    help="Smooth transition width (fraction of cycle) for the down/up stroke speed change.",
)
parser.add_argument(
    "--modulation-ff-amp",
    type=float,
    default=0.0,
    help="Phase feedforward amplitude (fraction of nominal phase speed, e.g. 0.2).",
)
parser.add_argument(
    "--modulation-ff-shift-deg",
    type=float,
    default=180.0,
    help="Phase feedforward shift (deg). Default 180 aligns max modulation with downstroke mid.",
)
parser.add_argument(
    "--modulation-f-ref-hz",
    type=float,
    default=4.5,
    help="Reference frequency for 'fixed_peak_qd' mode (defines peak |qdot| limit).",
)
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
parser.add_argument("--print-every", type=int, default=37, help="Print a summary every N steps (0 disables).")
parser.add_argument("--draw-forces", action="store_true", help="Draw force/torque arrows in the viewport (if available).")
parser.add_argument("--force-scale", type=float, default=0.02, help="Scale factor for drawing force vectors (m/N).")
parser.add_argument("--torque-scale", type=float, default=0.05, help="Scale factor for drawing torque vectors (m/(N·m)).")
parser.add_argument("--draw-frames", action="store_true", help="Draw wing link frame axes (viewport required).")
parser.add_argument("--draw-wang-frames", action="store_true", help="Also draw Wang co-rotating axes (cyan/magenta/yellow).")
parser.add_argument("--frame-axis-length", type=float, default=0.08, help="Axis length for frame drawing (m).")
parser.add_argument("--frame-axis-thickness", type=float, default=2.0, help="Line thickness for frame drawing.")
parser.add_argument(
    "--draw-world-frame",
    action="store_true",
    help="Draw world axes (uses origin or base_link as the anchor).",
)
parser.add_argument("--world-axis-length", type=float, default=None, help="World axis length (m). Defaults to --frame-axis-length.")
parser.add_argument(
    "--world-axis-thickness",
    type=float,
    default=None,
    help="World axis line thickness. Defaults to --frame-axis-thickness.",
)
parser.add_argument(
    "--world-frame-origin",
    type=str,
    choices=("origin", "base"),
    default="base",
    help="Anchor point for world axes (default: base_link position).",
)
parser.add_argument(
    "--drive-mode",
    type=str,
    choices=("target", "kinematic"),
    default="kinematic",
    help="How to drive the wing motion. 'kinematic' writes joint pos/vel to sim each step (rig/stand).",
)
parser.add_argument(
    "--qsm-from-command",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If enabled, compute QSM ω/α from commanded sweep rate (stable/symmetric). If disabled, use sim body ω/α.",
)

parser.add_argument(
    "--wang-axes",
    type=str,
    nargs=3,
    default=("y", "z", "x"),
    metavar=("WANG_X", "WANG_Y", "WANG_Z"),
    help=(
        "Co-rotating Wang frame axes expressed in the wing link frame (e.g. 'y z x'). "
        "Supports +/- prefixes. Note: argparse treats tokens starting with '-' as options, so use "
        "'mx'/'my'/'mz' as synonyms for '-x'/'-y'/'-z' when passing negative axes."
    ),
)
parser.add_argument(
    "--wang-axes-left",
    type=str,
    nargs=3,
    default=None,
    metavar=("WANG_X", "WANG_Y", "WANG_Z"),
    help=(
        "Override --wang-axes for the left wing only (supports +/- prefixes; use mx/my/mz for negatives)."
    ),
)
parser.add_argument(
    "--wang-axes-right",
    type=str,
    nargs=3,
    default=None,
    metavar=("WANG_X", "WANG_Y", "WANG_Z"),
    help=(
        "Override --wang-axes for the right wing only (supports +/- prefixes; use mx/my/mz for negatives)."
    ),
)
parser.add_argument(
    "--auto-mirror-wang-x",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If --wang-axes-left/right are not provided, mirror Wang x-axis sign for the right wing (common for mirrored link frames).",
)

parser.add_argument("--R", type=float, default=0.10, help="Span R (m).")
parser.add_argument("--c", type=float, default=0.05, help="Chord c (m).")
parser.add_argument("--dhat", type=float, default=0.0, help="Pitch axis offset d̂ (LE->axis)/c. Use 0 for leading edge.")
parser.add_argument(
    "--aspect_ratio",
    type=float,
    default=0.0,
    help="Effective aspect ratio used in eq. (2.8)/(2.18). If <=0, compute from geometry as AR=R^2/S.",
)
parser.add_argument(
    "--wing-geom-csv",
    type=Path,
    default=Path("outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv"),
    help=(
        "Wing geometry CSV from fit_wing_geom_from_points.py (columns: x_mid_m,c_m,dhat). "
        "Overrides --R/--c."
    ),
)

parser.add_argument(
    "--twist-mode",
    type=str,
    choices=("off", "quasi_static", "dynamic", "prescribed"),
    default="off",
    help=(
        "Enable an outboard twist model on span segment [--twist-x0, --twist-x1]. "
        "'quasi_static'/'dynamic' couple twist to the aerodynamic pitching moment (FSI-like). "
        "'prescribed' uses a simple open-loop η_tip(t) for fast comparison runs."
    ),
)
parser.add_argument("--twist-x0", type=float, default=0.0, help="Start of twisting span region (m). Default: 0 (full-span).")
parser.add_argument(
    "--twist-x1",
    type=float,
    default=None,
    help="End of twisting span region (m). Default: full-span (uses wing R inferred from geometry).",
)
parser.add_argument(
    "--twist-eta-shape",
    type=str,
    choices=("uniform", "linear", "smoothstep"),
    default="uniform",
    help=(
        "Spanwise twist shape g(x) in the twist region. "
        "'uniform' reproduces the previous behavior (η(x)=η_tip for all strips in [x0,x1]); "
        "'linear'/'smoothstep' ramp from 0 at x0 to 1 at x1: η(x)=g(x)*η_tip."
    ),
)
parser.add_argument("--twist-eta-limit-deg", type=float, default=20.0, help="Clamp |eta_tip| <= limit (deg).")
parser.add_argument("--twist-rest-deg", type=float, default=0.0, help="Rest twist angle eta0 (deg).")
parser.add_argument("--twist-k", type=float, default=0.3, help="Torsional stiffness k (N·m/rad).")
parser.add_argument("--twist-c", type=float, default=0.01, help="Torsional damping c (N·m·s/rad).")
parser.add_argument("--twist-I", type=float, default=1e-4, help="Torsional inertia I (kg·m^2).")
parser.add_argument("--twist-sign", type=float, default=1.0, help="Sign convention (use -1 to flip twist direction).")
parser.add_argument(
    "--twist-sign-left",
    type=float,
    default=None,
    help="Optional override for left wing twist sign (useful when left/right x_c are opposite).",
)
parser.add_argument(
    "--twist-sign-right",
    type=float,
    default=None,
    help="Optional override for right wing twist sign (useful when left/right x_c are opposite).",
)
parser.add_argument(
    "--auto-twist-sign",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Automatically choose per-wing twist sign so that positive η_tip moves the trailing edge 'up' in world (+Z) "
        "at the initial pose. Disable to use --twist-sign (or --twist-sign-left/right) verbatim."
    ),
)
parser.add_argument("--twist-solve-iters", type=int, default=3, help="Quasi-static fixed-point iterations per step.")
parser.add_argument("--twist-relax", type=float, default=0.8, help="Quasi-static relaxation factor (0,1].")
parser.add_argument(
    "--twist-prescribed",
    type=str,
    choices=("sin", "sign_qd", "qd_scaled"),
    default="sin",
    help=(
        "When --twist-mode prescribed: "
        "'sin' -> η_tip=η0+sign*η_amp*sin(2πft+phase); "
        "'sign_qd' -> η_tip=η0+sign*η_amp*sign(qd_cmd); "
        "'qd_scaled' -> η_tip=η0+sign*η_max*(qd_cmd/qd_ref) where qd_ref=amp*2π*f_ref."
    ),
)
parser.add_argument("--twist-eta-amp-deg", type=float, default=20.0, help="When --twist-mode prescribed: η_amp (deg).")
parser.add_argument("--twist-eta-phase-deg", type=float, default=0.0, help="When --twist-mode prescribed: phase shift (deg).")
parser.add_argument(
    "--twist-f-ref-hz",
    type=float,
    default=4.5,
    help="When --twist-prescribed qd_scaled: reference flapping frequency f_ref (Hz) that defines qd_ref.",
)
parser.add_argument(
    "--twist-eta-max-deg",
    type=float,
    default=40.0,
    help="When --twist-prescribed qd_scaled: |η_tip| at qd_cmd=qd_ref (deg) before applying --twist-eta-limit-deg.",
)

parser.add_argument("--disable-translation", action="store_true", help="Disable translation-induced load.")
parser.add_argument("--disable-rotation", action="store_true", help="Disable rotation-induced load.")
parser.add_argument("--disable-coupling", action="store_true", help="Disable coupling load.")
parser.add_argument("--disable-added-mass", action="store_true", help="Disable added-mass load.")
parser.add_argument("--include-wagner", action="store_true", help="Enable Wagner multiplier (requires t*; default=1).")
parser.add_argument("--aoa-alpha1-deg", type=float, default=15.0, help="AOA stats: pre-stall threshold α1 (deg).")
parser.add_argument("--aoa-alpha2-deg", type=float, default=35.0, help="AOA stats: post-stall threshold α2 (deg).")
parser.add_argument(
    "--profile-cd0",
    type=float,
    default=0.0,
    help=(
        "Optional profile/viscous drag coefficient CD0 per strip (dimensionless). "
        "If >0, adds dF_prof = -0.5*rho*|u|^2*(c*dx)*CD0*w(aoa)*u_hat to reduce over-predicted thrust "
        "at low/moderate AOA (extension, not in Wang2016)."
    ),
)
parser.add_argument("--profile-alpha1-deg", type=float, default=15.0, help="Profile drag weight: α1 (deg), w≈1 for aoa<=α1.")
parser.add_argument("--profile-alpha2-deg", type=float, default=35.0, help="Profile drag weight: α2 (deg), w≈0 for aoa>=α2.")
parser.add_argument("--delaurier-alpha0-deg", type=float, default=0.5, help="DeLaurier: zero-lift angle α0 (deg).")
parser.add_argument("--delaurier-eta-s", type=float, default=0.98, help="DeLaurier: leading-edge suction efficiency η_s.")
parser.add_argument(
    "--delaurier-cd-cf",
    type=float,
    default=1.98,
    help="DeLaurier: separated-flow crossflow drag coefficient C_d_cf.",
)
parser.add_argument(
    "--delaurier-alpha-stall-max-deg",
    type=float,
    default=13.0,
    help="DeLaurier: static stall angle α_stall_max (deg).",
)
parser.add_argument(
    "--delaurier-alpha-stall-min-deg",
    type=float,
    default=-180.0,
    help="DeLaurier: lower stall bound α_stall_min (deg).",
)
parser.add_argument("--delaurier-xi", type=float, default=0.0, help="DeLaurier: dynamic stall coefficient ξ.")
parser.add_argument("--delaurier-cmac", type=float, default=0.025, help="DeLaurier: pitching moment coefficient C_mac.")
parser.add_argument(
    "--delaurier-cd-f",
    type=float,
    default=None,
    help="DeLaurier: friction drag coefficient C_d_f. If omitted, compute from Re via Eq.(43).",
)
parser.add_argument(
    "--delaurier-nu",
    type=float,
    default=1.5e-5,
    help="DeLaurier: kinematic viscosity nu (m^2/s) used for C_d_f if not provided.",
)
parser.add_argument(
    "--delaurier-theta-a-deg",
    type=float,
    default=None,
    help="DeLaurier: flapping-axis angle theta_a (deg). Default: body_pitch_deg.",
)
parser.add_argument(
    "--delaurier-theta-w-deg",
    type=float,
    default=0.0,
    help="DeLaurier: mean chord angle relative to flapping axis theta_w (deg).",
)
parser.add_argument(
    "--delaurier-enable-separation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="DeLaurier: enable separated-flow model (Eq. 24-29).",
)
parser.add_argument(
    "--force-exit",
    action="store_true",
    help="Force terminate the process after closing Isaac Sim (workaround for rare shutdown hangs).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--sim-device",
    type=str,
    default=None,
    help=(
        "Override the physics simulation device used by IsaacLab SimulationCfg. "
        "This is useful when GPUs are busy: e.g. use `--device cuda:1 --sim-device cpu` "
        "to run Kit on GPU1 but force PhysX to CPU."
    ),
)
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
from flapping_bot.flapping_bot.physics.qsm_wang2016 import WingGeometry, compute_aero_wrench_from_omega_alpha, eta_shape_piecewise
from flapping_bot.flapping_bot.physics.phase_warp import PhaseWarp
from flapping_bot.flapping_bot.physics.qsm_delaurier1993 import DeLaurierParams, compute_aero_wrench_delaurier1993
from flapping_bot.flapping_bot.physics.virtual_twist import (
    VirtualTwistCfg,
    VirtualTwistState,
    solve_quasi_static_eta_tip,
    step_virtual_twist,
)


def _load_wing_geom_csv(path: Path) -> tuple[list[float], list[float], list[float]]:
    import csv

    xs: list[float] = []
    cs: list[float] = []
    dhats: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"x_mid_m", "c_m", "dhat"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(f"Invalid --wing-geom-csv header in {path}. Expected columns: {sorted(expected)}")
        for row in reader:
            xs.append(float(row["x_mid_m"]))
            cs.append(float(row["c_m"]))
            dhats.append(float(row["dhat"]))
    if len(xs) < 2:
        raise ValueError(f"--wing-geom-csv must contain at least 2 rows: {path}")
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    cs = [cs[i] for i in order]
    dhats = [dhats[i] for i in order]
    return xs, cs, dhats


def _infer_span_from_x_mid(xs: list[float]) -> float:
    if len(xs) < 2:
        raise ValueError("Need at least 2 x_mid samples to infer R.")
    dxs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    dxs_sorted = sorted(dxs)
    dx = float(dxs_sorted[len(dxs_sorted) // 2])  # median
    if dx <= 0:
        raise ValueError("x_mid must be strictly increasing.")
    return float(xs[-1] + 0.5 * dx)


def _resolve_run_dir(run_root: Path, run_id: int | None) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    if run_id is not None:
        return run_root / f"run_{run_id:04d}"
    existing: list[int] = []
    for p in run_root.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
            suffix = p.name[4:]
            if suffix.isdigit():
                existing.append(int(suffix))
    next_id = max(existing) + 1 if existing else 1
    return run_root / f"run_{next_id:04d}"


def _interp1d_torch(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """1D linear interpolation like numpy.interp, supports CUDA tensors."""
    if xp.ndim != 1 or fp.ndim != 1:
        raise ValueError("xp/fp must be 1D tensors.")
    if xp.numel() != fp.numel():
        raise ValueError("xp and fp must have the same length.")
    if xp.numel() < 2:
        raise ValueError("xp must have at least 2 points.")
    if not torch.all(xp[1:] > xp[:-1]):
        raise ValueError("xp must be strictly increasing.")

    x_clamped = torch.clamp(x, float(xp[0].item()), float(xp[-1].item()))
    idx_hi = torch.searchsorted(xp, x_clamped, right=True)
    idx_lo = torch.clamp(idx_hi - 1, 0, xp.numel() - 2)
    idx_hi = idx_lo + 1

    x0 = xp[idx_lo]
    x1 = xp[idx_hi]
    y0 = fp[idx_lo]
    y1 = fp[idx_hi]
    t = (x_clamped - x0) / torch.clamp(x1 - x0, min=1e-12)
    return y0 + t * (y1 - y0)


def _phase_ff_state(
    psi: float,
    w: float,
    ff_amp: float,
    down_scale: float,
    up_scale: float,
    ff_bias: float,
    ff_shift_rad: float,
) -> tuple[float, float, float, float]:
    """Return (psi_dot, psi_ddot, psi_dddot, u_phase) for phase feedforward modulation."""
    phi = psi + ff_shift_rad - math.pi
    u_raw = ff_amp * math.cos(phi)
    scale = down_scale if u_raw >= 0.0 else up_scale
    u_phase = u_raw * scale - ff_bias
    # Clamp to keep psi_dot positive (avoid direction reversal).
    u_phase = max(-0.95, min(0.95, u_phase))
    psi_dot = w * (1.0 + u_phase)

    du_raw = -ff_amp * math.sin(phi)
    du = scale * du_raw
    d2u_raw = -ff_amp * math.cos(phi)
    d2u = scale * d2u_raw

    psi_ddot = w * du * psi_dot
    psi_dddot = w * (d2u * psi_dot * psi_dot + du * psi_ddot)
    return psi_dot, psi_ddot, psi_dddot, u_phase


def _read_urdf_joint_limits(urdf_path: Path, joint_name: str) -> tuple[float | None, float | None]:
    import xml.etree.ElementTree as ET

    root = ET.parse(str(urdf_path)).getroot()
    j = root.find(f"./joint[@name='{joint_name}']")
    if j is None:
        raise ValueError(f"Joint '{joint_name}' not found in URDF: {urdf_path}")
    limit = j.find("limit")
    lower = upper = None
    if limit is not None:
        if "lower" in limit.attrib:
            lower = float(limit.attrib["lower"])
        if "upper" in limit.attrib:
            upper = float(limit.attrib["upper"])
    return lower, upper


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
    elif len(name) == 2 and name[0] == "m" and name[1] in ("x", "y", "z"):
        # Workaround for argparse: values starting with '-' are treated like options.
        # Allow "mx"/"my"/"mz" as synonyms for "-x"/"-y"/"-z".
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


def _rotate_about_x(v: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate vectors about +x by `angle` (radians)."""
    vx, vy, vz = v.unbind(-1)
    c = torch.cos(angle)
    s = torch.sin(angle)
    vy2 = c * vy - s * vz
    vz2 = s * vy + c * vz
    return torch.stack((vx, vy2, vz2), dim=-1)


def _accumulate_aoa_stats(
    omega_strip: torch.Tensor,
    v_forward_strip: torch.Tensor,
    wing_geom: WingGeometry,
    *,
    alpha1_deg: float,
    alpha2_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute area-weighted AOA stats from per-strip ω and v_forward.

    Returns:
        (w_sum, aoa_sum, pre_sum, mid_sum, post_sum, aoa_max) each of shape (2,).
    """
    if omega_strip.ndim != 3 or omega_strip.shape[0] != 2 or omega_strip.shape[-1] != 3:
        raise ValueError("omega_strip must have shape (2,N,3).")
    if v_forward_strip.shape != omega_strip.shape:
        raise ValueError("v_forward_strip must match omega_strip shape (2,N,3).")

    N = int(wing_geom.x_mid.numel())
    if omega_strip.shape[1] != N:
        raise ValueError("omega_strip strip count must match wing_geom.")

    x = wing_geom.x_mid.view(1, N)  # absolute x_c
    wy = omega_strip[..., 1]
    wz = omega_strip[..., 2]
    vxf = v_forward_strip[..., 0]
    vyf = v_forward_strip[..., 1]
    vzf = v_forward_strip[..., 2]

    v_y = x * wz + vyf
    v_z = -x * wy + vzf
    v2 = vxf * vxf + v_y * v_y + v_z * v_z
    denom = torch.sqrt(v2).clamp(min=1e-8)
    aoa = torch.acos(torch.clamp(torch.abs(v_z) / denom, -1.0, 1.0))  # rad, in [0,pi/2]
    aoa_deg = aoa * (180.0 / math.pi)

    weights = (wing_geom.c * wing_geom.dx).view(1, N)  # per-strip area weight
    w_sum = weights.sum(dim=-1).clamp(min=1e-12)  # (1,)
    w_sum = w_sum.expand(2)  # (2,)

    aoa_sum = torch.sum(aoa_deg * weights, dim=-1)  # (2,)

    a1 = float(alpha1_deg)
    a2 = float(alpha2_deg)
    pre = (aoa_deg <= a1).to(aoa_deg.dtype)
    post = (aoa_deg >= a2).to(aoa_deg.dtype)
    mid = ((aoa_deg > a1) & (aoa_deg < a2)).to(aoa_deg.dtype)
    pre_sum = torch.sum(pre * weights, dim=-1)
    mid_sum = torch.sum(mid * weights, dim=-1)
    post_sum = torch.sum(post * weights, dim=-1)
    aoa_max = torch.max(aoa_deg, dim=-1).values
    return w_sum, aoa_sum, pre_sum, mid_sum, post_sum, aoa_max


def main():
    # IMPORTANT: AppLauncher --device controls Kit's active GPU. SimulationCfg.device controls PhysX/torch device.
    # By default we keep them consistent, but users can override with --sim-device (e.g. GPU for rendering, CPU PhysX).
    sim_device = str(args_cli.sim_device) if args_cli.sim_device is not None else str(args_cli.device)
    sim_cfg = SimulationCfg(dt=args_cli.dt, device=sim_device)
    sim = SimulationContext(sim_cfg)
    # Only meaningful when a viewport exists.
    if not bool(getattr(args_cli, "headless", False)):
        sim.set_camera_view(eye=[1.0, -1.0, 0.8], target=[0.0, 0.0, 0.2])

    # Stage: ground + light
    # IMPORTANT: Avoid Isaac nucleus/cloud assets in sweeps/headless runs (network may be restricted).
    # GroundPlaneCfg defaults to a grid-world USD on Nucleus, so we use a simple local cuboid as "ground".
    ground_cfg = sim_utils.CuboidCfg(
        size=(8.0, 8.0, 0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.05))
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

    # Useful for interpreting "can it fly": compare mean lift against weight.
    try:
        total_mass_kg = float(robot.data.default_mass[0].sum().item())
        print(f"[INFO] Total mass ≈ {total_mass_kg:.6f} kg (mg ≈ {total_mass_kg * 9.81:.6f} N)")
    except Exception:
        pass

    # Optional debug draw (works when a viewport is active; may be unavailable in strict headless runs).
    draw_interface = None
    if args_cli.draw_forces or args_cli.draw_frames or args_cli.draw_wang_frames or args_cli.draw_world_frame:
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
    wing_geom_csv = args_cli.wing_geom_csv
    if wing_geom_csv is not None:
        xs, cs, _dhats = _load_wing_geom_csv(wing_geom_csv)
        R = _infer_span_from_x_mid(xs)
        print(f"[INFO] Loaded wing geometry from CSV: {wing_geom_csv} (inferred R≈{R:.4f} m)")
        print(f"[INFO] Chord check (root->tip): c_root≈{cs[0]:.4f} m, c_tip≈{cs[-1]:.4f} m")
        if cs[0] < cs[-1]:
            print("[WARN] Root chord is smaller than tip chord; check wing-geom orientation (root->tip should decrease).")
    elif args_cli.auto_geom_from_urdf:
        urdf_path = Path(robot_cfg.spawn.asset_path).resolve()
        aL, aR = _estimate_wing_area_from_urdf(urdf_path, mesh_scale=float(args_cli.mesh_scale))
        areas = [a for a in (aL, aR) if (a is not None and a > 0.0)]
        if areas:
            area_mean = float(sum(areas) / len(areas))
            AR = float(args_cli.aspect_ratio) if float(args_cli.aspect_ratio) > 0.0 else float(R / c)
            if AR > 0.0:
                # Use S = R * c_bar and AR = R / c_bar => R = sqrt(S*AR), c_bar = sqrt(S/AR).
                R = math.sqrt(area_mean * AR)
                c = math.sqrt(area_mean / AR)
                print(f"[INFO] Auto-geom from URDF: wing_area≈{area_mean:.6f} m^2 -> R≈{R:.4f} m, c≈{c:.4f} m (AR={AR:g})")
        else:
            print("[WARN] Auto-geom requested but failed to estimate wing area; falling back to --R/--c.")

    # Build chord distribution.
    chord_func = None
    if wing_geom_csv is not None:
        xp = torch.tensor(xs, device=sim.device, dtype=torch.float32)
        fp = torch.tensor(cs, device=sim.device, dtype=torch.float32)

        def chord_func(x_mid: torch.Tensor) -> torch.Tensor:
            return _interp1d_torch(xp, fp, x_mid)

        # Compute geometric area S and geometric AR from the CSV (per wing, x in [0,R]).
        # For uniform x_mid, S ≈ sum c(x_mid) * dx.
        dxs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        dxs_sorted = sorted(dxs)
        dx = float(dxs_sorted[len(dxs_sorted) // 2])
        S = float(sum(ci * dx for ci in cs))
        c_bar = S / R if R > 0 else float("nan")
        AR_geom = (R * R / S) if S > 0 else float("nan")
        if float(args_cli.aspect_ratio) <= 0.0:
            args_cli.aspect_ratio = float(AR_geom)
        print(f"[INFO] Planform (from CSV): S≈{S:.6f} m^2, c_bar≈{c_bar:.4f} m, AR_geom≈{AR_geom:.3f}")
    else:
        if float(args_cli.aspect_ratio) <= 0.0:
            args_cli.aspect_ratio = float(R / c)

    wing_geom = WingGeometry.from_input(
        {
            "R": R,
            "N": args_cli.N,
            "chord_func": chord_func if chord_func is not None else c,
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

    # Frame mapping (wing link -> Wang co-rotating frame), potentially different for left/right due to mirroring.
    wang_axes_L = tuple(args_cli.wang_axes_left) if args_cli.wang_axes_left is not None else tuple(args_cli.wang_axes)
    if args_cli.wang_axes_right is not None:
        wang_axes_R = tuple(args_cli.wang_axes_right)
    else:
        wang_axes_R = tuple(args_cli.wang_axes)
        # For mirrored left/right link frames, Wang x_c (spanwise) often needs a sign flip on the right wing.
        if args_cli.auto_mirror_wang_x and args_cli.wang_axes_left is None:
            x0 = wang_axes_R[0].strip()
            if x0.startswith("m") and len(x0) == 2:
                mirrored = x0[1:]  # mx -> x
            elif x0.startswith(("+", "-")):
                mirrored = x0[1:] if x0.startswith("-") else f"m{x0[1:]}"
            else:
                mirrored = f"m{x0}"
            wang_axes_R = (mirrored, wang_axes_R[1], wang_axes_R[2])
    A_l2w = torch.stack(
        (
            _link_to_wang_matrix(wang_axes_L, device=sim.device, dtype=torch.float32),
            _link_to_wang_matrix(wang_axes_R, device=sim.device, dtype=torch.float32),
        ),
        dim=0,
    )  # (2,3,3)
    A_w2l = A_l2w.transpose(-1, -2)  # (2,3,3)
    # NOTE on reflections (det=-1):
    # Left/right wing link frames are often mirrored. A mirrored axis mapping is an improper rotation (det=-1).
    # This is fine for forces (polar vectors), but torques/angular velocities are axial vectors and must pick up
    # an extra det(R) factor under reflections. We handle this by multiplying axial-vector transforms by det(A).
    detA = torch.sign(torch.det(A_l2w)).to(device=sim.device, dtype=torch.float32)  # (2,), each is ±1

    # Auto-select a per-wing twist sign convention to avoid left/right mirroring issues.
    # This is only used for twist-mode != off; users can override with --twist-sign-left/right.
    q_w_link0 = robot.data.body_quat_w[0, wing_body_ids, :]  # (2,4) at initial pose
    e_y_wang = torch.tensor([0.0, 1.0, 0.0], device=sim.device, dtype=torch.float32)
    y_c_link0 = torch.einsum("bij,j->bi", A_w2l, e_y_wang)  # (2,3)
    y_c_world0 = quat_apply(q_w_link0, y_c_link0)  # (2,3)
    # Desired: positive η_tip corresponds to TE moving "up" in world (+Z) at rest.
    # For Wang2016, +y_c is wing-surface normal; with z_c pointing to LE, a positive pitch about +x_c
    # lifts the trailing edge approximately along +y_c.
    corr = torch.sign(y_c_world0[:, 2])  # dot(+y_c, +Z_world)
    corr = torch.where(corr == 0.0, torch.ones_like(corr), corr)
    twist_sign_vec = float(args_cli.twist_sign) * corr  # (2,)
    if args_cli.twist_sign_left is not None:
        twist_sign_vec[0] = float(args_cli.twist_sign_left)
    if args_cli.twist_sign_right is not None:
        twist_sign_vec[1] = float(args_cli.twist_sign_right)
    if not bool(args_cli.auto_twist_sign):
        # Use provided values verbatim (either global or per-wing overrides).
        twist_sign_vec = torch.tensor(
            [
                float(args_cli.twist_sign_left) if args_cli.twist_sign_left is not None else float(args_cli.twist_sign),
                float(args_cli.twist_sign_right) if args_cli.twist_sign_right is not None else float(args_cli.twist_sign),
            ],
            device=sim.device,
            dtype=torch.float32,
        )
    print(
        "[INFO] twist sign (L,R)="
        f"({float(twist_sign_vec[0]):+.1f},{float(twist_sign_vec[1]):+.1f}) "
        f"(auto={bool(args_cli.auto_twist_sign)})"
    )

    # Wing joint profile (optionally phase-warped).
    mod_enabled = bool(args_cli.enable_modulation)
    mod_profile = str(args_cli.modulation_profile)
    mod_profile_code = 0 if mod_profile == "phase_warp" else 1
    mod_delta = float(args_cli.downstroke_ratio)
    mod_mode = str(args_cli.modulation_mode)
    mod_mode_code = 0 if mod_mode == "fixed_f" else 1
    mod_smooth = float(args_cli.modulation_smoothness)
    mod_f_ref = float(args_cli.modulation_f_ref_hz)
    mod_ff_amp = float(args_cli.modulation_ff_amp)
    mod_ff_shift_deg = float(args_cli.modulation_ff_shift_deg)
    mod_ff_shift_rad = math.radians(mod_ff_shift_deg)
    down_scale = mod_delta / 0.5
    up_scale = (1.0 - mod_delta) / 0.5
    mod_ff_bias = mod_ff_amp * (down_scale - up_scale) / math.pi
    phase_warp = None
    phase_ff_enabled = mod_enabled and (mod_profile == "phase_ff")
    if mod_enabled and mod_profile == "phase_warp":
        phase_warp = PhaseWarp(
            f_hz=float(args_cli.f_hz),
            delta=mod_delta,
            smoothness=mod_smooth,
            mode=mod_mode,
            f_ref_hz=mod_f_ref,
        )
        f_eff = phase_warp.f_eff
        f_down = phase_warp.f_down
        f_up = phase_warp.f_up
    elif phase_ff_enabled:
        mod_mode_code = 2
        f_eff = float(args_cli.f_hz)
        f_down = f_eff / (2.0 * mod_delta)
        f_up = f_eff / (2.0 * (1.0 - mod_delta))
    else:
        f_eff = float(args_cli.f_hz)
        f_down = f_eff
        f_up = f_eff
    w = 2.0 * math.pi * f_eff
    # Fixed ±30° as per your v50 rig, but clamp to URDF joint limits to avoid saturation/jitter.
    amp = math.radians(30.0)
    urdf_path = Path(robot_cfg.spawn.asset_path).resolve()
    lower_L, upper_L = _read_urdf_joint_limits(urdf_path, "left_wing")
    lower_R, upper_R = _read_urdf_joint_limits(urdf_path, "right_wing")
    lims = [v for v in (lower_L, upper_L, lower_R, upper_R) if v is not None]
    if lims:
        amp_limit = min(abs(v) for v in lims) - 1e-4
        if amp > amp_limit:
            print(f"[WARN] Clamping flap amplitude from {math.degrees(amp):.3f} deg to {math.degrees(amp_limit):.3f} deg due to URDF limits.")
            amp = max(0.0, float(amp_limit))
    if mod_enabled and mod_profile == "phase_warp" and phase_warp is not None:
        print(
            "[INFO] modulation enabled: "
            f"delta={mod_delta:.3f} mode={mod_mode} smooth={mod_smooth:.3f} "
            f"f_eff={f_eff:.3f} Hz (fd={f_down:.3f}, fu={f_up:.3f})"
        )
    elif phase_ff_enabled:
        print(
            "[INFO] modulation enabled: "
            f"profile=phase_ff delta={mod_delta:.3f} amp={mod_ff_amp:.3f} "
            f"shift={mod_ff_shift_deg:.1f} deg f_eff={f_eff:.3f} Hz"
        )

    # Wind-tunnel air flow in world frame (fixed, not body-aligned).
    # The paper's note under eq. (2.5) adds the wing's forward-flight velocity to v_c.
    # In a wind tunnel the wing is fixed while air moves, so we use: v_forward = -v_air.
    flow_dir = torch.tensor(args_cli.flow_dir_world, device=sim.device, dtype=torch.float32)
    flow_dir = flow_dir / torch.linalg.norm(flow_dir).clamp(min=1e-9)
    v_air_w = float(args_cli.airspeed) * flow_dir  # (3,)
    v_forward_w = -v_air_w  # (3,)

    # Virtual passive twist state (outer wing segment only).
    # Virtual twist parameters are an extension (not in Wang2016). Only used for twist-mode quasi_static/dynamic.
    twist_mode = str(args_cli.twist_mode)
    twist_cfg = VirtualTwistCfg(
        stiffness=float(args_cli.twist_k),
        damping=float(args_cli.twist_c),
        inertia=float(args_cli.twist_I),
        eta_limit=math.radians(float(args_cli.twist_eta_limit_deg)) if args_cli.twist_eta_limit_deg is not None else None,
        mode=twist_mode if twist_mode in ("quasi_static", "dynamic") else "dynamic",
        # NOTE: we apply per-wing sign in the script (twist_sign_vec); keep cfg.sign=+1 for clarity.
        sign=1.0,
    )
    twist_state = VirtualTwistState.zeros((2,), device=sim.device, dtype=torch.float32)
    eta_rest = math.radians(float(args_cli.twist_rest_deg))
    aero_model = str(args_cli.aero_model)
    if aero_model == "delaurier1993" and twist_mode in ("quasi_static", "dynamic"):
        raise ValueError("--aero-model delaurier1993 currently supports twist-mode off/prescribed only.")
    delaurier_params = DeLaurierParams(
        alpha0_rad=math.radians(float(args_cli.delaurier_alpha0_deg)),
        eta_s=float(args_cli.delaurier_eta_s),
        cd_cf=float(args_cli.delaurier_cd_cf),
        alpha_stall_min_rad=math.radians(float(args_cli.delaurier_alpha_stall_min_deg)),
        alpha_stall_max_rad=math.radians(float(args_cli.delaurier_alpha_stall_max_deg)),
        xi=float(args_cli.delaurier_xi),
        c_mac=float(args_cli.delaurier_cmac),
        nu=float(args_cli.delaurier_nu),
        cd_f=float(args_cli.delaurier_cd_f) if args_cli.delaurier_cd_f is not None else None,
    )
    theta_a_deg = float(args_cli.body_pitch_deg) if args_cli.delaurier_theta_a_deg is None else float(args_cli.delaurier_theta_a_deg)
    theta_a = math.radians(theta_a_deg)
    theta_w = math.radians(float(args_cli.delaurier_theta_w_deg))
    theta_bar = theta_a + theta_w

    # Split the wing into (inner rigid) + (outer twisting) segments.
    # Note: when x0==0 (full-span twist), the "inner" segment is empty and is skipped.
    eta_shape_outer = None  # (N_outer,) in [0,1] when twist enabled
    twist_x0 = 0.0
    twist_x1 = float(wing_geom.R)
    twist_shape_kind = str(args_cli.twist_eta_shape)
    if args_cli.twist_mode != "off":
        x0 = float(args_cli.twist_x0)
        x1 = float(wing_geom.R) if args_cli.twist_x1 is None else float(args_cli.twist_x1)
        if x1 <= x0:
            raise ValueError("--twist-x1 must be > --twist-x0.")
        if x0 < 0.0 or x1 < 0.0:
            raise ValueError("--twist-x0/--twist-x1 must be >= 0.")
        if x1 > float(wing_geom.R) + 1e-9:
            raise ValueError(
                f"Twist region [x0,x1]=[{x0:g},{x1:g}] exceeds wing span R={wing_geom.R:g}. "
                "Set a correct --R (or disable --auto-geom-from-urdf) before enabling --twist-mode."
            )
        wing_geom_inner = None if x0 <= 0.0 else wing_geom.subspan(0.0, x0)
        wing_geom_outer = wing_geom.subspan(x0, x1)
        twist_x0 = x0
        twist_x1 = x1
        twist_shape_kind = str(args_cli.twist_eta_shape)
        if twist_shape_kind == "uniform":
            eta_shape_outer = torch.ones_like(wing_geom_outer.x_mid)
        else:
            eta_shape_outer = eta_shape_piecewise(wing_geom_outer.x_mid, x0=x0, x1=x1, kind=twist_shape_kind)
    else:
        wing_geom_inner = wing_geom
        wing_geom_outer = None

    if args_cli.csv is None:
        run_dir = _resolve_run_dir(args_cli.run_root, args_cli.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        args_cli.csv = run_dir / "windtunnel_v50.csv"
        print(f"[INFO] Run directory: {run_dir}")
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
                "wing_q_R",
                "wing_qd_R",
                "wing_qdd_R",
                "airspeed",
                "v_air_wx",
                "v_air_wy",
                "v_air_wz",
                "body_pitch_deg",
                "mod_enabled",
                "mod_delta",
                "mod_mode",
                "mod_smooth",
                "mod_f_eff_hz",
                "mod_fd_hz",
                "mod_fu_hz",
                "omega_w_norm_L",
                "omega_w_norm_R",
                "omega_c_norm_L",
                "omega_c_norm_R",
                "joint_pos_L",
                "joint_pos_R",
                "joint_vel_L",
                "joint_vel_R",
                "eta_tip_L",
                "eta_tip_R",
                "tau_twist_x_L",
                "tau_twist_x_R",
                "tau_hinge_L",
                "tau_hinge_R",
                "power_in_L",
                "power_in_R",
                "power_in_T",
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
                "p_base_wx",
                "p_base_wy",
                "p_base_wz",
                "p_wing_L_wx",
                "p_wing_L_wy",
                "p_wing_L_wz",
                "p_wing_R_wx",
                "p_wing_R_wy",
                "p_wing_R_wz",
                "F_world_x_T",
                "F_world_y_T",
                "F_world_z_T",
                "tau_world_x_T",
                "tau_world_y_T",
                "tau_world_z_T",
                "tau_world_x_T_about_base",
                "tau_world_y_T_about_base",
                "tau_world_z_T_about_base",
                "aoa_mean_deg_L",
                "aoa_mean_deg_R",
                "aoa_max_deg_L",
                "aoa_max_deg_R",
                "aoa_frac_pre_L",
                "aoa_frac_pre_R",
                "aoa_frac_mid_L",
                "aoa_frac_mid_R",
                "aoa_frac_post_L",
                "aoa_frac_post_R",
                "del_sep_ratio_L",
                "del_sep_ratio_R",
                "del_Nc_L",
                "del_Nc_R",
                "del_Na_L",
                "del_Na_R",
                "del_Fx_suction_L",
                "del_Fx_suction_R",
                "del_Fx_camber_L",
                "del_Fx_camber_R",
                "del_Fx_friction_L",
                "del_Fx_friction_R",
                "del_Fx_total_L",
                "del_Fx_total_R",
                "del_k_mean_L",
                "del_k_mean_R",
                "del_alpha_prime_mean_L",
                "del_alpha_prime_mean_R",
                "del_alpha_le_mean_L",
                "del_alpha_le_mean_R",
                "del_alpha_tip_deg_L",
                "del_alpha_tip_deg_R",
                "del_alpha_prime_tip_deg_L",
                "del_alpha_prime_tip_deg_R",
                "del_alpha_le_tip_deg_L",
                "del_alpha_le_tip_deg_R",
                "del_theta_tip_deg_L",
                "del_theta_tip_deg_R",
                "mod_profile",
                "mod_ff_amp",
                "mod_ff_shift_deg",
            ]
        )

        psi_ff = 0.0
        for step in range(args_cli.steps):
            t = step * sim_dt

            # Prescribed wing motion (fixed amplitude, optional phase warp).
            if phase_warp is None and not phase_ff_enabled:
                # Use cosine for consistency with SCCPFM phase-warp baseline.
                q_cmd = amp * math.cos(w * t)
                qd_cmd = -amp * w * math.sin(w * t)
                qdd_cmd = -(w**2) * q_cmd
                qddd_cmd = -(w**2) * qd_cmd
                psi_dot = w
            elif phase_ff_enabled:
                psi_dot, psi_ddot, psi_dddot, _u_phase = _phase_ff_state(
                    psi_ff,
                    w,
                    mod_ff_amp,
                    down_scale,
                    up_scale,
                    mod_ff_bias,
                    mod_ff_shift_rad,
                )
                s = math.sin(psi_ff)
                c = math.cos(psi_ff)
                q_cmd = amp * s
                qd_cmd = amp * c * psi_dot
                qdd_cmd = amp * (-s * (psi_dot**2) + c * psi_ddot)
                qddd_cmd = amp * (-c * (psi_dot**3) - 3.0 * s * psi_dot * psi_ddot + c * psi_dddot)
                psi_ff = psi_ff + psi_dot * sim_dt + 0.5 * psi_ddot * sim_dt * sim_dt
            else:
                st = phase_warp.eval(t)
                # Use a cosine waveform for SCCPFM so the split occurs at stroke reversal (smooth position/velocity).
                s = math.sin(st.psi)
                c = math.cos(st.psi)
                q_cmd = amp * c
                qd_cmd = -amp * s * st.psi_dot
                qdd_cmd = -amp * (c * (st.psi_dot**2) + s * st.psi_ddot)
                qddd_cmd = amp * (
                    s * (st.psi_dot**3) - 3.0 * c * st.psi_dot * st.psi_ddot - s * st.psi_dddot
                )
                psi_dot = st.psi_dot

            # Drive the mechanism like a test stand: enforce the joint motion kinematically (default),
            # or use position targets (so aerodynamic torques can perturb wing motion).
            joint_pos = torch.tensor([[q_cmd, q_cmd, 0.0, 0.0]], device=sim.device, dtype=torch.float32)
            joint_vel = torch.tensor([[qd_cmd, qd_cmd, 0.0, 0.0]], device=sim.device, dtype=torch.float32)
            if args_cli.drive_mode == "kinematic":
                robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids)
            else:
                robot.set_joint_position_target(joint_pos, joint_ids=joint_ids)

            # Current wing orientation (from sim state at the start of this step).
            q_w_link = robot.data.body_quat_w[0, wing_body_ids, :]  # (2,4), wxyz

            # Angular velocity/acceleration of wings in world (sim-provided).
            omega_w = robot.data.body_ang_vel_w[0, wing_body_ids, :]  # (2,3)
            alpha_w = robot.data.body_ang_acc_w[0, wing_body_ids, :]  # (2,3)

            # Wing hinge axes in world (left/right use opposite signs).
            if base_body_ids:
                q_base_w = robot.data.body_quat_w[0, base_body_ids[0], :].view(1, 4).expand(2, 4)
            else:
                q_base_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device, dtype=torch.float32).expand(2, 4)
            axis_b = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=sim.device, dtype=torch.float32)  # (2,3)
            axis_w = quat_apply(q_base_w, axis_b)  # (2,3)

            if args_cli.qsm_from_command:
                # Use commanded kinematics, but keep the frame mapping consistent with --wang-axes.
                # The wing DOF rotates about the base_link x-axis, with opposite signs for left vs right (URDF axes).
                omega_w_cmd = axis_w * float(qd_cmd)
                alpha_w_cmd = axis_w * float(qdd_cmd)
                omega_l = quat_apply_inverse(q_w_link, omega_w_cmd)
                alpha_l = quat_apply_inverse(q_w_link, alpha_w_cmd)
                omega_c = detA.unsqueeze(-1) * torch.einsum("bij,bj->bi", A_l2w, omega_l)  # axial
                alpha_c = detA.unsqueeze(-1) * torch.einsum("bij,bj->bi", A_l2w, alpha_l)  # axial
            else:
                # Express ω, α in wing link frame, then map to Wang co-rotating frame.
                omega_l = quat_apply_inverse(q_w_link, omega_w)
                alpha_l = quat_apply_inverse(q_w_link, alpha_w)
                omega_c = detA.unsqueeze(-1) * torch.einsum("bij,bj->bi", A_l2w, omega_l)  # axial
                alpha_c = detA.unsqueeze(-1) * torch.einsum("bij,bj->bi", A_l2w, alpha_l)  # axial

            # Forward-flight velocity expressed in Wang co-rotating frame.
            v_forward_l = quat_apply_inverse(q_w_link, v_forward_w.view(1, 3).expand(2, 3))
            v_forward_c = torch.einsum("bij,bj->bi", A_l2w, v_forward_l)  # (2,3)

            # Body positions (world) for net wrench about base_link.
            p_wing_w = robot.data.body_pos_w[0, wing_body_ids, :]  # (2,3)
            if base_body_ids:
                p_base_w = robot.data.body_pos_w[0, base_body_ids[0], :]  # (3,)
            else:
                p_base_w = torch.zeros((3,), device=sim.device, dtype=torch.float32)

            # Virtual twist update (driven by the outer segment aerodynamic pitching moment about +x).
            eta_tip = twist_state.eta
            etad_tip = twist_state.etad
            etadd_tip = torch.zeros_like(etad_tip)
            tau_twist_x = torch.zeros((2,), device=sim.device, dtype=torch.float32)

            if args_cli.twist_mode != "off":
                assert wing_geom_outer is not None

                if args_cli.twist_mode == "prescribed":
                    # Fast open-loop twist for comparison: avoids extra QSM calls / iterations.
                    amp_eta = math.radians(float(args_cli.twist_eta_amp_deg))
                    phase = math.radians(float(args_cli.twist_eta_phase_deg))
                    sgn_L = float(twist_sign_vec[0])
                    sgn_R = float(twist_sign_vec[1])
                    twist_mode_name = str(args_cli.twist_prescribed)
                    if twist_mode_name == "sign_qd":
                        # Piecewise-constant twist sign from sweep rate sign (no η̇/η̈).
                        s = 1.0 if qd_cmd >= 0.0 else -1.0
                        eta_L = float(eta_rest) + sgn_L * amp_eta * s
                        eta_R = float(eta_rest) + sgn_R * amp_eta * s
                        eta_tip = torch.tensor([eta_L, eta_R], device=sim.device, dtype=torch.float32)
                        etad_tip = torch.zeros_like(eta_tip)
                        etadd_tip = torch.zeros_like(eta_tip)
                    elif twist_mode_name == "qd_scaled":
                        # η_tip driven by instantaneous flapping angular speed q̇_cmd (rad/s).
                        # Calibrate with a reference frequency f_ref such that |η_tip|=η_max at |q̇|=q̇_ref.
                        f_ref = float(args_cli.twist_f_ref_hz)
                        if f_ref <= 0.0:
                            raise ValueError("--twist-f-ref-hz must be positive for --twist-prescribed qd_scaled.")
                        eta_max = math.radians(float(args_cli.twist_eta_max_deg))
                        qd_ref = float(amp) * (2.0 * math.pi * f_ref)  # max |q̇| at f_ref for q=amp*sin(wt)
                        if qd_ref <= 0.0:
                            raise RuntimeError("Invalid qd_ref (check flap amplitude and --twist-f-ref-hz).")
                        # Scale factor; clamp so that at f>=f_ref the tip twist saturates at η_max.
                        s = float(qd_cmd) / qd_ref
                        s = max(-1.0, min(1.0, s))
                        eta_tip = torch.tensor(
                            [float(eta_rest) + sgn_L * eta_max * s, float(eta_rest) + sgn_R * eta_max * s],
                            device=sim.device,
                            dtype=torch.float32,
                        )
                        # If η=k*q̇ then η̇=k*q̈ and η̈=k*q⃛. For q=amp*sin(wt): q⃛ = -w^2*q̇.
                        k = eta_max / qd_ref
                        etad = k * float(qdd_cmd)
                        etadd = k * float(qddd_cmd)
                        etad_tip = torch.tensor([sgn_L * etad, sgn_R * etad], device=sim.device, dtype=torch.float32)
                        etadd_tip = torch.tensor([sgn_L * etadd, sgn_R * etadd], device=sim.device, dtype=torch.float32)
                    else:
                        # Sinusoidal twist at flap frequency with phase shift.
                        ang = w * t + phase
                        s = math.sin(ang)
                        c = math.cos(ang)
                        eta_tip = torch.tensor(
                            [float(eta_rest) + sgn_L * amp_eta * s, float(eta_rest) + sgn_R * amp_eta * s],
                            device=sim.device,
                            dtype=torch.float32,
                        )
                        etad_tip = torch.tensor(
                            [sgn_L * amp_eta * w * c, sgn_R * amp_eta * w * c],
                            device=sim.device,
                            dtype=torch.float32,
                        )
                        etadd_tip = torch.tensor(
                            [-sgn_L * amp_eta * (w**2) * s, -sgn_R * amp_eta * (w**2) * s],
                            device=sim.device,
                            dtype=torch.float32,
                        )

                    if twist_cfg.eta_limit is not None:
                        lim = float(twist_cfg.eta_limit)
                        eta_tip = torch.clamp(eta_tip, -lim, lim)
                        # If clamped, zero rates for robustness.
                        etad_tip = torch.where((eta_tip.abs() >= lim), torch.zeros_like(etad_tip), etad_tip)
                        etadd_tip = torch.where((eta_tip.abs() >= lim), torch.zeros_like(etadd_tip), etadd_tip)
                    tau_twist_x = torch.zeros((2,), device=sim.device, dtype=torch.float32)
                else:
                    def tau_of_eta_phys(eta: torch.Tensor) -> torch.Tensor:
                        if wing_geom_outer is None or eta_shape_outer is None:
                            raise RuntimeError("twist enabled but outer geometry/shape is missing.")
                        N_outer = int(wing_geom_outer.x_mid.numel())
                        eta_strip = eta.unsqueeze(-1) * eta_shape_outer.view(1, N_outer)  # (2,N)
                        if omega_c.ndim >= 3 and omega_c.shape[-2] == N_outer:
                            omega_base = omega_c
                        else:
                            omega_base = omega_c.unsqueeze(-2).expand(2, N_outer, 3)
                        if alpha_c.ndim >= 3 and alpha_c.shape[-2] == N_outer:
                            alpha_base = alpha_c
                        else:
                            alpha_base = alpha_c.unsqueeze(-2).expand(2, N_outer, 3)
                        if v_forward_c.ndim >= 3 and v_forward_c.shape[-2] == N_outer:
                            v_forward_base = v_forward_c
                        else:
                            v_forward_base = v_forward_c.unsqueeze(-2).expand(2, N_outer, 3)

                        omega_outer = _rotate_about_x(omega_base, -eta_strip)
                        alpha_outer = _rotate_about_x(alpha_base, -eta_strip)
                        v_forward_outer = _rotate_about_x(v_forward_base, -eta_strip)

                        _, tau_strip = compute_aero_wrench_from_omega_alpha(
                            omega_outer,
                            alpha_outer,
                            wing_geom_outer,
                            rho=args_cli.rho,
                            include_wagner=bool(args_cli.include_wagner),
                            v_forward_c=v_forward_outer,
                            return_per_strip=True,
                            profile_cd0=float(args_cli.profile_cd0) if float(args_cli.profile_cd0) > 0.0 else None,
                            profile_alpha1_deg=float(args_cli.profile_alpha1_deg),
                            profile_alpha2_deg=float(args_cli.profile_alpha2_deg),
                        )  # (2,N,3) in per-strip frames
                        # x-axis is common across strips (twist is about +x), so τ_x sums directly.
                        return tau_strip[..., 0].sum(dim=-1)

                    if args_cli.twist_mode == "quasi_static":
                        # Fixed-point solve for: η = η0 + (sign_i * τ_x(η)) / k   (per wing).
                        k = float(twist_cfg.stiffness)
                        if k <= 0:
                            raise ValueError("--twist-k must be positive in quasi_static mode.")
                        relax = float(args_cli.twist_relax)
                        max_iters = int(args_cli.twist_solve_iters)
                        eta = twist_state.eta
                        for _ in range(max(1, max_iters)):
                            tau_phys = tau_of_eta_phys(eta)
                            eta_target = float(eta_rest) + (twist_sign_vec * tau_phys) / k
                            eta = (1.0 - relax) * eta + relax * eta_target
                            if twist_cfg.eta_limit is not None:
                                lim = float(twist_cfg.eta_limit)
                                eta = torch.clamp(eta, -lim, lim)
                        twist_state.eta.copy_(eta)
                        twist_state.etad.zero_()
                        eta_tip = twist_state.eta
                        etad_tip = twist_state.etad
                        etadd_tip = torch.zeros_like(etad_tip)
                        tau_twist_x = tau_of_eta_phys(eta_tip)
                    else:
                        tau_twist_x = tau_of_eta_phys(eta_tip)
                        # Apply per-wing sign convention so that +η matches the intended physical direction.
                        tau_drive = twist_sign_vec * tau_twist_x
                        eta_tip, etad_tip, etadd_tip = step_virtual_twist(
                            twist_state,
                            tau_x=tau_drive,
                            dt=float(sim_dt),
                            cfg=twist_cfg,
                            eta_rest=eta_rest,
                        )

            # Compute aerodynamic wrench in Wang co-rotating frame, with optional outer-segment twist.
            del_sep_ratio = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
            power_in = None
            omega_outer = None
            v_forward_outer = None
            del_terms = None
            if aero_model == "delaurier1993":
                N_full = int(wing_geom.x_mid.numel())
                y_full = wing_geom.x_mid.view(1, N_full).expand(2, N_full)
                h_full = -float(q_cmd) * y_full
                hdot_full = -float(qd_cmd) * y_full
                hddot_full = -float(qdd_cmd) * y_full
                if args_cli.twist_mode == "off":
                    eta_shape_full = torch.zeros_like(wing_geom.x_mid)
                else:
                    if twist_shape_kind == "uniform":
                        eta_shape_full = torch.where(
                            (wing_geom.x_mid >= twist_x0) & (wing_geom.x_mid <= twist_x1),
                            torch.ones_like(wing_geom.x_mid),
                            torch.zeros_like(wing_geom.x_mid),
                        )
                    else:
                        eta_shape_full = eta_shape_piecewise(
                            wing_geom.x_mid, x0=twist_x0, x1=twist_x1, kind=twist_shape_kind
                        )
                        eta_shape_full = torch.where(
                            (wing_geom.x_mid < twist_x0) | (wing_geom.x_mid > twist_x1),
                            torch.zeros_like(eta_shape_full),
                            eta_shape_full,
                        )
                eta_strip_full = eta_tip.unsqueeze(-1) * eta_shape_full.view(1, N_full)
                etad_strip_full = etad_tip.unsqueeze(-1) * eta_shape_full.view(1, N_full)
                etadd_strip_full = etadd_tip.unsqueeze(-1) * eta_shape_full.view(1, N_full)

                theta_full = theta_bar + eta_strip_full
                thetad_full = etad_strip_full
                thetadd_full = etadd_strip_full
                F_c, tau_c, power_in, del_sep_ratio, del_terms = compute_aero_wrench_delaurier1993(
                    h_full,
                    hdot_full,
                    hddot_full,
                    theta_full,
                    thetad_full,
                    thetadd_full,
                    wing_geom,
                    rho=float(args_cli.rho),
                    U=float(args_cli.airspeed),
                    theta_a=theta_a,
                    theta_bar=theta_bar,
                    omega_ref=float(psi_dot),
                    params=delaurier_params,
                    enable_separation=bool(args_cli.delaurier_enable_separation),
                    return_terms=True,
                )

                if wing_geom_outer is not None and eta_shape_outer is not None:
                    N_outer = int(wing_geom_outer.x_mid.numel())
                    eta_strip = eta_tip.unsqueeze(-1) * eta_shape_outer.view(1, N_outer)  # (2,N)
                    etad_strip = etad_tip.unsqueeze(-1) * eta_shape_outer.view(1, N_outer)  # (2,N)
                    etadd_strip = etadd_tip.unsqueeze(-1) * eta_shape_outer.view(1, N_outer)  # (2,N)
                    if omega_c.ndim >= 3 and omega_c.shape[-2] == N_outer:
                        omega_base = omega_c
                    else:
                        omega_base = omega_c.unsqueeze(-2).expand(2, N_outer, 3)
                    if v_forward_c.ndim >= 3 and v_forward_c.shape[-2] == N_outer:
                        v_forward_base = v_forward_c
                    else:
                        v_forward_base = v_forward_c.unsqueeze(-2).expand(2, N_outer, 3)
                    omega_outer = _rotate_about_x(omega_base, -eta_strip)
                    omega_outer = omega_outer + torch.stack(
                        (etad_strip, torch.zeros_like(etad_strip), torch.zeros_like(etad_strip)), dim=-1
                    )
                    v_forward_outer = _rotate_about_x(v_forward_base, -eta_strip)
            else:
                if wing_geom_inner is None:
                    F_inner = torch.zeros((2, 3), device=sim.device, dtype=torch.float32)
                    tau_inner = torch.zeros((2, 3), device=sim.device, dtype=torch.float32)
                else:
                    F_inner, tau_inner = compute_aero_wrench_from_omega_alpha(
                        omega_c,
                        alpha_c,
                        wing_geom_inner,
                        rho=args_cli.rho,
                        include_wagner=bool(args_cli.include_wagner),
                        v_forward_c=v_forward_c,
                        profile_cd0=float(args_cli.profile_cd0) if float(args_cli.profile_cd0) > 0.0 else None,
                        profile_alpha1_deg=float(args_cli.profile_alpha1_deg),
                        profile_alpha2_deg=float(args_cli.profile_alpha2_deg),
                    )  # (2,3)

                if args_cli.twist_mode == "off":
                    F_c, tau_c = F_inner, tau_inner
                else:
                    if wing_geom_outer is None or eta_shape_outer is None:
                        raise RuntimeError("twist enabled but outer geometry/shape is missing.")
                    N_outer = int(wing_geom_outer.x_mid.numel())

                    # Distributed twist: η(x,t)=g(x)*η_tip(t) over the twist region.
                    eta_strip = eta_tip.unsqueeze(-1) * eta_shape_outer.view(1, N_outer)  # (2,N)
                    etad_strip = etad_tip.unsqueeze(-1) * eta_shape_outer.view(1, N_outer)  # (2,N)
                    etadd_strip = etadd_tip.unsqueeze(-1) * eta_shape_outer.view(1, N_outer)  # (2,N)

                    if omega_c.ndim >= 3 and omega_c.shape[-2] == N_outer:
                        omega_base = omega_c
                    else:
                        omega_base = omega_c.unsqueeze(-2).expand(2, N_outer, 3)
                    if alpha_c.ndim >= 3 and alpha_c.shape[-2] == N_outer:
                        alpha_base = alpha_c
                    else:
                        alpha_base = alpha_c.unsqueeze(-2).expand(2, N_outer, 3)
                    if v_forward_c.ndim >= 3 and v_forward_c.shape[-2] == N_outer:
                        v_forward_base = v_forward_c
                    else:
                        v_forward_base = v_forward_c.unsqueeze(-2).expand(2, N_outer, 3)

                    # Express base wing ω/α in each strip's twisted frame, then add local twist rates about +x.
                    omega_outer = _rotate_about_x(omega_base, -eta_strip)
                    alpha_outer = _rotate_about_x(alpha_base, -eta_strip)
                    omega_outer = omega_outer + torch.stack(
                        (etad_strip, torch.zeros_like(etad_strip), torch.zeros_like(etad_strip)), dim=-1
                    )
                    alpha_outer = alpha_outer + torch.stack(
                        (etadd_strip, torch.zeros_like(etad_strip), torch.zeros_like(etad_strip)), dim=-1
                    )
                    v_forward_outer = _rotate_about_x(v_forward_base, -eta_strip)

                    # Compute per-strip loads in each strip frame, rotate back to the base Wang frame, then sum.
                    F_outer_strip, tau_outer_strip = compute_aero_wrench_from_omega_alpha(
                        omega_outer,
                        alpha_outer,
                        wing_geom_outer,
                        rho=args_cli.rho,
                        include_wagner=bool(args_cli.include_wagner),
                        v_forward_c=v_forward_outer,
                        return_per_strip=True,
                        profile_cd0=float(args_cli.profile_cd0) if float(args_cli.profile_cd0) > 0.0 else None,
                        profile_alpha1_deg=float(args_cli.profile_alpha1_deg),
                        profile_alpha2_deg=float(args_cli.profile_alpha2_deg),
                    )  # (2,N,3) in per-strip frames
                    F_outer_strip = _rotate_about_x(F_outer_strip, eta_strip)
                    tau_outer_strip = _rotate_about_x(tau_outer_strip, eta_strip)
                    F_outer = F_outer_strip.sum(dim=-2)  # (2,3)
                    tau_outer = tau_outer_strip.sum(dim=-2)  # (2,3)

                    F_c = F_inner + F_outer
                    tau_c = tau_inner + tau_outer

            # AOA stats (area-weighted) using the same kinematics used by QSM:
            # AOA is computed from v_c components in each segment's local co-rotating frame.
            alpha1 = float(args_cli.aoa_alpha1_deg)
            alpha2 = float(args_cli.aoa_alpha2_deg)
            if alpha2 <= alpha1:
                raise ValueError("--aoa-alpha2-deg must be > --aoa-alpha1-deg.")

            w_sum = torch.zeros((2,), device=sim.device, dtype=torch.float32)
            aoa_sum = torch.zeros((2,), device=sim.device, dtype=torch.float32)
            pre_sum = torch.zeros((2,), device=sim.device, dtype=torch.float32)
            mid_sum = torch.zeros((2,), device=sim.device, dtype=torch.float32)
            post_sum = torch.zeros((2,), device=sim.device, dtype=torch.float32)
            aoa_max = torch.zeros((2,), device=sim.device, dtype=torch.float32)

            if wing_geom_inner is not None:
                N_in = int(wing_geom_inner.x_mid.numel())
                omega_in = omega_c.unsqueeze(-2).expand(2, N_in, 3)
                v_in = v_forward_c.unsqueeze(-2).expand(2, N_in, 3)
                w_s, a_s, p_s, m_s, po_s, a_m = _accumulate_aoa_stats(
                    omega_in, v_in, wing_geom_inner, alpha1_deg=alpha1, alpha2_deg=alpha2
                )
                w_sum = w_sum + w_s
                aoa_sum = aoa_sum + a_s
                pre_sum = pre_sum + p_s
                mid_sum = mid_sum + m_s
                post_sum = post_sum + po_s
                aoa_max = torch.maximum(aoa_max, a_m)

            if wing_geom_outer is not None and eta_shape_outer is not None:
                # Use the same per-strip omega/v_forward used by the QSM call above.
                # These are in the per-strip twisted frames (correct for distributed twist).
                w_s, a_s, p_s, m_s, po_s, a_m = _accumulate_aoa_stats(
                    omega_outer, v_forward_outer, wing_geom_outer, alpha1_deg=alpha1, alpha2_deg=alpha2
                )
                w_sum = w_sum + w_s
                aoa_sum = aoa_sum + a_s
                pre_sum = pre_sum + p_s
                mid_sum = mid_sum + m_s
                post_sum = post_sum + po_s
                aoa_max = torch.maximum(aoa_max, a_m)

            aoa_mean = aoa_sum / torch.clamp(w_sum, min=1e-12)
            frac_pre = pre_sum / torch.clamp(w_sum, min=1e-12)
            frac_mid = mid_sum / torch.clamp(w_sum, min=1e-12)
            frac_post = post_sum / torch.clamp(w_sum, min=1e-12)

            # Map back to link frame for application.
            F_l = torch.einsum("bij,bj->bi", A_w2l, F_c)
            tau_l = detA.unsqueeze(-1) * torch.einsum("bij,bj->bi", A_w2l, tau_c)  # axial under reflections
            F_w = quat_apply(q_w_link, F_l)
            tau_w = quat_apply(q_w_link, tau_l)

            # Hinge-axis torque and input power (positive = motor work against aero).
            if power_in is None:
                tau_hinge = torch.sum(tau_w * axis_w, dim=1)  # (2,)
                power_in = -tau_hinge * float(qd_cmd)  # (2,)
            else:
                tau_hinge = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
            power_in_T = power_in.sum()

            # Total wrench in world.
            F_w_T = F_w.sum(dim=0)  # (3,)
            tau_w_T = tau_w.sum(dim=0)  # (3,) sum of pure torques
            r_w = p_wing_w - p_base_w.view(1, 3)
            tau_w_T_about_base = tau_w_T + torch.linalg.cross(r_w[0], F_w[0]) + torch.linalg.cross(r_w[1], F_w[1])

            # Apply to wing links in link frame (local).
            robot.set_external_force_and_torque(
                forces=F_l.view(1, 2, 3), torques=tau_l.view(1, 2, 3), body_ids=wing_body_ids, is_global=False
            )

            # Push buffers and step.
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)

            if draw_interface is not None:
                try:
                    draw_interface.clear_lines()
                    p = robot.data.body_pos_w[0, wing_body_ids, :]  # (2,3)
                    if args_cli.draw_forces:
                        pF = p + float(args_cli.force_scale) * F_w
                        pT = p + float(args_cli.torque_scale) * tau_w
                        colors_F = [[0.1, 0.8, 1.0, 1.0]] * 2
                        colors_T = [[1.0, 0.6, 0.1, 1.0]] * 2
                        thickness = [3.0] * 2
                        draw_interface.draw_lines(p.tolist(), pF.tolist(), colors_F, thickness)
                        draw_interface.draw_lines(p.tolist(), pT.tolist(), colors_T, thickness)
                    if args_cli.draw_frames or args_cli.draw_wang_frames:
                        axis_len = float(args_cli.frame_axis_length)
                        axis_th = float(args_cli.frame_axis_thickness)
                        link_colors = [
                            [1.0, 0.2, 0.2, 1.0],
                            [0.2, 1.0, 0.2, 1.0],
                            [0.2, 0.2, 1.0, 1.0],
                        ]  # x,y,z (link)
                        wang_colors = [
                            [0.2, 1.0, 1.0, 1.0],
                            [1.0, 0.2, 1.0, 1.0],
                            [1.0, 1.0, 0.2, 1.0],
                        ]  # x_c,y_c,z_c
                        starts = []
                        ends = []
                        colors = []
                        thicknesses = []
                        eye = torch.eye(3, device=sim.device, dtype=torch.float32)
                        for i in range(2):
                            p0 = p[i]
                            q = q_w_link[i].view(1, 4)
                            if args_cli.draw_frames:
                                for axis_idx in range(3):
                                    v_link = eye[axis_idx].view(1, 3)
                                    v_w = quat_apply(q, v_link)[0]
                                    starts.append(p0.tolist())
                                    ends.append((p0 + axis_len * v_w).tolist())
                                    colors.append(link_colors[axis_idx])
                                    thicknesses.append(axis_th)
                            if args_cli.draw_wang_frames:
                                for axis_idx in range(3):
                                    v_link = A_l2w[i, axis_idx].view(1, 3)
                                    v_w = quat_apply(q, v_link)[0]
                                    starts.append(p0.tolist())
                                    ends.append((p0 + axis_len * v_w).tolist())
                                    colors.append(wang_colors[axis_idx])
                                    thicknesses.append(axis_th)
                        draw_interface.draw_lines(starts, ends, colors, thicknesses)
                    if args_cli.draw_world_frame:
                        axis_len = (
                            float(args_cli.world_axis_length)
                            if args_cli.world_axis_length is not None
                            else float(args_cli.frame_axis_length)
                        )
                        axis_th = (
                            float(args_cli.world_axis_thickness)
                            if args_cli.world_axis_thickness is not None
                            else float(args_cli.frame_axis_thickness)
                        )
                        if args_cli.world_frame_origin == "base":
                            p0 = p_base_w
                        else:
                            p0 = torch.zeros((3,), device=sim.device, dtype=torch.float32)
                        world_colors = [
                            [0.9, 0.9, 0.9, 1.0],
                            [0.6, 0.6, 0.6, 1.0],
                            [0.3, 0.3, 0.3, 1.0],
                        ]  # x,y,z (world)
                        starts = []
                        ends = []
                        colors = []
                        thicknesses = []
                        eye = torch.eye(3, device=sim.device, dtype=torch.float32)
                        for axis_idx in range(3):
                            v_w = eye[axis_idx]
                            starts.append(p0.tolist())
                            ends.append((p0 + axis_len * v_w).tolist())
                            colors.append(world_colors[axis_idx])
                            thicknesses.append(axis_th)
                        draw_interface.draw_lines(starts, ends, colors, thicknesses)
                except Exception:
                    # Keep simulation running even if drawing fails intermittently.
                    pass

            if args_cli.print_every > 0 and (step % int(args_cli.print_every) == 0):
                fy_tag = "del" if aero_model == "delaurier1993" else "wang"
                print(
                    f"[step {step:05d}] model={aero_model} t={t:7.3f} qL={q_cmd:+.3f} rad qdL={qd_cmd:+.3f} rad/s | "
                    f"omega_c_norm(L,R)=({float(torch.linalg.norm(omega_c[0]).item()):.3f},{float(torch.linalg.norm(omega_c[1]).item()):.3f}) | "
                    f"eta_tip(L,R)=({float(eta_tip[0]):+.3f},{float(eta_tip[1]):+.3f}) rad | "
                    f"Fy_{fy_tag}(L,R)=({float(F_c[0,1]):+.3f},{float(F_c[1,1]):+.3f}) N | "
                    f"F_world_T=({float(F_w_T[0]):+.3f},{float(F_w_T[1]):+.3f},{float(F_w_T[2]):+.3f}) N"
                )

            if del_terms is None:
                del_Nc = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_Na = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_Fx_suction = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_Fx_camber = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_Fx_friction = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_Fx_total = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_k_mean = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_alpha_prime_mean = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_alpha_le_mean = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_alpha_tip_deg = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_alpha_prime_tip_deg = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_alpha_le_tip_deg = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
                del_theta_tip_deg = torch.full((2,), float("nan"), device=sim.device, dtype=torch.float32)
            else:
                del_Nc = del_terms["N_c"]
                del_Na = del_terms["N_a"]
                del_Fx_suction = del_terms["Fx_suction"]
                del_Fx_camber = del_terms["Fx_camber"]
                del_Fx_friction = del_terms["Fx_friction"]
                del_Fx_total = del_terms["Fx_total"]
                del_k_mean = del_terms["k_mean"]
                del_alpha_prime_mean = del_terms["alpha_prime_mean"]
                del_alpha_le_mean = del_terms["alpha_le_mean"]
                rad2deg = 180.0 / math.pi
                del_alpha_tip_deg = del_terms["alpha_tip"] * rad2deg
                del_alpha_prime_tip_deg = del_terms["alpha_prime_tip"] * rad2deg
                del_alpha_le_tip_deg = del_terms["alpha_le_tip"] * rad2deg
                del_theta_tip_deg = theta_full[:, -1] * rad2deg

            writer.writerow(
                [
                    step,
                    float(t),
                    float(q_cmd),
                    float(qd_cmd),
                    float(qdd_cmd),
                    float(q_cmd),
                    float(qd_cmd),
                    float(qdd_cmd),
                    float(args_cli.airspeed),
                    float(v_air_w[0].item()),
                    float(v_air_w[1].item()),
                    float(v_air_w[2].item()),
                    float(args_cli.body_pitch_deg),
                    1 if mod_enabled else 0,
                    float(mod_delta),
                    float(mod_mode_code),
                    float(mod_smooth),
                    float(f_eff),
                    float(f_down),
                    float(f_up),
                    float(torch.linalg.norm(omega_w[0]).item()),
                    float(torch.linalg.norm(omega_w[1]).item()),
                    float(torch.linalg.norm(omega_c[0]).item()),
                    float(torch.linalg.norm(omega_c[1]).item()),
                    float(robot.data.joint_pos[0, joint_ids[0]].item()),
                    float(robot.data.joint_pos[0, joint_ids[1]].item()),
                    float(robot.data.joint_vel[0, joint_ids[0]].item()),
                    float(robot.data.joint_vel[0, joint_ids[1]].item()),
                    float(eta_tip[0].item()),
                    float(eta_tip[1].item()),
                    float(tau_twist_x[0].item()),
                    float(tau_twist_x[1].item()),
                    float(tau_hinge[0].item()),
                    float(tau_hinge[1].item()),
                    float(power_in[0].item()),
                    float(power_in[1].item()),
                    float(power_in_T.item()),
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
                    float(p_base_w[0].item()),
                    float(p_base_w[1].item()),
                    float(p_base_w[2].item()),
                    float(p_wing_w[0, 0].item()),
                    float(p_wing_w[0, 1].item()),
                    float(p_wing_w[0, 2].item()),
                    float(p_wing_w[1, 0].item()),
                    float(p_wing_w[1, 1].item()),
                    float(p_wing_w[1, 2].item()),
                    float(F_w_T[0].item()),
                    float(F_w_T[1].item()),
                    float(F_w_T[2].item()),
                    float(tau_w_T[0].item()),
                    float(tau_w_T[1].item()),
                    float(tau_w_T[2].item()),
                    float(tau_w_T_about_base[0].item()),
                    float(tau_w_T_about_base[1].item()),
                    float(tau_w_T_about_base[2].item()),
                    float(aoa_mean[0].item()),
                    float(aoa_mean[1].item()),
                    float(aoa_max[0].item()),
                    float(aoa_max[1].item()),
                    float(frac_pre[0].item()),
                    float(frac_pre[1].item()),
                    float(frac_mid[0].item()),
                    float(frac_mid[1].item()),
                    float(frac_post[0].item()),
                    float(frac_post[1].item()),
                    float(del_sep_ratio[0].item()),
                    float(del_sep_ratio[1].item()),
                    float(del_Nc[0].item()),
                    float(del_Nc[1].item()),
                    float(del_Na[0].item()),
                    float(del_Na[1].item()),
                    float(del_Fx_suction[0].item()),
                    float(del_Fx_suction[1].item()),
                    float(del_Fx_camber[0].item()),
                    float(del_Fx_camber[1].item()),
                    float(del_Fx_friction[0].item()),
                    float(del_Fx_friction[1].item()),
                    float(del_Fx_total[0].item()),
                    float(del_Fx_total[1].item()),
                    float(del_k_mean[0].item()),
                    float(del_k_mean[1].item()),
                    float(del_alpha_prime_mean[0].item()),
                    float(del_alpha_prime_mean[1].item()),
                    float(del_alpha_le_mean[0].item()),
                    float(del_alpha_le_mean[1].item()),
                    float(del_alpha_tip_deg[0].item()),
                    float(del_alpha_tip_deg[1].item()),
                    float(del_alpha_prime_tip_deg[0].item()),
                    float(del_alpha_prime_tip_deg[1].item()),
                    float(del_alpha_le_tip_deg[0].item()),
                    float(del_alpha_le_tip_deg[1].item()),
                    float(del_theta_tip_deg[0].item()),
                    float(del_theta_tip_deg[1].item()),
                    float(mod_profile_code),
                    float(mod_ff_amp),
                    float(mod_ff_shift_deg),
                ]
            )

    print(f"[OK] CSV written to: {args_cli.csv}")
    if base_body_ids:
        print(f"[INFO] base_link body id: {base_body_ids[0]} (fixed-base rig)")


if __name__ == "__main__":
    success = False
    try:
        main()
        success = True
    finally:
        # Ensure SimulationContext callbacks are released before closing Kit.
        try:
            SimulationContext.clear_instance()
        except Exception:
            pass
        simulation_app.close()
        if bool(getattr(args_cli, "force_exit", False)):
            # Workaround for rare shutdown hangs in Kit/Omni extensions.
            # IMPORTANT: do not mask failures as success; sweep runners rely on exit codes.
            import os

            os._exit(0 if success else 1)

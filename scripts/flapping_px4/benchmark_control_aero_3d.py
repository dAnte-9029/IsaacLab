"""Matched CPU float32 batch timings; excludes PhysX, includes no file output in timers."""
from pathlib import Path
import argparse
import json, math, time
import torch
from flapping_bot.analysis.crosswind_aero import Experiment, cases, CSV, URDF
from flapping_bot.physics.control_aero_3d import ControlAero3D, ControlAero3DConfig
from flapping_bot.physics.control_aero_kinematics import LinkKinematics3D

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("logs/control_aero_3d/kernel_timings.json"))
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
torch.set_num_threads(1)
batch = 4
steps = 480
rows = []
exp = Experiment(cases().iloc[:batch], 10)
# DeLaurier reference adapter uses the unchanged 80-strip production formula.
# Convert its geometry and inputs to the same runtime dtype as the 3D kernels.
exp.velocity = exp.velocity.float()
exp.omega = exp.omega.float()
exp.reference = exp.reference.float()
exp.hinges = exp.hinges.float()
exp.rolls = exp.rolls.float()
for k, v in vars(exp.dg).items():
    if isinstance(v, torch.Tensor):
        object.__setattr__(exp.dg, k, v.float())
for model in ("DeLaurier", "L3D", "I3D"):
    if model != "DeLaurier":
        cfg = ControlAero3DConfig(backend=model, twist_amplitude_deg=35)
        kin = LinkKinematics3D(CSV, URDF, cfg)
        backend = ControlAero3D(cfg)
    total = 0.0
    kernel = 0.0
    for j in range(steps + 10):
        t = time.perf_counter()
        if model == "DeLaurier":
            exp.delaurier(j / 480, 3.0)
            kernel_dt = time.perf_counter() - t
        else:
            g, m, rel = kin.prescribed(
                exp.velocity, exp.omega, exp.reference, torch.full((batch,), 2 * math.pi * 3 * j / 480), 3.0
            )
            k = time.perf_counter()
            backend.evaluate(g, m, exp.reference, rel)
            kernel_dt = time.perf_counter() - k
        end = time.perf_counter()
        if j >= 10:
            total += end - t
            kernel += kernel_dt
    rows.append(
        dict(
            model=model,
            batch=batch,
            steps=steps,
            seconds=total,
            kernel_batch_ms=kernel * 1000 / steps,
            with_kinematics_batch_ms=total * 1000 / steps,
            dtype="float32",
            threads=1,
        )
    )
    print(rows[-1], flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(rows, indent=2))

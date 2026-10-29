"""Small deterministic control-mechanism matrix, no flight-log or policy access."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from flapping_bot.analysis.crosswind_aero import Experiment, cases, CSV, URDF, CG, AXES
from flapping_bot.physics.control_aero_3d import ControlAero3D, ControlAero3DConfig, distributed_body_drag
from flapping_bot.physics.control_aero_kinematics import LinkKinematics3D

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--sections", type=int, default=10)
p.add_argument("--steps", type=int, default=128)
p.add_argument("--trim", type=Path)
p.add_argument("--resolution", action="store_true")
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
rows = []
means = []
timings = []
diagnostics = []
table = cases()
table = table[
    table.case.isin(
        [
            "neutral",
            "beta-10",
            "beta-5",
            "beta-0.1",
            "beta0.1",
            "beta5",
            "beta10",
            "p-0.25",
            "p0.25",
            "r-0.25",
            "r0.25",
            "rudder-10",
            "rudder10",
            "differential-10",
            "differential10",
        ]
    )
].reset_index(drop=True)
table.to_csv(args.output / "cases.csv", index=False)
for twist in (0.0, 35.0):
    for incidence in (
        (0.0,)
        if args.resolution
        else ((json.loads(args.trim.read_text())["I3D"]["alpha_deg"],) if args.trim else (0.0, 8.0))
    ):
        if args.trim:
            table["symmetric"] = json.loads(args.trim.read_text())["I3D"]["symmetric_tail_deg"]
        exp = Experiment(table, args.sections)
        exp.velocity[:, 2] = -exp.velocity[:, 0] * math.sin(math.radians(incidence))
        exp.velocity[:, 0] *= math.cos(math.radians(incidence))
        tail = sum(exp.tails().values())
        bf, bm = distributed_body_drag(exp.velocity, exp.omega, exp.reference)
        body = torch.cat([bf, bm], -1).numpy()
        for hz in (json.loads(args.trim.read_text())["I3D"]["frequency_hz"],) if args.trim else (3.0, 4.0):
            for model in ("L3D", "I3D"):
                cfg = ControlAero3DConfig(backend=model, sections_per_wing=args.sections, twist_amplitude_deg=twist)
                kin = LinkKinematics3D(CSV, URDF, cfg, dtype=torch.float64)
                backend = ControlAero3D(cfg)
                trace = []
                worst_cond = 0
                worst_res = 0
                maxalpha = 0
                separated = 0
                kernel = 0
                start = time.perf_counter()
                for j in range(args.steps):
                    phase = torch.full((len(table),), 2 * math.pi * j / args.steps, dtype=torch.float64)
                    g, m, rel = kin.prescribed(exp.velocity, exp.omega, exp.reference, phase, hz, exp.amplitude)
                    t = time.perf_counter()
                    o = backend.evaluate(g, m, exp.reference, rel)
                    kernel += time.perf_counter() - t
                    six = torch.cat([o.force, o.moment], -1)
                    wings = six.sum(1).numpy()
                    trace.append(wings)
                    worst_cond = max(worst_cond, float(o.condition.max()))
                    worst_res = max(worst_res, float(o.residual.max()))
                    maxalpha = max(maxalpha, float(o.alpha.abs().max()) * 180 / math.pi)
                    separated += float(o.separation.mean()) / args.steps
                    for component, data in [
                        ("wings", wings),
                        ("right", six[:, : args.sections].sum(1).numpy()),
                        ("left", six[:, args.sections :].sum(1).numpy()),
                        ("tail", tail),
                        ("body", body),
                        ("aircraft", wings + tail + body),
                    ]:
                        for i, case in table.iterrows():
                            rows.append(
                                dict(
                                    model=model,
                                    twist=twist,
                                    alpha=incidence,
                                    hz=hz,
                                    phase=360 * j / args.steps,
                                    case=case["case"],
                                    component=component,
                                    **dict(zip(AXES, data[i]))
                                )
                            )
                timings.append(
                    dict(
                        model=model,
                        twist=twist,
                        alpha=incidence,
                        hz=hz,
                        batch=len(table),
                        kernel_ms=kernel * 1000 / args.steps,
                        including_kinematics_and_output_ms=(time.perf_counter() - start) * 1000 / args.steps,
                    )
                )
                diagnostics.append(
                    dict(
                        model=model,
                        twist=twist,
                        alpha=incidence,
                        hz=hz,
                        max_condition=worst_cond,
                        max_residual=worst_res,
                        max_alpha_deg=maxalpha,
                        mean_separation=separated,
                    )
                )
                print(diagnostics[-1], flush=True)
phase = pd.DataFrame(rows)
phase.to_csv(args.output / "phase.csv.gz", index=False)
keys = ["model", "twist", "alpha", "hz", "case", "component"]
means = phase.groupby(keys)[AXES].mean().reset_index()
means.to_csv(args.output / "means.csv", index=False)
derivatives = []
for key, group in means.groupby(["model", "twist", "alpha", "hz", "component"]):
    idx = group.set_index("case")
    for channel, lo, hi, delta in [
        ("beta", "beta-0.1", "beta0.1", math.radians(0.2)),
        ("p", "p-0.25", "p0.25", 0.5),
        ("r", "r-0.25", "r0.25", 0.5),
        ("rudder", "rudder-10", "rudder10", math.radians(20)),
        ("differential", "differential-10", "differential10", math.radians(20)),
    ]:
        d = (idx.loc[hi, AXES] - idx.loc[lo, AXES]) / delta
        derivatives.append(
            dict(zip(["model", "twist", "alpha", "hz", "component"], key)) | dict(derivative=channel, **d.to_dict())
        )
pd.DataFrame(derivatives).to_csv(args.output / "derivatives.csv", index=False)
pd.DataFrame(timings).to_csv(args.output / "timings.csv", index=False)
(args.output / "summary.json").write_text(
    json.dumps(
        dict(
            config=asdict(cfg),
            diagnostics=diagnostics,
            reference=CG.tolist(),
            note="Prescribed operating points; shared I3D mean trim if --trim supplied",
            steps=args.steps,
        ),
        indent=2,
    )
)

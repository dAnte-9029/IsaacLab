"""Bounded numerical sensitivity on an input-selected flight window."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flapping_bot.config.ullt import ULLTConfig
from flapping_bot.physics.ullt_geometry import load_csv_urdf_geometry
from flapping_bot.physics.ullt_state_space import ULLTBackend, ULLTGeometry

from .ullt_flight_replay import (
    FlightReplayConfig,
    flight_kinematics,
    prepare_runs,
    sha256,
)


def run_sensitivity(replay_dir: Path, csv: Path, urdf: Path, output: Path) -> dict:
    """Check first alphabetic flight's first selected window, without retuning.

    Compare 10/20 sections per wing, 10/5 ms input integration and zero/steady
    initial wake. A bounded 4 s trace includes 2 s burn-in and 2 s scoring.
    """
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(replay_dir / "replay.parquet", use_threads=False)
    window = (
        frame[frame.run_id == frame.loc[frame.run_id >= 0, "run_id"].min()]
        .sort_values("time_s")
        .iloc[:401]
        .reset_index(drop=True)
    )
    runs, _ = prepare_runs(window, FlightReplayConfig(windowed_screen=False))
    if len(runs) != 1 or len(window) != 401:
        raise ValueError("Expected a complete first 4-second selected window")
    inputs = torch.from_numpy(runs[0]["inputs"])
    offset = inputs.new_tensor([0, 0, 0.030])
    root = ET.parse(urdf).getroot()
    hinges = (
        inputs.new_tensor([
            [float(v) for v in root.find(f"joint[@name='{name}']/origin").attrib["xyz"].split()]
            for name in ("right_wing", "left_wing")
        ])
        + offset
    )
    cg = inputs.new_tensor([[-0.12154, -0.00541, 0.04298]])
    traces = {}
    for name, sections, halves, initialization in [
        ("base", 10, 1, "zero"),
        ("time_half", 10, 2, "zero"),
        ("span_double", 20, 1, "zero"),
        ("steady_initial", 10, 1, "steady"),
    ]:
        neutral = load_csv_urdf_geometry(csv, urdf, sections_per_wing=sections)
        neutral = ULLTGeometry(*(
            (
                getattr(neutral, key) + offset
                if key in ("edge_start_w_m", "edge_end_w_m", "quarter_chord_w_m")
                else getattr(neutral, key)
            )
            for key in vars(neutral)
        ))
        backend = ULLTBackend(ULLTConfig(model="four_state", air_density_kg_m3=1))
        g, m = flight_kinematics(neutral, hinges, inputs[:1], np.pi / 6)
        state = backend.zero_state(g) if initialization == "zero" else backend.steady_state(g, m)
        values = []
        print(f"sensitivity: {name}", flush=True)
        for i in range(len(inputs)):
            g, m = flight_kinematics(neutral, hinges, inputs[i : i + 1], np.pi / 6)
            loads = backend.evaluate(g, m, state, cg)
            sign = inputs.new_tensor([1, -1, -1])
            values.append(
                torch.cat((loads.force_w_n * sign, loads.moment_about_reference_w_nm * sign), -1)[0].numpy()
                * float(inputs[i, 15])
            )
            if i < len(inputs) - 1:

                def sample(t: float):
                    x = inputs[i : i + 1] + (inputs[i + 1 : i + 2] - inputs[i : i + 1]) * (t / 0.01)
                    return flight_kinematics(neutral, hinges, x, np.pi / 6)

                for h in range(halves):
                    state = backend.advance(state, sample, h * 0.01 / halves, 0.01 / halves)
        traces[name] = np.asarray(values)
    metrics = {}
    for name in ("time_half", "span_double", "steady_initial"):
        rows = {}
        for axis, index in (("fx_b", 0), ("fz_b", 2), ("my_b", 4)):
            base, other = traces["base"][200:, index], traces[name][200:, index]
            rows[axis] = {
                "rms_difference": float(np.sqrt(np.mean((other - base) ** 2))),
                "relative_rms_difference": float(
                    np.sqrt(np.mean((other - base) ** 2)) / max(np.sqrt(np.mean(base**2)), 1e-12)
                ),
                "mean_difference": float(np.mean(other - base)),
            }
        metrics[name] = rows
    report = {
        "log_id": str(window.log_id.iloc[0]),
        "time_range_s": [float(window.time_s.iloc[0]), float(window.time_s.iloc[-1])],
        "scope": "one input-selected 4-second trace, final 2 seconds scored; not full-envelope convergence",
        "metrics": metrics,
        "source_replay_sha256": sha256(replay_dir / "replay.parquet"),
        "source_sha256": {str(p.resolve()): sha256(p) for p in (Path(__file__), csv, urdf)},
    }
    np.savez_compressed(output / "traces.npz", **traces)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

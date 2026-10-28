"""Small 10/20-panel check of area and masked section-load fractions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flapping_bot.physics.ullt_geometry import load_csv_urdf_geometry
from flapping_bot.physics.ullt_twist import PrescribedTwistConfig

from .ullt_flight_replay import AXES, FlightReplayConfig, replay_registered_flights, sha256
from .ullt_section_attribution import SectionCollector, THRESHOLDS


def check_section_resolution(source: Path, csv: Path, urdf: Path, output: Path) -> dict:
    """Use first archived selected window, 4 s replay with final 2 s scored.

    This isolated-window replay recomputes frequency derivatives locally. Both
    resolutions receive exactly the same inputs; it is not a replacement for
    the main full-contiguous-input derivative/parity comparison.
    """
    output.mkdir(parents=True, exist_ok=False)
    archived = source / "zero.parquet"
    expected = json.loads((source / "summary.json").read_text())["artifact_sha256"][archived.name]
    if sha256(archived) != expected:
        raise ValueError("Source replay hash mismatch")
    frame = pd.read_parquet(archived)
    run = frame.loc[frame.run_id >= 0, "run_id"].min()
    window = frame.loc[frame.run_id == run].sort_values("time_s").iloc[:401].reset_index(drop=True)
    records, summaries = [], []
    for candidate, twist in (("zero", PrescribedTwistConfig()), ("A35_k-0.5_p0", PrescribedTwistConfig(35, -0.5, 0))):
        loads = {}
        for n in (10, 20):
            config = FlightReplayConfig(sections_per_wing=n, windowed_screen=False, twist=twist)
            neutral = load_csv_urdf_geometry(csv, urdf, sections_per_wing=n)
            collector = SectionCollector(window, neutral)
            replay, diagnostic = replay_registered_flights(
                window, csv, urdf, config, lambda s: print(f"{candidate}/N{n}: {s}", flush=True), collector
            )
            if diagnostic["failures"]:
                raise ValueError("Resolution diagnostic replay failed")
            sections = collector.finish()
            sections.to_parquet(output / f"{candidate}_n{n}_sections.parquet", index=False)
            (output / f"{candidate}_n{n}.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
            t = int(replay.score_valid.sum())
            angles = abs(sections.alpha_deg.to_numpy().reshape(t, 2 * n))
            area = sections.reference_area_m2.to_numpy().reshape(t, 2 * n)
            values = sections[[f"total_{a}" for a in AXES]].to_numpy().reshape(t, 2 * n, 6)
            loads[n] = values.sum(axis=1)
            for threshold in THRESHOLDS:
                mask = angles > threshold
                fraction = float(np.mean((area * mask).sum(axis=1) / area.sum(axis=1)))
                numerator = (abs(values) * mask[..., None]).sum(axis=(0, 1))
                denominator = abs(values).sum(axis=(0, 1))
                shares = np.divide(numerator, denominator, out=np.full(6, np.nan), where=denominator > 1e-12)
                for i, axis in enumerate(AXES):
                    records.append(
                        {
                            "candidate": candidate,
                            "sections_per_wing": n,
                            "threshold_deg": threshold,
                            "axis": axis,
                            "mean_area_fraction": fraction,
                            "absolute_section_load_share": float(shares[i]),
                        }
                    )
        for i, axis in enumerate(AXES):
            difference = np.sqrt(np.mean((loads[20][:, i] - loads[10][:, i]) ** 2))
            scale = np.sqrt(np.mean(loads[10][:, i] ** 2))
            summaries.append(
                {
                    "candidate": candidate,
                    "axis": axis,
                    "rms_difference": float(difference),
                    "relative_rms_difference": float(difference / scale) if scale > 1e-12 else None,
                }
            )
    table = pd.DataFrame(records)
    table.to_csv(output / "threshold_comparison.csv", index=False)
    report = {
        "log_id": str(window.log_id.iloc[0]),
        "time_range_s": [float(window.time_s.min()), float(window.time_s.max())],
        "scope": "one input-selected four-second window, last two seconds scored; local derivative preprocessing",
        "resultant_differences": summaries,
        "source_sha256": {str(p.resolve()): sha256(p) for p in (Path(__file__), archived, csv, urdf)},
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

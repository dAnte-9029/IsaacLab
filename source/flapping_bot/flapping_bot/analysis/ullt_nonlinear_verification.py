"""Independent checks of saved rows, moments, components and solver acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ullt_flight_replay import AXES, sha256


def verify_nonlinear_artifacts(output: Path, source: Path) -> dict:
    """Verify full comparisons without rerunning aerodynamics or loading raw logs."""
    reference = pd.read_parquet(source / "A35_k-0.5_p0.parquet")
    windows = json.loads((source / "A35_k-0.5_p0.json").read_text())["selected_windows"]
    results = {}
    baseline_sections = pd.read_parquet(output / "linear_twist_sections.parquet")
    for name in ("linear_twist", "early", "late"):
        diagnostic = json.loads((output / f"{name}.json").read_text())
        assert diagnostic["failures"] == [] and diagnostic["selected_windows"] == windows
        frame = pd.read_parquet(output / f"{name}.parquet")
        assert np.array_equal(frame.score_valid, reference.score_valid)
        assert frame.partition.eq("train").all() and frame.log_id.nunique() == 3
        for column in (*AXES, *(f"delaurier_{a}" for a in AXES), "mechanical_phase_rad", "flap_frequency_hz"):
            assert np.array_equal(frame[column], reference[column], equal_nan=True), column
        sections = pd.read_parquet(output / f"{name}_sections.parquet")
        assert len(sections) == int(frame.score_valid.sum()) * 2 * diagnostic["configuration"]["sections_per_wing"]
        assert not sections[["row_id", "section_id"]].duplicated().any()
        assert np.array_equal(sections[["row_id", "section_id"]], baseline_sections[["row_id", "section_id"]])
        columns = [f"total_{axis}" for axis in AXES]
        sums = sections.groupby("row_id")[columns].sum()
        expected = frame.loc[sums.index, [f"ullt_{a}" for a in AXES]].to_numpy()
        closure = float(np.max(abs(sums.to_numpy() - expected)))
        assert closure < 1e-9
        component_error = 0.0
        for axis in AXES:
            delta = sections[f"total_{axis}"] - sum(
                sections[f"{part}_{axis}"] for part in ("circulatory", "added_mass", "profile_drag")
            )
            component_error = max(component_error, float(delta.abs().max()))
            assert np.array_equal(sections[f"added_mass_{axis}"], baseline_sections[f"added_mass_{axis}"])
        assert component_error < 1e-9
        assert (sections.nonlinear_residual_mps <= sections.nonlinear_tolerance_mps).all()
        assert np.isfinite(sections[columns].to_numpy()).all()
        assert (sections.section_cd >= 0).all()
        results[name] = dict(
            scored_rows=int(frame.score_valid.sum()),
            section_rows=len(sections),
            section_sum_max_error=closure,
            component_sum_max_error=component_error,
            added_mass_bitwise_unchanged=True,
            labels_and_frozen_delaurier_unchanged=True,
            all_reported_residuals_pass=True,
        )
    verification = dict(
        passed=True,
        candidates=results,
        artifacts_sha256={p.name: sha256(p) for p in output.iterdir() if p.suffix in (".parquet", ".csv")},
    )
    (output / "independent_verification.json").write_text(json.dumps(verification, indent=2))
    return verification

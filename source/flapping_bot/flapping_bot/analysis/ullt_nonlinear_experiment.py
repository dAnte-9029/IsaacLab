"""Fixed-prior, train-only nonlinear ULLT experiment on an archived twist screen."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flapping_bot.config.nonlinear_section import flat_plate_prior
from flapping_bot.config.ullt import ULLTConfig
from flapping_bot.physics.ullt_geometry import load_csv_urdf_geometry
from flapping_bot.physics.ullt_state_space import ULLTGeometry, ULLTLoads, ULLTMotion
from flapping_bot.physics.ullt_twist import PrescribedTwistConfig
from .ullt_flight_replay import FlightReplayConfig, replay_registered_flights, sha256
from .ullt_flight_report import compute_flight_tables, summarize_tables
from .ullt_section_attribution import SectionCollector


class NonlinearSectionCollector(SectionCollector):
    """Extend exact wrench decomposition with physical/equivalent-flow diagnostics."""

    def __call__(
        self,
        indices: np.ndarray,
        geometry: ULLTGeometry,
        motion: ULLTMotion,
        loads: ULLTLoads,
        density: torch.Tensor,
        cg: torch.Tensor,
    ) -> None:
        super().__call__(indices, geometry, motion, loads, density, cg)
        chunk = self.chunks[-1]
        for name in (
            "effective_incidence_rad",
            "equivalent_normalwash_mps",
            "section_cl",
            "section_cd",
            "nonlinear_residual_mps",
            "nonlinear_iterations",
            "nonlinear_tolerance_mps",
            "effective_to_incident_speed_ratio",
        ):
            value = getattr(loads, name).numpy()
            if value.ndim == 1:
                value = np.repeat(value[:, None], len(self.reference_area), axis=1)
            chunk[name] = value.ravel()


def run_experiment(source: Path, csv: Path, urdf: Path, output: Path, smoke: bool = False) -> None:
    """Replay two fixed priors and one timed linear control; never read a data registry."""
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    summary = json.loads((source / "summary.json").read_text())
    archives = {}
    for name in ("zero", "A35_k-0.5_p0"):
        path = source / f"{name}.parquet"
        if sha256(path) != summary["artifact_sha256"][path.name]:
            raise ValueError(f"Archive hash mismatch: {path}")
        archives[name] = pd.read_parquet(path)
        if not archives[name].partition.eq("train").all() or archives[name].log_id.nunique() != 3:
            raise ValueError("Expected precisely three archived training flights")
    raw = json.loads((source / "A35_k-0.5_p0.json").read_text())["configuration"]
    config = FlightReplayConfig(**{**raw, "twist": PrescribedTwistConfig(**raw["twist"])})
    if config.twist != PrescribedTwistConfig(35, -0.5, 0):
        raise ValueError("Twist must remain frozen at (35,-0.5,0)")
    if smoke:
        config = replace(config, maximum_rows_per_run=221)
    root = Path(__file__).resolve().parents[4]
    sources = [
        root / "scripts/flapping_px4/compare_ullt_nonlinear.py",
        csv,
        urdf,
        source / "summary.json",
        source / "A35_k-0.5_p0.json",
        *[source / f"{n}.parquet" for n in archives],
        *Path(__file__).parent.glob("ullt_*.py"),
        *Path(__file__).parents[1].joinpath("physics").glob("*ullt*.py"),
        Path(__file__).parents[1] / "physics/flat_plate_section.py",
        *Path(__file__).parents[1].joinpath("config").glob("*section*.py"),
        Path(__file__).parents[1] / "config/ullt.py",
    ]
    manifest = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        smoke=smoke,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        torch=torch.__version__,
        threads=1,
        command=sys.argv,
        python=sys.executable,
        source_sha256={str(p.resolve()): sha256(p) for p in sources},
        scope="Three train logs only; wing predictions versus reconstructed whole aircraft; no fitting",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for path in sources:
        if path.suffix == ".py":
            snapshot = output / "execution_source_snapshot" / path.resolve().relative_to(root)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
    archives["zero"].to_parquet(output / "zero.parquet", index=False)
    neutral = load_csv_urdf_geometry(csv, urdf, sections_per_wing=config.sections_per_wing)
    for name in ("linear_twist", "early", "late"):
        cfg = ULLTConfig(model=config.model, air_density_kg_m3=1.0, collect_solver_statistics=True)
        if name != "linear_twist":
            cfg = replace(cfg, section_model="nonlinear_flat_plate", nonlinear_polar=flat_plate_prior(name))
        replay_cfg = replace(config, backend_config=cfg)
        collector = NonlinearSectionCollector(archives["A35_k-0.5_p0"], neutral)
        start = time.perf_counter()
        try:
            result, diagnostic = replay_registered_flights(
                archives["A35_k-0.5_p0"], csv, urdf, replay_cfg, lambda s: print(f"{name}: {s}", flush=True), collector
            )
            diagnostic["wall_seconds_including_observer"] = time.perf_counter() - start
            diagnostic["section_closure_max"] = collector.maximum_section_closure_error
            diagnostic["complete_expected_mask"] = bool(
                np.array_equal(result.score_valid, archives["A35_k-0.5_p0"].score_valid)
            )
            if name == "linear_twist" and not smoke:
                cols = [f"ullt_{a}" for a in ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")]
                diagnostic["linear_archive_max_error"] = float(
                    np.nanmax(abs(result[cols].to_numpy() - archives["A35_k-0.5_p0"][cols].to_numpy()))
                )
                if diagnostic["linear_archive_max_error"] != 0:
                    raise ValueError("Linear baseline changed")
            result.to_parquet(output / f"{name}.parquet", index=False)
            if collector.chunks:
                collector.finish().to_parquet(output / f"{name}_sections.parquet", index=False)
            (output / f"{name}.json").write_text(json.dumps(diagnostic, indent=2))
            if diagnostic["failures"]:
                print(f"{name}: {len(diagnostic['failures'])} explicitly rejected runs", flush=True)
            if diagnostic["failures"] or (not smoke and not diagnostic["complete_expected_mask"]):
                (output / f"{name}_failure.json").write_text(
                    json.dumps(
                        dict(
                            error="Rejected integration or incomplete expected mask; excluded from full comparison",
                            failures=diagnostic["failures"],
                        ),
                        indent=2,
                    )
                )
            print(
                f'{name}: saved, {diagnostic["wall_seconds_including_observer"]:.1f}s; full mask={diagnostic["complete_expected_mask"]}',
                flush=True,
            )
        except (ValueError, FloatingPointError) as error:
            (output / f"{name}_failure.json").write_text(
                json.dumps(dict(error=str(error), wall_seconds=time.perf_counter() - start), indent=2)
            )
            print(f"{name}: EXPLICIT FAILURE: {error}", flush=True)
    if not smoke:
        write_experiment_report(output, config)
        if not any(output.glob("*_failure.json")):
            from .ullt_nonlinear_verification import verify_nonlinear_artifacts

            verify_nonlinear_artifacts(output, source)


def write_experiment_report(output: Path, config: FlightReplayConfig | None = None) -> None:
    """Create metrics only for full coverage; preserve failed candidates visibly."""
    if config is None:
        raw = json.loads((output / "linear_twist.json").read_text())["configuration"]
        config = FlightReplayConfig(**{**raw, "twist": PrescribedTwistConfig(**raw["twist"]), "backend_config": None})
    rows = []
    for name in ("zero", "linear_twist", "early", "late"):
        if not (output / f"{name}.parquet").exists():
            continue
        if name != "zero" and not json.loads((output / f"{name}.json").read_text())["complete_expected_mask"]:
            continue
        frame = pd.read_parquet(output / f"{name}.parquet")
        tables = compute_flight_tables(frame, config)
        summary = summarize_tables(tables, frame)
        (output / f"{name}_metrics.json").write_text(json.dumps(summary, indent=2))
        for key, table in tables.items():
            table.to_csv(output / f"{name}_{key}.csv", index=False)
        for metric in summary["metrics"]:
            for model in ("ullt", "delaurier") if name == "zero" else ("ullt",):
                rows.append(
                    dict(
                        candidate=name if model == "ullt" else "delaurier",
                        axis=metric["axis"],
                        **{
                            k.removeprefix(model + "_"): v
                            for k, v in metric.items()
                            if k.startswith(model + "_") and k.endswith("_macro")
                        },
                    )
                )
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    from .ullt_nonlinear_report import write_nonlinear_figures

    write_nonlinear_figures(output)

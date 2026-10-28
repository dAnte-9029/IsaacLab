"""Area/load-resolved diagnostics; no local truth or causal error allocation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flapping_bot.physics.ullt_load_diagnostics import section_load_diagnostics
from flapping_bot.physics.ullt_state_space import ULLTGeometry, ULLTLoads, ULLTMotion

from .ullt_flight_replay import AXES, FlightReplayConfig, replay_registered_flights, sha256
from flapping_bot.physics.ullt_geometry import load_csv_urdf_geometry
from flapping_bot.physics.ullt_twist import PrescribedTwistConfig

COMPONENTS = ("circulatory", "added_mass", "profile_drag")
THRESHOLDS = (25.0, 35.0, 45.0, 60.0)
BAND_EDGES = np.array([0, 15, 25, 35, 45, 60, 180.000001])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Inverse weighted empirical CDF (no interpolated phantom observations)."""
    if not 0 <= quantile <= 1 or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid quantile or weights")
    order = np.argsort(values)
    index = min(np.searchsorted(np.cumsum(weights[order]), quantile * weights.sum(), side="left"), len(order) - 1)
    return float(values[order[index]])


def masked_load_metrics(
    sections: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Six-axis metrics for (T,N,6) loads; subtraction freezes the wake/geometry.

    Absolute shares sum |section load| before summation over the wing. An
    undefined zero denominator stays NaN. Error changes are nonadditive.
    """
    region = (sections * mask[..., None]).sum(axis=1)
    absolute = np.abs(sections)
    numerator = (absolute * mask[..., None]).sum(axis=(0, 1))
    denominator = absolute.sum(axis=(0, 1))
    share = np.divide(numerator, denominator, out=np.full(6, np.nan), where=denominator > 1e-12)
    residual = prediction - target
    mse = np.mean(residual**2, axis=0)
    removed_mse = np.mean((residual - region) ** 2, axis=0)
    return {
        "signed_mean": region.mean(axis=0),
        "resultant_rms": np.sqrt(np.mean(region**2, axis=0)),
        "absolute_section_load_share": share,
        "original_rmse": np.sqrt(mse),
        "frozen_subtraction_rmse": np.sqrt(removed_mse),
        "frozen_subtraction_mse_reduction": mse - removed_mse,
        "residual_load_alignment": np.mean(residual * region, axis=0),
    }


class SectionCollector:
    """Read-only scored-step observer; retains material area separately from projected area."""

    def __init__(self, frame: pd.DataFrame, neutral: ULLTGeometry):
        self.frame = frame
        self.reference_area = (neutral.chord_m * neutral.width_m).numpy()[0]
        n = len(self.reference_area) // 2
        self.span = np.r_[(np.arange(n)[::-1] + 0.5) / n, (np.arange(n) + 0.5) / n]
        self.wing = np.array(["right"] * n + ["left"] * n)
        self.chunks: list[pd.DataFrame] = []
        self.maximum_section_closure_error = 0.0

    def __call__(
        self,
        indices: np.ndarray,
        geometry: ULLTGeometry,
        motion: ULLTMotion,
        loads: ULLTLoads,
        density: torch.Tensor,
        cg: torch.Tensor,
    ) -> None:
        fields = section_load_diagnostics(geometry, motion, loads, density, cg)
        sign = geometry.chord_m.new_tensor([1, -1, -1])
        expected = torch.cat((loads.section_force_w_n, loads.section_moment_about_reference_w_nm), -1)
        expected = expected * density[..., None] * torch.cat((sign, sign))
        error = float((fields["total"] - expected).abs().max())
        self.maximum_section_closure_error = max(error, self.maximum_section_closure_error)
        if error > 1e-9:
            raise ValueError("Per-section force/moment decomposition does not close")
        n = len(self.reference_area)
        data = {
            "row_id": np.repeat(indices, n),
            "section_id": np.tile(np.arange(n), len(indices)),
            "wing": np.tile(self.wing, len(indices)),
            "span_fraction": np.tile(self.span, len(indices)),
            "reference_area_m2": np.tile(self.reference_area, len(indices)),
        }
        for name in ("log_id", "timestamp_us", "time_s", "mechanical_phase_rad", "cycle_id", "segment_id"):
            data[name] = np.repeat(self.frame.loc[indices, name].to_numpy(), n)
        data["phase_bin"] = (np.mod(data["mechanical_phase_rad"], 2 * np.pi) * 36 / (2 * np.pi)).astype(int)
        for name, value in fields.items():
            array = value.numpy()
            if array.ndim == 2:
                data[name] = array.ravel()
            else:
                axes = AXES if array.shape[-1] == 6 else AXES[3:]
                for i, axis in enumerate(axes):
                    data[f"{name}_{axis}"] = array[..., i].ravel()
        data["circulation_m2_s"] = loads.circulation_m2_s.numpy().ravel()
        self.chunks.append(pd.DataFrame(data))

    def finish(self) -> pd.DataFrame:
        return pd.concat(self.chunks, ignore_index=True).sort_values(["row_id", "section_id"]).reset_index(drop=True)


def collect_section_case(
    source: Path,
    name: str,
    csv: Path,
    urdf: Path,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Replay a hashed prior screen with the observer and require frozen-output parity."""
    source_summary = json.loads((source / "summary.json").read_text())
    path = source / f"{name}.parquet"
    if sha256(path) != source_summary["artifact_sha256"][path.name]:
        raise ValueError(f"Prior replay hash mismatch: {path}")
    original = pd.read_parquet(path)
    if not original.partition.eq("train").all():
        raise ValueError("This diagnostic is restricted to the previously selected training flights")
    original_config = json.loads((source / f"{name}.json").read_text())["configuration"]
    config = FlightReplayConfig(**{**original_config, "twist": PrescribedTwistConfig(**original_config["twist"])})
    neutral = load_csv_urdf_geometry(csv, urdf, sections_per_wing=config.sections_per_wing)
    collector = SectionCollector(original, neutral)
    replay, diagnostic = replay_registered_flights(
        original, csv, urdf, config, lambda s: print(f"{name}: {s}", flush=True), collector
    )
    if not replay.score_valid.equals(original.score_valid) or diagnostic["failures"]:
        raise ValueError("Observer replay changed coverage or failed")
    mask = original.score_valid
    differences = {}
    for prefix in ("ullt", "ullt_added_mass", "ullt_circulatory"):
        columns = [f"{prefix}_{a}" for a in AXES]
        differences[prefix] = float(
            np.max(abs(replay.loc[mask, columns].to_numpy() - original.loc[mask, columns].to_numpy()))
        )
    if max(differences.values()) > 1e-9:
        raise ValueError(f"Observer replay does not match archived loads: {differences}")
    sections = collector.finish()
    summed = sections.groupby("row_id")[[f"total_{a}" for a in AXES]].sum().to_numpy()
    expected = replay.loc[mask, [f"ullt_{a}" for a in AXES]].to_numpy()
    if not np.allclose(summed, expected, rtol=1e-12, atol=1e-10):
        raise ValueError("Saved section loads do not reconstruct saved resultant")
    sections.to_parquet(output / f"{name}_sections.parquet", index=False)
    report = {
        "configuration": asdict(config),
        "replay": diagnostic,
        "archive_parity_max_abs": differences,
        "section_closure_max_abs": collector.maximum_section_closure_error,
        "resultant_closure_max_abs": float(np.max(abs(summed - expected))),
        "source_replay_sha256": sha256(path),
        "section_rows": len(sections),
    }
    (output / f"{name}_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return sections, replay, report


def summarize_sections(sections: pd.DataFrame, replay: pd.DataFrame, candidate: str) -> dict[str, pd.DataFrame]:
    """Keep per-log, per-component and per-threshold evidence without mixing units."""
    threshold_rows, band_rows, area_rows, phase_rows, scalar_rows = [], [], [], [], []
    n = sections.section_id.nunique()
    for log, group in sections.groupby("log_id", sort=True):
        group = group.sort_values(["row_id", "section_id"])
        ids = group.row_id.unique()
        source = replay.loc[ids]
        count = len(ids)
        alpha = group.alpha_deg.to_numpy().reshape(count, n)
        area = group.reference_area_m2.to_numpy().reshape(count, n)
        abs_alpha = abs(alpha)
        weights = area / area.sum(axis=1, keepdims=True)
        prediction = source[[f"ullt_{a}" for a in AXES]].to_numpy()
        target = source[list(AXES)].to_numpy()
        components = {
            c: group[[f"{c}_{a}" for a in AXES]].to_numpy().reshape(count, n, 6)
            for c in (*COMPONENTS, "total", "circulatory_flow_parallel", "profile_drag_per_cd")
        }
        scalar_rows.append(
            {
                "candidate": candidate,
                "log_id": log,
                "scored_times": count,
                "reference_wing_area_m2": float(area[0].sum()),
                "area_weighted_abs_alpha_p50": weighted_quantile(abs_alpha.ravel(), weights.ravel(), 0.5),
                "area_weighted_abs_alpha_p95": weighted_quantile(abs_alpha.ravel(), weights.ravel(), 0.95),
                "max_alpha_p95": float(np.quantile(abs_alpha.max(axis=1), 0.95)),
            }
        )
        for threshold in (-1.0, *THRESHOLDS):
            mask = abs_alpha > threshold
            fraction = (weights * mask).sum(axis=1)
            area_rows.append(
                {
                    "candidate": candidate,
                    "log_id": log,
                    "threshold_deg": threshold,
                    "mean_area_fraction": float(fraction.mean()),
                    "p95_area_fraction": float(np.quantile(fraction, 0.95)),
                }
            )
            time = pd.DataFrame(
                {
                    "candidate": candidate,
                    "log_id": log,
                    "row_id": ids,
                    "threshold_deg": threshold,
                    "phase_bin": (np.mod(source.mechanical_phase_rad, 2 * np.pi) * 36 / (2 * np.pi))
                    .astype(int)
                    .to_numpy(),
                    "area_fraction": fraction,
                }
            )
            for component, values in components.items():
                metrics = masked_load_metrics(values, mask, prediction, target)
                if component not in (*COMPONENTS, "total"):
                    metrics = {
                        key: value
                        for key, value in metrics.items()
                        if key in ("signed_mean", "resultant_rms", "absolute_section_load_share")
                    }
                for i, axis in enumerate(AXES):
                    threshold_rows.append(
                        {
                            "candidate": candidate,
                            "log_id": log,
                            "component": component,
                            "threshold_deg": threshold,
                            "axis": axis,
                            **{k: float(v[i]) for k, v in metrics.items()},
                        }
                    )
                if component in (*COMPONENTS, "total"):
                    high = (values * mask[..., None]).sum(axis=1)
                    for i, axis in enumerate(AXES):
                        time[f"{component}_{axis}"] = high[:, i]
            phase_rows.append(
                time.groupby(["candidate", "log_id", "threshold_deg", "phase_bin"], as_index=False)
                .mean(numeric_only=True)
                .drop(columns="row_id")
            )
        for lower, upper in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
            mask = (abs_alpha >= lower) & (abs_alpha < upper)
            fraction = float((weights * mask).sum(axis=1).mean())
            for component in COMPONENTS:
                values = components[component]
                metrics = masked_load_metrics(values, mask, prediction, target)
                for i, axis in enumerate(AXES):
                    band_rows.append(
                        {
                            "candidate": candidate,
                            "log_id": log,
                            "lower_deg": lower,
                            "upper_deg": min(upper, 180),
                            "area_fraction": fraction,
                            "component": component,
                            "axis": axis,
                            "signed_mean": float(metrics["signed_mean"][i]),
                            "absolute_section_load_share": float(metrics["absolute_section_load_share"][i]),
                        }
                    )
    cells = sections.copy()
    cells["abs_alpha_deg"] = abs(cells.alpha_deg)
    cells["over35_probability"] = (cells.abs_alpha_deg > 35).astype(float)
    cells["candidate"] = candidate
    keys = ["candidate", "log_id", "wing", "section_id", "span_fraction", "phase_bin"]
    values = [
        "alpha_deg",
        "abs_alpha_deg",
        "over35_probability",
        "reference_area_m2",
        "effective_area_m2",
        "dynamic_pressure_pa",
        *[f"{c}_{a}" for c in (*COMPONENTS, "total") for a in AXES],
    ]
    phase_span = cells.groupby(keys, as_index=False)[values].mean()
    return {
        "threshold_metrics": pd.DataFrame(threshold_rows),
        "incidence_bands": pd.DataFrame(band_rows),
        "area_fractions": pd.DataFrame(area_rows),
        "phase_threshold": pd.concat(phase_rows, ignore_index=True),
        "incidence_summary": pd.DataFrame(scalar_rows),
        "phase_span": phase_span,
    }


def write_attribution_tables(tables: list[dict[str, pd.DataFrame]], output: Path) -> dict[str, pd.DataFrame]:
    """Save per-flight evidence and equal-flight aggregates with named grouping keys."""
    result = {key: pd.concat([t[key] for t in tables], ignore_index=True) for key in tables[0]}
    groupings = {
        "threshold_metrics": ["candidate", "component", "threshold_deg", "axis"],
        "incidence_bands": ["candidate", "lower_deg", "upper_deg", "component", "axis"],
        "area_fractions": ["candidate", "threshold_deg"],
        "phase_threshold": ["candidate", "threshold_deg", "phase_bin"],
        "incidence_summary": ["candidate"],
        "phase_span": ["candidate", "wing", "section_id", "span_fraction", "phase_bin"],
    }
    for name, table in list(result.items()):
        table.to_csv(output / f"{name}_per_flight.csv", index=False)
        macro = table.drop(columns="log_id").groupby(groupings[name], as_index=False).mean(numeric_only=True)
        macro.to_csv(output / f"{name}_macro.csv", index=False)
        result[name + "_macro"] = macro
    return result

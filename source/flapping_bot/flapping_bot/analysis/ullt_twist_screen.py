"""Bounded exploratory twist screen, separate from held-out validation."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from flapping_bot.physics.ullt_twist import PrescribedTwistConfig
from .ullt_flight_replay import FlightReplayConfig, prepare_runs, replay_registered_flights, sha256
from .ullt_flight_report import compute_flight_tables


def screen_candidates() -> dict[str, PrescribedTwistConfig]:
    """Fixed candidates declared before inspecting this screen's load errors."""
    return {
        "zero": PrescribedTwistConfig(),
        "A20_k0_p-25": PrescribedTwistConfig(20, 0, -25),
        "A35_k-0.5_p-25": PrescribedTwistConfig(35, -0.5, -25),
        "A40_k0_p-25": PrescribedTwistConfig(40, 0, -25),
        "A35_k-0.5_p0": PrescribedTwistConfig(35, -0.5, 0),
        "A35_k-0.5_p-45": PrescribedTwistConfig(35, -0.5, -45),
        "A35_k0_p-25": PrescribedTwistConfig(35, 0, -25),
    }


def select_screen_flights(frame: pd.DataFrame, config: FlightReplayConfig) -> tuple[pd.DataFrame, list[dict]]:
    """Choose low/median/high train-flight frequency, without looking at loads."""
    train = frame.loc[frame.partition == "train"].reset_index(drop=True)
    runs, _ = prepare_runs(train, config)
    eligible = sorted({str(r["log_id"]) for r in runs})
    ranking = train[train.log_id.isin(eligible)].groupby("log_id").flap_frequency_hz.median().sort_values(kind="stable")
    if len(ranking) < 3:
        raise ValueError("Need three eligible train flights")
    chosen = ranking.iloc[[0, len(ranking) // 2, len(ranking) - 1]]
    records = [{"log_id": str(log), "median_frequency_hz": float(f)} for log, f in chosen.items()]
    return train[train.log_id.isin(chosen.index)].reset_index(drop=True), records


def run_twist_screen(
    frame: pd.DataFrame,
    csv: Path,
    urdf: Path,
    output: Path,
    config: FlightReplayConfig = FlightReplayConfig(),
) -> dict:
    """Replay fixed candidates; save raw results before reporting any selection."""
    selected, selection = select_screen_flights(frame, config)
    candidates = screen_candidates()
    (output / "design.json").write_text(
        json.dumps(
            {
                "selection_rule": "train-only low/median/high median logged frequency; no load selection",
                "flights": selection,
                "configuration": asdict(config),
                "candidates": {name: asdict(value) for name, value in candidates.items()},
            },
            indent=2,
        )
        + "\n"
    )
    results, diagnostics = {}, {}
    for name, twist in candidates.items():
        print(f"Candidate {name}", flush=True)
        result, diagnostic = replay_registered_flights(
            selected, csv, urdf, replace(config, twist=twist), lambda s: print(f"{name}: {s}", flush=True)
        )
        result.to_parquet(output / f"{name}.parquet", index=False)
        (output / f"{name}.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        results[name], diagnostics[name] = result, diagnostic
    return report_twist_screen(results, diagnostics, output, config, selection)


def report_twist_screen(
    results: dict[str, pd.DataFrame],
    diagnostics: dict,
    output: Path,
    config: FlightReplayConfig,
    selection: list[dict],
    replay_dir: Path | None = None,
) -> dict:
    """Score all candidates on one common mask; report full coverage separately."""
    common = np.logical_and.reduce([r.score_valid.to_numpy() for r in results.values()])
    axes = ("fx_b", "fz_b", "my_b")
    tables_all, per_flight, cycles_all, transitions_all, incidence = {}, [], [], [], []
    for name, original in results.items():
        result = original.copy()
        result["score_valid"] = common
        tables = compute_flight_tables(result, config)
        tables_all[name] = tables
        models = ("delaurier", "ullt") if name == "zero" else ("ullt",)
        for model in models:
            label = "delaurier" if model == "delaurier" else name
            flight = tables["per_flight"].query("model == @model").copy()
            flight["candidate"] = label
            per_flight.append(flight)
            cycles = tables["cycles"].query("source == @model").copy()
            cycles["candidate"] = label
            cycles_all.append(cycles)
            if not tables["transitions"].empty:
                transitions = tables["transitions"].query("model == @model").copy()
                transitions["candidate"] = label
                transitions_all.append(transitions)
        for log, group in result.loc[common].groupby("log_id"):
            incidence.append(
                {
                    "candidate": name,
                    "log_id": log,
                    "max_local_incidence_p50_deg": group.ullt_max_incidence_deg.median(),
                    "max_local_incidence_p95_deg": group.ullt_max_incidence_deg.quantile(0.95),
                }
            )
    flights, cycles = pd.concat(per_flight), pd.concat(cycles_all)
    flights.to_csv(output / "per_flight.csv", index=False)
    cycles.to_csv(output / "cycles.csv", index=False)
    pd.DataFrame(incidence).to_csv(output / "incidence.csv", index=False)
    if transitions_all:
        pd.concat(transitions_all).to_csv(output / "transitions.csv", index=False)
    rows = []
    for (candidate, axis), group in flights.groupby(["candidate", "axis"], sort=False):
        cycle = cycles[(cycles.candidate == candidate) & (cycles.axis == axis)].groupby("log_id")
        rows.append(
            {
                "candidate": candidate,
                "axis": axis,
                "rmse": group.rmse.mean(),
                "bias": group.bias.mean(),
                "cycle_mean_rmse": cycle.mean_error.apply(lambda x: np.sqrt(np.mean(x**2))).mean(),
                "cycle_peak_to_peak_mae": cycle.peak_to_peak_error.apply(lambda x: x.abs().mean()).mean(),
                "h1_phase_mae_deg": cycle.h1_phase_error_deg.apply(lambda x: x.abs().mean()).mean(),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    comparison = metrics.pivot(index="candidate", columns="axis", values="rmse")[list(axes)]
    score = comparison.div(comparison.loc["zero"]).mean(axis=1)
    ordered = score.drop(index=["zero", "delaurier"]).sort_values()
    summary = {
        "scope": "Exploratory train-only wing predictions vs reconstructed whole-aircraft loads",
        "selected_flights": selection,
        "common_scored_rows": int(common.sum()),
        "baseline_scored_rows": int(results["zero"].score_valid.sum()),
        "coverage": {k: int(v.score_valid.sum()) for k, v in results.items()},
        "failures": {k: v["failures"] for k, v in diagnostics.items()},
        "equal_flight_rmse": comparison.reset_index().to_dict("records"),
        "ranking_definition": "mean of Fx/Fz/My equal-flight RMSE divided by zero-twist RMSE; exploratory only",
        "normalized_rmse_score": score.to_dict(),
        "best_screen_candidate": ordered.index[0],
        "source_replay_directory": str((replay_dir or output).resolve()),
        "artifact_sha256": {p.name: sha256(p) for p in (replay_dir or output).glob("*.parquet")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_screen(results, tables_all, ordered.index[:2].tolist(), output, common)
    images = "".join(
        f'<h2>{html.escape(s["log_id"])}</h2><img src="flight_{i}.png" width="1100">' for i, s in enumerate(selection)
    )
    (output / "report.html").write_text(
        '<meta charset="utf-8"><title>ULLT prescribed twist screen</title>'
        "<h1>Three-flight exploratory twist comparison</h1><p>"
        + html.escape(summary["scope"])
        + "</p><p>All metrics use identical rows, equal flight weights; no held-out validation or promotion.</p>"
        + comparison.round(3).to_html()
        + "<p>Full candidate, cycle and transition metrics are in adjacent CSV files.</p>"
        + images
    )
    return summary


def _plot_screen(results: dict, tables: dict, best: list[str], output: Path, common: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = json.loads((output / "design.json").read_text())["flights"]
    for i, item in enumerate(order):
        log = item["log_id"]
        fig, panels = plt.subplots(3, 3, figsize=(16, 9), constrained_layout=True)
        base = results["zero"]
        scored = base.loc[common & (base.log_id == log)]
        # Show first 1.5 s from each selected window, preserving the time gap.
        windows = [g[g.time_s < g.time_s.min() + 1.5] for _, g in scored.groupby("run_id")]
        for row, axis in enumerate(("fx_b", "fz_b", "my_b")):
            for name, prefix, color in (
                ("log", "", "black"),
                ("delaurier", "delaurier_", "#d97706"),
                ("zero", "ullt_", "#2563eb"),
                (best[0], "ullt_", "#16a34a"),
                (best[1], "ullt_", "#a855f7"),
            ):
                source = results.get(name, base)
                for j, window in enumerate(windows):
                    panels[row, j].plot(
                        window.time_s,
                        source.loc[window.index, prefix + axis],
                        color=color,
                        label=name if j == 0 else None,
                        linewidth=1,
                    )
                phase = tables.get(name, tables["zero"])["phase_bins"]
                phase_source = name if name in ("log", "delaurier") else "ullt"
                p = phase[(phase.log_id == log) & (phase.axis == axis) & (phase.source == phase_source)]
                panels[row, 2].plot(np.degrees(p.phase_rad), p["mean"], color=color, label=name)
            for panel in panels[row]:
                panel.set_ylabel(axis + (" [N m]" if axis == "my_b" else " [N]"))
                panel.grid(alpha=0.25)
        panels[0, 0].legend(fontsize=7)
        for j in range(2):
            panels[0, j].set_title(f"Selected window {j + 1}: first 1.5 scored seconds")
            panels[2, j].set_xlabel("Time [s]")
        panels[0, 2].set_title("Mean phase waveform from all scored rows")
        panels[2, 2].set_xlabel("Mechanical phase [deg]")
        fig.suptitle(log + " — wings vs whole-aircraft log")
        fig.savefig(output / f"flight_{i}.png", dpi=150)
        plt.close(fig)

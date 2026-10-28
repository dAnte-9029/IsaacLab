"""Reproducible figures and signed-peak diagnostics for the fixed nonlinear screen."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flapping_bot.config.nonlinear_section import flat_plate_prior
from flapping_bot.physics.flat_plate_section import flat_plate_polar
from .ullt_section_attribution import weighted_quantile

AXES = ("fx_b", "fz_b", "my_b")
NAMES = ("zero", "linear_twist", "early", "late")


def write_nonlinear_figures(output: Path) -> None:
    """Plot static priors, phase/span physical flow and exact component wrenches."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    images = []
    alpha = torch.linspace(-np.pi / 2, np.pi / 2, 721, dtype=torch.float64)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(
        np.rad2deg(alpha), 2 * np.pi * alpha.sin(), label="original normalwash equivalent", color="gray", ls="--"
    )
    for name in ("early", "late"):
        polar = flat_plate_polar(alpha, flat_plate_prior(name))
        axs[0].plot(np.rad2deg(alpha), polar.cl, label=name)
        axs[1].plot(np.rad2deg(alpha), polar.cd, label=name)
    for ax, label in zip(axs, ("Cl", "Cd")):
        ax.set(xlabel="effective incidence (deg)", ylabel=label)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Uncalibrated section hypotheses; no dynamic-stall claim")
    fig.tight_layout()
    fig.savefig(figures / "polars.png", dpi=160)
    plt.close(fig)
    images.append("polars.png")
    frames = {}
    runtime, stats, peaks, harmonic_rows = [], [], [], []
    distribution, distribution_axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in NAMES:
        if not (output / f"{name}_metrics.json").exists():
            continue
        frame = pd.read_parquet(output / f"{name}.parquet")
        frames[name] = frame
        cycles = pd.read_csv(output / f"{name}_cycles.csv")
        for model in ("ullt", "delaurier") if name == "zero" else ("ullt",):
            predicted = cycles[cycles.source.eq(model)]
            observed = cycles[cycles.source.eq("log")][["run_id", "cycle_id", "axis", "h1_amplitude"]]
            paired = predicted.merge(observed, on=["run_id", "cycle_id", "axis"], suffixes=("", "_log"))
            for (log, axis), group in paired.groupby(["log_id", "axis"]):
                harmonic_rows.append(
                    dict(
                        candidate=name if model == "ullt" else "delaurier",
                        log_id=log,
                        axis=axis,
                        cycles=len(group),
                        identifiable_h1_phase_cycles=int(group.h1_phase_error_deg.notna().sum()),
                        h1_amplitude_mae=float((group.h1_amplitude - group.h1_amplitude_log).abs().mean()),
                    )
                )
        keys = cycles[["run_id", "cycle_id"]].drop_duplicates()
        good = frame[frame.score_valid].merge(keys, on=["run_id", "cycle_id"], how="inner")
        for (log, run, cycle), group in good.groupby(["log_id", "run_id", "cycle_id"]):
            for axis in AXES:
                for model in ("ullt", "delaurier") if name == "zero" else ("ullt",):
                    ref = group[axis]
                    pred = group[f"{model}_{axis}"]
                    peaks.append(
                        dict(
                            candidate=name if model == "ullt" else "delaurier",
                            log_id=log,
                            run_id=run,
                            cycle_id=cycle,
                            axis=axis,
                            reference_max=ref.max(),
                            reference_min=ref.min(),
                            prediction_max=pred.max(),
                            prediction_min=pred.min(),
                            maximum_error=pred.max() - ref.max(),
                            minimum_error=pred.min() - ref.min(),
                        )
                    )
        if name == "zero":
            continue
        diag = json.loads((output / f"{name}.json").read_text())
        runtime.append(
            dict(
                candidate=name,
                wall_seconds=diag["wall_seconds_including_observer"],
                closure_max=diag["section_closure_max"],
                backend_seconds=diag["backend_compute_seconds_including_rk_kinematics"],
                **diag["solver_statistics"],
            )
        )
        sections = pd.read_parquet(output / f"{name}_sections.parquet")
        weights = sections.reference_area_m2.to_numpy()
        angle = np.rad2deg(sections.effective_incidence_rad.abs().to_numpy())
        speed = sections.effective_to_incident_speed_ratio.to_numpy()
        bins = np.arange(0, 95, 5)
        histogram, _ = np.histogram(angle, bins=bins, weights=weights / weights.sum())
        distribution_axes[0].step((bins[:-1] + bins[1:]) / 2, histogram * 100, where="mid", label=name)
        area = sections.assign(high_area=weights * (angle > 25)).groupby("phase_bin")
        share = area.high_area.sum() / area.reference_area_m2.sum()
        distribution_axes[1].plot((share.index + 0.5) * 10, share * 100, label=name)
        row = dict(
            candidate=name,
            effective_angle_area_p50=weighted_quantile(angle, weights, 0.5),
            effective_angle_area_p95=weighted_quantile(angle, weights, 0.95),
            speed_ratio_area_p05=weighted_quantile(speed, weights, 0.05),
            speed_ratio_area_p95=weighted_quantile(speed, weights, 0.95),
            speed_ratio_min=speed.min(),
            speed_ratio_max=speed.max(),
            residual_max=sections.nonlinear_residual_mps.max(),
            iterations_max=sections.nonlinear_iterations.max(),
        )
        for threshold in (25, 35):
            mask = angle > threshold
            row[f"effective_gt{threshold}_area_share"] = float(weights[mask].sum() / weights.sum())
            for axis in AXES:
                load = sections[f"total_{axis}"].abs()
                row[f"effective_gt{threshold}_{axis}_absolute_share"] = float(load[mask].sum() / load.sum())
        for component in ("circulatory", "added_mass", "profile_drag"):
            values = sections.groupby("row_id")[[f"{component}_{a}" for a in AXES]].sum()
            for axis in AXES:
                row[f"{component}_{axis}_mean"] = values[f"{component}_{axis}"].mean()
                row[f"{component}_{axis}_rms"] = np.sqrt(np.mean(values[f"{component}_{axis}"] ** 2))
        stats.append(row)
        for log, group in sections.groupby("log_id", sort=True):
            # Keep the two wings separate: section ID is right tip->root, left root->tip.
            quantities = [
                ("effective_incidence_rad", "effective alpha (deg)"),
                ("section_cd", "section Cd"),
                ("total_fx_b", "section Fx (N)"),
                ("total_fz_b", "section Fz (N)"),
                ("total_my_b", "section My (N m)"),
                ("circulation_m2_s", "circulation (m2/s)"),
            ]
            fig, axs = plt.subplots(2, 3, figsize=(15, 7))
            for ax, (column, title) in zip(axs.flat, quantities):
                matrix = group.groupby(["section_id", "phase_bin"])[column].mean().unstack().reindex(columns=range(36))
                values = matrix.to_numpy()
                if column == "effective_incidence_rad":
                    values = np.rad2deg(values)
                bound = max(np.nanmax(abs(values)), 1e-12)
                im = ax.imshow(
                    values,
                    origin="lower",
                    aspect="auto",
                    extent=(0, 360, -0.5, len(matrix) - 0.5),
                    cmap="magma" if column == "section_cd" else "coolwarm",
                    vmin=0 if column == "section_cd" else -bound,
                    vmax=bound,
                )
                ax.axhline(9.5, color="black", lw=0.7)
                ax.set(title=title, xlabel="mechanical phase (deg)", ylabel="section ID")
                fig.colorbar(im, ax=ax)
            fig.suptitle(f"{name} | {log} | right 0-9, left 10-19; signed phase means")
            fig.tight_layout()
            path = f"{name}_{log}_sections.png"
            fig.savefig(figures / path, dpi=150)
            plt.close(fig)
            images.append(path)
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        for ax, axis in zip(axs, AXES):
            for component in ("circulatory", "added_mass", "profile_drag"):
                value = (
                    sections.groupby(["row_id", "phase_bin"])[f"{component}_{axis}"].sum().groupby("phase_bin").mean()
                )
                ax.plot((value.index + 0.5) * 10, value, label=component)
            ax.set(title=axis, xlabel="mechanical phase (deg)")
            ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle(f"{name} | component resultants, pooled phase mean")
        fig.tight_layout()
        path = f"{name}_components.png"
        fig.savefig(figures / path, dpi=150)
        plt.close(fig)
        images.append(path)
    distribution_axes[0].set(xlabel="absolute effective incidence (deg)", ylabel="area-time share per 5 deg bin (%)")
    distribution_axes[1].set(xlabel="mechanical phase (deg)", ylabel="area share |effective alpha| >25 deg (%)")
    for ax in distribution_axes:
        ax.legend()
        ax.grid(alpha=0.2)
    distribution.tight_layout()
    distribution.savefig(figures / "effective_incidence_distribution.png", dpi=160)
    plt.close(distribution)
    images.append("effective_incidence_distribution.png")
    pd.DataFrame(harmonic_rows).to_csv(output / "h1_amplitude_and_phase_counts.csv", index=False)
    pd.DataFrame(runtime).to_csv(output / "runtime.csv", index=False)
    pd.DataFrame(stats).to_csv(output / "section_summary.csv", index=False)
    peak_table = pd.DataFrame(peaks)
    peak_table.to_csv(output / "signed_cycle_peaks.csv", index=False)
    if not peak_table.empty:
        per_flight = peak_table.groupby(["candidate", "log_id", "axis"])[["maximum_error", "minimum_error"]].agg(
            lambda x: x.abs().mean()
        )
        per_flight.to_csv(output / "signed_peak_per_flight.csv")
        per_flight.groupby(["candidate", "axis"]).mean().to_csv(output / "signed_peak_macro.csv")
    if frames:
        reference = frames["zero"]
        for log in reference.log_id.unique():
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            for ax, axis in zip(axs, AXES):
                for name, frame in [("log", reference), ("delaurier", reference), *frames.items()]:
                    group = frame[frame.score_valid & frame.log_id.eq(log)]
                    col = axis if name == "log" else f"delaurier_{axis}" if name == "delaurier" else f"ullt_{axis}"
                    bins = (np.mod(group.mechanical_phase_rad, 2 * np.pi) * 36 / (2 * np.pi)).astype(int)
                    value = group[col].groupby(bins).mean()
                    ax.plot((value.index + 0.5) * 10, value, label=name)
                ax.set(title=axis, xlabel="mechanical phase (deg)")
                ax.grid(alpha=0.2)
            axs[0].legend(fontsize=8)
            fig.suptitle(f"{log} | log = whole aircraft; predictions = wings")
            fig.tight_layout()
            path = f"{log}_comparison.png"
            fig.savefig(figures / path, dpi=150)
            plt.close(fig)
            images.append(path)
    comparison = pd.read_csv(output / "comparison.csv")
    failure_text = "\n".join(f"{p.name}: {p.read_text()}" for p in output.glob("*failure.json"))
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>ULLT nonlinear version A</title>'
        "<style>body{font:15px sans-serif;max-width:1400px;margin:30px auto}img{max-width:100%}td,th{padding:5px}</style>"
        "<h1>ULLT version A: fixed-prior training diagnostics</h1>"
        "<p>Twist fixed at (35°, -0.5, 0°). Wing-only predictions versus reconstructed whole-aircraft loads. "
        "No fitting, no held-out validation, no dynamic-stall claim. Only complete expected masks are scored. "
        "Phase errors are amplitude-gated by the existing report. Static closure can have multiple roots.</p>"
        + "<pre>"
        + html.escape(failure_text)
        + "</pre>"
        + comparison.to_html(index=False, float_format=lambda x: f"{x:.4f}")
        + "<h2>Timing (same observer and replay)</h2>"
        + pd.DataFrame(runtime).to_html(index=False)
        + "".join(f'<h2>{html.escape(p)}</h2><img src="figures/{p}">' for p in images)
    )

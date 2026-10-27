"""Paired three-way flight diagnostics, with whole-aircraft/wing scope labels."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .ullt_flight_replay import AXES, FlightReplayConfig

SOURCES = ("log", "delaurier", "ullt")
LABELS = {
    "log": "Log: reconstructed whole aircraft",
    "delaurier": "DeLaurier: wings only",
    "ullt": "ULLT-4: wings only",
}
COLORS = {"log": "#222222", "delaurier": "#d97706", "ullt": "#2563eb"}


def source_values(frame: pd.DataFrame, source: str, axis: str) -> np.ndarray:
    """Return an unchanged label or corresponding keyed wing-model column."""
    return frame[axis if source == "log" else f"{source}_{axis}"].to_numpy(float)


def harmonic_features(phase: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    """Fit constant plus first two harmonics; phase is maximum of each harmonic.

    A*cos(k*phase)+B*sin(k*phase) peaks at atan2(B,A)/k. The reported
    `harmonic_phase` is atan2(B,A), so its difference wraps at 2*pi for both k.
    This decomposition describes a waveform; it does not calibrate either model.
    """
    design = np.column_stack([np.ones(len(phase)), np.cos(phase), np.sin(phase), np.cos(2 * phase), np.sin(2 * phase)])
    fit = np.linalg.lstsq(design, signal, rcond=None)[0]
    return {
        "mean": float(np.mean(signal)),
        "peak_to_peak": float(np.ptp(signal)),
        "h1_amplitude": float(np.hypot(fit[1], fit[2])),
        "h2_amplitude": float(np.hypot(fit[3], fit[4])),
        "h1_phase_rad": float(np.arctan2(fit[2], fit[1])),
        "h2_phase_rad": float(np.arctan2(fit[4], fit[3])),
    }


def paired_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Physical-unit error; do not invent correlation for a constant channel."""
    error = prediction - reference
    correlation = (
        float(np.corrcoef(reference, prediction)[0, 1])
        if np.std(reference) > 1e-10 and np.std(prediction) > 1e-10
        else np.nan
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs(error))),
        "bias": float(np.mean(error)),
        "correlation": correlation,
    }


def compute_flight_tables(frame: pd.DataFrame, config: FlightReplayConfig) -> dict[str, pd.DataFrame]:
    """Score identical rows and complete cycles; all phase thresholds are fixed."""
    good = frame.loc[frame.score_valid].copy()
    if good.empty:
        raise ValueError("No scored rows after warmup")
    flights, cycles, rejected, phases, transitions = [], [], [], [], []
    for (partition, log), group in good.groupby(["partition", "log_id"], sort=True):
        for axis in AXES:
            y = source_values(group, "log", axis)
            for model in ("delaurier", "ullt"):
                flights.append({
                    "partition": partition,
                    "log_id": log,
                    "axis": axis,
                    "model": model,
                    "rows": len(group),
                    **paired_metrics(y, source_values(group, model, axis)),
                })
        bins = (np.mod(group.mechanical_phase_rad.to_numpy(), 2 * np.pi) * config.phase_bins / (2 * np.pi)).astype(int)
        for b in range(config.phase_bins):
            subset = group.iloc[np.flatnonzero(bins == b)]
            if subset.empty:
                continue
            for axis in AXES:
                for source in SOURCES:
                    y = source_values(subset, source, axis)
                    phases.append({
                        "partition": partition,
                        "log_id": log,
                        "phase_bin": b,
                        "phase_rad": (b + 0.5) * 2 * np.pi / config.phase_bins,
                        "axis": axis,
                        "source": source,
                        "mean": float(y.mean()),
                        "std": float(y.std()),
                        "rows": len(y),
                    })
    full_counts = frame.groupby(["log_id", "segment_id", "cycle_id"]).size().to_dict()
    for (partition, log, run, cycle), group in good.groupby(["partition", "log_id", "run_id", "cycle_id"], sort=False):
        phase = np.mod(group.mechanical_phase_rad.to_numpy(), 2 * np.pi)
        sorted_phase = np.sort(phase)
        circular_gap = np.max(np.diff(np.r_[sorted_phase, sorted_phase[0] + 2 * np.pi]))
        key = {"partition": partition, "log_id": log, "run_id": run, "cycle_id": cycle}
        # Whole canonical cycle must survive warmup/rejection; no boundary fragments.
        if len(group) < 12 or len(group) != full_counts[(log, group.segment_id.iloc[0], cycle)] or circular_gap > 0.65:
            rejected.append({
                **key,
                "rows": len(group),
                "maximum_phase_gap_rad": circular_gap,
                "reason": "incomplete_or_sparse_cycle",
            })
            continue
        for axis in AXES:
            features = {source: harmonic_features(phase, source_values(group, source, axis)) for source in SOURCES}
            for source, values in features.items():
                row = {
                    **key,
                    "time_s": float(group.time_s.mean()),
                    "frequency_hz": float(group.flap_frequency_hz.mean()),
                    "axis": axis,
                    "source": source,
                    **values,
                }
                for h in (1, 2):
                    delta = values[f"h{h}_phase_rad"] - features["log"][f"h{h}_phase_rad"]
                    # Phase is unidentifiable when either harmonic is negligible.
                    scale = max(features["log"]["peak_to_peak"], 1e-9)
                    reliable = min(values[f"h{h}_amplitude"], features["log"][f"h{h}_amplitude"]) > 0.01 * scale
                    row[f"h{h}_phase_error_deg"] = (
                        float(np.degrees(np.arctan2(np.sin(delta), np.cos(delta))) / h) if reliable else np.nan
                    )
                row["mean_error"] = values["mean"] - features["log"]["mean"]
                row["peak_to_peak_error"] = values["peak_to_peak"] - features["log"]["peak_to_peak"]
                cycles.append(row)
    # Frequency events are chosen from inputs, never from a model residual.
    for run, group in good.groupby("run_id", sort=False):
        if len(group) < 501:
            continue
        freq = group.flap_frequency_hz.to_numpy()
        smooth = pd.Series(freq).rolling(101, center=True, min_periods=101).mean().to_numpy()
        change = np.full(len(group), np.nan)
        change[50:-50] = smooth[100:] - smooth[:-100]
        candidates, _ = find_peaks(np.nan_to_num(abs(change)), height=config.transition_change_hz, distance=400)
        candidates = [i for i in candidates if 200 <= i < len(group) - 200]
        candidates = sorted(candidates, key=lambda i: (-abs(change[i]), i))[:3]
        for i in candidates:
            window = group.iloc[i - 200 : i + 201]
            event_id = f"run{int(run)}_row{i}"
            for axis in AXES:
                for model in ("delaurier", "ullt"):
                    transitions.append({
                        "event_id": event_id,
                        "partition": group.partition.iloc[0],
                        "log_id": group.log_id.iloc[0],
                        "run_id": run,
                        "center_time_s": float(group.time_s.iloc[i]),
                        "frequency_change_hz": float(change[i]),
                        "axis": axis,
                        "model": model,
                        **paired_metrics(source_values(window, "log", axis), source_values(window, model, axis)),
                    })
    return {
        "per_flight": pd.DataFrame(flights),
        "cycles": pd.DataFrame(cycles),
        "cycle_rejections": pd.DataFrame(rejected),
        "phase_bins": pd.DataFrame(phases),
        "transitions": pd.DataFrame(transitions),
    }


def summarize_tables(tables: dict[str, pd.DataFrame], frame: pd.DataFrame) -> dict:
    """Flight-equal metrics, paired gains and whole-flight bootstrap intervals."""
    metrics = []
    rng = np.random.default_rng(20261027)
    for (partition, axis), group in tables["per_flight"].groupby(["partition", "axis"]):
        pivot = group.pivot(index="log_id", columns="model", values="rmse")
        differences = (pivot.delaurier - pivot.ullt).to_numpy()
        boot = np.mean(rng.choice(differences, (2000, len(differences)), replace=True), axis=1)
        row = {
            "partition": partition,
            "axis": axis,
            "flights": len(pivot),
            "delaurier_rmse_macro": float(pivot.delaurier.mean()),
            "ullt_rmse_macro": float(pivot.ullt.mean()),
            "ullt_lower_rmse_flights": int((differences > 0).sum()),
            "delaurier_minus_ullt_rmse_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
        sub = tables["cycles"]
        sub = sub[(sub.partition == partition) & (sub.axis == axis)]
        for model in ("delaurier", "ullt"):
            per_log = sub[sub.source == model].groupby("log_id")
            row[f"{model}_cycle_mean_rmse_macro"] = float(
                per_log.mean_error.apply(lambda x: np.sqrt(np.mean(x**2))).mean()
            )
            row[f"{model}_cycle_amplitude_mae_macro"] = float(
                per_log.peak_to_peak_error.apply(lambda x: np.mean(abs(x))).mean()
            )
            for h in (1, 2):
                row[f"{model}_h{h}_phase_mae_deg_macro"] = float(
                    per_log[f"h{h}_phase_error_deg"].apply(lambda x: x.abs().mean()).mean()
                )
        metrics.append(row)
    transition_macro = []
    if not tables["transitions"].empty:
        per_flight = tables["transitions"].groupby(["partition", "log_id", "model", "axis"]).rmse.mean().reset_index()
        for (partition, model, axis), group in per_flight.groupby(["partition", "model", "axis"]):
            transition_macro.append({
                "partition": partition,
                "model": model,
                "axis": axis,
                "flights": len(group),
                "rmse_macro": float(group.rmse.mean()),
            })
    good = frame[frame.score_valid]
    return {
        "schema_version": "ullt_flight_comparison_v1",
        "metrics": metrics,
        "transition_metrics": transition_macro,
        "scored_rows": len(good),
        "scored_flights": int(good.log_id.nunique()),
        "complete_cycles": int(tables["cycles"][["run_id", "cycle_id"]].drop_duplicates().shape[0]),
        "transition_events": int(tables["transitions"].event_id.nunique()) if not tables["transitions"].empty else 0,
        "local_incidence_deg_quantiles": good.ullt_max_incidence_deg.quantile([0.5, 0.95, 1]).to_dict(),
        "phase_rate_difference_rad_s_rms": float(np.sqrt(np.mean(good.phase_rate_difference_rad_s**2))),
        "claim": (
            "wing-only predictions compared with reconstructed whole-aircraft loads; no isolated-wing ground truth"
        ),
        "scope": "offline prescribed-motion consistency, not closed-loop or independent acceleration prediction",
    }


def write_flight_report(frame: pd.DataFrame, tables: dict[str, pd.DataFrame], summary: dict, output: Path) -> None:
    """Export scientific figures and a local HTML index without external assets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir()
    links = []
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    # One phase figure and one representative contiguous time window per flight.
    for number, (log, group) in enumerate(frame[frame.score_valid].groupby("log_id", sort=True)):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        phase_table = tables["phase_bins"]
        phase_table = phase_table[phase_table.log_id == log]
        for axis, ax in zip(AXES, axes.flat):
            for source in SOURCES:
                row = phase_table[(phase_table.axis == axis) & (phase_table.source == source)].sort_values("phase_rad")
                ax.plot(row.phase_rad, row["mean"], color=COLORS[source], label=LABELS[source])
            ax.set(title=axis, xlabel="mechanical phase (rad)", ylabel="N" if axis.startswith("f") else "N m")
            ax.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{log} | phase averages | whole aircraft vs wings only")
        path = f"figures/{number:02d}_phase.png"
        fig.savefig(output / path, dpi=130)
        plt.close(fig)
        links.append((log + " — phase", path))
        longest = max((g for _, g in group.groupby("run_id")), key=len)
        window = longest.iloc[: min(400, len(longest))]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        for axis, ax in zip(AXES, axes.flat):
            for source in SOURCES:
                ax.plot(window.time_s, source_values(window, source, axis), color=COLORS[source], label=LABELS[source])
            ax.set(title=axis, xlabel="log time (s)", ylabel="N" if axis.startswith("f") else "N m")
            ax.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{log} | first 4 s of longest scored run (input-selected)")
        path = f"figures/{number:02d}_trace.png"
        fig.savefig(output / path, dpi=130)
        plt.close(fig)
        links.append((log + " — time", path))
    # Cross-flight error distribution; equal weight per flight, all six axes.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, ax in zip(AXES, axes.flat):
        part = tables["per_flight"]
        part = part[part.axis == axis]
        pivot = part.pivot(index="log_id", columns="model", values="rmse")
        for model in ("delaurier", "ullt"):
            ax.plot(np.arange(len(pivot)), pivot[model], "o-", label=LABELS[model], color=COLORS[model], markersize=3)
        ax.set(
            title=axis + " RMSE vs log",
            xlabel="flight index (alphabetical)",
            ylabel="N" if axis.startswith("f") else "N m",
        )
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(figure_dir / "cross_flight.png", dpi=140)
    plt.close(fig)
    links.insert(0, ("Cross-flight stability", "figures/cross_flight.png"))
    # Cycle means (all flights), three sources, sorted within each flight.
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
    for axis, ax in zip(("fx_b", "fz_b", "my_b"), axes):
        cy = tables["cycles"]
        cy = cy[cy.axis == axis]
        for source in SOURCES:
            row = cy[cy.source == source].sort_values(["log_id", "time_s"])
            ax.plot(np.arange(len(row)), row["mean"], ".", ms=1.5, color=COLORS[source], label=LABELS[source])
        ax.set(
            title=axis + " cycle means",
            xlabel="cycle index (ordered by flight)",
            ylabel="N" if axis.startswith("f") else "N m",
        )
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(figure_dir / "cycle_means.png", dpi=140)
    plt.close(fig)
    links.insert(1, ("Cycle means", "figures/cycle_means.png"))
    if not tables["transitions"].empty:
        events = (
            tables["transitions"]
            .drop_duplicates("event_id")
            .sort_values("frequency_change_hz", key=abs, ascending=False)
        )
        for number, event in enumerate(events.head(12).itertuples()):
            win = frame[
                (frame.run_id == event.run_id)
                & frame.score_valid
                & (abs(frame.time_s - event.center_time_s) <= 2.00001)
            ]
            fig, axes = plt.subplots(4, 1, figsize=(12, 9), constrained_layout=True)
            axes[0].plot(win.time_s - event.center_time_s, win.flap_frequency_hz, color="purple")
            axes[0].set(ylabel="frequency (Hz)", title=f"{event.log_id} | input-selected transition")
            for axis, ax in zip(("fx_b", "fz_b", "my_b"), axes[1:]):
                for source in SOURCES:
                    ax.plot(
                        win.time_s - event.center_time_s,
                        source_values(win, source, axis),
                        color=COLORS[source],
                        label=LABELS[source],
                    )
                ax.set(ylabel=axis, xlabel="relative time (s)")
                ax.grid(alpha=0.25)
            axes[1].legend(fontsize=8)
            path = f"figures/transition_{number:02d}.png"
            fig.savefig(output / path, dpi=130)
            plt.close(fig)
            links.append((event.event_id + " — frequency transition", path))
    metric_table = pd.DataFrame(summary["metrics"]).to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0)
    transition_table = pd.DataFrame(summary["transition_metrics"]).to_html(
        index=False, float_format=lambda x: f"{x:.4g}", border=0
    )
    images = "\n".join(
        f'<details><summary>{html.escape(title)}</summary><a href="{path}"><img loading="lazy" src="{path}"'
        ' style="width:100%;max-width:1400px"></a></details>'
        for title, path in links
    )
    page = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>ULLT 实飞三方比较</title><style>body{font:16px'
        " sans-serif;max-width:1500px;margin:30px auto;padding:0"
        " 20px;color:#172033}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px"
        " solid #ddd}details{margin:20px 0}summary{cursor:pointer}"
        " .scroll{overflow:auto}</style><h1>实飞载荷三方比较</h1><p>黑色：日志反算整机非重力载荷；橙色：DeLaurier"
        " 机翼；蓝色：ULLT-4 机翼。没有尾翼/机身扣除或参数拟合。</p><p>仅已注册四月 train/validation 飞行；封存 test"
        f' 未读取。误差不是纯机翼气动误差；结果不代表自由飞行动力学验证。</p><p>计分行 {summary["scored_rows"]}；飞行'
        f' {summary["scored_flights"]}；完整周期 {summary["complete_cycles"]}；变频事件'
        f' {summary["transition_events"]}。</p><p>RMSE 按飞行等权。正的 DeLaurier−ULLT 差值表示 ULLT 更接近日志。'
        "相位误差仅对可辨识谐波报告。详见 manifest.json 和 CSV。</p><div"
        f' class="scroll">{metric_table}</div><h2>变频过渡（按飞行等权）</h2>{transition_table}{images}</html>'
    )
    (output / "report.html").write_text(page)

"""Scientific figures for phase/span incidence and diagnostic load attribution."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from .ullt_flight_replay import AXES


def write_section_report(tables: dict[str, pd.DataFrame], output: Path) -> None:
    """Export standalone figures and a local HTML report; never alter replay data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = tables["area_fractions_macro"].candidate.unique()
    figure_names = []
    for candidate in candidates:
        data = tables["phase_span_macro"].query("candidate == @candidate")
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for row, wing in enumerate(("right", "left")):
            subset = data.query("wing == @wing")
            for col, (field, title) in enumerate(
                (
                    ("alpha_deg", "Mean signed incidence [deg]"),
                    ("abs_alpha_deg", "Mean absolute incidence [deg]"),
                    ("over35_probability", "Fraction of samples with |alpha| > 35 deg"),
                )
            ):
                matrix = subset.pivot(index="span_fraction", columns="phase_bin", values=field).sort_index()
                cmap = "RdBu_r" if field == "alpha_deg" else "viridis"
                kwargs = (
                    {"vmin": -70, "vmax": 70}
                    if field == "alpha_deg"
                    else ({"vmin": 0, "vmax": 1} if field == "over35_probability" else {"vmin": 0, "vmax": 70})
                )
                im = axes[row, col].imshow(
                    matrix, origin="lower", aspect="auto", extent=(0, 360, 0, 1), cmap=cmap, **kwargs
                )
                fig.colorbar(im, ax=axes[row, col])
                axes[row, col].set(title=title, ylabel=wing + " root-to-tip s", xlabel="Mechanical phase [deg]")
        fig.suptitle(candidate + " — equal-flight phase/span incidence")
        name = candidate + "_incidence.png"
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        figure_names.append(name)
        for wing in ("right", "left"):
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
            subset = data.query("wing == @wing")
            for ax, axis in zip(axes.flat, AXES):
                matrix = subset.pivot(index="span_fraction", columns="phase_bin", values="total_" + axis).sort_index()
                limit = max(float(np.nanmax(abs(matrix.to_numpy()))), 1e-12)
                im = ax.imshow(
                    matrix, origin="lower", aspect="auto", extent=(0, 360, 0, 1), cmap="RdBu_r", vmin=-limit, vmax=limit
                )
                fig.colorbar(im, ax=ax)
                ax.set(
                    title=axis + (" [N m per section]" if axis.startswith("m") else " [N per section]"),
                    xlabel="Mechanical phase [deg]",
                    ylabel="Root-to-tip s",
                )
            fig.suptitle(candidate + " — " + wing + " wing: signed section wrench about neutral CG")
            name = candidate + "_" + wing + "_wrench.png"
            fig.savefig(output / name, dpi=150)
            plt.close(fig)
            figure_names.append(name)
    fig, axes = plt.subplots(len(candidates), 3, figsize=(14, 7), constrained_layout=True, squeeze=False)
    for row, candidate in enumerate(candidates):
        area = tables["area_fractions_macro"].query("candidate == @candidate and threshold_deg > 0")
        metrics = tables["threshold_metrics_macro"].query(
            "candidate == @candidate and component == 'total' and threshold_deg > 0"
        )
        for col, axis in enumerate(("fx_b", "fz_b", "my_b")):
            m = metrics.query("axis == @axis")
            axes[row, col].plot(
                area.threshold_deg, 100 * area.mean_area_fraction, "o-", label="Reference area-time fraction"
            )
            axes[row, col].plot(
                m.threshold_deg, 100 * m.absolute_section_load_share, "s-", label="Absolute section-load share"
            )
            axes[row, col].set(
                title=candidate + " / " + axis, xlabel="Threshold |alpha| > [deg]", ylabel="Percent", ylim=(0, 100)
            )
            axes[row, col].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output / "area_vs_load.png", dpi=150)
    plt.close(fig)
    figure_names.insert(0, "area_vs_load.png")
    area = tables["area_fractions_macro"].query("threshold_deg > 0")
    metric = tables["threshold_metrics_macro"].query(
        "axis in ['fx_b','fz_b','my_b'] and component in ['total','circulatory','added_mass','profile_drag']"
    )
    selected = metric[
        [
            "candidate",
            "component",
            "threshold_deg",
            "axis",
            "signed_mean",
            "resultant_rms",
            "absolute_section_load_share",
            "original_rmse",
            "frozen_subtraction_rmse",
        ]
    ]
    images = "".join(
        f'<h2>{html.escape(name)}</h2><img src="{html.escape(name)}" style="max-width:100%">' for name in figure_names
    )
    (output / "report.html").write_text(
        '<meta charset="utf-8"><title>ULLT section attribution</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse}td,th{padding:5px}</style>'
        "<h1>ULLT 翼段迎角与载荷诊断</h1>"
        "<p>同三条训练日志、相同计分行。参考面积来自 CSV；中心空隙不计面积。25/35/45/60° 是描述性阈值，不是失速判据。</p>"
        "<p>只把环量、附加质量和剖面阻力相加。流向环量投影是环量的子项，不能再次相加。当前剖面 Cd=0；per_cd 字段只给导数，不代表真实缺失阻力。</p>"
        "<p>frozen_subtraction_rmse 是保持尾迹和运动学后的代数减项，不是物理删翼或失速修正。减项的误差变化不具可加性，不能解释为因果误差百分比。日志为整机重建载荷，非翼段测力。</p>"
        "<h2>逐飞行等权面积比例（0–1）</h2>"
        + area.round(4).to_html(index=False)
        + "<h2>载荷与减项诊断</h2><p>threshold=-1 表示所有翼段；力 N、力矩 N·m。完整六分量和逐飞行结果见 CSV。</p>"
        + selected.round(4).to_html(index=False)
        + images
    )

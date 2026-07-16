"""Matplotlib figures for the WT0 tail unit audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "fixed_horizontal": "#4C78A8",
    "left_elevon": "#F58518",
    "right_elevon": "#54A24B",
    "fixed_vertical": "#B279A2",
    "rudder": "#E45756",
    "aggregate": "#222222",
}
SURFACES = tuple(COLORS)


def _rows(data: Sequence[dict[str, Any]], surface: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in data if row.get("surface") == surface),
        key=lambda row: (str(row.get("rate_axis", "")), float(row.get("independent_value", 0.0))),
    )


def _xy(
    data: Sequence[dict[str, Any]],
    surface: str,
    x: str,
    y: str,
    *,
    rate_axis: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = _rows(data, surface)
    if rate_axis is not None:
        selected = [row for row in selected if row.get("rate_axis") == rate_axis]
    return (
        np.asarray([float(row[x]) for row in selected], dtype=float),
        np.asarray([float(row[y]) for row in selected], dtype=float),
    )


def _style_axis(ax: plt.Axes, *, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.axhline(0.0, color="0.5", linewidth=0.8)


def _finish(fig: plt.Figure, path: Path, *, title: str, run_id: str, nominal: str) -> None:
    fig.suptitle(f"{title}\nrun={run_id}; {nominal}", fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_geometry(path: Path, geometry: Sequence[dict[str, Any]], run_id: str, nominal: str) -> None:
    fig, (top, side) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    top.set_title("Top view: body x-y and aerodynamic centers")
    side.set_title("Side view: body x-z and aerodynamic centers")
    top_offsets = {
        "fixed_horizontal": (6, 10),
        "left_elevon": (6, 10),
        "right_elevon": (6, 10),
        "fixed_vertical": (6, -18),
        "rudder": (-6, 18),
    }
    side_offsets = {
        "fixed_horizontal": (6, -30),
        "left_elevon": (6, -18),
        "right_elevon": (6, -6),
        "fixed_vertical": (6, 8),
        "rudder": (6, 20),
    }
    for row in geometry:
        name = str(row["surface"])
        x = float(row["position_b_from_origin_m_x"])
        y = float(row["position_b_from_origin_m_y"])
        z = float(row["position_b_from_origin_m_z"])
        top.scatter(x, y, s=80, color=COLORS[name], label=name)
        top.annotate(
            name,
            (x, y),
            xytext=top_offsets[name],
            textcoords="offset points",
            fontsize=8,
            ha="right" if name == "rudder" else "left",
        )
        side.scatter(x, z, s=80, color=COLORS[name], label=name)
        side.annotate(name, (x, z), xytext=side_offsets[name], textcoords="offset points", fontsize=8)
    top.scatter(0.0, 0.0, marker="x", s=80, color="black", label="base origin")
    side.scatter(0.0, 0.0, marker="x", s=80, color="black", label="base origin")
    for ax, ordinate, label in ((top, 1, "+y left"), (side, 2, "+z up")):
        ax.arrow(0.0, 0.0, 0.16, 0.0, width=0.0015, head_width=0.015, color="black", length_includes_head=True)
        ax.text(0.17, 0.0, "+x forward / +Mx", va="center", fontsize=9)
        ax.arrow(0.0, 0.0, 0.0, 0.10, width=0.0015, head_width=0.015, color="black", length_includes_head=True)
        ax.text(0.0, 0.105, f"{label} / +M{'y' if ordinate == 1 else 'z'}", ha="center", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("body x [m]")
    top.set_ylabel("body y [m]")
    side.set_ylabel("body z [m]")
    top.set_xlim(-0.65, 0.22)
    top.set_ylim(-0.20, 0.20)
    side.set_xlim(-0.65, 0.22)
    side.set_ylim(-0.08, 0.16)
    side.set_aspect("auto")
    top.legend(fontsize=7, loc="best")
    side.text(
        0.48,
        0.02,
        "Positive commands:\n"
        "symmetric: L=+delta, R=+delta\n"
        "differential: L=+delta, R=-delta\n"
        "rudder: +delta\n"
        "surface angle = incidence - command (moving surfaces)",
        transform=side.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    _finish(fig, path, title="Figure 1 - Tail geometry and sign convention", run_id=run_id, nominal=nominal)


def _plot_nominal(path: Path, data: Sequence[dict[str, Any]], run_id: str, nominal: str) -> None:
    surface_rows = [row for row in data if row["surface"] != "aggregate"]
    labels = [str(row["surface"]) for row in surface_rows]
    x = np.arange(len(labels))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for index, component in enumerate("xyz"):
        axes[0].bar(
            x + (index - 1) * width,
            [float(row[f"force_b_N_{component}"]) for row in surface_rows],
            width,
            label=f"F{component}",
        )
        axes[1].bar(
            x + (index - 1) * width,
            [float(row[f"moment_b_Nm_{component}"]) for row in surface_rows],
            width,
            label=f"M{component}",
        )
    for ax, ylabel in zip(axes, ("Force [N]", "Moment about COM [N m]")):
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
    axes[0].set_title("Per-surface force")
    axes[1].set_title("Per-surface r x F moment")
    _finish(fig, path, title="Figure 2 - Zero-input surface decomposition", run_id=run_id, nominal=nominal)


def _plot_symmetry(path: Path, data: Sequence[dict[str, Any]], run_id: str, nominal: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    quantities = {
        "force_b_N_x": "Fx",
        "force_b_N_z": "Fz",
        "moment_b_Nm_x": "Mx",
        "moment_b_Nm_y": "My",
        "moment_b_Nm_z": "Mz",
    }
    for reference, linestyle in (("base_origin", "-"), ("configured_com", "--")):
        for quantity, label in quantities.items():
            selected = [row for row in data if row["reference"] == reference and row["quantity"] == quantity]
            axes[0, 0 if quantity.startswith("force") else 1].semilogy(
                [float(row["symmetric_elevon_deg"]) for row in selected],
                np.maximum([float(row["normalized_error"]) for row in selected], 1.0e-16),
                linestyle=linestyle,
                label=f"{label}, {reference}",
            )
    for ax in axes[0]:
        ax.set_xlabel("Symmetric elevon [deg]")
        ax.set_ylabel("Parity normalized error [-]")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    axes[0, 0].set_title("Force parity")
    axes[0, 1].set_title("Moment parity and COM-reference effect")
    for quantity, ax, ylabel in (
        ("force_b_N_z", axes[1, 0], "Fz [N]"),
        ("moment_b_Nm_x", axes[1, 1], "Mx [N m]"),
    ):
        selected = [
            row
            for row in data
            if row["reference"] == "configured_com" and row["quantity"] == quantity
        ]
        x = [float(row["symmetric_elevon_deg"]) for row in selected]
        ax.plot(x, [float(row["left_value"]) for row in selected], label="left")
        ax.plot(x, [float(row["parity_adjusted_right_value"]) for row in selected], "--", label="parity-adjusted right")
        _style_axis(ax, xlabel="Symmetric elevon [deg]", ylabel=ylabel)
        ax.legend()
    _finish(fig, path, title="Figure 3 - Left-right symmetry", run_id=run_id, nominal=nominal)


def _plot_symmetric(path: Path, data: Sequence[dict[str, Any]], run_id: str, nominal: str) -> None:
    aggregate = _rows(data, "aggregate")
    x = np.asarray([float(row["symmetric_elevon_deg"]) for row in aggregate])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(x, [float(row["moment_b_Nm_y"]) for row in aggregate], color=COLORS["aggregate"], label="aggregate My")
    axes[0, 1].plot(x, [float(row["force_b_N_z"]) for row in aggregate], label="Fz")
    axes[0, 1].plot(x, [float(row["force_b_N_x"]) for row in aggregate], label="Fx")
    axes[1, 0].plot(x, [float(row["moment_b_Nm_x"]) for row in aggregate], label="Mx")
    axes[1, 0].plot(x, [float(row["moment_b_Nm_z"]) for row in aggregate], label="Mz")
    for surface in ("left_elevon", "right_elevon"):
        sx, alpha = _xy(data, surface, "symmetric_elevon_deg", "alpha_deg")
        axes[1, 1].plot(sx, alpha, color=COLORS[surface], label=f"{surface} alpha")
    axes[1, 1].axhline(25.0, color="red", linestyle=":", label="alpha clip")
    axes[1, 1].axhline(-25.0, color="red", linestyle=":")
    labels = (("My [N m]",), ("Force [N]",), ("Non-target moment [N m]",), ("Local alpha [deg]",))
    for ax, label in zip(axes.flat, labels):
        _style_axis(ax, xlabel="Symmetric elevon [deg]", ylabel=label[0])
        ax.legend(fontsize=8)
    _finish(fig, path, title="Figure 4 - Symmetric elevon sweep", run_id=run_id, nominal=nominal)


def _plot_differential(path: Path, data: Sequence[dict[str, Any]], run_id: str, nominal: str) -> None:
    aggregate = _rows(data, "aggregate")
    x = np.asarray([float(row["differential_elevon_deg"]) for row in aggregate])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(x, [float(row["moment_b_Nm_x"]) for row in aggregate], label="aggregate Mx")
    axes[0, 1].plot(x, [float(row["moment_b_Nm_y"]) for row in aggregate], label="My")
    axes[0, 1].plot(x, [float(row["moment_b_Nm_z"]) for row in aggregate], label="Mz")
    for surface in ("left_elevon", "right_elevon"):
        sx, sy = _xy(data, surface, "differential_elevon_deg", "moment_b_Nm_x")
        axes[1, 0].plot(sx, sy, color=COLORS[surface], label=f"{surface} Mx")
        sx, fz = _xy(data, surface, "differential_elevon_deg", "force_b_N_z")
        axes[1, 1].plot(sx, fz, color=COLORS[surface], label=f"{surface} Fz")
    for ax, ylabel in zip(axes.flat, ("Mx [N m]", "Coupled moment [N m]", "Surface Mx [N m]", "Surface Fz [N]")):
        _style_axis(ax, xlabel="Differential elevon [deg]", ylabel=ylabel)
        ax.legend(fontsize=8)
    _finish(fig, path, title="Figure 5 - Differential elevon sweep", run_id=run_id, nominal=nominal)


def _plot_rudder(path: Path, data: Sequence[dict[str, Any]], run_id: str, nominal: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for surface in ("fixed_vertical", "rudder", "aggregate"):
        x, fy = _xy(data, surface, "rudder_deg", "force_b_N_y")
        axes[0, 0].plot(x, fy, color=COLORS[surface], label=f"{surface} Fy")
        x, mz = _xy(data, surface, "rudder_deg", "moment_b_Nm_z")
        axes[0, 1].plot(x, mz, color=COLORS[surface], label=f"{surface} Mz")
    aggregate = _rows(data, "aggregate")
    x = [float(row["rudder_deg"]) for row in aggregate]
    for component in "xz":
        axes[1, 0].plot(x, [float(row[f"force_b_N_{component}"]) for row in aggregate], label=f"F{component}")
    for component in "xy":
        axes[1, 0].plot(x, [float(row[f"moment_b_Nm_{component}"]) for row in aggregate], linestyle="--", label=f"M{component}")
    rudder = _rows(data, "rudder")
    axes[1, 1].plot(x, [float(row["alpha_deg"]) for row in rudder], label="rudder alpha")
    axes[1, 1].axhline(25.0, color="red", linestyle=":", label="alpha clip")
    axes[1, 1].axhline(-25.0, color="red", linestyle=":")
    for ax, ylabel in zip(axes.flat, ("Fy [N]", "Mz [N m]", "Side effects [mixed units]", "Rudder alpha [deg]")):
        _style_axis(ax, xlabel="Rudder deflection [deg]", ylabel=ylabel)
        ax.legend(fontsize=8)
    _finish(fig, path, title="Figure 6 - Rudder sweep", run_id=run_id, nominal=nominal)


def _plot_airspeed(path: Path, data: Sequence[dict[str, Any]], run_id: str, nominal: str) -> None:
    aggregate = _rows(data, "aggregate")
    u = np.asarray([float(row["airspeed_mps"]) for row in aggregate])
    force = np.asarray([float(row["force_norm_N"]) for row in aggregate])
    moment = np.asarray([float(row["moment_norm_Nm"]) for row in aggregate])
    positive = u > 0.0
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(u, force, "o-", label="|Ftail|")
    axes[0, 1].plot(u, moment, "o-", label="|Mtail|")
    axes[1, 0].plot(u[positive], force[positive] / u[positive] ** 2, "o-", label="|F|/U^2")
    axes[1, 1].plot(u[positive], moment[positive] / u[positive] ** 2, "o-", label="|M|/U^2")
    for ax, ylabel in zip(axes.flat, ("Force [N]", "Moment [N m]", "N s^2/m^2", "N m s^2/m^2")):
        _style_axis(ax, xlabel="Airspeed [m/s]", ylabel=ylabel)
        ax.legend()
    _finish(fig, path, title="Figure 7 - Airspeed scaling", run_id=run_id, nominal=nominal)


def _plot_incidence(path: Path, data: Sequence[dict[str, Any]], run_id: str, nominal: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for surface in ("fixed_horizontal", "left_elevon", "right_elevon"):
        x, alpha = _xy(data, surface, "body_flow_angle_deg", "alpha_deg")
        axes[0, 0].plot(x, alpha, color=COLORS[surface], label=f"{surface} alpha")
        x, cl = _xy(data, surface, "body_flow_angle_deg", "lift_coefficient")
        axes[0, 1].plot(x, cl, color=COLORS[surface], label=f"{surface} CL")
        _, cd = _xy(data, surface, "body_flow_angle_deg", "drag_coefficient")
        axes[0, 1].plot(x, cd, color=COLORS[surface], linestyle="--", label=f"{surface} CD")
    aggregate = _rows(data, "aggregate")
    x = [float(row["body_flow_angle_deg"]) for row in aggregate]
    axes[1, 0].plot(x, [float(row["force_b_N_x"]) for row in aggregate], label="Fx")
    axes[1, 0].plot(x, [float(row["force_b_N_z"]) for row in aggregate], label="Fz")
    axes[1, 1].plot(x, [float(row["moment_b_Nm_y"]) for row in aggregate], label="My")
    axes[0, 0].axhline(25.0, color="red", linestyle=":", label="alpha limits")
    axes[0, 0].axhline(-25.0, color="red", linestyle=":")
    for ax, ylabel in zip(axes.flat, ("Local alpha [deg]", "Coefficient [-]", "Force [N]", "My [N m]")):
        _style_axis(ax, xlabel="Body-flow angle atan2(vz,vx) [deg]", ylabel=ylabel)
        ax.legend(fontsize=7)
    _finish(fig, path, title="Figure 8 - Horizontal-tail incidence sweep", run_id=run_id, nominal=nominal)


def _plot_sideslip(path: Path, data: Sequence[dict[str, Any]], run_id: str, nominal: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for surface in ("fixed_vertical", "rudder", "aggregate"):
        for ax, field, label in zip(
            axes,
            ("force_b_N_y", "moment_b_Nm_z", "force_b_N_x"),
            ("Fy", "Mz", "Fx"),
        ):
            x, y = _xy(data, surface, "sideslip_deg", field)
            ax.plot(x, y, color=COLORS[surface], label=f"{surface} {label}")
    for ax, ylabel in zip(axes, ("Fy [N]", "Mz [N m]", "Fx [N]")):
        _style_axis(ax, xlabel="Sideslip coordinate [deg]", ylabel=ylabel)
        ax.legend(fontsize=7)
    _finish(fig, path, title="Figure 9 - Sideslip sweep", run_id=run_id, nominal=nominal)


def _plot_rates(path: Path, data: Sequence[dict[str, Any]], run_id: str, nominal: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, rate_axis, field, ylabel in zip(
        axes,
        ("q", "p", "r"),
        ("moment_b_Nm_y", "moment_b_Nm_x", "moment_b_Nm_z"),
        ("My [N m]", "Mx [N m]", "Mz [N m]"),
    ):
        x, y = _xy(data, "aggregate", "independent_value", field, rate_axis=rate_axis)
        ax.plot(x, y, label=f"{ylabel.split()[0]} vs {rate_axis}")
        _style_axis(ax, xlabel=f"{rate_axis} [rad/s]", ylabel=ylabel)
        ax.legend()
    _finish(fig, path, title="Figure 10 - Angular-rate damping", run_id=run_id, nominal=nominal)


def _plot_effectiveness(path: Path, summary: dict[str, Any], run_id: str, nominal: str) -> None:
    d = summary["control_derivatives"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    moment_labels = ("dMy/dsym", "dMx/ddiff", "dMz/drud")
    moment_values = (
        d["dMy_d_symmetric_Nm_per_rad"],
        d["dMx_d_differential_Nm_per_rad"],
        d["dMz_d_rudder_Nm_per_rad"],
    )
    force_labels = ("dFz/dsym", "dFx/dsym", "dFy/drud", "dFy/ddiff")
    force_values = (
        d["dFz_d_symmetric_N_per_rad"],
        d["dFx_d_symmetric_N_per_rad"],
        d["dFy_d_rudder_N_per_rad"],
        d["dFy_d_differential_N_per_rad"],
    )
    axes[0].bar(moment_labels, moment_values, color=("#4C78A8", "#F58518", "#E45756"))
    axes[1].bar(force_labels, force_values, color=("#4C78A8", "#72B7B2", "#E45756", "#F58518"))
    for ax, ylabel in zip(axes, ("Moment derivative [N m/rad]", "Force derivative [N/rad]")):
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.tick_params(axis="x", rotation=15)
    _finish(fig, path, title="Figure 11 - Control effectiveness summary", run_id=run_id, nominal=nominal)


def _plot_continuity(
    path: Path,
    symmetric: Sequence[dict[str, Any]],
    airspeed: Sequence[dict[str, Any]],
    run_id: str,
    nominal: str,
) -> None:
    sym = _rows(symmetric, "aggregate")
    x = np.asarray([float(row["symmetric_elevon_deg"]) for row in sym])
    my = np.asarray([float(row["moment_b_Nm_y"]) for row in sym])
    derivative = np.gradient(my, np.deg2rad(x))
    air = _rows(airspeed, "aggregate")
    u = np.asarray([float(row["airspeed_mps"]) for row in air])
    force = np.asarray([float(row["force_norm_N"]) for row in air])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(x, my, label="My")
    axes[0, 0].axvline(-25.0, color="red", linestyle=":", label="elevon alpha clipping starts")
    axes[0, 0].axvline(25.0, color="red", linestyle=":")
    axes[0, 1].plot(x, derivative, label="dMy/d(delta)")
    axes[0, 1].axvline(-25.0, color="red", linestyle=":")
    axes[0, 1].axvline(25.0, color="red", linestyle=":")
    low = u <= 0.5
    axes[1, 0].plot(u[low], force[low], "o-", label="|F| near zero speed")
    positive = (u > 0.0) & (u <= 2.0)
    axes[1, 1].plot(u[positive], force[positive] / u[positive] ** 2, "o-", label="|F|/U^2")
    for ax, xlabel, ylabel in zip(
        axes.flat,
        ("Symmetric elevon [deg]", "Symmetric elevon [deg]", "Airspeed [m/s]", "Airspeed [m/s]"),
        ("My [N m]", "N m/rad", "Force [N]", "N s^2/m^2"),
    ):
        _style_axis(ax, xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=8)
    _finish(fig, path, title="Figure 12 - Continuity diagnostics", run_id=run_id, nominal=nominal)


def generate_tail_audit_figures(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    csv_rows: dict[str, list[dict[str, Any]]],
) -> None:
    """Generate the twelve required PNG figures from serialized audit rows."""

    figures = output_dir / "figures"
    run_id = output_dir.name
    nominal = f"rho={manifest['density_kg_m3']} kg/m^3, U={manifest['nominal_conditions']['airspeed_mps']} m/s"
    _plot_geometry(figures / "figure_01_geometry_sign_convention.png", csv_rows["surface_geometry.csv"], run_id, nominal)
    _plot_nominal(figures / "figure_02_zero_input_decomposition.png", csv_rows["nominal_surface_wrench.csv"], run_id, nominal)
    _plot_symmetry(figures / "figure_03_left_right_symmetry.png", csv_rows["zero_input_symmetry.csv"], run_id, nominal)
    _plot_symmetric(figures / "figure_04_symmetric_elevon.png", csv_rows["symmetric_elevon_sweep.csv"], run_id, nominal)
    _plot_differential(figures / "figure_05_differential_elevon.png", csv_rows["differential_elevon_sweep.csv"], run_id, nominal)
    _plot_rudder(figures / "figure_06_rudder.png", csv_rows["rudder_sweep.csv"], run_id, nominal)
    _plot_airspeed(figures / "figure_07_airspeed_scaling.png", csv_rows["airspeed_sweep.csv"], run_id, nominal)
    _plot_incidence(figures / "figure_08_horizontal_incidence.png", csv_rows["incidence_sweep.csv"], run_id, nominal)
    _plot_sideslip(figures / "figure_09_sideslip.png", csv_rows["sideslip_sweep.csv"], run_id, nominal)
    _plot_rates(figures / "figure_10_angular_rate_damping.png", csv_rows["angular_rate_sweep.csv"], run_id, nominal)
    _plot_effectiveness(figures / "figure_11_control_effectiveness.png", summary, run_id, nominal)
    _plot_continuity(
        figures / "figure_12_continuity.png",
        csv_rows["symmetric_elevon_sweep.csv"],
        csv_rows["airspeed_sweep.csv"],
        run_id,
        nominal,
    )

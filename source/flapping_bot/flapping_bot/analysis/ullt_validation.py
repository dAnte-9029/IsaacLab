"""Prescribed-motion ULLT benchmarks; no controller, Isaac Sim, or training."""

from __future__ import annotations

import math
import numpy as np
import torch
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from flapping_bot.config.ullt import ULLTConfig
from flapping_bot.physics.ullt_geometry import (
    elliptic_geometry,
    load_csv_urdf_geometry,
    opposed_flapping_motion,
    rigid_motion,
    rotation_xyz,
)
from flapping_bot.physics.ullt_state_space import ULLTBackend, ULLTGeometry, ULLTMotion


def paper_motion(
    case: str, geometry: ULLTGeometry
) -> tuple[Callable[[float], tuple[ULLTGeometry, ULLTMotion]], float, Callable[[float], torch.Tensor]]:
    """Sections III.A--D: c_mid=1 m, U=1 m/s, AR=3, radians and seconds.

    Surge/pitch use k_mid=2; heave-pitch uses St=.3 and h/c=.75. The pitching
    axis is quarter chord. Rotation is nose-up positive, opposite global +y.
    Impulsive start omits the singular acceleration impulse at t=0; regular
    t>=0 loads are evaluated with zero post-start acceleration.
    """
    if case not in {"impulsive_start", "surge", "pitch", "heave_pitch"}:
        raise ValueError(f"Unknown paper case {case}")
    omega = 4.0 if case in {"surge", "pitch"} else 0.3 * math.pi / 0.75
    duration = 7.5 if case == "impulsive_start" else 3 * 2 * math.pi / omega
    angle_max = -math.atan(omega * 0.75) + math.radians(15)
    batch = geometry.chord_m.shape[0]

    def values(time: float) -> tuple[float, ...]:
        theta, td, tdd, x, u, ud, z, zd, zdd = math.radians(5), 0.0, 0.0, time, 1.0, 0.0, 0.0, 0.0, 0.0
        if case == "surge":
            x = time + 0.5 / omega * (1 - math.cos(omega * time))
            u, ud = 1 + 0.5 * math.sin(omega * time), 0.5 * omega * math.cos(omega * time)
        if case in {"pitch", "heave_pitch"}:
            amplitude = math.radians(5) if case == "pitch" else angle_max
            theta = amplitude * math.sin(omega * time)
            td = amplitude * omega * math.cos(omega * time)
            tdd = -amplitude * omega**2 * math.sin(omega * time)
        if case == "heave_pitch":
            z, zd, zdd = (
                0.75 * math.cos(omega * time),
                -0.75 * omega * math.sin(omega * time),
                -0.75 * omega**2 * math.cos(omega * time),
            )
        return theta, td, tdd, x, u, ud, z, zd, zdd

    def vector(x: float, y: float, z: float) -> torch.Tensor:
        return geometry.chord_m.new_tensor([[x, y, z]]).repeat(batch, 1)

    def sample(time: float) -> tuple[ULLTGeometry, ULLTMotion]:
        theta, td, tdd, x, u, ud, z, zd, zdd = values(time)
        return rigid_motion(
            geometry,
            rotation_w_from_reference=rotation_xyz(vector(0, -theta, 0)),
            origin_w_m=vector(x, 0, z),
            origin_velocity_w_mps=vector(u, 0, zd),
            origin_acceleration_w_mps2=vector(ud, 0, zdd),
            angular_velocity_w_rad_s=vector(0, -td, 0),
            angular_acceleration_w_rad_s2=vector(0, -tdd, 0),
        )

    def origin(time: float) -> torch.Tensor:
        val = values(time)
        return vector(val[3], 0, val[6])

    return sample, duration, origin


def run_trace(
    case: str, config: ULLTConfig, *, sections: int, steps: int, dtype: torch.dtype = torch.float64
) -> tuple[np.ndarray, dict[str, float]]:
    """Return [t,Fxyz,Mxyz,Gamma_mid] and diagnostics for one prescribed case."""
    base = elliptic_geometry(num_sections=sections, dtype=dtype)
    sample, duration, origin = paper_motion(case, base)
    backend = ULLTBackend(config)
    state = backend.zero_state(base)
    trace = []
    max_incidence, fit_rms = 0.0, 0.0
    for i in range(steps + 1):
        time = duration * i / steps
        g, m = sample(time)
        result = backend.evaluate(g, m, state, origin(time))
        max_incidence = max(max_incidence, float(result.geometric_incidence_rad.abs().max()))
        fit_rms = max(fit_rms, float(result.correction_fit_rms.max()))
        trace.append([
            time,
            *result.force_w_n[0].tolist(),
            *result.moment_about_reference_w_nm[0].tolist(),
            float(result.circulation_m2_s[0, sections // 2]),
        ])
        if i < steps:
            state = backend.advance(state, sample, time, duration / steps)
    array = np.asarray(trace)
    return array, {
        "duration_s": duration,
        "dt_s": duration / steps,
        "maximum_geometric_incidence_deg": math.degrees(max_incidence),
        "correction_fit_rms": fit_rms,
        "peak_abs_force_n": float(np.abs(array[:, 1:4]).max()),
        "peak_abs_moment_nm": float(np.abs(array[:, 4:7]).max()),
        "finite": bool(np.isfinite(array).all()),
    }


def spatial_convergence() -> list[dict[str, float]]:
    """Independent analytical Prandtl elliptic-wing limit at small incidence."""
    rows = []
    exact = 2 * math.pi / (1 + 2 / 3)
    area = math.pi * (3 * math.pi / 4) / 4
    for n in (9, 17, 33, 65, 129):
        base = elliptic_geometry(num_sections=n)
        angle = 1e-4
        zero = base.chord_m.new_zeros((1, 3))
        g, m = rigid_motion(
            base,
            rotation_w_from_reference=rotation_xyz(base.chord_m.new_tensor([[0, -angle, 0]])),
            origin_w_m=zero,
            origin_velocity_w_mps=base.chord_m.new_tensor([[1, 0, 0]]),
            origin_acceleration_w_mps2=zero,
            angular_velocity_w_rad_s=zero,
            angular_acceleration_w_rad_s2=zero,
        )
        backend = ULLTBackend(ULLTConfig(model="quasi_steady"))
        result = backend.evaluate(g, m, backend.zero_state(g), zero)
        slope = float(result.force_w_n[0, 2]) / (0.5 * 1.225 * area * math.sin(angle))
        rows.append({
            "sections": n,
            "lift_slope_per_rad": slope,
            "analytic_lift_slope_per_rad": exact,
            "relative_error": abs(slope - exact) / exact,
        })
    return rows


def run_stage1_validation(
    *,
    sections: int = 9,
    steps: int = 384,
    models: tuple[str, ...] = ("one_state", "four_state", "six_state", "quasi_steady"),
    progress: Callable[[str], None] | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Generate paper cases and convergence evidence, with explicit claim limits."""
    if sections < 3 or sections % 2 == 0 or steps < 24:
        raise ValueError("Use an odd section count >=3 and steps >=24")
    if not models:
        raise ValueError("At least one model is required")
    traces, cases, configurations = {}, {}, {}
    for model in models:
        cfg = ULLTConfig(model=model, added_mass_factor=0.85)
        configurations[model] = asdict(cfg)
        for case in ("impulsive_start", "surge", "pitch", "heave_pitch"):
            key = f"{case}__{model}"
            if progress:
                progress(key)
            trace, diagnostic = run_trace(case, cfg, sections=sections, steps=steps)
            traces[key], cases[key] = trace, diagnostic
    space = spatial_convergence()
    temporal = {}
    for model in models:
        if model == "quasi_steady":
            continue
        cfg = ULLTConfig(model=model, added_mass_factor=0.85)
        coarse = traces[f"pitch__{model}"]
        if progress:
            progress(f"pitch time refinement: {model}")
        fine, _ = run_trace("pitch", cfg, sections=sections, steps=2 * steps)
        errors = {}
        for quantity, columns in (("force", slice(1, 4)), ("moment", slice(4, 7)), ("circulation", slice(7, 8))):
            scale = max(float(np.max(np.abs(fine[:, columns]))), 1e-12)
            errors[f"relative_max_{quantity}_difference"] = (
                float(np.max(np.abs(coarse[:, columns] - fine[::2, columns]))) / scale
            )
        temporal[model] = {"refinement_ratio": 2, **errors, "accepted": all(v < 1e-3 for v in errors.values())}
    accepted = (
        all(row["finite"] for row in cases.values())
        and all(row["accepted"] for row in temporal.values())
        and space[-1]["relative_error"] < 0.005
        and all(a["relative_error"] > b["relative_error"] for a, b in zip(space, space[1:]))
    )
    summary = {
        "schema_version": "ullt_stage1_v1",
        "all_numerical_gates_passed": accepted,
        "backend": "Izraelevitz2017_prescribed_motion",
        "sections": sections,
        "steps": steps,
        "dtype": "float64",
        "device": "cpu",
        "initialization": "zero_wake",
        "configurations": configurations,
        "cases": cases,
        "spatial_convergence": space,
        "time_convergence": temporal,
        "trace_columns": ["time_s", "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm", "Gamma_mid_m2_s"],
        "evidence": {
            "analytical_prandtl_limit": "verified by refinement",
            "paper_sections_III_A_to_D": "prescribed scenarios generated; not digitized UVLM-curve agreement",
            "paper_section_IV_flapping": "not an exact reproduction in this stage",
            "paper_section_V_experiments": "not reproduced; no independent source data loaded",
            "real_wing_validation": False,
            "free_body_coupling": False,
            "rl_training": False,
        },
    }
    return summary, traces


def plot_stage1(traces: dict[str, np.ndarray], output_path: Path) -> None:
    """Write a standalone load-history figure; no recomputation of traces."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    for row, case in enumerate(("impulsive_start", "surge", "pitch", "heave_pitch")):
        for key, trace in traces.items():
            name, model = key.split("__")
            if name != case:
                continue
            axes[row, 0].plot(trace[:, 0], trace[:, 3], label=model)
            axes[row, 1].plot(trace[:, 0], trace[:, 1], label=model)
        for col, label in enumerate(("Fz (N)", "Fx (N), positive forward")):
            axes[row, col].set(title=case, xlabel="time (s)", ylabel=label)
            axes[row, col].grid(alpha=0.3)
        axes[row, 0].legend(fontsize=8)
    figure.suptitle("ULLT prescribed-motion benchmarks (not experimental validation)")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_csv_flapping_checks(csv_path: Path, urdf_path: Path, *, steps: int = 128) -> tuple[dict, dict[str, np.ndarray]]:
    """Numerical prescribed-motion check of confirmed geometry; not force validation."""
    geometry = load_csv_urdf_geometry(csv_path, urdf_path, sections_per_wing=10)
    urdf = ET.parse(urdf_path).getroot()
    origins = geometry.chord_m.new_tensor([[
        [float(v) for v in urdf.find(f"joint[@name='{name}']/origin").attrib["xyz"].split()]
        for name in ("right_wing", "left_wing")
    ]])
    traces, cases = {}, {}
    for model in ("one_state", "four_state"):
        # Infinite-section added mass is a declared prior here; no AR interpolation/calibration.
        backend = ULLTBackend(ULLTConfig(model=model, added_mass_factor=1.0))
        for frequency in (2.0, 4.0, 5.0):

            def sample(t):
                return opposed_flapping_motion(
                    geometry, origins, time_s=t, speed_mps=8.0, frequency_hz=frequency, amplitude_rad=math.radians(30)
                )

            state = backend.zero_state(geometry)
            trace, incidences = [], []
            for i in range(steps + 1):
                time = i / steps / frequency
                g, m = sample(time)
                reference = geometry.chord_m.new_tensor([[8 * time, 0, 0]])
                loads = backend.evaluate(g, m, state, reference)
                trace.append([
                    time,
                    *loads.force_w_n[0].tolist(),
                    *loads.moment_about_reference_w_nm[0].tolist(),
                    float(loads.circulation_m2_s[0, 0]),
                ])
                incidences.append(float(loads.geometric_incidence_rad.abs().max()))
                if i < steps:
                    state = backend.advance(state, sample, time, 1 / steps / frequency)
            key = f"csv_flap_{frequency:g}Hz__{model}"
            array = np.asarray(trace)
            traces[key] = array
            cases[key] = {
                "finite": bool(np.isfinite(array).all()),
                "speed_mps": 8.0,
                "frequency_hz": frequency,
                "flap_amplitude_deg": 30.0,
                "maximum_geometric_incidence_deg": math.degrees(max(incidences)),
                "twist": "disabled",
                "added_mass_factor": 1.0,
                "scope": "numerical and geometry check; high incidence may violate attached-flow prior",
            }
    return {
        "sections_per_wing": 10,
        "trace_columns": ["time_s", "Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm", "Gamma_right_tip_panel_m2_s"],
        "configurations": {
            model: asdict(ULLTConfig(model=model, added_mass_factor=1.0)) for model in ("one_state", "four_state")
        },
        "hinge_origins_w_m": origins.tolist(),
        "cases": cases,
        "six_state": "explicitly unsupported on this nonplanar/curved lifting line",
        "all_finite": all(c["finite"] for c in cases.values()),
    }, traces

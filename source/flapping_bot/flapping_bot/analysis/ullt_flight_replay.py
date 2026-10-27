"""Offline wing-only replay against frozen whole-aircraft FRD wrench labels.

No tail subtraction, model calibration, or closed-loop prediction. Inertial
vectors are expressed in axes coincident with the instantaneous body FLU axes;
accelerations include transport terms, not rotating-coordinate derivatives.
Scalar wake states are invariant to this common rotation between samples.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import yaml
from flapping_bot.config.ullt import ULLTConfig
from flapping_bot.physics.ullt_geometry import load_csv_urdf_geometry, rotation_xyz
from flapping_bot.physics.ullt_state_space import (
    ULLTBackend,
    ULLTGeometry,
    ULLTMotion,
    ULLTState,
)
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from flapping_bot.physics.ullt_twist import PrescribedTwistConfig, twist_wing

AXES = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")


@dataclass(frozen=True)
class FlightReplayConfig:
    """Frozen offline diagnostics; no values are selected from load residuals."""

    model: str = "four_state"
    sections_per_wing: int = 10
    warmup_s: float = 2.0
    amplitude_rad: float = math.pi / 6
    frequency_derivative_window_s: float = 0.07
    maximum_gap_s: float = 0.015
    transition_change_hz: float = 0.3
    phase_bins: int = 36
    windowed_screen: bool = True
    maximum_rows_per_run: int | None = None  # smoke only; full run has no truncation
    twist: PrescribedTwistConfig = PrescribedTwistConfig()


def sha256(path: Path) -> str:
    """Hash one explicitly selected file without touching other partitions."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registered_comparison(root: Path) -> tuple[pd.DataFrame, dict]:
    """Resolve active registries, verify train/val hashes and join exact keys."""
    registry_paths = [
        root / "configs/data/canonical_dataset_registry.yaml",
        root / "configs/physics/delaurier_prior_registry.yaml",
    ]
    data_reg, prior_reg = (yaml.safe_load(p.read_text()) for p in registry_paths)
    data_id, prior_id = data_reg["default_dataset_id"], prior_reg["default_prior_id"]
    data, prior = data_reg["datasets"][data_id], prior_reg["priors"][prior_id]
    if data["lifecycle_status"] != "active" or prior["lifecycle_status"] != "active" or prior["dataset_id"] != data_id:
        raise ValueError("Active dataset/prior registry mismatch")
    data_root, prior_root = root / data["repository_relative_root"], root / prior["artifact_root"]
    dm_path, pm_path = root / data["manifest_path"], prior_root / "manifest.json"
    hashes = {}

    def verify(path: Path, expected: str) -> None:
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Hash mismatch: {path}")
        hashes[str(path.resolve())] = actual

    verify(dm_path, data["manifest_sha256"])
    verify(pm_path, prior["artifact_manifest_sha256"])
    dm, pm = json.loads(dm_path.read_text()), json.loads(pm_path.read_text())
    needed = [
        "log_id",
        "timestamp_us",
        "time_s",
        "segment_id",
        "mechanical_phase_rad",
        "flap_frequency_hz",
        "cycle_id",
        "label_valid",
        "phase_valid",
        "cycle_valid",
        "vehicle_land_detected.landed",
        "vehicle_air_data.rho",
        "wind.windspeed_north",
        "wind.windspeed_east",
        *AXES,
        *[f"vehicle_attitude.q[{i}]" for i in range(4)],
        *[f"vehicle_local_position.v{x}" for x in "xyz"],
        *[f"vehicle_local_position.a{x}_smooth" for x in "xyz"],
        *[f"vehicle_angular_velocity.xyz[{i}]" for i in range(3)],
        *[f"vehicle_angular_velocity.xyz_derivative_smooth[{i}]" for i in range(3)],
    ]
    frames = []
    for split, stem in (("train", "train"), ("validation", "val")):
        sample_path, prediction_path = data_root / f"{stem}_samples.parquet", prior_root / f"{stem}_predictions.parquet"
        verify(sample_path, data["artifact_sha256"][sample_path.name])
        verify(prediction_path, pm["prediction_sha256"][split])
        samples = pd.read_parquet(sample_path, columns=needed, use_threads=False)
        predictions = pd.read_parquet(prediction_path, use_threads=False)
        keys = ["log_id", "timestamp_us"]
        if samples.duplicated(keys).any() or predictions.duplicated(keys).any():
            raise ValueError("Duplicate sample/prior keys")
        predictions = predictions[keys + list(AXES)].rename(columns={a: f"delaurier_{a}" for a in AXES})
        sample_keys = pd.MultiIndex.from_frame(samples[keys])
        prediction_keys = pd.MultiIndex.from_frame(predictions[keys])
        if len(sample_keys) != len(prediction_keys) or not sample_keys.isin(prediction_keys).all():
            raise ValueError("Dataset/prior key sets differ")
        joined = samples.merge(predictions, on=keys, how="left", validate="one_to_one")
        joined["partition"] = split
        frames.append(joined)
    frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["log_id", "segment_id", "timestamp_us"])
        .reset_index(drop=True)
    )
    if frame.groupby("log_id").partition.nunique().max() != 1:
        raise ValueError("Flight crosses train/validation split")
    return frame, {
        "dataset_id": data_id,
        "prior_id": prior_id,
        "dataset_registry": data,
        "prior_registry": prior,
        "dataset_manifest": dm,
        "prior_manifest": pm,
        "input_sha256": hashes,
        "registry_sha256": {str(p): sha256(p) for p in registry_paths},
        "test_rows_loaded": 0,
        "label_semantics": (
            "registered fixed-neutral-inertia whole-aircraft reconstructed non-gravity wrench; not wing truth"
        ),
    }


def flight_kinematics(
    neutral: ULLTGeometry,
    hinges_flu_m: torch.Tensor,
    inputs: torch.Tensor,
    amplitude_rad: float,
    twist: PrescribedTwistConfig = PrescribedTwistConfig(),
) -> tuple[ULLTGeometry, ULLTMotion]:
    """Map (B,16) offline inputs into section kinematics in instantaneous FLU.

    Columns: phase, phase rate, phase acceleration, air-relative IMU velocity
    (3), inertial IMU acceleration relative to air (3), body omega (3), body
    alpha (3), density. Geometry and hinges have the IMU as their origin.
    Atmospheric acceleration is approximated as zero; wind sets velocity only.
    """
    phase, rate, phase_acc = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    velocity, acceleration, omega, alpha = (
        inputs[:, sl] for sl in (slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15))
    )
    q = amplitude_rad * phase.sin()
    qd = amplitude_rad * rate * phase.cos()
    qdd = amplitude_rad * (phase_acc * phase.cos() - rate.square() * phase.sin())
    n = neutral.chord_m.shape[1] // 2
    geometry_parts, motion_parts = [], []
    for wing, sign in ((0, -1), (1, 1)):
        sl = slice(wing * n, (wing + 1) * n)
        root = hinges_flu_m[wing].view(1, 1, 3)
        angles = torch.zeros((len(inputs), 3), dtype=inputs.dtype, device=inputs.device)
        angles[:, 0] = sign * q
        rotation = rotation_xyz(angles)

        def rotate(v: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bij,bnj->bni", rotation, v.expand(len(inputs), -1, -1))

        local = ULLTGeometry(*(getattr(neutral, name)[:, sl] for name in vars(neutral)))
        internal = None
        if twist.amplitude_deg != 0:
            local, internal = twist_wing(local, hinges_flu_m[wing], phase, rate, phase_acc, twist)
        positions = [
            rotate(v - root) + root for v in (local.edge_start_w_m, local.edge_end_w_m, local.quarter_chord_w_m)
        ]
        g = ULLTGeometry(*positions, rotate(local.chord_forward_w), local.chord_m.expand(len(inputs), -1))
        flap_omega, flap_alpha = torch.zeros_like(angles), torch.zeros_like(angles)
        flap_omega[:, 0], flap_alpha[:, 0] = sign * qd, sign * qdd
        wb, ab, wf, af = (v[:, None].expand(-1, n, -1) for v in (omega, alpha, flap_omega, flap_alpha))

        def flow(point: torch.Tensor, internal_flow: torch.Tensor | None = None) -> torch.Tensor:
            value = -(velocity[:, None] + torch.linalg.cross(wb, point) + torch.linalg.cross(wf, point - root))
            return value if internal_flow is None else value + rotate(internal_flow)

        point = g.three_quarter_chord_w_m
        relative_velocity = torch.linalg.cross(wf, point - root)
        acc = (
            acceleration[:, None]
            + torch.linalg.cross(ab, point)
            + torch.linalg.cross(wb, torch.linalg.cross(wb, point))
            + 2 * torch.linalg.cross(wb, relative_velocity)
            + torch.linalg.cross(af, point - root)
            + torch.linalg.cross(wf, relative_velocity)
        )
        total_omega = wb + wf
        total_alpha = ab + af + torch.linalg.cross(wb, wf)
        if internal is not None:
            twist_velocity = -rotate(internal.flow_three_quarter_w_mps)
            acc = acc + 2 * torch.linalg.cross(wb + wf, twist_velocity) + rotate(internal.acceleration_w_mps2)
            twist_omega = rotate(internal.angular_velocity_w_rad_s)
            total_omega = total_omega + twist_omega
            total_alpha = (
                total_alpha + rotate(internal.angular_acceleration_w_rad_s2) + torch.linalg.cross(wb + wf, twist_omega)
            )
        geometry_parts.append(g)
        motion_parts.append(
            ULLTMotion(
                flow(point, None if internal is None else internal.flow_three_quarter_w_mps),
                flow(g.edge_start_w_m, None if internal is None else internal.flow_edge_start_w_mps),
                flow(g.edge_end_w_m, None if internal is None else internal.flow_edge_end_w_mps),
                acc,
                total_omega,
                total_alpha,
            )
        )
    return (
        ULLTGeometry(*(torch.cat([getattr(g, name) for g in geometry_parts], 1) for name in vars(neutral))),
        ULLTMotion(*(torch.cat([getattr(m, name) for m in motion_parts], 1) for name in vars(motion_parts[0]))),
    )


def prepare_runs(frame: pd.DataFrame, config: FlightReplayConfig) -> tuple[list[dict], pd.DataFrame]:
    """Split gaps/invalid inputs; retain every original row with an exclusion reason."""
    frame = frame.copy()
    frame["replay_reason"] = "invalid_input"
    frame["run_id"] = -1
    frame["score_valid"] = False
    frame["phase_rate_difference_rad_s"] = np.nan
    qcols = [f"vehicle_attitude.q[{i}]" for i in range(4)]
    vcols = [f"vehicle_local_position.v{x}" for x in "xyz"]
    acols = [f"vehicle_local_position.a{x}_smooth" for x in "xyz"]
    wcols = [f"vehicle_angular_velocity.xyz[{i}]" for i in range(3)]
    alcols = [f"vehicle_angular_velocity.xyz_derivative_smooth[{i}]" for i in range(3)]
    fields = (
        qcols
        + vcols
        + acols
        + wcols
        + alcols
        + [
            "wind.windspeed_north",
            "wind.windspeed_east",
            "vehicle_air_data.rho",
            "mechanical_phase_rad",
            "flap_frequency_hz",
        ]
    )
    finite = np.isfinite(frame[fields].to_numpy(float)).all(1)
    valid = finite & frame.label_valid.astype(bool) & frame.phase_valid.astype(bool) & frame.cycle_valid.astype(bool)
    valid &= (
        (frame.flap_frequency_hz > 0.5)
        & (frame["vehicle_air_data.rho"] > 0)
        & (frame["vehicle_land_detected.landed"] == 0)
    )
    valid = np.asarray(valid, dtype=bool)
    c = np.array([1.0, -1.0, -1.0])
    runs = []
    for (_, _), group in frame.groupby(["log_id", "segment_id"], sort=False):
        indices = group.index.to_numpy()
        good = valid[indices]
        breaks = np.r_[
            True, (np.diff(group.time_s) > config.maximum_gap_s) | (np.diff(group.time_s) <= 0) | ~good[:-1] | ~good[1:]
        ]
        for _, chunk in group.groupby(np.cumsum(breaks), sort=False):
            idx = chunk.index.to_numpy()
            if not valid[idx].all():
                continue
            if len(chunk) < 9:
                frame.loc[idx, "replay_reason"] = "short_contiguous_run"
                continue
            if config.maximum_rows_per_run is not None:
                frame.loc[idx[config.maximum_rows_per_run :], "replay_reason"] = "smoke_truncation"
                chunk = chunk.iloc[: config.maximum_rows_per_run]
                idx = chunk.index.to_numpy()
            times = chunk.time_s.to_numpy()
            if not np.allclose(np.diff(times), 0.01, atol=1e-7, rtol=0):
                raise ValueError("Registered replay currently requires native canonical 100 Hz runs")
            rot = Rotation.from_quat(chunk[qcols].to_numpy()[:, [1, 2, 3, 0]])
            wind = np.c_[chunk["wind.windspeed_north"], chunk["wind.windspeed_east"], np.zeros(len(chunk))]
            air = rot.inv().apply(chunk[vcols].to_numpy() - wind) * c
            acc = rot.inv().apply(chunk[acols].to_numpy()) * c
            phase = np.unwrap(chunk.mechanical_phase_rad.to_numpy())
            rate = 2 * np.pi * chunk.flap_frequency_hz.to_numpy()
            window = max(5, int(round(config.frequency_derivative_window_s / 0.01)) | 1)
            window = min(window, len(chunk) if len(chunk) % 2 else len(chunk) - 1)
            rate_dot = savgol_filter(rate, window, 3, deriv=1, delta=0.01)
            phase_rate = np.gradient(phase, times)
            x = np.column_stack(
                (
                    phase,
                    rate,
                    rate_dot,
                    air,
                    acc,
                    chunk[wcols].to_numpy() * c,
                    chunk[alcols].to_numpy() * c,
                    chunk["vehicle_air_data.rho"].to_numpy(),
                )
            )
            run_id = len(runs)
            frame.loc[idx, "run_id"] = run_id
            frame.loc[idx, "phase_rate_difference_rad_s"] = phase_rate - rate
            frame.loc[idx, "replay_reason"] = "warmup"
            runs.append({"indices": idx, "inputs": x, "run_id": run_id})
    if config.windowed_screen and config.maximum_rows_per_run is None:
        runs, frame = select_flight_windows(runs, frame)
    return runs, frame


def select_flight_windows(runs: list[dict], frame: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Choose two disjoint input-only 12 s windows per flight, with 2 s warmup.

    Stable minimizes frequency standard deviation over the scored 10 s.
    Transition maximizes the difference of adjacent one-second frequency means
    at the scored-window center. Candidates start every second; ties keep the
    first time. Selection never reads any predicted or reconstructed load.
    """
    length = 1201
    chosen = []
    frame["replay_reason"] = np.where(frame.run_id >= 0, "outside_selected_windows", frame.replay_reason)
    frame["run_id"] = -1
    frame["window_kind"] = ""
    by_log: dict[str, list[dict]] = {}
    for run in runs:
        log = frame.loc[run["indices"][0], "log_id"]
        by_log.setdefault(log, []).append(run)
    for log, candidates in by_log.items():
        stable, transient = [], []
        for run in candidates:
            x = run["inputs"]
            if len(x) < length:
                continue
            frequency = x[:, 1] / (2 * np.pi)
            for start in range(0, len(x) - length + 1, 100):
                stable.append((float(np.std(frequency[start + 200 : start + length])), run, start))
                center = start + 700
                delta = float(np.mean(frequency[center : center + 100]) - np.mean(frequency[center - 100 : center]))
                transient.append((-abs(delta), run, start))
        if not stable:
            continue
        first = min(stable, key=lambda item: item[0])
        occupied = set(first[1]["indices"][first[2] : first[2] + length])
        candidates2 = [
            item for item in transient if not occupied.intersection(item[1]["indices"][item[2] : item[2] + length])
        ]
        selections = [("stable", first)]
        if candidates2:
            selections.append(("frequency_change", min(candidates2, key=lambda item: item[0])))
        for kind, (score, run, start) in selections:
            idx = run["indices"][start : start + length]
            new_id = len(chosen)
            frame.loc[idx, "run_id"] = new_id
            frame.loc[idx, "replay_reason"] = "warmup"
            frame.loc[idx, "window_kind"] = kind
            chosen.append(
                {
                    "indices": idx,
                    "inputs": run["inputs"][start : start + length],
                    "run_id": new_id,
                    "window_kind": kind,
                    "selection_score": score,
                    "log_id": log,
                }
            )
    if not chosen:
        raise ValueError("No continuous 12-second input-valid windows")
    return chosen, frame


def replay_registered_flights(
    frame: pd.DataFrame, csv: Path, urdf: Path, config: FlightReplayConfig, progress: Callable[[str], None] = print
) -> tuple[pd.DataFrame, dict]:
    """Replay complete continuous runs in parallel, zero wake and 2 s burn-in.

    Canonical phase, frequency and frequency derivative are interpolated between
    recorded rows, as separate measured kinematic channels. Their inconsistency
    is reported, not fitted away. Density scales loads exactly (state equations
    do not depend on rho). No model coefficients are fitted.
    """
    import xml.etree.ElementTree as ET

    neutral = load_csv_urdf_geometry(csv, urdf, sections_per_wing=config.sections_per_wing)
    # URDF base origin to IMU: +0.030 m in FRD z, hence IMU is -0.030 in FLU.
    offset = neutral.chord_m.new_tensor([0.0, 0.0, 0.030])
    neutral = ULLTGeometry(
        *(
            getattr(neutral, name) + offset if name.endswith("_m") and name != "chord_m" else getattr(neutral, name)
            for name in vars(neutral)
        )
    )
    root = ET.parse(urdf).getroot()
    hinges = (
        neutral.chord_m.new_tensor(
            [
                [float(v) for v in root.find(f"joint[@name='{name}']/origin").attrib["xyz"].split()]
                for name in ("right_wing", "left_wing")
            ]
        )
        + offset
    )
    cg = neutral.chord_m.new_tensor([-0.12154, -0.00541, 0.04298])
    backend = ULLTBackend(ULLTConfig(model=config.model, air_density_kg_m3=1.0))
    runs, result = prepare_runs(frame, config)
    for axis in AXES:
        for component in ("ullt", "ullt_circulatory", "ullt_added_mass"):
            result[f"{component}_{axis}"] = np.nan
    result["ullt_max_incidence_deg"] = np.nan
    if not runs:
        raise ValueError("No valid contiguous runs")
    # One independent environment per contiguous run. Completed runs are removed.
    count, length = len(runs), max(len(r["indices"]) for r in runs)
    inputs = torch.zeros((count, length, 16), dtype=torch.float64)
    for i, run in enumerate(runs):
        n = len(run["indices"])
        inputs[i, :n] = torch.from_numpy(run["inputs"])
        inputs[i, n:] = inputs[i, n - 1]
    state = ULLTState(torch.zeros((count, 2 * config.sections_per_wing, backend.state_count), dtype=torch.float64))
    alive = np.ones(count, bool)
    sign = neutral.chord_m.new_tensor([1, -1, -1])
    failures = []
    for step in range(length):
        active = np.array([i for i, run in enumerate(runs) if alive[i] and step < len(run["indices"])], dtype=int)
        if not len(active):
            break
        ids = torch.from_numpy(active)
        x = inputs[ids, step]
        g, m = flight_kinematics(neutral, hinges, x, config.amplitude_rad, config.twist)
        forward = -(m.flow_three_quarter_w_mps * g.chord_forward_w).sum(-1)
        okay = (forward > backend.config.minimum_speed_mps).all(1)
        if not bool(okay.all()):
            for run_index in active[~okay.numpy()]:
                alive[run_index] = False
                result.loc[runs[run_index]["indices"][step:], "replay_reason"] = "reverse_or_zero_chordwise_flow"
                failures.append({"run_id": int(run_index), "step": step, "reason": "reverse_or_zero_chordwise_flow"})
            active = active[okay.numpy()]
            if not len(active):
                continue
            ids = torch.from_numpy(active)
            x = inputs[ids, step]
            g, m = flight_kinematics(neutral, hinges, x, config.amplitude_rad, config.twist)
        loads = backend.evaluate(g, m, ULLTState(state.values_mps[ids]), cg.expand(len(active), -1))
        rho = x[:, 15, None]
        force = loads.force_w_n * rho * sign
        moment = loads.moment_about_reference_w_nm * rho * sign
        fa = loads.added_mass_force_w_n
        ma = (torch.linalg.cross(g.three_quarter_chord_w_m - cg, fa) + loads.added_mass_couple_three_quarter_w_nm).sum(
            1
        )
        added = torch.cat((fa.sum(1) * rho * sign, ma * rho * sign), -1)
        total = torch.cat((force, moment), -1)
        indices = np.array([runs[i]["indices"][step] for i in active])
        for prefix, values in (("ullt", total), ("ullt_added_mass", added), ("ullt_circulatory", total - added)):
            result.loc[indices, [f"{prefix}_{a}" for a in AXES]] = values.numpy()
        result.loc[indices, "ullt_max_incidence_deg"] = (
            loads.geometric_incidence_rad.abs().amax(1).numpy() * 180 / np.pi
        )
        scored = step * 0.01 >= config.warmup_s
        result.loc[indices, "score_valid"] = scored
        result.loc[indices, "replay_reason"] = "scored" if scored else "warmup"
        if step < length - 1:
            # Inputs are prescribed offline channels. Sample all RK stages at
            # their interpolation fraction; the same section identity persists.
            x0, x1 = inputs[ids, step], inputs[ids, step + 1]

            samples = {}

            def sample(t: float) -> tuple[ULLTGeometry, ULLTMotion]:
                if t not in samples:
                    samples[t] = flight_kinematics(
                        neutral, hinges, x0 + (x1 - x0) * (t / 0.01), config.amplitude_rad, config.twist
                    )
                return samples[t]

            try:
                updated = backend.advance(ULLTState(state.values_mps[ids]), sample, 0.0, 0.01)
                state.values_mps[ids] = updated.values_mps
            except (ValueError, FloatingPointError) as error:
                # Isolate failures: one bad run must not remove unrelated flights.
                for j, run_index in enumerate(active):

                    def single(t: float) -> tuple[ULLTGeometry, ULLTMotion]:
                        return flight_kinematics(
                            neutral,
                            hinges,
                            x0[j : j + 1] + (x1[j : j + 1] - x0[j : j + 1]) * (t / 0.01),
                            config.amplitude_rad,
                            config.twist,
                        )

                    try:
                        state.values_mps[run_index : run_index + 1] = backend.advance(
                            ULLTState(state.values_mps[run_index : run_index + 1]), single, 0, 0.01
                        ).values_mps
                    except (ValueError, FloatingPointError) as local_error:
                        alive[run_index] = False
                        result.loc[runs[run_index]["indices"][step + 1 :], "replay_reason"] = "integration_rejected"
                        failures.append(
                            {
                                "run_id": int(run_index),
                                "step": step,
                                "reason": str(local_error),
                                "batch_error": str(error),
                            }
                        )
        if step % 250 == 0:
            progress(f"replay {step}/{length}, active runs={len(active)}, scored rows={int(result.score_valid.sum())}")
    return result, {
        "configuration": asdict(config),
        "backend": asdict(backend.config),
        "run_count": count,
        "selected_windows": [
            {
                "run_id": r["run_id"],
                "log_id": str(result.loc[r["indices"][0], "log_id"]),
                "kind": r.get("window_kind", "continuous_run"),
                "start_time_s": float(result.loc[r["indices"][0], "time_s"]),
                "end_time_s": float(result.loc[r["indices"][-1], "time_s"]),
                "selection_score": r.get("selection_score"),
            }
            for r in runs
        ],
        "failures": failures,
        "row_reasons": result.replay_reason.value_counts().to_dict(),
        "density": "same-row logged rho; exact linear load scaling from rho=1",
        "moment_reference_frd_m": [-0.12154, 0.00541, -0.04298],
        "frame": "FRD",
        "force_scope": "two wings only",
        "label_scope": "whole aircraft, gravity excluded",
    }

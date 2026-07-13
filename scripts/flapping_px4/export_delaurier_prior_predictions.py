#!/usr/bin/env python3
"""Export log-conditioned DeLaurier force-prior predictions for PX4 flight-log splits.

This script is intentionally a prior exporter, not a closed-loop simulator.  It
reads the canonical flight-log split, reconstructs prescribed wing kinematics
from logged mechanical phase and flapping frequency, evaluates the DeLaurier
strip-theory wing model, and writes keyed prior wrench parquets for downstream
system-identification scripts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLAPPING_BOT_SOURCE = PROJECT_ROOT / "source" / "flapping_bot"
if str(FLAPPING_BOT_SOURCE) not in sys.path:
    sys.path.insert(0, str(FLAPPING_BOT_SOURCE))

from flapping_bot.physics.qsm_delaurier1993 import DeLaurierParams, compute_aero_wrench_delaurier1993
from flapping_bot.physics.wing_geom_csv import build_wing_geometry_from_csv


SPLITS = ("train", "val", "test")
KEY_COLUMNS = ("dataset_id", "log_id", "segment_id", "time_s", "timestamp_us")
TARGET_COLUMNS = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")
PHASE_PRIORITY = ("mechanical_phase_rad", "wing_phase.phase_rad", "drive_phase_rad")
FREQUENCY_PRIORITY = ("flap_frequency_hz", "wing_phase.flap_frequency_hz")


def _load_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - environment configuration failure
            raise ModuleNotFoundError("PyYAML is required to read aircraft metadata YAML files") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"metadata must be a mapping: {path}")
    return data


def _metadata_value(metadata: dict[str, Any], path: tuple[str, ...], default: float) -> float:
    node: Any = metadata
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return float(default)
        node = node[key]
    if isinstance(node, dict) and "value" in node:
        node = node["value"]
    return float(node)


def choose_first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for column in candidates:
        if column in frame.columns and frame[column].notna().any():
            return column
    raise ValueError(f"None of the candidate columns are available: {candidates}")


def stroke_kinematics_from_phase(
    phase_rad: np.ndarray,
    flap_frequency_hz: np.ndarray,
    amplitude_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q, qdot, qddot for q=A sin(phi), phi=0 neutral and starting upstroke."""

    phi = np.asarray(phase_rad, dtype=float)
    freq = np.asarray(flap_frequency_hz, dtype=float)
    omega = 2.0 * np.pi * freq
    amp = float(amplitude_rad)
    q = amp * np.sin(phi)
    qdot = amp * omega * np.cos(phi)
    qddot = -amp * omega * omega * np.sin(phi)
    return q, qdot, qddot


def _git_status(root: Path) -> dict[str, Any]:
    git = shutil.which("git")
    if git is None:
        return {"available": False}
    try:
        commit = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output([git, "status", "--porcelain"], cwd=root, text=True).strip())
    except subprocess.CalledProcessError:
        return {"available": False}
    return {"available": True, "commit": commit, "dirty": dirty}


def _column_or_default(frame: pd.DataFrame, column: str, default: float, *, n: int) -> np.ndarray:
    if column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, float(default))
    return np.full(n, float(default), dtype=float)


def compute_delaurier_force_prior(
    frame: pd.DataFrame,
    *,
    phase_column: str,
    frequency_column: str,
    amplitude_rad: float,
    wing_geom_csv: Path,
    num_strips: int,
    chunk_size: int,
    device: str,
    alpha0_deg: float = 0.0,
    eta_s: float = 0.65,
    cd_cf: float = 1.95,
    alpha_stall_min_deg: float | None = None,
    alpha_stall_max_deg: float = 12.0,
    xi: float = 0.0,
    c_mac: float = 0.0,
    cd_f: float | None = 0.028,
    theta_w_deg: float = 0.0,
    twist_eta_max_deg: float = 10.0,
    twist_eta_limit_deg: float = 10.0,
    twist_f_ref_hz: float = 4.0,
    enable_separation: bool = False,
    stall_smoothing_width_deg: float = 0.0,
    include_diagnostics: bool = False,
) -> pd.DataFrame:
    """Compute a body-frame wing-force prior with zero moment placeholder columns."""

    resolved_alpha_stall_min_deg = (
        -float(alpha_stall_max_deg) if alpha_stall_min_deg is None else float(alpha_stall_min_deg)
    )
    n_total = len(frame)
    outputs: list[pd.DataFrame] = []
    torch_device = torch.device(device)
    dtype = torch.float32
    wing_geom, _geom_info = build_wing_geometry_from_csv(
        wing_geom_csv,
        N=int(num_strips),
        device=torch_device,
        dtype=dtype,
    )
    delaurier_params = DeLaurierParams(
        alpha0_rad=math.radians(float(alpha0_deg)),
        eta_s=float(eta_s),
        cd_cf=float(cd_cf),
        alpha_stall_min_rad=math.radians(resolved_alpha_stall_min_deg),
        alpha_stall_max_rad=math.radians(float(alpha_stall_max_deg)),
        xi=float(xi),
        c_mac=float(c_mac),
        nu=1.5e-5,
        cd_f=None if cd_f is None else float(cd_f),
        stall_smoothing_width_rad=math.radians(float(stall_smoothing_width_deg)),
    )
    n_strips = int(wing_geom.x_mid.numel())
    y = wing_geom.x_mid.view(1, n_strips)

    for start in range(0, n_total, int(chunk_size)):
        stop = min(start + int(chunk_size), n_total)
        chunk = frame.iloc[start:stop]
        n = len(chunk)
        phase = chunk[phase_column].to_numpy(dtype=float)
        frequency = chunk[frequency_column].to_numpy(dtype=float)
        q, qd, qdd = stroke_kinematics_from_phase(phase, frequency, amplitude_rad)

        q_t = torch.as_tensor(q, device=torch_device, dtype=dtype)
        qd_t = torch.as_tensor(qd, device=torch_device, dtype=dtype)
        qdd_t = torch.as_tensor(qdd, device=torch_device, dtype=dtype)
        omega_t = torch.as_tensor(2.0 * np.pi * frequency, device=torch_device, dtype=dtype)

        true_airspeed = _column_or_default(chunk, "airspeed_validated.true_airspeed_m_s", 7.0, n=n)
        rho = _column_or_default(chunk, "vehicle_air_data.rho", 1.225, n=n)
        pitch = _column_or_default(chunk, "airspeed_validated.pitch_filtered", 0.0, n=n)

        u_t = torch.as_tensor(np.clip(true_airspeed, 0.5, None), device=torch_device, dtype=dtype).view(n, 1)
        rho_t = torch.as_tensor(rho, device=torch_device, dtype=dtype).view(n, 1)
        theta_a = torch.as_tensor(pitch, device=torch_device, dtype=dtype).view(n, 1)
        theta_bar = theta_a + math.radians(float(theta_w_deg))

        h = -q_t.view(n, 1) * y
        hdot = -qd_t.view(n, 1) * y
        hddot = -qdd_t.view(n, 1) * y

        eta_max = math.radians(float(twist_eta_max_deg))
        eta_lim = math.radians(float(twist_eta_limit_deg))
        qd_ref = max(float(amplitude_rad) * (2.0 * math.pi * float(twist_f_ref_hz)), 1.0e-6)
        s = torch.clamp(qd_t / qd_ref, -1.0, 1.0)
        eta_tip = torch.clamp(float(eta_max) * s, -float(eta_lim), float(eta_lim))
        k_twist = float(eta_max) / qd_ref
        qddd = -(omega_t * omega_t) * qd_t
        etad_tip = k_twist * qdd_t
        etadd_tip = k_twist * qddd
        theta = (theta_bar + eta_tip.view(n, 1)).expand(n, n_strips)
        thetad = etad_tip.view(n, 1).expand(n, n_strips)
        thetadd = etadd_tip.view(n, 1).expand(n, n_strips)
        omega_ref = omega_t.view(n, 1).expand(n, n_strips)

        result = compute_aero_wrench_delaurier1993(
            h,
            hdot,
            hddot,
            theta,
            thetad,
            thetadd,
            wing_geom,
            rho=rho_t,
            U=u_t,
            theta_a=theta_a,
            theta_bar=theta_bar,
            omega_ref=omega_ref,
            params=delaurier_params,
            enable_separation=bool(enable_separation),
            return_terms=bool(include_diagnostics),
        )
        if include_diagnostics:
            force_c, _tau_c, _power, sep_ratio, terms = result
        else:
            force_c, _tau_c, _power, sep_ratio = result

        force_np = force_c.detach().cpu().numpy()
        prediction = pd.DataFrame(
            {
                "fx_b": 2.0 * force_np[:, 2],
                "fy_b": np.zeros(n, dtype=float),
                "fz_b": -2.0 * force_np[:, 1],
                "mx_b": np.zeros(n, dtype=float),
                "my_b": np.zeros(n, dtype=float),
                "mz_b": np.zeros(n, dtype=float),
            },
            index=chunk.index,
        )
        if include_diagnostics:
            prediction["sep_ratio"] = sep_ratio.detach().cpu().numpy()
            for name, values in terms.items():
                prediction[name] = values.detach().cpu().numpy()
        outputs.append(prediction)

    return pd.concat(outputs, axis=0).reset_index(drop=True)


def export_delaurier_prior_predictions(
    *,
    split_root: Path,
    metadata: Path,
    output_root: Path,
    overwrite: bool = False,
    wing_geom_csv: Path | None = None,
    num_strips: int = 80,
    chunk_size: int = 20000,
    device: str = "cpu",
    max_rows_for_tests: int | None = None,
    alpha0_deg: float = 0.0,
    eta_s: float = 0.65,
    cd_cf: float = 1.95,
    alpha_stall_min_deg: float | None = None,
    alpha_stall_max_deg: float = 12.0,
    xi: float = 0.0,
    c_mac: float = 0.0,
    cd_f: float | None = 0.028,
    theta_w_deg: float = 0.0,
    twist_eta_max_deg: float = 10.0,
    twist_eta_limit_deg: float = 10.0,
    twist_f_ref_hz: float = 4.0,
    enable_separation: bool = False,
    stall_smoothing_width_deg: float = 0.0,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_dict = _load_metadata(metadata)
    encoder_to_drive_ratio = _metadata_value(
        metadata_dict,
        ("flapping_drive", "encoder_to_drive_ratio"),
        8.0,
    )
    amplitude_rad = _metadata_value(
        metadata_dict,
        ("flapping_drive", "wing_stroke_amplitude_rad"),
        math.radians(30.0),
    )
    geom_path = wing_geom_csv or (PROJECT_ROOT / "outputs_DeLaurier" / "right_wing_te_fit_poly5_gap50.csv")
    resolved_alpha_stall_min_deg = (
        -float(alpha_stall_max_deg) if alpha_stall_min_deg is None else float(alpha_stall_min_deg)
    )

    phase_column_used: str | None = None
    frequency_column_used: str | None = None
    row_counts: dict[str, int] = {}
    for split in SPLITS:
        samples_path = split_root / f"{split}_samples.parquet"
        samples = pd.read_parquet(samples_path)
        if max_rows_for_tests is not None:
            samples = samples.head(int(max_rows_for_tests)).copy()
        phase_column = choose_first_column(samples, PHASE_PRIORITY)
        frequency_column = choose_first_column(samples, FREQUENCY_PRIORITY)
        phase_column_used = phase_column if phase_column_used is None else phase_column_used
        frequency_column_used = frequency_column if frequency_column_used is None else frequency_column_used
        if phase_column != phase_column_used:
            raise ValueError(f"inconsistent phase column across splits: {phase_column_used} vs {phase_column}")
        if frequency_column != frequency_column_used:
            raise ValueError(f"inconsistent frequency column across splits: {frequency_column_used} vs {frequency_column}")

        prior = compute_delaurier_force_prior(
            samples,
            phase_column=phase_column,
            frequency_column=frequency_column,
            amplitude_rad=amplitude_rad,
            wing_geom_csv=geom_path,
            num_strips=num_strips,
            chunk_size=chunk_size,
            device=device,
            alpha0_deg=alpha0_deg,
            eta_s=eta_s,
            cd_cf=cd_cf,
            alpha_stall_min_deg=resolved_alpha_stall_min_deg,
            alpha_stall_max_deg=alpha_stall_max_deg,
            xi=xi,
            c_mac=c_mac,
            cd_f=cd_f,
            theta_w_deg=theta_w_deg,
            twist_eta_max_deg=twist_eta_max_deg,
            twist_eta_limit_deg=twist_eta_limit_deg,
            twist_f_ref_hz=twist_f_ref_hz,
            enable_separation=enable_separation,
            stall_smoothing_width_deg=stall_smoothing_width_deg,
            include_diagnostics=include_diagnostics,
        )
        keys = [column for column in KEY_COLUMNS if column in samples.columns]
        prediction = pd.concat([samples.loc[:, keys].reset_index(drop=True), prior], axis=1)
        prediction.to_parquet(output_root / f"{split}_predictions.parquet", index=False)
        row_counts[split] = int(len(prediction))

    manifest = {
        "source_split_root": str(split_root),
        "metadata_path": str(metadata),
        "output_root": str(output_root),
        "phase_column": phase_column_used,
        "frequency_column": frequency_column_used,
        "encoder_to_drive_ratio": float(encoder_to_drive_ratio),
        "wing_stroke_amplitude_rad": float(amplitude_rad),
        "stroke_convention": "q=A*sin(phi), phi=0 neutral starting upstroke",
        "target_columns": TARGET_COLUMNS,
        "force_prior_semantics": "two-wing_delaurier_strip_theory_mapped_to_body_force_prior",
        "moment_prior_semantics": "zero_moment_placeholder_for_direct_moment_head",
        "wing_geom_csv": str(geom_path),
        "num_strips": int(num_strips),
        "delaurier_parameters": {
            "alpha0_deg": float(alpha0_deg),
            "eta_s": float(eta_s),
            "cd_cf": float(cd_cf),
            "alpha_stall_min_deg": resolved_alpha_stall_min_deg,
            "alpha_stall_max_deg": float(alpha_stall_max_deg),
            "xi": float(xi),
            "c_mac": float(c_mac),
            "cd_f": None if cd_f is None else float(cd_f),
            "theta_w_deg": float(theta_w_deg),
            "twist_eta_max_deg": float(twist_eta_max_deg),
            "twist_eta_limit_deg": float(twist_eta_limit_deg),
            "twist_f_ref_hz": float(twist_f_ref_hz),
            "enable_separation": bool(enable_separation),
            "stall_smoothing_width_deg": float(stall_smoothing_width_deg),
        },
        "include_diagnostics": bool(include_diagnostics),
        "row_counts": row_counts,
        "isaaclab_git": _git_status(PROJECT_ROOT),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wing-geom-csv", type=Path, default=None)
    parser.add_argument("--num-strips", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--alpha0-deg", type=float, default=0.0)
    parser.add_argument("--eta-s", type=float, default=0.65)
    parser.add_argument("--cd-cf", type=float, default=1.95)
    parser.add_argument("--alpha-stall-min-deg", type=float, default=None)
    parser.add_argument("--alpha-stall-max-deg", type=float, default=12.0)
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--c-mac", type=float, default=0.0)
    parser.add_argument("--cd-f", type=float, default=0.028)
    parser.add_argument("--theta-w-deg", type=float, default=0.0)
    parser.add_argument("--twist-eta-max-deg", type=float, default=10.0)
    parser.add_argument("--twist-eta-limit-deg", type=float, default=10.0)
    parser.add_argument("--twist-f-ref-hz", type=float, default=4.0)
    parser.add_argument("--enable-separation", action="store_true")
    parser.add_argument("--stall-smoothing-width-deg", type=float, default=0.0)
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--max-rows-for-tests", type=int, default=None)
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = _parse_args()
    manifest = export_delaurier_prior_predictions(
        split_root=args.split_root,
        metadata=args.metadata,
        output_root=args.output_root,
        overwrite=args.overwrite,
        wing_geom_csv=args.wing_geom_csv,
        num_strips=args.num_strips,
        chunk_size=args.chunk_size,
        device=args.device,
        alpha0_deg=args.alpha0_deg,
        eta_s=args.eta_s,
        cd_cf=args.cd_cf,
        alpha_stall_min_deg=args.alpha_stall_min_deg,
        alpha_stall_max_deg=args.alpha_stall_max_deg,
        xi=args.xi,
        c_mac=args.c_mac,
        cd_f=args.cd_f,
        theta_w_deg=args.theta_w_deg,
        twist_eta_max_deg=args.twist_eta_max_deg,
        twist_eta_limit_deg=args.twist_eta_limit_deg,
        twist_f_ref_hz=args.twist_f_ref_hz,
        enable_separation=args.enable_separation,
        stall_smoothing_width_deg=args.stall_smoothing_width_deg,
        include_diagnostics=args.include_diagnostics,
        max_rows_for_tests=args.max_rows_for_tests,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

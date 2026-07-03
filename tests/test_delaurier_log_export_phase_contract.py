from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPORT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "flapping_px4"
    / "export_delaurier_prior_predictions.py"
)


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_delaurier_prior_predictions", EXPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {EXPORT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_kinematics_uses_sine_neutral_upstroke_convention() -> None:
    module = _load_export_module()

    phi = np.array([0.0, np.pi / 2.0, np.pi])
    frequency_hz = np.full_like(phi, 5.0)
    amp = 0.5
    q, qd, qdd = module.stroke_kinematics_from_phase(phi, frequency_hz, amp)

    omega = 2.0 * np.pi * frequency_hz
    np.testing.assert_allclose(q, amp * np.sin(phi))
    np.testing.assert_allclose(qd, amp * omega * np.cos(phi), atol=1.0e-12)
    np.testing.assert_allclose(qdd, -amp * omega * omega * np.sin(phi), atol=1.0e-12)


def test_export_prefers_mechanical_phase_and_canonical_frequency(tmp_path: Path) -> None:
    module = _load_export_module()
    split_root = tmp_path / "split"
    output_root = tmp_path / "prior"
    split_root.mkdir()

    samples = pd.DataFrame(
        {
            "dataset_id": ["ratio8_test"] * 4,
            "log_id": ["log_a"] * 4,
            "segment_id": [1] * 4,
            "timestamp_us": [10, 20, 30, 40],
            "time_s": [0.0, 0.01, 0.02, 0.03],
            "mechanical_phase_rad": [0.0, 0.1, 0.2, 0.3],
            "wing_phase.phase_rad": [1.0, 1.1, 1.2, 1.3],
            "drive_phase_rad": [2.0, 2.1, 2.2, 2.3],
            "flap_frequency_hz": [5.0, 5.0, 5.0, 5.0],
            "flap_frequency_topic_hz": [5.33, 5.33, 5.33, 5.33],
            "encoder_rpm_est": [2250.0] * 4,
            "airspeed_validated.true_airspeed_m_s": [7.0] * 4,
            "vehicle_air_data.rho": [1.2] * 4,
        }
    )
    for split in ("train", "val", "test"):
        samples.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    metadata = {
        "flapping_drive": {
            "encoder_to_drive_ratio": {"value": 8.0},
            "wing_stroke_amplitude_rad": {"value": 0.5},
        }
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    manifest = module.export_delaurier_prior_predictions(
        split_root=split_root,
        metadata=metadata_path,
        output_root=output_root,
        overwrite=True,
        max_rows_for_tests=4,
    )

    assert manifest["phase_column"] == "mechanical_phase_rad"
    assert manifest["frequency_column"] == "flap_frequency_hz"
    assert manifest["encoder_to_drive_ratio"] == 8.0
    assert manifest["moment_prior_semantics"] == "zero_moment_placeholder_for_direct_moment_head"

    pred = pd.read_parquet(output_root / "test_predictions.parquet")
    assert {"dataset_id", "log_id", "segment_id", "time_s", "timestamp_us"}.issubset(pred.columns)
    assert {"fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"}.issubset(pred.columns)


def test_export_defaults_to_symmetric_stall_bounds(tmp_path: Path) -> None:
    module = _load_export_module()
    split_root = tmp_path / "split"
    split_root.mkdir()

    samples = pd.DataFrame(
        {
            "dataset_id": ["ratio8_test"] * 2,
            "log_id": ["log_a"] * 2,
            "segment_id": [1] * 2,
            "timestamp_us": [10, 20],
            "time_s": [0.0, 0.01],
            "mechanical_phase_rad": [0.0, 0.1],
            "flap_frequency_hz": [5.0, 5.0],
            "airspeed_validated.true_airspeed_m_s": [7.0, 7.0],
            "vehicle_air_data.rho": [1.2, 1.2],
        }
    )
    for split in ("train", "val", "test"):
        samples.to_parquet(split_root / f"{split}_samples.parquet", index=False)

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "flapping_drive": {
                    "encoder_to_drive_ratio": {"value": 8.0},
                    "wing_stroke_amplitude_rad": {"value": 0.5},
                }
            }
        ),
        encoding="utf-8",
    )

    symmetric = module.export_delaurier_prior_predictions(
        split_root=split_root,
        metadata=metadata_path,
        output_root=tmp_path / "prior_symmetric",
        overwrite=True,
        max_rows_for_tests=2,
        alpha_stall_max_deg=18.0,
    )
    assert symmetric["delaurier_parameters"]["alpha_stall_min_deg"] == -18.0
    assert symmetric["delaurier_parameters"]["alpha_stall_max_deg"] == 18.0

    explicit = module.export_delaurier_prior_predictions(
        split_root=split_root,
        metadata=metadata_path,
        output_root=tmp_path / "prior_explicit",
        overwrite=True,
        max_rows_for_tests=2,
        alpha_stall_min_deg=-5.0,
        alpha_stall_max_deg=18.0,
    )
    assert explicit["delaurier_parameters"]["alpha_stall_min_deg"] == -5.0

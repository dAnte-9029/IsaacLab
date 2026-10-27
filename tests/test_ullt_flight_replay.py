"""Independent kinematics and metric checks for offline flight comparison."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from flapping_bot.analysis.ullt_flight_replay import (
    FlightReplayConfig,
    flight_kinematics,
    prepare_runs,
    select_flight_windows,
)
from flapping_bot.analysis.ullt_flight_report import (
    compute_flight_tables,
    harmonic_features,
    paired_metrics,
)
from flapping_bot.physics.ullt_geometry import (
    elliptic_geometry,
    load_csv_urdf_geometry,
    opposed_flapping_motion,
)


def test_harmonic_amplitude_phase_and_bias():
    phase = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    y = 3 + 2 * np.cos(phase - 0.7) + 0.5 * np.cos(2 * phase + 0.2)
    f = harmonic_features(phase, y)
    assert f["mean"] == pytest.approx(3)
    assert f["h1_amplitude"] == pytest.approx(2)
    assert f["h2_amplitude"] == pytest.approx(0.5)
    assert f["h1_phase_rad"] == pytest.approx(0.7)
    assert f["h2_phase_rad"] == pytest.approx(-0.2)
    m = paired_metrics(y, y + 2)
    assert m["rmse"] == pytest.approx(2)
    assert m["bias"] == pytest.approx(2)
    assert np.isnan(paired_metrics(np.ones(20), np.ones(20))["correlation"])


def test_flight_adapter_matches_independent_prescribed_flapping():
    root = Path(__file__).resolve().parents[1]
    g = load_csv_urdf_geometry(
        root / "outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv",
        root / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf",
        sections_per_wing=4,
    )
    hinges = torch.tensor([[0, -0.056, 0], [0, 0.056, 0]], dtype=torch.float64)
    t, f, amplitude = 0.173, 3.0, math.pi / 6
    x = torch.zeros((1, 16), dtype=torch.float64)
    x[:, 0], x[:, 1], x[:, 3], x[:, 15] = 2 * math.pi * f * t, 2 * math.pi * f, 8.0, 1.2
    actual_g, actual_m = flight_kinematics(g, hinges, x, amplitude)
    expected_g, expected_m = opposed_flapping_motion(
        g, hinges[None], time_s=t, speed_mps=8.0, frequency_hz=f, amplitude_rad=amplitude
    )
    for name in vars(g):
        expected = getattr(expected_g, name)
        if name in ("edge_start_w_m", "edge_end_w_m", "quarter_chord_w_m"):
            expected = expected - expected.new_tensor([8 * t, 0, 0])
        torch.testing.assert_close(getattr(actual_g, name), expected, atol=1e-12, rtol=1e-12)
    for name in vars(actual_m):
        torch.testing.assert_close(getattr(actual_m, name), getattr(expected_m, name), atol=1e-11, rtol=1e-11)


def test_transport_acceleration_and_body_rotation():
    g = elliptic_geometry(num_sections=6)
    x = torch.zeros((1, 16), dtype=torch.float64)
    x[0, 3:6] = torch.tensor([8.0, 1.0, -0.2])
    x[0, 6:9] = torch.tensor([0.4, -0.3, 0.7])
    x[0, 9:12] = torch.tensor([0.2, -0.5, 0.3])
    x[0, 12:15] = torch.tensor([-0.1, 0.2, 0.4])
    _, m = flight_kinematics(g, torch.zeros((2, 3), dtype=torch.float64), x, 0.0)
    p = g.three_quarter_chord_w_m.numpy()[0]
    w, a = x.numpy()[0, 9:12], x.numpy()[0, 12:15]
    expected_v = x.numpy()[0, 3:6] + np.cross(w, p)
    expected_a = x.numpy()[0, 6:9] + np.cross(a, p) + np.cross(w, np.cross(w, p))
    np.testing.assert_allclose(m.flow_three_quarter_w_mps.numpy()[0], -expected_v, atol=1e-12)
    np.testing.assert_allclose(m.acceleration_w_mps2.numpy()[0], expected_a, atol=1e-12)


def test_prepare_runs_splits_missing_time_without_phase_reset():
    n = 60
    d = pd.DataFrame({
        "log_id": ["a"] * n,
        "segment_id": [1] * n,
        "time_s": np.r_[np.arange(30) * 0.01, 1 + np.arange(30) * 0.01],
        "label_valid": True,
        "phase_valid": True,
        "cycle_valid": True,
        "flap_frequency_hz": 3.0,
        "vehicle_air_data.rho": 1.2,
        "vehicle_land_detected.landed": 0.0,
        "mechanical_phase_rad": np.mod(np.arange(n) * 0.01 * 6 * np.pi, 2 * np.pi),
        "wind.windspeed_north": 0.0,
        "wind.windspeed_east": 0.0,
    })
    for i in range(4):
        d[f"vehicle_attitude.q[{i}]"] = 1.0 if i == 0 else 0.0
    for axis in "xyz":
        d[f"vehicle_local_position.v{axis}"] = 8.0 if axis == "x" else 0.0
        d[f"vehicle_local_position.a{axis}_smooth"] = 0.0
    for i in range(3):
        d[f"vehicle_angular_velocity.xyz[{i}]"] = 0.0
        d[f"vehicle_angular_velocity.xyz_derivative_smooth[{i}]"] = 0.0
    runs, out = prepare_runs(d, FlightReplayConfig(windowed_screen=False))
    assert len(runs) == 2
    assert len(runs[0]["indices"]) == 30
    assert runs[1]["inputs"][0, 0] == pytest.approx(d.mechanical_phase_rad.iloc[30])
    assert not out.score_valid.any()


def test_window_selection_is_load_independent_and_disjoint():
    n = 5000
    frame = pd.DataFrame({"log_id": ["a"] * n, "run_id": 0, "replay_reason": "warmup", "fx_b": np.arange(n)})
    inputs = np.zeros((n, 16))
    inputs[:, 1] = 2 * np.pi * (3 + np.clip((np.arange(n) - 3000) / 100, 0, 1))
    runs = [{"indices": np.arange(n), "inputs": inputs, "run_id": 0}]
    chosen, _ = select_flight_windows(runs, frame.copy())
    frame["fx_b"] = -1.0e9
    changed, _ = select_flight_windows(runs, frame.copy())
    assert len(chosen) == 2
    assert not set(chosen[0]["indices"]).intersection(chosen[1]["indices"])
    assert chosen[1]["selection_score"] < -0.5
    for a, b in zip(chosen, changed):
        np.testing.assert_array_equal(a["indices"], b["indices"])
        assert len(a["indices"]) == 1201


def test_window_boundary_cycle_is_rejected_against_original_rows():
    n = 80
    phase = np.arange(n) % 20 * (2 * np.pi / 20)
    frame = pd.DataFrame({
        "log_id": "a",
        "partition": "validation",
        "segment_id": 1,
        "run_id": 0,
        "cycle_id": np.arange(n) // 20,
        "mechanical_phase_rad": phase,
        "score_valid": True,
        "time_s": np.arange(n) * 0.01,
        "flap_frequency_hz": 5.0,
    })
    frame.loc[0, ["run_id", "score_valid"]] = [-1, False]
    for a in ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"):
        for prefix in ("", "delaurier_", "ullt_"):
            frame[prefix + a] = np.sin(phase)
    tables = compute_flight_tables(frame, FlightReplayConfig())
    assert set(tables["cycles"].cycle_id) == {1, 2, 3}
    assert 0 in tables["cycle_rejections"].cycle_id.values

"""Independent contracts for diagnostic decomposition, weighting and replay parity."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from flapping_bot.analysis.ullt_flight_replay import AXES, FlightReplayConfig, replay_registered_flights
from flapping_bot.analysis.ullt_section_attribution import SectionCollector, masked_load_metrics, weighted_quantile
from flapping_bot.config.ullt import ULLTConfig
from flapping_bot.physics.ullt_geometry import elliptic_geometry, load_csv_urdf_geometry, uniform_translation
from flapping_bot.physics.ullt_load_diagnostics import section_load_diagnostics
from flapping_bot.physics.ullt_state_space import ULLTBackend


def test_wrench_closure_and_unit_cd_against_backend_evaluations():
    g = elliptic_geometry(num_sections=6)
    m = uniform_translation(g, 8)
    m.flow_three_quarter_w_mps[..., 2] = 0.8
    m.acceleration_w_mps2[..., 2] = 2.0
    m.angular_velocity_w_rad_s[..., 1] = 0.3
    m.angular_acceleration_w_rad_s2[..., 1] = -0.7
    cg = torch.tensor([[0.1, -0.02, 0.03]], dtype=torch.float64)
    cfg = ULLTConfig(model="four_state", air_density_kg_m3=1)
    backend = ULLTBackend(cfg)
    state = backend.zero_state(g)
    loads = backend.evaluate(g, m, state, cg)
    density = torch.tensor([[1.17]], dtype=torch.float64)
    fields = section_load_diagnostics(g, m, loads, density, cg)
    sign = torch.tensor([1, -1, -1, 1, -1, -1], dtype=torch.float64)
    expected = torch.cat((loads.force_w_n, loads.moment_about_reference_w_nm), -1) * density * sign
    torch.testing.assert_close(fields["total"].sum(1), expected, atol=1e-12, rtol=1e-12)
    drag_backend = ULLTBackend(replace(cfg, profile_drag_coefficient=0.07))
    with_drag = drag_backend.evaluate(g, m, state, cg)
    independent_difference = torch.cat(
        (
            with_drag.force_w_n - loads.force_w_n,
            with_drag.moment_about_reference_w_nm - loads.moment_about_reference_w_nm,
        ),
        -1,
    )
    torch.testing.assert_close(
        fields["profile_drag_per_cd"].sum(1) * 0.07, independent_difference * density * sign, atol=1e-12, rtol=1e-12
    )
    assert fields["profile_drag"].abs().max() == 0
    assert fields["added_mass_free_couple"].abs().max() > 0
    # Independently reconstruct one added-mass torque from numpy r x F + free couple.
    force = loads.added_mass_force_w_n.numpy()[0, 0]
    arm = g.three_quarter_chord_w_m.numpy()[0, 0] - cg.numpy()[0]
    couple = loads.added_mass_couple_three_quarter_w_nm.numpy()[0, 0]
    torque = (np.cross(arm, force) + couple) * 1.17 * np.array([1, -1, -1])
    np.testing.assert_allclose(fields["added_mass"].numpy()[0, 0, 3:], torque, atol=1e-12)


def test_tiny_high_angle_area_does_not_dominate_weighted_quantiles():
    values, weights = np.array([10.0, 60.0]), np.array([0.99, 0.01])
    assert weighted_quantile(values, weights, 0.95) == 10
    assert weighted_quantile(values, weights, 0.999) == 60


def test_absolute_share_handles_cancellation_and_error_changes_are_nonadditive():
    sections = np.zeros((2, 2, 6))
    sections[:, 0, 0], sections[:, 1, 0] = 10, -10
    prediction = sections.sum(1)
    target = np.zeros((2, 6))
    a = masked_load_metrics(sections, np.array([[True, False]] * 2), prediction, target)
    b = masked_load_metrics(sections, np.array([[False, True]] * 2), prediction, target)
    all_sections = masked_load_metrics(sections, np.ones((2, 2), bool), prediction, target)
    assert a["absolute_section_load_share"][0] == pytest.approx(0.5)
    assert np.isnan(a["absolute_section_load_share"][1])
    assert a["frozen_subtraction_rmse"][0] == pytest.approx(10)
    assert a["frozen_subtraction_mse_reduction"][0] + b["frozen_subtraction_mse_reduction"][0] == -200
    assert all_sections["frozen_subtraction_mse_reduction"][0] == 0


def test_zero_mean_component_can_have_large_rms():
    sections = np.zeros((2, 1, 6))
    sections[:, 0, 2] = [8, -8]
    metrics = masked_load_metrics(sections, np.ones((2, 1), bool), sections.sum(1), np.zeros((2, 6)))
    assert metrics["signed_mean"][2] == 0
    assert metrics["resultant_rms"][2] == 8
    assert metrics["frozen_subtraction_rmse"][2] == 0


def test_observer_replay_is_bitwise_identical():
    root = Path(__file__).resolve().parents[1]
    csv = root / "outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv"
    urdf = root / "source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf"
    n = 12
    frame = pd.DataFrame(
        {
            "log_id": "synthetic",
            "segment_id": 0,
            "cycle_id": 0,
            "timestamp_us": np.arange(n) * 10000,
            "time_s": np.arange(n) * 0.01,
            "mechanical_phase_rad": np.arange(n) * 0.01 * 2 * np.pi,
            "flap_frequency_hz": 1.0,
            "label_valid": True,
            "phase_valid": True,
            "cycle_valid": True,
            "vehicle_land_detected.landed": 0.0,
            "vehicle_air_data.rho": 1.2,
            "wind.windspeed_north": 0.0,
            "wind.windspeed_east": 0.0,
        }
    )
    for i in range(4):
        frame[f"vehicle_attitude.q[{i}]"] = 1.0 if i == 0 else 0.0
    for axis in "xyz":
        frame[f"vehicle_local_position.v{axis}"] = 8.0 if axis == "x" else 0.0
        frame[f"vehicle_local_position.a{axis}_smooth"] = 0.0
    for i in range(3):
        frame[f"vehicle_angular_velocity.xyz[{i}]"] = 0.0
        frame[f"vehicle_angular_velocity.xyz_derivative_smooth[{i}]"] = 0.0
    config = FlightReplayConfig(sections_per_wing=3, amplitude_rad=0.0, warmup_s=0.0, windowed_screen=False)
    collector = SectionCollector(frame, load_csv_urdf_geometry(csv, urdf, sections_per_wing=3))
    plain, _ = replay_registered_flights(frame, csv, urdf, config, lambda _: None)
    observed, _ = replay_registered_flights(frame, csv, urdf, config, lambda _: None, collector)
    pd.testing.assert_frame_equal(plain, observed, check_exact=True)
    sections = collector.finish()
    assert len(sections) == 12 * 6
    summed = sections.groupby("row_id")[["total_" + a for a in AXES]].sum().to_numpy()
    np.testing.assert_allclose(summed, observed[["ullt_" + a for a in AXES]].to_numpy(), atol=1e-12)

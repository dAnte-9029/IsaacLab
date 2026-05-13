import numpy as np
import pandas as pd

from flapping_bot.analysis.delaurier_gain_calibration import (
    PHYSICAL_PARAMETER_BOUNDS,
    TARGET_COLUMNS,
    OfflineDeLaurierConfig,
    PhysicalCalibrationParameters,
    apply_channel_gains,
    fit_channel_gains,
    metrics_by_channel,
    normalized_wrench_objective,
    parameter_vector_to_config,
    random_search_physical_calibration,
)


def test_fit_channel_gains_recovers_per_target_scales_with_nan_rows() -> None:
    predictions = pd.DataFrame(
        {
            "fx_b": [1.0, 2.0, 3.0, np.nan],
            "fy_b": [2.0, -1.0, 4.0, 3.0],
            "fz_b": [0.5, 1.0, 1.5, 2.0],
            "mx_b": [1.0, 0.0, -1.0, 2.0],
            "my_b": [3.0, 3.0, 3.0, 3.0],
            "mz_b": [-2.0, -1.0, 1.0, 2.0],
        }
    )
    true_gains = pd.Series(
        {
            "fx_b": 2.0,
            "fy_b": -0.5,
            "fz_b": 4.0,
            "mx_b": 1.5,
            "my_b": 0.25,
            "mz_b": -2.0,
        }
    )
    targets = predictions.mul(true_gains, axis=1)
    targets.loc[1, "fy_b"] = np.nan

    gains = fit_channel_gains(predictions, targets)

    for column in TARGET_COLUMNS:
        assert np.isclose(gains[column], true_gains[column])


def test_apply_channel_gains_and_metrics_handle_zero_prediction_channel() -> None:
    predictions = pd.DataFrame({column: [0.0, 0.0, 0.0] for column in TARGET_COLUMNS})
    targets = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in TARGET_COLUMNS})

    gains = fit_channel_gains(predictions, targets)
    calibrated = apply_channel_gains(predictions, gains)
    metrics = metrics_by_channel(targets, calibrated)

    assert all(gains[column] == 0.0 for column in TARGET_COLUMNS)
    assert calibrated.to_numpy().sum() == 0.0
    assert set(metrics["target"]) == set(TARGET_COLUMNS)
    assert set(metrics["n"]) == {3}
    assert np.isclose(float(metrics.loc[metrics["target"] == "fx_b", "rmse"].iloc[0]), np.sqrt(14.0 / 3.0))


def test_physical_parameters_map_to_offline_config() -> None:
    base_cfg = OfflineDeLaurierConfig(twist_eta_limit_deg=10.0)
    params = PhysicalCalibrationParameters(
        wing_normal_force_scale=1.25,
        wing_chordwise_force_scale=0.75,
        delaurier_theta_w_deg=3.0,
        twist_eta_max_deg=14.0,
        delaurier_induced_drag_efficiency=0.7,
        fuselage_drag_cda=0.006,
        tail_lift_scale=1.4,
        phase_delay_s=0.012,
    )

    cfg = parameter_vector_to_config(params, base_cfg)

    assert cfg.theta_w_deg == 3.0
    assert cfg.twist_eta_max_deg == 14.0
    assert cfg.twist_eta_limit_deg == 14.0
    assert cfg.induced_drag_efficiency == 0.7
    assert cfg.fuselage_drag_cda == 0.006
    assert cfg.tail_fixed_horizontal_effectiveness == 0.5
    assert cfg.tail_elevon_effectiveness == 1.2
    assert cfg.tail_force_scale == 1.4
    assert cfg.phase_delay_s == 0.012
    assert cfg.wing_normal_force_scale == 1.25
    assert cfg.wing_chordwise_force_scale == 0.75


def test_normalized_wrench_objective_uses_train_scale_and_regularization() -> None:
    predictions = pd.DataFrame({column: [2.0, 4.0] for column in TARGET_COLUMNS})
    targets = pd.DataFrame({column: [1.0, 2.0] for column in TARGET_COLUMNS})
    scale = pd.Series({column: 2.0 for column in TARGET_COLUMNS})
    params = PhysicalCalibrationParameters(
        wing_normal_force_scale=PHYSICAL_PARAMETER_BOUNDS["wing_normal_force_scale"].nominal,
        wing_chordwise_force_scale=PHYSICAL_PARAMETER_BOUNDS["wing_chordwise_force_scale"].nominal,
        delaurier_theta_w_deg=PHYSICAL_PARAMETER_BOUNDS["delaurier_theta_w_deg"].nominal,
        twist_eta_max_deg=PHYSICAL_PARAMETER_BOUNDS["twist_eta_max_deg"].nominal,
        delaurier_induced_drag_efficiency=PHYSICAL_PARAMETER_BOUNDS["delaurier_induced_drag_efficiency"].nominal,
        fuselage_drag_cda=PHYSICAL_PARAMETER_BOUNDS["fuselage_drag_cda"].nominal,
        tail_lift_scale=PHYSICAL_PARAMETER_BOUNDS["tail_lift_scale"].nominal,
        phase_delay_s=PHYSICAL_PARAMETER_BOUNDS["phase_delay_s"].nominal,
    )

    objective = normalized_wrench_objective(
        predictions,
        targets,
        scale,
        params,
        regularization_weight=0.1,
    )

    expected = np.mean(np.array([0.5, 1.0]) ** 2)
    assert objective == expected


def test_phase_delay_shifts_wing_state_without_crossing_logs() -> None:
    from flapping_bot.analysis.delaurier_gain_calibration import _wing_state

    frame = pd.DataFrame(
        {
            "log_id": ["a", "a", "a", "b", "b", "b"],
            "time_s": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "wing_stroke_angle_rad": [0.0, 10.0, 20.0, 100.0, 110.0, 120.0],
            "wing_stroke_velocity_rad_s": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            "wing_stroke_acceleration_rad_s2": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
        }
    )

    q, qd, qdd = _wing_state(frame, phase_delay_s=0.5)

    np.testing.assert_allclose(q, [0.0, 5.0, 15.0, 100.0, 105.0, 115.0])
    np.testing.assert_allclose(qd, [0.0, 0.5, 1.5, 10.0, 10.5, 11.5])
    np.testing.assert_allclose(qdd, [0.0, 0.05, 0.15, 1.0, 1.05, 1.15])


def test_random_search_physical_calibration_is_deterministic_with_fake_predictor() -> None:
    frame = pd.DataFrame({column: [0.0, 0.0] for column in TARGET_COLUMNS})
    scale = pd.Series({column: 1.0 for column in TARGET_COLUMNS})

    def fake_predictor(_frame: pd.DataFrame, params: PhysicalCalibrationParameters) -> pd.DataFrame:
        value = params.wing_normal_force_scale - 1.0
        return pd.DataFrame({column: [value, value] for column in TARGET_COLUMNS})

    first = random_search_physical_calibration(
        frame,
        frame,
        scale,
        n_candidates=4,
        seed=7,
        predictor=fake_predictor,
    )
    second = random_search_physical_calibration(
        frame,
        frame,
        scale,
        n_candidates=4,
        seed=7,
        predictor=fake_predictor,
    )

    assert list(first.trace.columns[:2]) == ["candidate", "objective"]
    assert len(first.trace) == 5
    assert first.best_objective == second.best_objective
    assert first.best_parameters.as_dict() == second.best_parameters.as_dict()

import numpy as np
import pandas as pd

from flapping_bot.analysis.delaurier_gain_calibration import (
    TARGET_COLUMNS,
    apply_channel_gains,
    fit_channel_gains,
    metrics_by_channel,
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

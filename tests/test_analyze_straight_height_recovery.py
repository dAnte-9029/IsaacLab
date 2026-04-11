import math

import pytest

from scripts.flapping_px4.analyze_straight_height_recovery import compute_metrics


def _row(
    *,
    t: float,
    height_error_m: float,
    lateral_error_m: float,
    action_elevon_pitch: float,
    rudder: float = 0.0,
    elevon_roll: float = 0.0,
) -> dict[str, float]:
    return {
        "t": t,
        "height_error_m": height_error_m,
        "progress_s": 8.0 * t,
        "freq_hz": 4.5,
        "exec_freq_hz": 4.5,
        "action_elevon_pitch": action_elevon_pitch,
        "exec_action_elevon_pitch": action_elevon_pitch,
        "rudder": rudder,
        "elevon_roll": elevon_roll,
        "lateral_error_m": lateral_error_m,
        "speed": 8.0,
        "airspeed": 8.0,
        "tecs_pitch_sp_deg": -12.0,
        "pitch_sp_deg": -12.0,
    }


def test_straight_height_recovery_metrics_include_second_dip_chatter_and_lateral_bias():
    rows: list[dict[str, float]] = []
    dt = 1.0 / 60.0
    for idx in range(int(8.0 / dt) + 1):
        t = idx * dt
        initial_drop = -0.35 * math.exp(-((t - 2.0) / 0.45) ** 2)
        second_dip = -0.22 * math.exp(-((t - 4.5) / 0.35) ** 2)
        height_error_m = initial_drop + second_dip
        lateral_error_m = 0.2 if t < 5.0 else 0.8
        chatter = 0.12 * math.sin(2.0 * math.pi * 4.8 * t)
        rows.append(
            _row(
                t=t,
                height_error_m=height_error_m,
                lateral_error_m=lateral_error_m,
                action_elevon_pitch=chatter,
                rudder=0.04 * math.sin(2.0 * math.pi * 5.2 * t),
                elevon_roll=0.03 * math.sin(2.0 * math.pi * 4.5 * t),
            )
        )

    metrics = compute_metrics(
        rows=rows,
        summary={"status": "test", "progress_ratio_final": 1.0},
        height_band_m=0.05,
        sustain_band_m=0.10,
        sustain_time_s=0.5,
        freq_sat_hz=4.99,
        action_sat_abs=0.98,
    )

    assert metrics["min_cycle_mean_height_error_after_4s_m"] == pytest.approx(-0.18, abs=0.04)
    assert metrics["time_at_min_cycle_mean_height_error_after_4s_s"] == pytest.approx(4.5, abs=0.2)
    assert metrics["elevon_pitch_chatter_rms_4to8hz"] > 0.07
    assert metrics["rudder_chatter_rms_4to8hz"] > 0.02
    assert metrics["elevon_roll_chatter_rms_4to8hz"] > 0.015
    assert metrics["post5_mean_lateral_error_m"] == pytest.approx(0.8, abs=0.02)
    assert metrics["post5_final_lateral_error_m"] == pytest.approx(0.8, abs=0.02)
    assert metrics["post5_mean_abs_lateral_error_m"] == pytest.approx(0.8, abs=0.02)

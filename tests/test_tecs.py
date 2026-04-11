from __future__ import annotations

import math

import pytest
import torch

from flapping_bot.px4_like.tecs import PX4LikeTECS, PX4LikeTECSCfg


def test_tecs_accepts_batched_height_setpoints() -> None:
    tecs = PX4LikeTECS(PX4LikeTECSCfg(), device=torch.device("cpu"))

    pitch_sp, throttle_sp, diag = tecs.update(
        dt=1.0 / 120.0,
        altitude=torch.tensor([10.0, 10.0]),
        altitude_rate=torch.tensor([0.0, 0.0]),
        tas=torch.tensor([7.0, 7.0]),
        height_sp_m=torch.tensor([10.0, 12.0]),
        speed_sp_mps=torch.tensor([7.0, 7.0]),
    )

    assert pitch_sp.shape == (2,)
    assert throttle_sp.shape == (2,)
    assert diag["tecs_height_err_raw"].shape == (2,)


def test_tecs_load_factor_compensation_increases_throttle_demand() -> None:
    tecs = PX4LikeTECS(PX4LikeTECSCfg(), device=torch.device("cpu"))

    _, throttle_nominal, diag_nominal = tecs.update(
        dt=1.0 / 120.0,
        altitude=torch.tensor([10.0]),
        altitude_rate=torch.tensor([0.0]),
        tas=torch.tensor([7.0]),
        height_sp_m=torch.tensor([10.0]),
        speed_sp_mps=torch.tensor([7.0]),
        load_factor=torch.tensor([1.0]),
        load_factor_correction=torch.tensor([0.5]),
    )

    tecs.reset()

    _, throttle_loaded, diag_loaded = tecs.update(
        dt=1.0 / 120.0,
        altitude=torch.tensor([10.0]),
        altitude_rate=torch.tensor([0.0]),
        tas=torch.tensor([7.0]),
        height_sp_m=torch.tensor([10.0]),
        speed_sp_mps=torch.tensor([7.0]),
        load_factor=torch.tensor([1.3]),
        load_factor_correction=torch.tensor([0.5]),
    )

    assert diag_nominal["tecs_load_factor"].shape == (1,)
    assert diag_loaded["tecs_load_factor"].shape == (1,)
    assert float(diag_loaded["tecs_load_factor"][0]) > float(diag_nominal["tecs_load_factor"][0])
    assert float(diag_loaded["tecs_load_factor_energy_bias"][0]) > 0.0
    assert float(throttle_loaded[0]) > float(throttle_nominal[0])


def test_tecs_load_factor_compensation_increases_pitch_demand_when_enabled() -> None:
    tecs_nominal = PX4LikeTECS(
        PX4LikeTECSCfg(
            pitch_sp_filter_tau_s=0.0,
            pitch_sp_filter_tau_capture_s=0.0,
            pitch_sp_rate_limit_deg_s=0.0,
            pitch_sp_rate_limit_capture_deg_s=0.0,
            load_factor_pitch_compensation_gain=0.0,
        ),
        device=torch.device("cpu"),
    )
    tecs_compensated = PX4LikeTECS(
        PX4LikeTECSCfg(
            pitch_sp_filter_tau_s=0.0,
            pitch_sp_filter_tau_capture_s=0.0,
            pitch_sp_rate_limit_deg_s=0.0,
            pitch_sp_rate_limit_capture_deg_s=0.0,
            load_factor_pitch_compensation_gain=0.75,
        ),
        device=torch.device("cpu"),
    )

    pitch_nominal, _, diag_nominal = tecs_nominal.update(
        dt=1.0 / 120.0,
        altitude=torch.tensor([10.0]),
        altitude_rate=torch.tensor([0.0]),
        tas=torch.tensor([7.0]),
        height_sp_m=torch.tensor([10.0]),
        speed_sp_mps=torch.tensor([7.0]),
        load_factor=torch.tensor([1.3]),
        load_factor_correction=torch.tensor([0.5]),
    )
    pitch_loaded, _, diag_loaded = tecs_compensated.update(
        dt=1.0 / 120.0,
        altitude=torch.tensor([10.0]),
        altitude_rate=torch.tensor([0.0]),
        tas=torch.tensor([7.0]),
        height_sp_m=torch.tensor([10.0]),
        speed_sp_mps=torch.tensor([7.0]),
        load_factor=torch.tensor([1.3]),
        load_factor_correction=torch.tensor([0.5]),
    )

    assert float(diag_loaded["tecs_load_factor"][0]) > 1.0
    assert float(diag_loaded["tecs_load_factor_pitch_bias"][0]) > 0.0
    assert float(pitch_loaded[0]) < float(pitch_nominal[0])


def test_tecs_capture_tuning_reduces_large_height_error_pitch_sp_lag() -> None:
    baseline = PX4LikeTECS(
        PX4LikeTECSCfg(
            pitch_sp_filter_tau_s=0.35,
            pitch_sp_filter_tau_capture_s=0.35,
            pitch_sp_rate_limit_deg_s=20.0,
            pitch_sp_rate_limit_capture_deg_s=20.0,
        ),
        device=torch.device("cpu"),
    )
    capture_tuned = PX4LikeTECS(
        PX4LikeTECSCfg(
            pitch_sp_filter_tau_s=0.35,
            pitch_sp_filter_tau_capture_s=0.08,
            pitch_sp_rate_limit_deg_s=20.0,
            pitch_sp_rate_limit_capture_deg_s=45.0,
        ),
        device=torch.device("cpu"),
    )

    pitch_sp_baseline = None
    pitch_sp_tuned = None
    for _ in range(10):
        pitch_sp_baseline, _, _ = baseline.update(
            dt=1.0 / 120.0,
            altitude=torch.tensor([6.0]),
            altitude_rate=torch.tensor([0.0]),
            tas=torch.tensor([7.0]),
            height_sp_m=torch.tensor([10.0]),
            speed_sp_mps=torch.tensor([7.0]),
        )
        pitch_sp_tuned, _, _ = capture_tuned.update(
            dt=1.0 / 120.0,
            altitude=torch.tensor([6.0]),
            altitude_rate=torch.tensor([0.0]),
            tas=torch.tensor([7.0]),
            height_sp_m=torch.tensor([10.0]),
            speed_sp_mps=torch.tensor([7.0]),
        )

    assert pitch_sp_baseline is not None
    assert pitch_sp_tuned is not None
    assert math.degrees(float(pitch_sp_tuned[0])) < math.degrees(float(pitch_sp_baseline[0])) - 1.5


def test_tecs_capture_tuning_stays_neutral_near_altitude_hold_band() -> None:
    baseline = PX4LikeTECS(
        PX4LikeTECSCfg(
            pitch_sp_filter_tau_s=0.35,
            pitch_sp_filter_tau_capture_s=0.35,
            pitch_sp_rate_limit_deg_s=20.0,
            pitch_sp_rate_limit_capture_deg_s=20.0,
        ),
        device=torch.device("cpu"),
    )
    capture_tuned = PX4LikeTECS(
        PX4LikeTECSCfg(
            pitch_sp_filter_tau_s=0.35,
            pitch_sp_filter_tau_capture_s=0.08,
            pitch_sp_rate_limit_deg_s=20.0,
            pitch_sp_rate_limit_capture_deg_s=45.0,
        ),
        device=torch.device("cpu"),
    )

    pitch_sp_baseline, _, _ = baseline.update(
        dt=1.0 / 120.0,
        altitude=torch.tensor([9.9]),
        altitude_rate=torch.tensor([0.0]),
        tas=torch.tensor([7.0]),
        height_sp_m=torch.tensor([10.0]),
        speed_sp_mps=torch.tensor([7.0]),
    )
    pitch_sp_tuned, _, _ = capture_tuned.update(
        dt=1.0 / 120.0,
        altitude=torch.tensor([9.9]),
        altitude_rate=torch.tensor([0.0]),
        tas=torch.tensor([7.0]),
        height_sp_m=torch.tensor([10.0]),
        speed_sp_mps=torch.tensor([7.0]),
    )

    assert torch.allclose(pitch_sp_tuned, pitch_sp_baseline, atol=1.0e-4)


def test_tecs_capture_blend_persists_until_recovered() -> None:
    tecs = PX4LikeTECS(
        PX4LikeTECSCfg(
            altitude_hold_error_band_m=0.05,
            altitude_capture_error_m=0.8,
            altitude_capture_release_error_m=0.02,
            altitude_capture_release_time_s=0.3,
            altitude_capture_persistence_gain=1.0,
        ),
        device=torch.device("cpu"),
    )

    _, _, diag_drop = tecs.update(
        dt=0.1,
        altitude=torch.tensor([9.6]),
        altitude_rate=torch.tensor([0.0]),
        tas=torch.tensor([7.0]),
        height_sp_m=torch.tensor([10.0]),
        speed_sp_mps=torch.tensor([7.0]),
    )
    _, _, diag_mid = tecs.update(
        dt=0.1,
        altitude=torch.tensor([9.92]),
        altitude_rate=torch.tensor([0.0]),
        tas=torch.tensor([7.0]),
        height_sp_m=torch.tensor([10.0]),
        speed_sp_mps=torch.tensor([7.0]),
    )

    assert float(diag_drop["tecs_capture_active"][0]) == pytest.approx(1.0)
    assert float(diag_mid["tecs_capture_active"][0]) == pytest.approx(1.0)
    assert float(diag_mid["tecs_capture_blend"][0]) > 0.9

    diag_releasing = diag_mid
    for _ in range(2):
        _, _, diag_releasing = tecs.update(
            dt=0.1,
            altitude=torch.tensor([9.99]),
            altitude_rate=torch.tensor([0.0]),
            tas=torch.tensor([7.0]),
            height_sp_m=torch.tensor([10.0]),
            speed_sp_mps=torch.tensor([7.0]),
        )

    assert float(diag_releasing["tecs_capture_active"][0]) == pytest.approx(1.0)

    for _ in range(2):
        _, _, diag_releasing = tecs.update(
            dt=0.1,
            altitude=torch.tensor([9.99]),
            altitude_rate=torch.tensor([0.0]),
            tas=torch.tensor([7.0]),
            height_sp_m=torch.tensor([10.0]),
            speed_sp_mps=torch.tensor([7.0]),
        )

    assert float(diag_releasing["tecs_capture_active"][0]) == pytest.approx(0.0)
    assert float(diag_releasing["tecs_capture_blend"][0]) < 0.1

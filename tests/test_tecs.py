from __future__ import annotations

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

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

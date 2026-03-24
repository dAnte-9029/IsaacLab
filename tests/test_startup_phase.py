from __future__ import annotations

import math

import torch

from flapping_bot.direct.flapping_bot.startup_phase import advance_flap_phase


def test_advance_flap_phase_holds_phase_while_frozen() -> None:
    phase = torch.zeros(1, dtype=torch.float32)
    freq_hz = torch.tensor([2.5], dtype=torch.float32)
    freeze_steps = torch.tensor([240], dtype=torch.int32)

    for _ in range(240):
        phase = advance_flap_phase(
            phase=phase,
            freq_hz=freq_hz,
            physics_dt_s=1.0 / 240.0,
            freeze_steps=freeze_steps,
        )

    assert torch.allclose(phase, torch.zeros_like(phase), atol=1.0e-6)


def test_advance_flap_phase_restarts_from_zero_after_freeze_release() -> None:
    phase = torch.zeros(1, dtype=torch.float32)
    freq_hz = torch.tensor([2.5], dtype=torch.float32)

    phase = advance_flap_phase(
        phase=phase,
        freq_hz=freq_hz,
        physics_dt_s=1.0 / 240.0,
        freeze_steps=torch.tensor([0], dtype=torch.int32),
    )

    expected = torch.tensor([2.0 * math.pi * 2.5 / 240.0], dtype=torch.float32)
    assert torch.allclose(phase, expected, atol=1.0e-6)

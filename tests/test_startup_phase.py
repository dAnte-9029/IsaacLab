from __future__ import annotations

import math

import pytest
import torch

from flapping_bot.direct.flapping_bot.startup_phase import (
    LEGACY_COSINE_ENDPOINT_ZERO,
    MECHANICAL_SINE_NEUTRAL_UPSTROKE,
    advance_flap_phase,
    compute_prescribed_flap_kinematics,
)


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


def test_sine_flap_kinematics_match_mechanical_quadrature_contract() -> None:
    phase = torch.tensor(
        [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0],
        dtype=torch.float64,
    )
    freq_hz = torch.full_like(phase, 2.0)
    amplitude = 0.4
    omega = 4.0 * math.pi

    q, qd, qdd = compute_prescribed_flap_kinematics(
        phase=phase,
        freq_hz=freq_hz,
        amplitude_rad=amplitude,
        convention=MECHANICAL_SINE_NEUTRAL_UPSTROKE,
    )

    torch.testing.assert_close(q, torch.tensor([0.0, 0.4, 0.0, -0.4], dtype=phase.dtype), atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        qd,
        torch.tensor([amplitude * omega, 0.0, -amplitude * omega, 0.0], dtype=phase.dtype),
        atol=1.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        qdd,
        torch.tensor([0.0, -amplitude * omega**2, 0.0, amplitude * omega**2], dtype=phase.dtype),
        atol=1.0e-11,
        rtol=0.0,
    )


def test_legacy_cosine_flap_kinematics_remain_available() -> None:
    phase = torch.tensor([0.0], dtype=torch.float64)
    freq_hz = torch.tensor([2.0], dtype=torch.float64)

    q, qd, qdd = compute_prescribed_flap_kinematics(
        phase=phase,
        freq_hz=freq_hz,
        amplitude_rad=0.4,
        convention=LEGACY_COSINE_ENDPOINT_ZERO,
    )

    assert q.item() == pytest.approx(0.4)
    assert qd.item() == pytest.approx(0.0)
    assert qdd.item() == pytest.approx(-0.4 * (4.0 * math.pi) ** 2)


def test_flap_off_is_neutral_for_both_phase_conventions() -> None:
    phase = torch.tensor([0.7], dtype=torch.float32)
    freq_hz = torch.zeros_like(phase)

    for convention in (MECHANICAL_SINE_NEUTRAL_UPSTROKE, LEGACY_COSINE_ENDPOINT_ZERO):
        kinematics = compute_prescribed_flap_kinematics(
            phase=phase,
            freq_hz=freq_hz,
            amplitude_rad=0.4,
            convention=convention,
        )
        for value in kinematics:
            torch.testing.assert_close(value, torch.zeros_like(value))


def test_unsupported_flap_phase_convention_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported flap phase convention"):
        compute_prescribed_flap_kinematics(
            phase=torch.zeros(1),
            freq_hz=torch.ones(1),
            amplitude_rad=0.4,
            convention="unlabeled_phase",
        )

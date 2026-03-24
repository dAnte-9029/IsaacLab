"""Startup helpers for flapping-wing reset and release behavior."""

from __future__ import annotations

import math

import torch

Tensor = torch.Tensor


def advance_flap_phase(
    *,
    phase: Tensor,
    freq_hz: Tensor,
    physics_dt_s: float,
    freeze_steps: Tensor | None = None,
) -> Tensor:
    """Advance flap phase, optionally holding frozen environments fixed.

    When ``freeze_steps`` is positive for an environment, its flap phase is held
    constant so release does not occur at an arbitrary phase accumulated during
    the reset hold window.
    """

    if physics_dt_s <= 0.0:
        raise ValueError("physics_dt_s must be positive.")

    phase_next = phase.clone()
    if freeze_steps is None:
        free_mask = torch.ones_like(freq_hz, dtype=torch.bool)
    else:
        free_mask = freeze_steps <= 0

    if torch.any(free_mask):
        phase_step = 2.0 * math.pi * freq_hz[free_mask] * float(physics_dt_s)
        phase_next[free_mask] = torch.remainder(phase[free_mask] + phase_step, 2.0 * math.pi)
    return phase_next

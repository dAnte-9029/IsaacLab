"""Utilities for teacher-guided RL training schedules and action envelopes."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def linear_anneal(step: int, *, start: float, end: float, duration_steps: int) -> float:
    """Linearly interpolate from ``start`` to ``end`` over ``duration_steps``."""
    if duration_steps <= 0:
        return float(end)
    alpha = min(max(float(step) / float(duration_steps), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * alpha


def apply_teacher_action_envelope(teacher_actions: Tensor, rl_actions: Tensor, delta: float | Tensor) -> Tensor:
    """Limit executed actions to a bounded deviation around teacher actions."""
    if teacher_actions.shape != rl_actions.shape:
        raise ValueError("teacher_actions and rl_actions must have the same shape.")

    if not torch.is_tensor(delta):
        delta = torch.tensor(delta, dtype=teacher_actions.dtype, device=teacher_actions.device)
    else:
        delta = delta.to(device=teacher_actions.device, dtype=teacher_actions.dtype)

    bounded_delta = torch.clamp(rl_actions - teacher_actions, min=-delta, max=delta)
    return torch.clamp(teacher_actions + bounded_delta, min=-1.0, max=1.0)

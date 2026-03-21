"""Utilities for teacher-guided RL training schedules and action envelopes."""

from __future__ import annotations

from collections.abc import Sequence

import torch

Tensor = torch.Tensor


def linear_anneal(step: int, *, start: float, end: float, duration_steps: int) -> float:
    """Linearly interpolate from ``start`` to ``end`` over ``duration_steps``."""
    if duration_steps <= 0:
        return float(end)
    alpha = min(max(float(step) / float(duration_steps), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * alpha


def piecewise_linear_anneal(step: int, *, steps: Sequence[int], values: Sequence[float]) -> float:
    """Linearly interpolate across a sequence of schedule knots."""
    if len(steps) != len(values):
        raise ValueError("steps and values must have the same length.")
    if len(steps) == 0:
        raise ValueError("steps and values must not be empty.")
    if any(int(steps[i]) > int(steps[i + 1]) for i in range(len(steps) - 1)):
        raise ValueError("steps must be monotonically non-decreasing.")

    step = int(step)
    if step <= int(steps[0]):
        return float(values[0])

    for idx in range(len(steps) - 1):
        left_step = int(steps[idx])
        right_step = int(steps[idx + 1])
        left_val = float(values[idx])
        right_val = float(values[idx + 1])

        if step <= right_step:
            duration = max(right_step - left_step, 1)
            alpha = float(step - left_step) / float(duration)
            return left_val + (right_val - left_val) * alpha

    return float(values[-1])


def teacher_guidance_is_active(step: int, *, enabled: bool, disable_after_steps: int) -> bool:
    """Return whether teacher guidance should still be applied at ``step``."""
    if not enabled:
        return False
    if int(disable_after_steps) >= 0 and int(step) >= int(disable_after_steps):
        return False
    return True


def compute_teacher_guidance_delta(
    step: int,
    *,
    enabled: bool,
    delta_init: float,
    delta_final: float,
    anneal_steps: int,
    schedule_steps: Sequence[int],
    schedule_deltas: Sequence[float],
    disable_after_steps: int,
) -> float:
    """Compute the current teacher-action envelope width."""
    if not teacher_guidance_is_active(step, enabled=enabled, disable_after_steps=disable_after_steps):
        return float(delta_final)

    if len(schedule_steps) > 0 or len(schedule_deltas) > 0:
        if len(schedule_steps) != len(schedule_deltas):
            raise ValueError("schedule_steps and schedule_deltas must have the same length.")
        return piecewise_linear_anneal(step, steps=schedule_steps, values=schedule_deltas)

    return linear_anneal(step, start=delta_init, end=delta_final, duration_steps=anneal_steps)


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


def compute_recovery_teacher_mask(
    *,
    lateral_error: Tensor,
    height_error: Tensor,
    airspeed: Tensor,
    tilt_deg: Tensor,
    ang_rate_deg_s: Tensor,
    lateral_error_trigger_m: float,
    height_error_trigger_m: float,
    min_safe_airspeed_mps: float,
    tilt_trigger_deg: float,
    ang_rate_trigger_deg_s: float,
) -> Tensor:
    """Return a boolean mask for states that should use recovery teacher guidance."""
    return (
        (torch.abs(lateral_error) > lateral_error_trigger_m)
        | (torch.abs(height_error) > height_error_trigger_m)
        | (airspeed < min_safe_airspeed_mps)
        | (torch.abs(tilt_deg) > tilt_trigger_deg)
        | (torch.abs(ang_rate_deg_s) > ang_rate_trigger_deg_s)
    )

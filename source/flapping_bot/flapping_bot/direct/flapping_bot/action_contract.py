"""Public normalized-action contracts for the flapping-bot direct environments."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

Tensor = torch.Tensor

MIXED_ELEVON_ACTION = "mixed_elevon"
DIRECT_TAIL_SURFACE_ACTION = "direct_tail_surface"
COMMAND_TAIL_AERO_DEFLECTION = "command"
ACTUAL_JOINT_TAIL_AERO_DEFLECTION = "actual_joint_position"


class FrequencyGovernorStep(NamedTuple):
    """Result of one physical-frequency slew-governor step."""

    requested_frequency_hz: Tensor
    applied_frequency_hz: Tensor
    slew_hz_per_s: Tensor
    limited: Tensor


def validate_action_interface(value: str) -> str:
    """Return a supported four-channel action-interface identifier."""

    resolved = str(value)
    if resolved not in {MIXED_ELEVON_ACTION, DIRECT_TAIL_SURFACE_ACTION}:
        raise ValueError(
            "action_interface must be 'mixed_elevon' or 'direct_tail_surface'."
        )
    return resolved


def validate_tail_aero_deflection_source(value: str) -> str:
    """Return a supported source for tail aerodynamic deflections."""

    resolved = str(value)
    if resolved not in {
        COMMAND_TAIL_AERO_DEFLECTION,
        ACTUAL_JOINT_TAIL_AERO_DEFLECTION,
    }:
        raise ValueError(
            "tail_aero_deflection_source must be 'command' or "
            "'actual_joint_position'."
        )
    return resolved


def _validate_frequency_bounds(
    *,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
) -> tuple[float, float]:
    minimum = float(minimum_frequency_hz)
    maximum = float(maximum_frequency_hz)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("Frequency bounds must be finite.")
    if minimum < 0.0:
        raise ValueError("minimum_frequency_hz must be non-negative.")
    if maximum <= minimum:
        raise ValueError("maximum_frequency_hz must be greater than minimum_frequency_hz.")
    return minimum, maximum


def normalized_action_to_frequency_hz(
    action: Tensor,
    *,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
) -> Tensor:
    """Map normalized frequency actions to hertz while preserving tensor metadata."""

    if not isinstance(action, torch.Tensor):
        raise TypeError("action must be a torch tensor.")
    if not bool(torch.all(torch.isfinite(action))):
        raise ValueError("action must be finite.")
    minimum, maximum = _validate_frequency_bounds(
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
    )
    normalized = torch.clamp(action, min=-1.0, max=1.0)
    return minimum + 0.5 * (normalized + 1.0) * (maximum - minimum)


def frequency_hz_to_normalized_action(
    frequency_hz: Tensor,
    *,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
) -> Tensor:
    """Map hertz to the normalized frequency action and clamp to ``[-1, 1]``."""

    if not isinstance(frequency_hz, torch.Tensor):
        raise TypeError("frequency_hz must be a torch tensor.")
    if not bool(torch.all(torch.isfinite(frequency_hz))):
        raise ValueError("frequency_hz must be finite.")
    minimum, maximum = _validate_frequency_bounds(
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
    )
    action = 2.0 * (frequency_hz - minimum) / (maximum - minimum) - 1.0
    return torch.clamp(action, min=-1.0, max=1.0)


def apply_frequency_slew_governor(
    requested_frequency_hz: Tensor,
    *,
    previous_frequency_hz: Tensor,
    policy_step_dt_s: float,
    maximum_rise_rate_hz_per_s: float,
    maximum_fall_rate_hz_per_s: float,
) -> FrequencyGovernorStep:
    """Limit a requested frequency change in physical ``Hz/s`` units.

    The operation is elementwise and never overshoots a request inside the
    configured rise or fall envelope. Tensor shape, device, and dtype are
    preserved.
    """

    _validate_frequency_tensor("requested_frequency_hz", requested_frequency_hz)
    _validate_frequency_tensor("previous_frequency_hz", previous_frequency_hz)
    if requested_frequency_hz.shape != previous_frequency_hz.shape:
        raise ValueError("Frequency tensors must have the same shape.")
    if requested_frequency_hz.device != previous_frequency_hz.device:
        raise ValueError("Frequency tensors must use the same device.")
    if requested_frequency_hz.dtype != previous_frequency_hz.dtype:
        raise ValueError("Frequency tensors must use the same dtype.")

    policy_dt = float(policy_step_dt_s)
    if not math.isfinite(policy_dt) or policy_dt <= 0.0:
        raise ValueError("policy_step_dt_s must be finite and positive.")
    rise_rate = float(maximum_rise_rate_hz_per_s)
    fall_rate = float(maximum_fall_rate_hz_per_s)
    if any(not math.isfinite(value) or value <= 0.0 for value in (rise_rate, fall_rate)):
        raise ValueError("Frequency governor rate limits must be finite and positive.")

    requested_delta_hz = requested_frequency_hz - previous_frequency_hz
    applied_delta_hz = torch.clamp(
        requested_delta_hz,
        min=-fall_rate * policy_dt,
        max=rise_rate * policy_dt,
    )
    applied_frequency_hz = previous_frequency_hz + applied_delta_hz
    return FrequencyGovernorStep(
        requested_frequency_hz=requested_frequency_hz,
        applied_frequency_hz=applied_frequency_hz,
        slew_hz_per_s=applied_delta_hz / policy_dt,
        limited=applied_delta_hz != requested_delta_hz,
    )


def normalized_action_to_joint_position(
    action: Tensor,
    *,
    lower_limit_rad: Tensor | float,
    upper_limit_rad: Tensor | float,
) -> Tensor:
    """Map normalized actions to zero-centered joint limits in radians.

    Positive actions scale toward the upper limit and negative actions scale
    toward the lower limit. This preserves zero as the neutral command even
    when a URDF joint has asymmetric limits. Inputs and outputs use the same
    tensor device and dtype.
    """

    lower, upper = _joint_limit_tensors(
        action,
        lower_limit_rad=lower_limit_rad,
        upper_limit_rad=upper_limit_rad,
    )
    normalized = torch.clamp(action, min=-1.0, max=1.0)
    return torch.where(normalized >= 0.0, normalized * upper, -normalized * lower)


def joint_position_to_normalized_action(
    joint_position_rad: Tensor,
    *,
    lower_limit_rad: Tensor | float,
    upper_limit_rad: Tensor | float,
) -> Tensor:
    """Map a zero-centered joint position in radians to `[-1, 1]`."""

    lower, upper = _joint_limit_tensors(
        joint_position_rad,
        lower_limit_rad=lower_limit_rad,
        upper_limit_rad=upper_limit_rad,
    )
    position = torch.minimum(torch.maximum(joint_position_rad, lower), upper)
    normalized = torch.where(position >= 0.0, position / upper, -position / lower)
    return torch.clamp(normalized, min=-1.0, max=1.0)


def _joint_limit_tensors(
    value: Tensor,
    *,
    lower_limit_rad: Tensor | float,
    upper_limit_rad: Tensor | float,
) -> tuple[Tensor, Tensor]:
    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a torch tensor.")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("value must be finite.")
    lower = torch.as_tensor(lower_limit_rad, device=value.device, dtype=value.dtype)
    upper = torch.as_tensor(upper_limit_rad, device=value.device, dtype=value.dtype)
    try:
        torch.broadcast_shapes(value.shape, lower.shape, upper.shape)
    except RuntimeError as exc:
        raise ValueError("Joint limits must broadcast with value.") from exc
    if not bool(torch.all(torch.isfinite(lower))) or not bool(torch.all(torch.isfinite(upper))):
        raise ValueError("Joint limits must be finite.")
    if not bool(torch.all(lower < 0.0)) or not bool(torch.all(upper > 0.0)):
        raise ValueError("Joint limits must strictly contain zero.")
    return lower, upper


def _validate_frequency_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor.")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating dtype.")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must be finite.")
    if bool(torch.any(value < 0.0)):
        raise ValueError(f"{name} must be non-negative.")


__all__ = [
    "ACTUAL_JOINT_TAIL_AERO_DEFLECTION",
    "COMMAND_TAIL_AERO_DEFLECTION",
    "DIRECT_TAIL_SURFACE_ACTION",
    "FrequencyGovernorStep",
    "MIXED_ELEVON_ACTION",
    "apply_frequency_slew_governor",
    "frequency_hz_to_normalized_action",
    "joint_position_to_normalized_action",
    "normalized_action_to_frequency_hz",
    "normalized_action_to_joint_position",
    "validate_action_interface",
    "validate_tail_aero_deflection_source",
]

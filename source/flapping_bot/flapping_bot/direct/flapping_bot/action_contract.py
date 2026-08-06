"""Public normalized-action contracts for the flapping-bot direct environments."""

from __future__ import annotations

import math

import torch

Tensor = torch.Tensor

MIXED_ELEVON_ACTION = "mixed_elevon"
DIRECT_TAIL_SURFACE_ACTION = "direct_tail_surface"
COMMAND_TAIL_AERO_DEFLECTION = "command"
ACTUAL_JOINT_TAIL_AERO_DEFLECTION = "actual_joint_position"


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


__all__ = [
    "ACTUAL_JOINT_TAIL_AERO_DEFLECTION",
    "COMMAND_TAIL_AERO_DEFLECTION",
    "DIRECT_TAIL_SURFACE_ACTION",
    "MIXED_ELEVON_ACTION",
    "frequency_hz_to_normalized_action",
    "joint_position_to_normalized_action",
    "normalized_action_to_frequency_hz",
    "normalized_action_to_joint_position",
    "validate_action_interface",
    "validate_tail_aero_deflection_source",
]

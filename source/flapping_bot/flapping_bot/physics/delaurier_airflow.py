"""Explicit body-to-section airflow convention for the DeLaurier model."""

from __future__ import annotations

from typing import Literal

import torch

Tensor = torch.Tensor
BodyFrameConvention = Literal["FLU", "FRD"]


def body_air_velocity_to_delaurier_section_velocity(
    air_velocity_body: Tensor,
    *,
    body_frame: BodyFrameConvention,
) -> Tensor:
    """Express vehicle air-relative velocity in the DeLaurier section frame.

    Args:
        air_velocity_body: Vehicle velocity relative to the air, shape
            ``(...,3)``, in m/s. This is a polar vector.
        body_frame: Convention of the input vector. ``FLU`` means ``+x``
            forward, ``+y`` left, ``+z`` up. ``FRD`` means ``+x`` forward,
            ``+y`` right, ``+z`` down.

    Returns:
        The same physical polar vector in the internal DeLaurier section
        convention ``(+x forward, +y right, +z down)``, shape ``(...,3)``.

    Notes:
        For an FLU input, ``v_D = diag(1,-1,-1) v_FLU``. The transform is an
        explicit frame conversion; it is not an empirical aerodynamic sign.
    """

    if not isinstance(air_velocity_body, torch.Tensor):
        raise TypeError("air_velocity_body must be a torch tensor.")
    if air_velocity_body.ndim < 1 or air_velocity_body.shape[-1] != 3:
        raise ValueError("air_velocity_body must have shape (...,3).")
    if body_frame == "FRD":
        return air_velocity_body.clone()
    if body_frame == "FLU":
        section_velocity = air_velocity_body.clone()
        section_velocity[..., 1:] = -section_velocity[..., 1:]
        return section_velocity
    raise ValueError(f"Unsupported body frame {body_frame!r}; expected 'FLU' or 'FRD'.")


def compute_delaurier_axis_incidence(
    *,
    air_velocity_body: Tensor,
    body_frame: BodyFrameConvention,
    minimum_forward_speed_mps: float = 1.0e-3,
) -> Tensor:
    """Return DeLaurier flapping-axis incidence ``theta_a`` in radians.

    The internal definition is ``theta_a = atan2(w_D, u_D)`` where ``u_D``
    is forward and ``w_D`` is positive downward. Positive incidence therefore
    corresponds to a vehicle air-relative velocity component below the body
    ``+x`` axis (the usual positive angle-of-attack geometry).

    Args:
        air_velocity_body: Vehicle air-relative velocity, shape ``(...,3)``,
            in m/s.
        body_frame: ``FLU`` or ``FRD`` convention of the input.
        minimum_forward_speed_mps: Positive lower bound applied only to
            ``u_D`` in the ``atan2`` denominator, matching the environment's
            historical low-speed guard.

    Returns:
        Incidence tensor with shape ``(...)`` in rad.
    """

    minimum_forward_speed = float(minimum_forward_speed_mps)
    if minimum_forward_speed <= 0.0:
        raise ValueError("minimum_forward_speed_mps must be positive.")
    section_velocity = body_air_velocity_to_delaurier_section_velocity(
        air_velocity_body,
        body_frame=body_frame,
    )
    forward_velocity = torch.clamp(section_velocity[..., 0], min=minimum_forward_speed)
    downward_velocity = section_velocity[..., 2]
    return torch.atan2(downward_velocity, forward_velocity)

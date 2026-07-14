"""Prescribed pitch kinematics for the DeLaurier wing model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


DELAURIER_DYNAMIC_TWIST_MODES = frozenset(
    {
        "disabled",
        "delaurier_linear_spanwise",
        "legacy_qd_scaled_proxy",
    }
)


@dataclass(frozen=True)
class DeLaurierTwistKinematics:
    """Prescribed strip pitch and its analytical time derivatives.

    ``theta``, ``theta_dot``, ``theta_ddot`` and the corresponding ``delta``
    fields have shape ``(B, N)`` and units rad, rad/s and rad/s².  Positive
    pitch is the right-hand rotation about Wang ``+x``, the spanwise pitching
    axis from root to tip. ``span_fraction`` is dimensionless with the same
    shape. ``phase``, ``phase_rate`` and ``phase_acceleration`` have shape
    ``(B, 1)`` and use rad, rad/s and rad/s².

    This object describes prescribed kinematics only.  It is not a passive
    aeroelastic state or a structural-dynamics solution.
    """

    theta: Tensor
    theta_dot: Tensor
    theta_ddot: Tensor
    delta_theta: Tensor
    delta_theta_dot: Tensor
    delta_theta_ddot: Tensor
    span_fraction: Tensor
    phase: Tensor
    phase_rate: Tensor
    phase_acceleration: Tensor


@dataclass(frozen=True)
class LegacyQdScaledTwistKinematics:
    """Frozen full-span-uniform kinematics of the historical ``qd`` proxy.

    Every field has shape ``(B, N)``. Angles use rad and their derivatives
    use rad/s and rad/s². The helper intentionally preserves the historical
    behavior in which only ``delta_theta`` is clamped: its derivative fields
    remain proportional to the supplied joint acceleration and jerk even when
    the angle is saturated. This contract exists for result reproduction, not
    as a claim of passive-wing physics.
    """

    theta: Tensor
    theta_dot: Tensor
    theta_ddot: Tensor
    delta_theta: Tensor
    delta_theta_dot: Tensor
    delta_theta_ddot: Tensor


def validate_delaurier_dynamic_twist_mode(mode: str) -> str:
    """Return a supported dynamic-twist mode or raise a clear error."""

    resolved_mode = str(mode)
    if resolved_mode not in DELAURIER_DYNAMIC_TWIST_MODES:
        supported = ", ".join(sorted(DELAURIER_DYNAMIC_TWIST_MODES))
        raise ValueError(f"Unsupported DeLaurier dynamic_twist_mode {resolved_mode!r}; expected one of: {supported}.")
    return resolved_mode


def resolve_delaurier_phase(
    *,
    current_phase: Tensor,
    current_phase_rate: Tensor,
    current_phase_acceleration: Tensor,
    phase_direction: float,
    phase_offset_rad: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map the engineering stroke phase to the DeLaurier phase convention.

    The returned tensors preserve the broadcast shape, device and dtype of the
    three input tensors.  The mapping is
    ``phi_D = phase_direction * current_phase + phase_offset_rad``; its first
    and second derivatives use the same direction. ``phase_direction`` must be
    exactly ``-1`` or ``+1`` so a sign convention cannot be hidden in an
    arbitrary scale factor.
    """

    if not isinstance(current_phase, torch.Tensor):
        raise TypeError("current_phase must be a torch tensor.")
    direction = float(phase_direction)
    if direction not in {-1.0, 1.0}:
        raise ValueError("phase_direction must be -1.0 or +1.0.")

    phase_rate = torch.as_tensor(
        current_phase_rate,
        device=current_phase.device,
        dtype=current_phase.dtype,
    )
    phase_acceleration = torch.as_tensor(
        current_phase_acceleration,
        device=current_phase.device,
        dtype=current_phase.dtype,
    )
    current_phase, phase_rate, phase_acceleration = torch.broadcast_tensors(
        current_phase,
        phase_rate,
        phase_acceleration,
    )
    phase = direction * current_phase + float(phase_offset_rad)
    return phase, direction * phase_rate, direction * phase_acceleration


def _as_strip_matrix(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor.")
    if value.ndim == 1:
        return value.view(1, -1)
    if value.ndim == 2:
        return value
    raise ValueError(f"{name} must have shape (N,) or (B,N).")


def _as_batch_column(value: Tensor | float, *, reference: Tensor, name: str) -> Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        return tensor.view(1, 1)
    if tensor.ndim == 1:
        return tensor.view(-1, 1)
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        return tensor
    raise ValueError(f"{name} must be scalar or have shape (B,) or (B,1).")


def _as_mean_pitch_matrix(value: Tensor | float, *, reference: Tensor, num_strips: int) -> Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        return tensor.view(1, 1)
    if tensor.ndim == 1:
        return tensor.view(-1, 1)
    if tensor.ndim == 2 and tensor.shape[1] in {1, num_strips}:
        return tensor
    raise ValueError("mean_pitch_rad must be scalar or have shape (B,), (B,1), or (B,N).")


def _resolve_batch_size(*tensors: Tensor) -> int:
    non_singleton_sizes = {int(tensor.shape[0]) for tensor in tensors if tensor.shape[0] != 1}
    if len(non_singleton_sizes) > 1:
        raise ValueError(f"Incompatible batch dimensions: {sorted(non_singleton_sizes)}.")
    return non_singleton_sizes.pop() if non_singleton_sizes else 1


def _expand_batch(tensor: Tensor, *, batch_size: int, name: str) -> Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    raise ValueError(f"{name} batch dimension must be 1 or {batch_size}.")


def compute_legacy_qd_scaled_twist(
    *,
    num_strips: int,
    mean_pitch_rad: Tensor | float,
    joint_velocity_rad_s: Tensor,
    joint_acceleration_rad_s2: Tensor,
    joint_jerk_rad_s3: Tensor,
    reference_velocity_rad_s: Tensor | float,
    maximum_twist_rad: Tensor | float,
    twist_limit_rad: Tensor | float,
    twist_sign: Tensor | float,
) -> LegacyQdScaledTwistKinematics:
    """Reproduce the historical full-span-uniform ``qd`` twist proxy.

    Args:
        num_strips: Number of output strips ``N``.
        mean_pitch_rad: Mean pitch, scalar or shape ``(B,)``, ``(B,1)`` or
            ``(B,N)``, in rad.
        joint_velocity_rad_s: Flapping-joint velocity, shape ``(B,)`` or
            ``(B,1)``, in rad/s.
        joint_acceleration_rad_s2: Flapping-joint acceleration with the same
            batch shape, in rad/s².
        joint_jerk_rad_s3: Flapping-joint jerk with the same batch shape, in
            rad/s³.
        reference_velocity_rad_s: Positive ``qd_ref`` used to normalize and
            clamp ``qd/qd_ref`` to ``[-1,1]``. Historical behavior clamps this
            denominator to at least ``1e-6`` rad/s.
        maximum_twist_rad: Historical ``eta_max`` scale, in rad.
        twist_limit_rad: Final absolute angle clamp ``eta_lim``, in rad.
        twist_sign: Per-wing historical scalar sign, scalar or batch-shaped.

    Returns:
        Frozen legacy pitch kinematics, all shaped ``(B,N)``.

    Notes:
        This helper deliberately has no spanwise distribution. It also does
        not differentiate either clamp; derivative outputs retain the exact
        historical algebra used before this function was extracted.
    """

    if int(num_strips) <= 0:
        raise ValueError("num_strips must be positive.")
    if not isinstance(joint_velocity_rad_s, torch.Tensor):
        raise TypeError("joint_velocity_rad_s must be a torch tensor.")

    reference = joint_velocity_rad_s
    joint_velocity = _as_batch_column(reference, reference=reference, name="joint_velocity_rad_s")
    joint_acceleration = _as_batch_column(
        joint_acceleration_rad_s2,
        reference=reference,
        name="joint_acceleration_rad_s2",
    )
    joint_jerk = _as_batch_column(joint_jerk_rad_s3, reference=reference, name="joint_jerk_rad_s3")
    reference_velocity = _as_batch_column(
        reference_velocity_rad_s,
        reference=reference,
        name="reference_velocity_rad_s",
    )
    maximum_twist = _as_batch_column(maximum_twist_rad, reference=reference, name="maximum_twist_rad")
    twist_limit = _as_batch_column(twist_limit_rad, reference=reference, name="twist_limit_rad")
    resolved_twist_sign = _as_batch_column(twist_sign, reference=reference, name="twist_sign")
    mean_pitch = _as_mean_pitch_matrix(mean_pitch_rad, reference=reference, num_strips=int(num_strips))

    batch_size = _resolve_batch_size(
        joint_velocity,
        joint_acceleration,
        joint_jerk,
        reference_velocity,
        maximum_twist,
        twist_limit,
        resolved_twist_sign,
        mean_pitch,
    )
    joint_velocity = _expand_batch(joint_velocity, batch_size=batch_size, name="joint_velocity_rad_s")
    joint_acceleration = _expand_batch(
        joint_acceleration,
        batch_size=batch_size,
        name="joint_acceleration_rad_s2",
    )
    joint_jerk = _expand_batch(joint_jerk, batch_size=batch_size, name="joint_jerk_rad_s3")
    reference_velocity = _expand_batch(
        reference_velocity,
        batch_size=batch_size,
        name="reference_velocity_rad_s",
    )
    maximum_twist = _expand_batch(maximum_twist, batch_size=batch_size, name="maximum_twist_rad")
    twist_limit = _expand_batch(twist_limit, batch_size=batch_size, name="twist_limit_rad")
    resolved_twist_sign = _expand_batch(resolved_twist_sign, batch_size=batch_size, name="twist_sign")
    mean_pitch = _expand_batch(mean_pitch, batch_size=batch_size, name="mean_pitch_rad")
    if mean_pitch.shape[1] == 1:
        mean_pitch = mean_pitch.expand(batch_size, int(num_strips))

    if torch.any(maximum_twist < 0.0):
        raise ValueError("maximum_twist_rad must be non-negative.")
    if torch.any(twist_limit < 0.0):
        raise ValueError("twist_limit_rad must be non-negative.")

    safe_reference_velocity = torch.clamp(reference_velocity, min=1.0e-6)
    normalized_flap_rate = torch.clamp(joint_velocity / safe_reference_velocity, min=-1.0, max=1.0)
    delta_theta_scalar = torch.clamp(
        maximum_twist * resolved_twist_sign * normalized_flap_rate,
        min=-twist_limit,
        max=twist_limit,
    )
    twist_gain = maximum_twist / safe_reference_velocity
    delta_theta_dot_scalar = twist_gain * resolved_twist_sign * joint_acceleration
    delta_theta_ddot_scalar = twist_gain * resolved_twist_sign * joint_jerk

    delta_theta = delta_theta_scalar.expand(batch_size, int(num_strips))
    delta_theta_dot = delta_theta_dot_scalar.expand(batch_size, int(num_strips))
    delta_theta_ddot = delta_theta_ddot_scalar.expand(batch_size, int(num_strips))
    return LegacyQdScaledTwistKinematics(
        theta=mean_pitch + delta_theta,
        theta_dot=delta_theta_dot,
        theta_ddot=delta_theta_ddot,
        delta_theta=delta_theta,
        delta_theta_dot=delta_theta_dot,
        delta_theta_ddot=delta_theta_ddot,
    )


def compute_delaurier_dynamic_twist(
    *,
    strip_span_m: Tensor,
    strip_width_m: Tensor,
    mean_pitch_rad: Tensor | float,
    tip_twist_amplitude_rad: Tensor | float,
    phase_rad: Tensor,
    phase_rate_rad_s: Tensor,
    phase_acceleration_rad_s2: Tensor,
    enabled: bool,
    semi_span_m: Tensor | float | None = None,
) -> DeLaurierTwistKinematics:
    """Compute DeLaurier's prescribed linear-spanwise dynamic twist.

    Args:
        strip_span_m: Strip-center distance from the wing-root pitching-axis
            origin, shape ``(N,)`` or ``(B,N)``, in m.
        strip_width_m: Strip width, shape ``(N,)`` or ``(B,N)``, in m.
        mean_pitch_rad: Mean strip pitch ``theta_bar``, scalar or shape ``(B,)``,
            ``(B,1)`` or ``(B,N)``, in rad.
        tip_twist_amplitude_rad: Dynamic-twist amplitude at the theoretical
            geometric tip ``y=R``, scalar or shape ``(B,)``/``(B,1)``, in rad.
        phase_rad: DeLaurier phase, shape ``(B,)`` or ``(B,1)``, in rad.
        phase_rate_rad_s: First phase derivative, same batch shape, in rad/s.
        phase_acceleration_rad_s2: Second phase derivative, same batch shape,
            in rad/s².  It may be non-zero for variable-frequency motion.
        enabled: If false, return ``theta=theta_bar`` and zero dynamic terms.
        semi_span_m: Optional explicit geometric semi-span ``R`` in m.  If
            omitted, infer it from the outermost strip edge
            ``max(y_i + 0.5*width_i)`` rather than from the last strip center.

    Returns:
        Strip pitch kinematics with all strip fields shaped ``(B,N)``.

    Notes:
        The enabled model is ``delta_theta = -theta_tip*(y/R)*sin(phi_D)``.
        Its derivatives are analytical; no numerical differencing is used.
    """

    strip_span = _as_strip_matrix(strip_span_m, name="strip_span_m")
    strip_width = _as_strip_matrix(strip_width_m, name="strip_width_m").to(
        device=strip_span.device,
        dtype=strip_span.dtype,
    )
    if strip_span.shape[1] != strip_width.shape[1]:
        raise ValueError("strip_span_m and strip_width_m must have the same number of strips.")
    if torch.any(strip_span < 0.0):
        raise ValueError("strip_span_m must be measured outward from the wing root and be non-negative.")
    if torch.any(strip_width <= 0.0):
        raise ValueError("strip_width_m must be positive.")

    num_strips = int(strip_span.shape[1])
    mean_pitch = _as_mean_pitch_matrix(mean_pitch_rad, reference=strip_span, num_strips=num_strips)
    tip_amplitude = _as_batch_column(
        tip_twist_amplitude_rad,
        reference=strip_span,
        name="tip_twist_amplitude_rad",
    )
    phase = _as_batch_column(phase_rad, reference=strip_span, name="phase_rad")
    phase_rate = _as_batch_column(phase_rate_rad_s, reference=strip_span, name="phase_rate_rad_s")
    phase_acceleration = _as_batch_column(
        phase_acceleration_rad_s2,
        reference=strip_span,
        name="phase_acceleration_rad_s2",
    )

    batch_size = _resolve_batch_size(
        strip_span,
        strip_width,
        mean_pitch,
        tip_amplitude,
        phase,
        phase_rate,
        phase_acceleration,
    )
    strip_span = _expand_batch(strip_span, batch_size=batch_size, name="strip_span_m")
    strip_width = _expand_batch(strip_width, batch_size=batch_size, name="strip_width_m")
    mean_pitch = _expand_batch(mean_pitch, batch_size=batch_size, name="mean_pitch_rad")
    tip_amplitude = _expand_batch(tip_amplitude, batch_size=batch_size, name="tip_twist_amplitude_rad")
    phase = _expand_batch(phase, batch_size=batch_size, name="phase_rad")
    phase_rate = _expand_batch(phase_rate, batch_size=batch_size, name="phase_rate_rad_s")
    phase_acceleration = _expand_batch(
        phase_acceleration,
        batch_size=batch_size,
        name="phase_acceleration_rad_s2",
    )
    if mean_pitch.shape[1] == 1:
        mean_pitch = mean_pitch.expand(batch_size, num_strips)

    if semi_span_m is None:
        semi_span = torch.max(strip_span + 0.5 * strip_width, dim=1, keepdim=True).values
    else:
        semi_span = _as_batch_column(semi_span_m, reference=strip_span, name="semi_span_m")
        semi_span = _expand_batch(semi_span, batch_size=batch_size, name="semi_span_m")
    if torch.any(semi_span <= 0.0):
        raise ValueError("semi_span_m must be positive.")

    span_fraction = strip_span / semi_span
    if torch.any(span_fraction > 1.0 + 1.0e-6):
        raise ValueError("A strip center lies beyond the supplied geometric semi-span.")

    if enabled:
        delta_theta = -tip_amplitude * span_fraction * torch.sin(phase)
        delta_theta_dot = -tip_amplitude * span_fraction * torch.cos(phase) * phase_rate
        delta_theta_ddot = tip_amplitude * span_fraction * (
            torch.sin(phase) * phase_rate.square()
            - torch.cos(phase) * phase_acceleration
        )
    else:
        delta_theta = torch.zeros_like(span_fraction)
        delta_theta_dot = torch.zeros_like(span_fraction)
        delta_theta_ddot = torch.zeros_like(span_fraction)

    return DeLaurierTwistKinematics(
        theta=mean_pitch + delta_theta,
        theta_dot=delta_theta_dot,
        theta_ddot=delta_theta_ddot,
        delta_theta=delta_theta,
        delta_theta_dot=delta_theta_dot,
        delta_theta_ddot=delta_theta_ddot,
        span_fraction=span_fraction,
        phase=phase,
        phase_rate=phase_rate,
        phase_acceleration=phase_acceleration,
    )

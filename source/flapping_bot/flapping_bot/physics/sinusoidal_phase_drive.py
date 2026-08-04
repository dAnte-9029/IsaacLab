"""Pure phase-speed drive model for an ideal sinusoidal wing mechanism."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .ideal_torque_drive import IDEAL_TORQUE_EQUIVALENT_INERTIA_KG_M2
from .ideal_torque_drive import ideal_torque_drive_gains

Tensor = torch.Tensor


@dataclass(frozen=True)
class SinusoidalPhaseSpeedDriveConfig:
    """Configuration for the common phase-speed drive.

    Torque quantities are generalized about the dimensionless phase coordinate
    and are expressed in N m. The effort limit and phase-inertia floor are
    numerical settings, not identified motor parameters.
    """

    amplitude_rad: float = math.radians(30.0)
    max_frequency_hz: float = 5.0
    common_joint_inertia_kg_m2: float = IDEAL_TORQUE_EQUIVALENT_INERTIA_KG_M2
    phase_inertia_floor_ratio: float = 0.01
    constant_phase_inertia_kg_m2: float | None = None
    phase_viscous_damping_nm_s: float = 0.0
    frequency_settling_time_s: float = 0.15
    frequency_damping_ratio: float = 1.0
    effort_limit_nm: float = 1000.0

    def __post_init__(self) -> None:
        if self.amplitude_rad <= 0.0:
            raise ValueError("amplitude_rad must be positive.")
        if self.max_frequency_hz <= 0.0:
            raise ValueError("max_frequency_hz must be positive.")
        if self.common_joint_inertia_kg_m2 < 0.0:
            raise ValueError("common_joint_inertia_kg_m2 must be nonnegative.")
        if self.phase_inertia_floor_ratio <= 0.0:
            raise ValueError("phase_inertia_floor_ratio must be positive.")
        if self.phase_viscous_damping_nm_s < 0.0:
            raise ValueError("phase_viscous_damping_nm_s must be nonnegative.")
        if self.frequency_settling_time_s <= 0.0:
            raise ValueError("frequency_settling_time_s must be positive.")
        if self.frequency_damping_ratio <= 0.0:
            raise ValueError("frequency_damping_ratio must be positive.")
        if self.effort_limit_nm <= 0.0:
            raise ValueError("effort_limit_nm must be positive.")
        if (
            self.constant_phase_inertia_kg_m2 is not None
            and self.constant_phase_inertia_kg_m2 <= 0.0
        ):
            raise ValueError("constant_phase_inertia_kg_m2 must be positive when provided.")
        if self.phase_inertia_floor_kg_m2 <= 0.0:
            raise ValueError(
                "The resolved phase inertia floor must be positive; provide "
                "constant_phase_inertia_kg_m2 when common_joint_inertia_kg_m2 is zero."
            )

    @property
    def peak_transformed_wing_inertia_kg_m2(self) -> float:
        """Return the peak wing inertia about phase."""

        return self.common_joint_inertia_kg_m2 * self.amplitude_rad**2

    @property
    def phase_inertia_floor_kg_m2(self) -> float:
        """Return the explicit numerical phase-inertia floor."""

        if self.constant_phase_inertia_kg_m2 is not None:
            return float(self.constant_phase_inertia_kg_m2)
        return self.phase_inertia_floor_ratio * self.peak_transformed_wing_inertia_kg_m2


@dataclass(frozen=True)
class SinusoidalPhaseDriveState:
    """Batched common mechanism state.

    Fields have shape ``(...)`` and units rad, rad/s and rad, respectively.
    The phase is intentionally unwrapped.
    """

    phase_rad: Tensor
    phase_rate_rad_s: Tensor
    speed_error_integral_rad: Tensor

    @classmethod
    def zeros(
        cls,
        batch_shape: tuple[int, ...],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "SinusoidalPhaseDriveState":
        """Construct a zero state on the requested device and dtype."""

        zero = torch.zeros(batch_shape, device=device, dtype=dtype)
        return cls(
            phase_rad=zero,
            phase_rate_rad_s=zero.clone(),
            speed_error_integral_rad=zero.clone(),
        )


@dataclass(frozen=True)
class OpposedWingKinematics:
    """Sinusoidal wing kinematics in URDF joint coordinates.

    All tensors have shape ``(...)``. Positions, velocities and accelerations
    use rad, rad/s and rad/s^2. Positive common position maps to left ``+q``
    and right ``-q``.
    """

    common_position_rad: Tensor
    common_velocity_rad_s: Tensor
    common_acceleration_rad_s2: Tensor
    left_position_rad: Tensor
    right_position_rad: Tensor
    left_velocity_rad_s: Tensor
    right_velocity_rad_s: Tensor
    left_acceleration_rad_s2: Tensor
    right_acceleration_rad_s2: Tensor


@dataclass(frozen=True)
class PhaseSpeedPIResult:
    """PI output and saturation diagnostics, each with shape ``(...)``."""

    drive_torque_nm: Tensor
    unsaturated_drive_torque_nm: Tensor
    next_speed_error_integral_rad: Tensor
    saturated: Tensor


@dataclass(frozen=True)
class SinusoidalConstraintEffort:
    """Nonlinear phase-to-wing constraint output with shape ``(...)``.

    Position error is in rad, velocity error in rad/s, common wing effort in
    N m and phase reaction in N m about the dimensionless phase coordinate.
    """

    position_error_rad: Tensor
    velocity_error_rad_s: Tensor
    common_wing_effort_nm: Tensor
    phase_reaction_torque_nm: Tensor
    saturated: Tensor


@dataclass(frozen=True)
class SinusoidalPhaseDriveStep:
    """One pure-model step and its physical diagnostics.

    ``phase_external_torque_nm`` is the signed generalized phase torque. A
    positive value does positive work when ``phase_rate_rad_s`` is positive.
    """

    next_state: SinusoidalPhaseDriveState
    kinematics: OpposedWingKinematics
    target_frequency_hz: Tensor
    actual_frequency_hz: Tensor
    target_phase_rate_rad_s: Tensor
    phase_acceleration_rad_s2: Tensor
    phase_inertia_kg_m2: Tensor
    phase_inertia_derivative_kg_m2: Tensor
    phase_external_torque_nm: Tensor
    drive_torque_nm: Tensor
    unsaturated_drive_torque_nm: Tensor
    drive_saturated: Tensor


def _validate_shared_tensor_contract(reference: Tensor, *others: Tensor) -> None:
    if not isinstance(reference, torch.Tensor) or any(not isinstance(value, torch.Tensor) for value in others):
        raise TypeError("All state and input values must be torch tensors.")
    if any(value.shape != reference.shape for value in others):
        raise ValueError("All state and input tensors must share shape.")
    if any(value.device != reference.device or value.dtype != reference.dtype for value in others):
        raise ValueError("All state and input tensors must share device and dtype.")
    if not bool(torch.all(torch.isfinite(reference))) or any(
        not bool(torch.all(torch.isfinite(value))) for value in others
    ):
        raise ValueError("State and input tensors must be finite.")


def map_virtual_throttle_to_frequency_hz(
    *,
    throttle_01: Tensor,
    max_frequency_hz: float,
) -> Tensor:
    """Map virtual throttle to frequency setpoint in Hz.

    Input and output have shape ``(...)``. Values outside ``[0,1]`` are
    explicitly clamped.
    """

    if not isinstance(throttle_01, torch.Tensor):
        raise TypeError("throttle_01 must be a torch tensor.")
    if not bool(torch.all(torch.isfinite(throttle_01))):
        raise ValueError("throttle_01 must be finite.")
    if max_frequency_hz <= 0.0:
        raise ValueError("max_frequency_hz must be positive.")
    return torch.clamp(throttle_01, min=0.0, max=1.0) * float(max_frequency_hz)


def compute_opposed_wing_kinematics(
    *,
    phase_rad: Tensor,
    phase_rate_rad_s: Tensor,
    phase_acceleration_rad_s2: Tensor,
    amplitude_rad: float,
    left_joint_mid_rad: float = 0.0,
    right_joint_mid_rad: float = 0.0,
) -> OpposedWingKinematics:
    """Return exact variable-frequency sine kinematics for both wings."""

    _validate_shared_tensor_contract(phase_rad, phase_rate_rad_s, phase_acceleration_rad_s2)
    amplitude = float(amplitude_rad)
    if amplitude <= 0.0:
        raise ValueError("amplitude_rad must be positive.")

    sin_phase = torch.sin(phase_rad)
    cos_phase = torch.cos(phase_rad)
    common_position = amplitude * sin_phase
    common_velocity = amplitude * cos_phase * phase_rate_rad_s
    common_acceleration = amplitude * (
        cos_phase * phase_acceleration_rad_s2
        - sin_phase * phase_rate_rad_s.square()
    )
    return OpposedWingKinematics(
        common_position_rad=common_position,
        common_velocity_rad_s=common_velocity,
        common_acceleration_rad_s2=common_acceleration,
        left_position_rad=float(left_joint_mid_rad) + common_position,
        right_position_rad=float(right_joint_mid_rad) - common_position,
        left_velocity_rad_s=common_velocity,
        right_velocity_rad_s=-common_velocity,
        left_acceleration_rad_s2=common_acceleration,
        right_acceleration_rad_s2=-common_acceleration,
    )


def project_common_joint_torque_to_phase(
    *,
    common_joint_torque_nm: Tensor,
    phase_rad: Tensor,
    amplitude_rad: float,
) -> Tensor:
    """Project signed common-joint torque into phase by virtual work."""

    _validate_shared_tensor_contract(common_joint_torque_nm, phase_rad)
    amplitude = float(amplitude_rad)
    if amplitude <= 0.0:
        raise ValueError("amplitude_rad must be positive.")
    return amplitude * torch.cos(phase_rad) * common_joint_torque_nm


def compute_sinusoidal_constraint_effort(
    *,
    actual_common_position_rad: Tensor,
    actual_common_velocity_rad_s: Tensor,
    reference_common_position_rad: Tensor,
    reference_common_velocity_rad_s: Tensor,
    phase_rad: Tensor,
    amplitude_rad: float,
    equivalent_common_inertia_kg_m2: float,
    natural_frequency_hz: float,
    damping_ratio: float,
    effort_limit_nm: float,
    physics_dt_s: float | None = None,
) -> SinusoidalConstraintEffort:
    """Return a power-consistent penalty effort for ``q=Gamma*sin(phi)``."""

    _validate_shared_tensor_contract(
        actual_common_position_rad,
        actual_common_velocity_rad_s,
        reference_common_position_rad,
        reference_common_velocity_rad_s,
        phase_rad,
    )
    amplitude = float(amplitude_rad)
    effort_limit = float(effort_limit_nm)
    if amplitude <= 0.0:
        raise ValueError("amplitude_rad must be positive.")
    if effort_limit <= 0.0:
        raise ValueError("effort_limit_nm must be positive.")
    stiffness, damping = ideal_torque_drive_gains(
        equivalent_inertia_kg_m2=equivalent_common_inertia_kg_m2,
        natural_frequency_hz=natural_frequency_hz,
        damping_ratio=damping_ratio,
        physics_dt_s=physics_dt_s,
    )
    position_error = actual_common_position_rad - reference_common_position_rad
    velocity_error = actual_common_velocity_rad_s - reference_common_velocity_rad_s
    unsaturated_effort = -stiffness * position_error - damping * velocity_error
    common_effort = torch.clamp(
        unsaturated_effort,
        min=-effort_limit,
        max=effort_limit,
    )
    phase_reaction = -amplitude * torch.cos(phase_rad) * common_effort
    return SinusoidalConstraintEffort(
        position_error_rad=position_error,
        velocity_error_rad_s=velocity_error,
        common_wing_effort_nm=common_effort,
        phase_reaction_torque_nm=phase_reaction,
        saturated=torch.abs(unsaturated_effort) > effort_limit,
    )


def compute_phase_inertia_terms(
    *,
    phase_rad: Tensor,
    common_joint_inertia_kg_m2: float,
    amplitude_rad: float,
    phase_inertia_floor_kg_m2: float,
) -> tuple[Tensor, Tensor]:
    """Return ``J_phi`` and ``dJ_phi/dphi`` with shape ``(...)``."""

    if not isinstance(phase_rad, torch.Tensor):
        raise TypeError("phase_rad must be a torch tensor.")
    if not bool(torch.all(torch.isfinite(phase_rad))):
        raise ValueError("phase_rad must be finite.")
    inertia = float(common_joint_inertia_kg_m2)
    amplitude = float(amplitude_rad)
    floor = float(phase_inertia_floor_kg_m2)
    if inertia < 0.0:
        raise ValueError("common_joint_inertia_kg_m2 must be nonnegative.")
    if amplitude <= 0.0:
        raise ValueError("amplitude_rad must be positive.")
    if floor <= 0.0:
        raise ValueError("phase_inertia_floor_kg_m2 must be positive.")

    scale = inertia * amplitude * amplitude
    sin_phase = torch.sin(phase_rad)
    cos_phase = torch.cos(phase_rad)
    phase_inertia = floor + scale * cos_phase.square()
    phase_inertia_derivative = -2.0 * scale * sin_phase * cos_phase
    return phase_inertia, phase_inertia_derivative


def phase_speed_pi_gains(
    *,
    settling_time_s: float,
    damping_ratio: float,
) -> tuple[float, float]:
    """Return acceleration-domain PI gains for a second-order speed response."""

    settling_time = float(settling_time_s)
    damping_ratio_value = float(damping_ratio)
    if settling_time <= 0.0:
        raise ValueError("settling_time_s must be positive.")
    if damping_ratio_value <= 0.0:
        raise ValueError("damping_ratio must be positive.")

    natural_frequency_rad_s = 4.0 / (damping_ratio_value * settling_time)
    proportional_gain = 2.0 * damping_ratio_value * natural_frequency_rad_s
    integral_gain = natural_frequency_rad_s**2
    return proportional_gain, integral_gain


def compute_phase_speed_pi_torque(
    *,
    phase_rate_rad_s: Tensor,
    target_phase_rate_rad_s: Tensor,
    speed_error_integral_rad: Tensor,
    phase_inertia_kg_m2: Tensor,
    phase_inertia_derivative_kg_m2: Tensor,
    physics_dt_s: float,
    proportional_gain_s_inv: float,
    integral_gain_s_inv2: float,
    phase_viscous_damping_nm_s: float,
    effort_limit_nm: float,
) -> PhaseSpeedPIResult:
    """Compute inertia-compensated PI torque with conditional anti-windup."""

    _validate_shared_tensor_contract(
        phase_rate_rad_s,
        target_phase_rate_rad_s,
        speed_error_integral_rad,
        phase_inertia_kg_m2,
        phase_inertia_derivative_kg_m2,
    )
    dt = float(physics_dt_s)
    proportional_gain = float(proportional_gain_s_inv)
    integral_gain = float(integral_gain_s_inv2)
    viscous_damping = float(phase_viscous_damping_nm_s)
    effort_limit = float(effort_limit_nm)
    if dt <= 0.0:
        raise ValueError("physics_dt_s must be positive.")
    if proportional_gain < 0.0 or integral_gain < 0.0:
        raise ValueError("PI gains must be nonnegative.")
    if viscous_damping < 0.0:
        raise ValueError("phase_viscous_damping_nm_s must be nonnegative.")
    if effort_limit <= 0.0:
        raise ValueError("effort_limit_nm must be positive.")
    if bool(torch.any(phase_inertia_kg_m2 <= 0.0)):
        raise ValueError("phase_inertia_kg_m2 must be positive.")

    speed_error = target_phase_rate_rad_s - phase_rate_rad_s
    candidate_integral = speed_error_integral_rad + speed_error * dt
    configuration_inertia_torque = (
        0.5 * phase_inertia_derivative_kg_m2 * phase_rate_rad_s.square()
    )
    candidate_regulated_acceleration = (
        proportional_gain * speed_error + integral_gain * candidate_integral
    )
    candidate_unsaturated = (
        phase_inertia_kg_m2 * candidate_regulated_acceleration
        + configuration_inertia_torque
        + viscous_damping * phase_rate_rad_s
    )
    drives_high_saturation = (candidate_unsaturated > effort_limit) & (speed_error > 0.0)
    drives_low_saturation = (candidate_unsaturated < -effort_limit) & (speed_error < 0.0)
    hold_integral = drives_high_saturation | drives_low_saturation
    next_integral = torch.where(
        hold_integral,
        speed_error_integral_rad,
        candidate_integral,
    )
    regulated_acceleration = (
        proportional_gain * speed_error + integral_gain * next_integral
    )
    unsaturated = (
        phase_inertia_kg_m2 * regulated_acceleration
        + configuration_inertia_torque
        + viscous_damping * phase_rate_rad_s
    )
    drive_torque = torch.clamp(unsaturated, min=-effort_limit, max=effort_limit)
    return PhaseSpeedPIResult(
        drive_torque_nm=drive_torque,
        unsaturated_drive_torque_nm=unsaturated,
        next_speed_error_integral_rad=next_integral,
        saturated=torch.abs(unsaturated) > effort_limit,
    )


def compute_phase_acceleration(
    *,
    phase_rate_rad_s: Tensor,
    drive_torque_nm: Tensor,
    phase_external_torque_nm: Tensor,
    phase_inertia_kg_m2: Tensor,
    phase_inertia_derivative_kg_m2: Tensor,
    phase_viscous_damping_nm_s: float,
) -> Tensor:
    """Return phase acceleration from the configuration-dependent inertia."""

    _validate_shared_tensor_contract(
        phase_rate_rad_s,
        drive_torque_nm,
        phase_external_torque_nm,
        phase_inertia_kg_m2,
        phase_inertia_derivative_kg_m2,
    )
    viscous_damping = float(phase_viscous_damping_nm_s)
    if viscous_damping < 0.0:
        raise ValueError("phase_viscous_damping_nm_s must be nonnegative.")
    if bool(torch.any(phase_inertia_kg_m2 <= 0.0)):
        raise ValueError("phase_inertia_kg_m2 must be positive.")

    configuration_inertia_torque = (
        0.5 * phase_inertia_derivative_kg_m2 * phase_rate_rad_s.square()
    )
    return (
        drive_torque_nm
        + phase_external_torque_nm
        - viscous_damping * phase_rate_rad_s
        - configuration_inertia_torque
    ) / phase_inertia_kg_m2


def step_sinusoidal_phase_speed_drive(
    *,
    state: SinusoidalPhaseDriveState,
    throttle_01: Tensor,
    common_joint_external_torque_nm: Tensor,
    physics_dt_s: float,
    config: SinusoidalPhaseSpeedDriveConfig,
    left_joint_mid_rad: float = 0.0,
    right_joint_mid_rad: float = 0.0,
) -> SinusoidalPhaseDriveStep:
    """Advance the pure phase drive by one semi-implicit Euler step.

    The external torque is the signed common wing-coordinate torque in N m,
    where positive torque does positive work for positive common wing velocity.
    No aerodynamic load feedforward is applied to the speed regulator.
    """

    _validate_shared_tensor_contract(
        state.phase_rad,
        state.phase_rate_rad_s,
        state.speed_error_integral_rad,
        throttle_01,
        common_joint_external_torque_nm,
    )
    if physics_dt_s <= 0.0:
        raise ValueError("physics_dt_s must be positive.")

    target_frequency_hz = map_virtual_throttle_to_frequency_hz(
        throttle_01=throttle_01,
        max_frequency_hz=config.max_frequency_hz,
    )
    target_phase_rate = 2.0 * math.pi * target_frequency_hz
    phase_inertia, phase_inertia_derivative = compute_phase_inertia_terms(
        phase_rad=state.phase_rad,
        common_joint_inertia_kg_m2=config.common_joint_inertia_kg_m2,
        amplitude_rad=config.amplitude_rad,
        phase_inertia_floor_kg_m2=config.phase_inertia_floor_kg_m2,
    )
    proportional_gain, integral_gain = phase_speed_pi_gains(
        settling_time_s=config.frequency_settling_time_s,
        damping_ratio=config.frequency_damping_ratio,
    )
    pi_result = compute_phase_speed_pi_torque(
        phase_rate_rad_s=state.phase_rate_rad_s,
        target_phase_rate_rad_s=target_phase_rate,
        speed_error_integral_rad=state.speed_error_integral_rad,
        phase_inertia_kg_m2=phase_inertia,
        phase_inertia_derivative_kg_m2=phase_inertia_derivative,
        physics_dt_s=physics_dt_s,
        proportional_gain_s_inv=proportional_gain,
        integral_gain_s_inv2=integral_gain,
        phase_viscous_damping_nm_s=config.phase_viscous_damping_nm_s,
        effort_limit_nm=config.effort_limit_nm,
    )
    phase_external_torque = project_common_joint_torque_to_phase(
        common_joint_torque_nm=common_joint_external_torque_nm,
        phase_rad=state.phase_rad,
        amplitude_rad=config.amplitude_rad,
    )
    phase_acceleration = compute_phase_acceleration(
        phase_rate_rad_s=state.phase_rate_rad_s,
        drive_torque_nm=pi_result.drive_torque_nm,
        phase_external_torque_nm=phase_external_torque,
        phase_inertia_kg_m2=phase_inertia,
        phase_inertia_derivative_kg_m2=phase_inertia_derivative,
        phase_viscous_damping_nm_s=config.phase_viscous_damping_nm_s,
    )
    kinematics = compute_opposed_wing_kinematics(
        phase_rad=state.phase_rad,
        phase_rate_rad_s=state.phase_rate_rad_s,
        phase_acceleration_rad_s2=phase_acceleration,
        amplitude_rad=config.amplitude_rad,
        left_joint_mid_rad=left_joint_mid_rad,
        right_joint_mid_rad=right_joint_mid_rad,
    )
    next_phase_rate = state.phase_rate_rad_s + phase_acceleration * float(physics_dt_s)
    next_phase = state.phase_rad + next_phase_rate * float(physics_dt_s)
    next_state = SinusoidalPhaseDriveState(
        phase_rad=next_phase,
        phase_rate_rad_s=next_phase_rate,
        speed_error_integral_rad=pi_result.next_speed_error_integral_rad,
    )
    return SinusoidalPhaseDriveStep(
        next_state=next_state,
        kinematics=kinematics,
        target_frequency_hz=target_frequency_hz,
        actual_frequency_hz=state.phase_rate_rad_s / (2.0 * math.pi),
        target_phase_rate_rad_s=target_phase_rate,
        phase_acceleration_rad_s2=phase_acceleration,
        phase_inertia_kg_m2=phase_inertia,
        phase_inertia_derivative_kg_m2=phase_inertia_derivative,
        phase_external_torque_nm=phase_external_torque,
        drive_torque_nm=pi_result.drive_torque_nm,
        unsaturated_drive_torque_nm=pi_result.unsaturated_drive_torque_nm,
        drive_saturated=pi_result.saturated,
    )


__all__ = [
    "OpposedWingKinematics",
    "PhaseSpeedPIResult",
    "SinusoidalPhaseDriveState",
    "SinusoidalPhaseDriveStep",
    "SinusoidalPhaseSpeedDriveConfig",
    "SinusoidalConstraintEffort",
    "compute_opposed_wing_kinematics",
    "compute_phase_acceleration",
    "compute_phase_inertia_terms",
    "compute_phase_speed_pi_torque",
    "compute_sinusoidal_constraint_effort",
    "map_virtual_throttle_to_frequency_hz",
    "phase_speed_pi_gains",
    "project_common_joint_torque_to_phase",
    "step_sinusoidal_phase_speed_drive",
]

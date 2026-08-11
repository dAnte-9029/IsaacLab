"""Ideal frequency source and reduced inverse dynamics for a sinusoidal wing mechanism."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .sinusoidal_phase_drive import OpposedWingKinematics, compute_opposed_wing_kinematics

Tensor = torch.Tensor


@dataclass(frozen=True)
class IdealInverseDynamicsPhaseDriveConfig:
    """Numerical settings for the ideal common-coordinate mechanism.

    The frequency response and tracking bandwidth are ideal-mechanism settings,
    not identified motor or gearbox parameters. Generalized effort is in N m
    about the upstroke-positive common wing coordinate.
    """

    amplitude_rad: float = math.radians(30.0)
    max_frequency_hz: float = 5.0
    frequency_settling_time_s: float = 0.15
    tracking_natural_frequency_hz: float = 50.0
    tracking_damping_ratio: float = 1.0
    effort_limit_nm: float = 1000.0

    def __post_init__(self) -> None:
        if self.amplitude_rad <= 0.0:
            raise ValueError("amplitude_rad must be positive.")
        if self.max_frequency_hz <= 0.0:
            raise ValueError("max_frequency_hz must be positive.")
        if self.frequency_settling_time_s <= 0.0:
            raise ValueError("frequency_settling_time_s must be positive.")
        if self.tracking_natural_frequency_hz <= 0.0:
            raise ValueError("tracking_natural_frequency_hz must be positive.")
        if not math.isclose(self.tracking_damping_ratio, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("tracking_damping_ratio must equal one for discrete pole matching.")
        if self.effort_limit_nm <= 0.0:
            raise ValueError("effort_limit_nm must be positive.")

    @property
    def frequency_time_constant_s(self) -> float:
        """Return the first-order time constant from the two-percent settling time."""

        return self.frequency_settling_time_s / 4.0


@dataclass(frozen=True)
class IdealFrequencyPhaseState:
    """Unwrapped mechanism phase in rad and instantaneous frequency in Hz."""

    phase_rad: Tensor
    frequency_hz: Tensor


@dataclass(frozen=True)
class IdealFrequencyPhaseStep:
    """Current ideal reference and the exactly integrated next phase state."""

    next_state: IdealFrequencyPhaseState
    kinematics: OpposedWingKinematics
    next_kinematics: OpposedWingKinematics
    target_frequency_hz: Tensor
    frequency_rate_hz_s: Tensor
    phase_rate_rad_s: Tensor
    phase_acceleration_rad_s2: Tensor


@dataclass(frozen=True)
class ReducedCommonInverseDynamics:
    """Reduced common-coordinate inverse-dynamics terms and limited effort.

    All output tensors have shape ``(N,)``. Inertia is in kg m^2; bias,
    external generalized load, unlimited effort and effort are in N m. Positive
    effort acts in the upstroke-positive common coordinate.
    """

    common_inertia_kg_m2: Tensor
    common_bias_effort_nm: Tensor
    common_external_effort_nm: Tensor
    desired_acceleration_rad_s2: Tensor
    unlimited_effort_nm: Tensor
    effort_nm: Tensor
    saturated: Tensor


@dataclass(frozen=True)
class CommonConstraintLoadEstimate:
    """Ideal common-coordinate constraint-load estimate.

    Every tensor has shape ``(N,)``. ``common_inertia_kg_m2`` is the reduced
    articulation inertia. The four torque terms are in N m about the ideal
    upstroke-positive common wing coordinate, with
    ``equivalent_constraint_torque_nm = inertia_torque_nm + bias_torque_nm -
    external_load_torque_nm``. ``mechanical_power_w`` is that estimated torque
    times the actual common-coordinate angular velocity. This is a multibody
    inverse-dynamics estimate at the wing mechanism output; it is neither the
    PhysX solver's constraint multiplier nor motor-shaft torque or power.
    """

    common_inertia_kg_m2: Tensor
    inertia_torque_nm: Tensor
    bias_torque_nm: Tensor
    external_load_torque_nm: Tensor
    equivalent_constraint_torque_nm: Tensor
    mechanical_power_w: Tensor


def _validate_same_tensor_contract(reference: Tensor, *others: Tensor) -> None:
    if not isinstance(reference, torch.Tensor) or any(not isinstance(value, torch.Tensor) for value in others):
        raise TypeError("All values must be torch tensors.")
    if any(value.shape != reference.shape for value in others):
        raise ValueError("All tensors must share shape.")
    if any(value.device != reference.device or value.dtype != reference.dtype for value in others):
        raise ValueError("All tensors must share device and dtype.")
    if any(not bool(torch.all(torch.isfinite(value))) for value in (reference, *others)):
        raise ValueError("All tensors must be finite.")


def step_ideal_frequency_phase(
    *,
    state: IdealFrequencyPhaseState,
    throttle_01: Tensor,
    physics_dt_s: float,
    config: IdealInverseDynamicsPhaseDriveConfig,
    left_joint_mid_rad: float = 0.0,
    right_joint_mid_rad: float = 0.0,
) -> IdealFrequencyPhaseStep:
    """Return the current sinusoidal reference and exact first-order phase step."""

    _validate_same_tensor_contract(state.phase_rad, state.frequency_hz, throttle_01)
    dt = float(physics_dt_s)
    if dt <= 0.0:
        raise ValueError("physics_dt_s must be positive.")

    target_frequency_hz = torch.clamp(throttle_01, min=0.0, max=1.0) * float(
        config.max_frequency_hz
    )
    time_constant_s = config.frequency_time_constant_s
    frequency_rate_hz_s = (target_frequency_hz - state.frequency_hz) / time_constant_s
    phase_rate_rad_s = 2.0 * math.pi * state.frequency_hz
    phase_acceleration_rad_s2 = 2.0 * math.pi * frequency_rate_hz_s
    kinematics = compute_opposed_wing_kinematics(
        phase_rad=state.phase_rad,
        phase_rate_rad_s=phase_rate_rad_s,
        phase_acceleration_rad_s2=phase_acceleration_rad_s2,
        amplitude_rad=config.amplitude_rad,
        left_joint_mid_rad=left_joint_mid_rad,
        right_joint_mid_rad=right_joint_mid_rad,
    )

    decay = math.exp(-dt / time_constant_s)
    next_frequency_hz = target_frequency_hz + (state.frequency_hz - target_frequency_hz) * decay
    frequency_integral_hz_s = (
        target_frequency_hz * dt
        + (state.frequency_hz - target_frequency_hz) * time_constant_s * (1.0 - decay)
    )
    next_phase_rad = state.phase_rad + 2.0 * math.pi * frequency_integral_hz_s
    next_frequency_rate_hz_s = (target_frequency_hz - next_frequency_hz) / time_constant_s
    next_phase_rate_rad_s = 2.0 * math.pi * next_frequency_hz
    next_phase_acceleration_rad_s2 = 2.0 * math.pi * next_frequency_rate_hz_s
    next_kinematics = compute_opposed_wing_kinematics(
        phase_rad=next_phase_rad,
        phase_rate_rad_s=next_phase_rate_rad_s,
        phase_acceleration_rad_s2=next_phase_acceleration_rad_s2,
        amplitude_rad=config.amplitude_rad,
        left_joint_mid_rad=left_joint_mid_rad,
        right_joint_mid_rad=right_joint_mid_rad,
    )
    return IdealFrequencyPhaseStep(
        next_state=IdealFrequencyPhaseState(
            phase_rad=next_phase_rad,
            frequency_hz=next_frequency_hz,
        ),
        kinematics=kinematics,
        next_kinematics=next_kinematics,
        target_frequency_hz=target_frequency_hz,
        frequency_rate_hz_s=frequency_rate_hz_s,
        phase_rate_rad_s=phase_rate_rad_s,
        phase_acceleration_rad_s2=phase_acceleration_rad_s2,
    )


def discrete_tracking_acceleration_gains(
    *,
    natural_frequency_hz: float,
    damping_ratio: float,
    physics_dt_s: float,
) -> tuple[float, float]:
    """Return position and velocity gains for one repeated discrete pole."""

    frequency_hz = float(natural_frequency_hz)
    damping = float(damping_ratio)
    dt = float(physics_dt_s)
    if frequency_hz <= 0.0:
        raise ValueError("natural_frequency_hz must be positive.")
    if not math.isclose(damping, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("damping_ratio must equal one for discrete pole matching.")
    if dt <= 0.0:
        raise ValueError("physics_dt_s must be positive.")
    pole = math.exp(-2.0 * math.pi * frequency_hz * dt)
    return (1.0 - pole) ** 2 / (dt * dt), (1.0 - pole * pole) / dt


def compute_desired_common_acceleration(
    *,
    actual_position_rad: Tensor,
    actual_velocity_rad_s: Tensor,
    reference_position_rad: Tensor,
    reference_velocity_rad_s: Tensor,
    reference_acceleration_rad_s2: Tensor,
    natural_frequency_hz: float,
    damping_ratio: float,
    physics_dt_s: float,
) -> Tensor:
    """Add discrete tracking feedback to the analytical common acceleration."""

    _validate_same_tensor_contract(
        actual_position_rad,
        actual_velocity_rad_s,
        reference_position_rad,
        reference_velocity_rad_s,
        reference_acceleration_rad_s2,
    )
    position_gain, velocity_gain = discrete_tracking_acceleration_gains(
        natural_frequency_hz=natural_frequency_hz,
        damping_ratio=damping_ratio,
        physics_dt_s=physics_dt_s,
    )
    return (
        reference_acceleration_rad_s2
        + position_gain * (reference_position_rad - actual_position_rad)
        + velocity_gain * (reference_velocity_rad_s - actual_velocity_rad_s)
    )


def _reduce_common_dynamics_terms(
    *,
    generalized_mass_matrix: Tensor,
    generalized_bias_effort: Tensor,
    external_generalized_effort: Tensor,
    joint_direction: Tensor,
    common_value: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return inertia, bias and external load reduced to one common DOF.

    ``generalized_mass_matrix`` has shape ``(N,D,D)``. The bias and external
    effort tensors have shape ``(N,D)``. ``joint_direction`` has shape ``(J,)``
    and maps the common coordinate to articulation joints; this project uses
    ``+1`` for the left wing and ``-1`` for the right wing. The function accepts
    either ``D=J`` (fixed base) or ``D=J+6`` (floating base). Floating-base
    generalized forces use world-frame root force then root moment in their
    first six components, matching the PhysX tensor API.
    """

    if generalized_mass_matrix.ndim != 3 or generalized_mass_matrix.shape[-1] != generalized_mass_matrix.shape[-2]:
        raise ValueError("generalized_mass_matrix must have shape (N,D,D).")
    batch_size, generalized_size, _ = generalized_mass_matrix.shape
    if generalized_bias_effort.shape != (batch_size, generalized_size):
        raise ValueError("generalized_bias_effort must have shape (N,D).")
    if external_generalized_effort.shape != (batch_size, generalized_size):
        raise ValueError("external_generalized_effort must have shape (N,D).")
    if joint_direction.ndim != 1:
        raise ValueError("joint_direction must have shape (J,).")
    if common_value.shape != (batch_size,):
        raise ValueError("common_value must have shape (N,).")
    if any(
        value.device != generalized_mass_matrix.device
        or value.dtype != generalized_mass_matrix.dtype
        for value in (
            generalized_bias_effort,
            external_generalized_effort,
            joint_direction,
            common_value,
        )
    ):
        raise ValueError("Inverse-dynamics tensors must share device and dtype.")
    if any(
        not bool(torch.all(torch.isfinite(value)))
        for value in (
            generalized_mass_matrix,
            generalized_bias_effort,
            external_generalized_effort,
            joint_direction,
            common_value,
        )
    ):
        raise ValueError("Inverse-dynamics tensors must be finite.")

    joint_count = int(joint_direction.numel())
    base_dof_count = generalized_size - joint_count
    if base_dof_count not in (0, 6):
        raise ValueError("Expected a fixed base (D=J) or floating base (D=J+6).")
    direction = joint_direction.view(1, joint_count, 1).expand(batch_size, -1, -1)

    mass_joint_joint = generalized_mass_matrix[:, base_dof_count:, base_dof_count:]
    bias_joint = generalized_bias_effort[:, base_dof_count:]
    external_joint = external_generalized_effort[:, base_dof_count:]
    if base_dof_count == 0:
        reduced_mass = mass_joint_joint
        reduced_bias = bias_joint
        reduced_external = external_joint
    else:
        mass_base_base = generalized_mass_matrix[:, :6, :6]
        mass_base_joint = generalized_mass_matrix[:, :6, 6:]
        mass_joint_base = generalized_mass_matrix[:, 6:, :6]
        solve_mass_base_joint = torch.linalg.solve(mass_base_base, mass_base_joint)
        reduced_mass = mass_joint_joint - mass_joint_base @ solve_mass_base_joint
        solve_bias_base = torch.linalg.solve(
            mass_base_base,
            generalized_bias_effort[:, :6].unsqueeze(-1),
        )
        reduced_bias = bias_joint - (mass_joint_base @ solve_bias_base).squeeze(-1)
        solve_external_base = torch.linalg.solve(
            mass_base_base,
            external_generalized_effort[:, :6].unsqueeze(-1),
        )
        reduced_external = external_joint - (mass_joint_base @ solve_external_base).squeeze(-1)

    common_inertia = (direction.transpose(1, 2) @ reduced_mass @ direction).reshape(batch_size)
    common_bias = (direction.transpose(1, 2) @ reduced_bias.unsqueeze(-1)).reshape(batch_size)
    common_external = (direction.transpose(1, 2) @ reduced_external.unsqueeze(-1)).reshape(batch_size)
    if bool(torch.any(common_inertia <= 0.0)):
        raise ValueError("Reduced common-coordinate inertia must be positive.")
    return common_inertia, common_bias, common_external


def estimate_common_constraint_load(
    *,
    generalized_mass_matrix: Tensor,
    generalized_bias_effort: Tensor,
    external_generalized_effort: Tensor,
    joint_direction: Tensor,
    prescribed_common_acceleration_rad_s2: Tensor,
    actual_common_velocity_rad_s: Tensor,
) -> CommonConstraintLoadEstimate:
    """Estimate the ideal trajectory-constraint load and mechanical power.

    The input generalized quantities follow the fixed/floating-base PhysX
    convention documented by :func:`reduce_common_inverse_dynamics`. Positive
    external generalized load assists positive upstroke. Therefore it is
    subtracted from the torque the ideal mechanism must supply.
    """

    _validate_same_tensor_contract(
        prescribed_common_acceleration_rad_s2,
        actual_common_velocity_rad_s,
    )
    common_inertia, common_bias, common_external = _reduce_common_dynamics_terms(
        generalized_mass_matrix=generalized_mass_matrix,
        generalized_bias_effort=generalized_bias_effort,
        external_generalized_effort=external_generalized_effort,
        joint_direction=joint_direction,
        common_value=prescribed_common_acceleration_rad_s2,
    )
    inertia_torque = common_inertia * prescribed_common_acceleration_rad_s2
    equivalent_constraint_torque = inertia_torque + common_bias - common_external
    return CommonConstraintLoadEstimate(
        common_inertia_kg_m2=common_inertia,
        inertia_torque_nm=inertia_torque,
        bias_torque_nm=common_bias,
        external_load_torque_nm=common_external,
        equivalent_constraint_torque_nm=equivalent_constraint_torque,
        mechanical_power_w=equivalent_constraint_torque * actual_common_velocity_rad_s,
    )


def reduce_common_inverse_dynamics(
    *,
    generalized_mass_matrix: Tensor,
    generalized_bias_effort: Tensor,
    external_generalized_effort: Tensor,
    joint_direction: Tensor,
    desired_common_acceleration_rad_s2: Tensor,
    effort_limit_nm: float,
) -> ReducedCommonInverseDynamics:
    """Reduce fixed- or floating-base PhysX inverse dynamics to one wing DOF."""

    effort_limit = float(effort_limit_nm)
    if effort_limit <= 0.0:
        raise ValueError("effort_limit_nm must be positive.")
    common_inertia, common_bias, common_external = _reduce_common_dynamics_terms(
        generalized_mass_matrix=generalized_mass_matrix,
        generalized_bias_effort=generalized_bias_effort,
        external_generalized_effort=external_generalized_effort,
        joint_direction=joint_direction,
        common_value=desired_common_acceleration_rad_s2,
    )
    unlimited_effort = (
        common_inertia * desired_common_acceleration_rad_s2
        + common_bias
        - common_external
    )
    effort = torch.clamp(unlimited_effort, min=-effort_limit, max=effort_limit)
    return ReducedCommonInverseDynamics(
        common_inertia_kg_m2=common_inertia,
        common_bias_effort_nm=common_bias,
        common_external_effort_nm=common_external,
        desired_acceleration_rad_s2=desired_common_acceleration_rad_s2,
        unlimited_effort_nm=unlimited_effort,
        effort_nm=effort,
        saturated=torch.abs(unlimited_effort) > effort_limit,
    )


__all__ = [
    "CommonConstraintLoadEstimate",
    "IdealFrequencyPhaseState",
    "IdealFrequencyPhaseStep",
    "IdealInverseDynamicsPhaseDriveConfig",
    "ReducedCommonInverseDynamics",
    "compute_desired_common_acceleration",
    "discrete_tracking_acceleration_gains",
    "estimate_common_constraint_load",
    "reduce_common_inverse_dynamics",
    "step_ideal_frequency_phase",
]

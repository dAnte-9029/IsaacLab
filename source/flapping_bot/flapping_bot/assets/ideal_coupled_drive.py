"""Project-local asset adapter for an ideal opposed wing coupling."""

from __future__ import annotations

from typing import Any

from isaaclab.actuators import ImplicitActuatorCfg

from ..physics import (
    IDEAL_TORQUE_DAMPING_RATIO,
    IDEAL_TORQUE_EQUIVALENT_INERTIA_KG_M2,
    IDEAL_TORQUE_NATURAL_FREQUENCY_HZ,
    SINGLE_WING_HINGE_INERTIA_KG_M2,
    apply_quintic_amplitude_ramp,
    compute_aerodynamic_joint_hinge_torques,
    compute_common_aerodynamic_hinge_torque,
    compute_ideal_torque_drive_effort,
    ideal_torque_drive_gains,
)
from .flapping_bot_cfg import FlappingBotCfg

KINEMATIC_WING_OVERRIDE = "kinematic_override"
IDEAL_COUPLED_WING_DRIVE = "ideal_coupled_drive"
PRESCRIBED_COUPLED_WING_DRIVE = "prescribed_coupled_drive"
IDEAL_TORQUE_COUPLED_WING_DRIVE = "ideal_torque_coupled_drive"
SINUSOIDAL_PHASE_SPEED_WING_DRIVE = "sinusoidal_phase_speed_drive"
IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE = "ideal_inverse_dynamics_phase_drive"
NATIVE_HOLONOMIC_WING_DRIVE = "native_holonomic_drive"
WING_DRIVE_VARIANTS = frozenset(
    {
        KINEMATIC_WING_OVERRIDE,
        IDEAL_COUPLED_WING_DRIVE,
        PRESCRIBED_COUPLED_WING_DRIVE,
        IDEAL_TORQUE_COUPLED_WING_DRIVE,
        SINUSOIDAL_PHASE_SPEED_WING_DRIVE,
        IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE,
        NATIVE_HOLONOMIC_WING_DRIVE,
    }
)

IDEAL_DRIVER_STIFFNESS_NM_PER_RAD = 2_000.0
IDEAL_DRIVER_DAMPING_NM_S_PER_RAD = 20.0
IDEAL_DRIVER_EFFORT_LIMIT_NM = 1_000.0
IDEAL_DRIVER_VELOCITY_LIMIT_RAD_S = 200.0
_MIMIC_NATURAL_FREQUENCY_ATTR = "physxMimicJoint:rotX:naturalFrequency"
_MIMIC_DAMPING_RATIO_ATTR = "physxMimicJoint:rotX:dampingRatio"
NATIVE_HOLONOMIC_SOLVER_POSITION_ITERATIONS = 16
NATIVE_HOLONOMIC_SOLVER_VELOCITY_ITERATIONS = 4


def validate_wing_drive_variant(wing_drive_variant: str) -> str:
    """Validate and return a wing-drive variant name."""

    variant = str(wing_drive_variant)
    if variant not in WING_DRIVE_VARIANTS:
        expected = ", ".join(sorted(WING_DRIVE_VARIANTS))
        raise ValueError(f"Unknown wing-drive variant {variant!r}; expected one of: {expected}.")
    return variant


IdealCoupledFlappingBotCfg = FlappingBotCfg.replace(
    spawn=FlappingBotCfg.spawn.replace(
        articulation_props=FlappingBotCfg.spawn.articulation_props.replace(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    actuators={
        "wing_driver": ImplicitActuatorCfg(
            joint_names_expr=["left_wing"],
            effort_limit_sim=IDEAL_DRIVER_EFFORT_LIMIT_NM,
            velocity_limit_sim=IDEAL_DRIVER_VELOCITY_LIMIT_RAD_S,
            stiffness=IDEAL_DRIVER_STIFFNESS_NM_PER_RAD,
            damping=IDEAL_DRIVER_DAMPING_NM_S_PER_RAD,
        ),
        "passive_right_wing": ImplicitActuatorCfg(
            joint_names_expr=["right_wing"],
            effort_limit_sim=IDEAL_DRIVER_EFFORT_LIMIT_NM,
            velocity_limit_sim=IDEAL_DRIVER_VELOCITY_LIMIT_RAD_S,
            stiffness=0.0,
            damping=0.0,
        ),
        "tail_servos": FlappingBotCfg.actuators["tail_servos"],
        "rudder_servo": FlappingBotCfg.actuators["rudder_servo"],
    },
)

PrescribedCoupledFlappingBotCfg = IdealCoupledFlappingBotCfg.replace(
    actuators={
        "passive_wings": ImplicitActuatorCfg(
            joint_names_expr=["left_wing", "right_wing"],
            effort_limit_sim=IDEAL_DRIVER_EFFORT_LIMIT_NM,
            velocity_limit_sim=IDEAL_DRIVER_VELOCITY_LIMIT_RAD_S,
            stiffness=0.0,
            damping=0.0,
        ),
        "tail_servos": FlappingBotCfg.actuators["tail_servos"],
        "rudder_servo": FlappingBotCfg.actuators["rudder_servo"],
    },
)

IdealTorqueCoupledFlappingBotCfg = IdealCoupledFlappingBotCfg.replace(
    actuators={
        "wing_driver": ImplicitActuatorCfg(
            joint_names_expr=["left_wing"],
            effort_limit_sim=IDEAL_DRIVER_EFFORT_LIMIT_NM,
            velocity_limit_sim=IDEAL_DRIVER_VELOCITY_LIMIT_RAD_S,
            stiffness=0.0,
            damping=0.0,
        ),
        "passive_right_wing": ImplicitActuatorCfg(
            joint_names_expr=["right_wing"],
            effort_limit_sim=IDEAL_DRIVER_EFFORT_LIMIT_NM,
            velocity_limit_sim=IDEAL_DRIVER_VELOCITY_LIMIT_RAD_S,
            stiffness=0.0,
            damping=0.0,
        ),
        "tail_servos": FlappingBotCfg.actuators["tail_servos"],
        "rudder_servo": FlappingBotCfg.actuators["rudder_servo"],
    },
)

SinusoidalPhaseSpeedCoupledFlappingBotCfg = IdealTorqueCoupledFlappingBotCfg.replace()
IdealInverseDynamicsPhaseCoupledFlappingBotCfg = IdealTorqueCoupledFlappingBotCfg.replace()
NativeHolonomicCoupledFlappingBotCfg = PrescribedCoupledFlappingBotCfg.replace(
    spawn=PrescribedCoupledFlappingBotCfg.spawn.replace(
        articulation_props=PrescribedCoupledFlappingBotCfg.spawn.articulation_props.replace(
            solver_position_iteration_count=NATIVE_HOLONOMIC_SOLVER_POSITION_ITERATIONS,
            solver_velocity_iteration_count=NATIVE_HOLONOMIC_SOLVER_VELOCITY_ITERATIONS,
        ),
    ),
)


def apply_hard_opposed_wing_mimic(stage: Any, *, articulation_root_path: str | None = None) -> str:
    """Apply a hard PhysX relation ``q_left + q_right = 0`` before initialization.

    The upstream robot asset intentionally remains unchanged. This adapter is
    called after the source environment is spawned and before PhysX creates
    the articulation. The source environment is then cloned with the schema.

    Returns:
        Path of the right-wing joint carrying the mimic API.
    """

    from pxr import PhysxSchema, Sdf, UsdPhysics

    root_prefix = None if articulation_root_path is None else f"{articulation_root_path.rstrip('/')}/"
    left_joints = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == "left_wing"
        and prim.IsA(UsdPhysics.Joint)
        and (root_prefix is None or str(prim.GetPath()).startswith(root_prefix))
    ]
    right_joints = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == "right_wing"
        and prim.IsA(UsdPhysics.Joint)
        and (root_prefix is None or str(prim.GetPath()).startswith(root_prefix))
    ]
    if len(left_joints) != 1 or len(right_joints) != 1:
        raise RuntimeError(
            "Expected exactly one scoped left_wing and right_wing joint "
            f"before cloning; found {len(left_joints)} and {len(right_joints)}."
        )

    left_joint = left_joints[0]
    right_joint = right_joints[0]
    mimic_api = PhysxSchema.PhysxMimicJointAPI.Apply(right_joint, UsdPhysics.Tokens.rotX)
    mimic_api.CreateReferenceJointRel().SetTargets([left_joint.GetPath()])
    # PhysX uses gearing*q_reference + q_mimic + offset = 0.
    mimic_api.CreateGearingAttr(1.0)
    mimic_api.CreateOffsetAttr(0.0)
    right_joint.CreateAttribute(_MIMIC_NATURAL_FREQUENCY_ATTR, Sdf.ValueTypeNames.Float).Set(0.0)
    right_joint.CreateAttribute(_MIMIC_DAMPING_RATIO_ATTR, Sdf.ValueTypeNames.Float).Set(0.0)
    return str(right_joint.GetPath())


__all__ = [
    "KINEMATIC_WING_OVERRIDE",
    "IDEAL_COUPLED_WING_DRIVE",
    "PRESCRIBED_COUPLED_WING_DRIVE",
    "IDEAL_TORQUE_COUPLED_WING_DRIVE",
    "SINUSOIDAL_PHASE_SPEED_WING_DRIVE",
    "IDEAL_INVERSE_DYNAMICS_PHASE_WING_DRIVE",
    "NATIVE_HOLONOMIC_WING_DRIVE",
    "WING_DRIVE_VARIANTS",
    "IDEAL_DRIVER_STIFFNESS_NM_PER_RAD",
    "IDEAL_DRIVER_DAMPING_NM_S_PER_RAD",
    "IDEAL_DRIVER_EFFORT_LIMIT_NM",
    "IDEAL_DRIVER_VELOCITY_LIMIT_RAD_S",
    "SINGLE_WING_HINGE_INERTIA_KG_M2",
    "IDEAL_TORQUE_EQUIVALENT_INERTIA_KG_M2",
    "IDEAL_TORQUE_NATURAL_FREQUENCY_HZ",
    "IDEAL_TORQUE_DAMPING_RATIO",
    "IdealCoupledFlappingBotCfg",
    "PrescribedCoupledFlappingBotCfg",
    "IdealTorqueCoupledFlappingBotCfg",
    "SinusoidalPhaseSpeedCoupledFlappingBotCfg",
    "IdealInverseDynamicsPhaseCoupledFlappingBotCfg",
    "NativeHolonomicCoupledFlappingBotCfg",
    "ideal_torque_drive_gains",
    "compute_ideal_torque_drive_effort",
    "compute_common_aerodynamic_hinge_torque",
    "compute_aerodynamic_joint_hinge_torques",
    "apply_quintic_amplitude_ramp",
    "validate_wing_drive_variant",
    "apply_hard_opposed_wing_mimic",
]

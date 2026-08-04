"""Project-local USD adapter for the native wing trajectory constraint."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

NATIVE_HOLONOMIC_EXTENSION_ID = "omni.flapping_bot.holonomic_constraint"
NATIVE_HOLONOMIC_JOINT_NAME = "flapping_wing_trajectory"


def native_holonomic_extension_root() -> Path:
    """Return the repository-local Kit extension directory."""

    return (
        Path(__file__).resolve().parents[2]
        / "native_extensions"
        / NATIVE_HOLONOMIC_EXTENSION_ID
    )


def native_holonomic_kit_args() -> str:
    """Return Kit arguments required before launching Isaac Sim."""

    extension_parent = native_holonomic_extension_root().parent
    return f"--ext-folder {extension_parent} --enable {NATIVE_HOLONOMIC_EXTENSION_ID}"


def require_native_holonomic_extension() -> ModuleType:
    """Return the loaded native bindings or fail with the launch contract."""

    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled(NATIVE_HOLONOMIC_EXTENSION_ID):
        raise RuntimeError(
            f"{NATIVE_HOLONOMIC_EXTENSION_ID} must be enabled when Kit starts. "
            f"Pass kit_args={native_holonomic_kit_args()!r} to AppLauncher."
        )
    from omni.flapping_bot.holonomic_constraint import _native

    expected_type = "FlappingWingTrajectoryJoint"
    if _native.get_joint_type_name() != expected_type:
        raise RuntimeError(
            "Loaded holonomic extension exposes an unexpected USD joint type: "
            f"{_native.get_joint_type_name()!r}."
        )
    return _native


def apply_native_holonomic_wing_constraint(
    stage: Any,
    *,
    articulation_root_path: str,
    constraint_prim_path: str | None = None,
    native_module: ModuleType,
) -> str:
    """Author one trajectory constraint parallel to the left revolute joint.

    The custom constraint contributes only the left-wing hinge angular row.
    The existing revolute joint retains the other five relative constraints,
    and the right-wing PhysX mimic retains the opposed-wing relation.

    Returns:
        Path of the authored custom joint.
    """

    from pxr import Sdf, UsdPhysics

    root = articulation_root_path.rstrip("/")
    left_joints = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == "left_wing"
        and prim.IsA(UsdPhysics.RevoluteJoint)
        and str(prim.GetPath()).startswith(f"{root}/")
    ]
    if len(left_joints) != 1:
        raise RuntimeError(
            "Expected exactly one scoped left_wing revolute joint before cloning; "
            f"found {len(left_joints)}."
        )

    source_joint = UsdPhysics.Joint(left_joints[0])
    constraint_path = (
        constraint_prim_path.rstrip("/")
        if constraint_prim_path is not None
        else f"{root}/{NATIVE_HOLONOMIC_JOINT_NAME}"
    )
    if stage.GetPrimAtPath(constraint_path):
        raise RuntimeError(f"Native holonomic joint already exists at {constraint_path}.")
    constraint_prim = stage.DefinePrim(constraint_path, native_module.get_joint_type_name())
    if not constraint_prim.IsA(UsdPhysics.Joint):
        raise RuntimeError(
            f"USD type {native_module.get_joint_type_name()!r} is not registered as a PhysicsJoint. "
            "The native extension must be enabled during Kit startup."
        )

    constraint = UsdPhysics.Joint(constraint_prim)
    body0_targets = source_joint.GetBody0Rel().GetTargets()
    body1_targets = source_joint.GetBody1Rel().GetTargets()
    if len(body0_targets) != 1 or len(body1_targets) != 1:
        raise RuntimeError("The left_wing joint must connect exactly two rigid bodies.")
    source_joint_path = source_joint.GetPrim().GetPath()
    absolute_body0 = body0_targets[0].MakeAbsolutePath(source_joint_path)
    absolute_body1 = body1_targets[0].MakeAbsolutePath(source_joint_path)
    constraint.CreateBody0Rel().SetTargets([absolute_body0])
    constraint.CreateBody1Rel().SetTargets([absolute_body1])
    constraint.CreateLocalPos0Attr(source_joint.GetLocalPos0Attr().Get())
    constraint.CreateLocalRot0Attr(source_joint.GetLocalRot0Attr().Get())
    constraint.CreateLocalPos1Attr(source_joint.GetLocalPos1Attr().Get())
    constraint.CreateLocalRot1Attr(source_joint.GetLocalRot1Attr().Get())
    constraint.CreateCollisionEnabledAttr(False)
    constraint.CreateExcludeFromArticulationAttr(True)
    constraint_prim.CreateAttribute(
        "flappingBot:constraintType",
        Sdf.ValueTypeNames.String,
    ).Set("wingTrajectory")
    return constraint_path


def cloned_native_holonomic_joint_paths(env_prim_paths: list[str]) -> list[str]:
    """Return joint paths outside the live-replicated environment subtrees."""

    paths: list[str] = []
    for env_path in env_prim_paths:
        env_name = env_path.rstrip("/").rsplit("/", maxsplit=1)[-1]
        paths.append(
            f"/World/flapping_bot_constraints/{env_name}/{NATIVE_HOLONOMIC_JOINT_NAME}"
        )
    return paths


__all__ = [
    "NATIVE_HOLONOMIC_EXTENSION_ID",
    "NATIVE_HOLONOMIC_JOINT_NAME",
    "apply_native_holonomic_wing_constraint",
    "cloned_native_holonomic_joint_paths",
    "native_holonomic_extension_root",
    "native_holonomic_kit_args",
    "require_native_holonomic_extension",
]

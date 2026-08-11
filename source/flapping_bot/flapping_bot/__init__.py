# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom flapping-wing robot tasks and utilities.

This package is intentionally *lazy* about importing IsaacSim-dependent modules.
That keeps pure-Python utilities (e.g., aerodynamic models) importable in a
standard Python environment without IsaacSim (`carb`, `omni.*`).
"""

from pathlib import Path

import toml

_EXT_DIR = Path(__file__).resolve().parent.parent
_EXT_METADATA = toml.load(_EXT_DIR / "config" / "extension.toml")
__version__ = _EXT_METADATA["package"]["version"]

__all__ = [
    "__version__",
    "FlappingBotCfg",
    "FlappingRoomSceneCfg",
    "FlappingBotEnvCfg",
    "FlappingBotEnv",
    "FlappingBotPathTrackingEnvCfg",
    "FlappingBotPathTrackingEnv",
    "FlappingBotPathTrackingWeakTeacherRLEnvCfg",
    "FlappingBotPathTrackingPureRLEnvCfg",
    "FlappingBotPathTrackingPrimitiveTeacherRLEnvCfg",
    "FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg",
    "FlappingBotPathTrackingPrimitivePureRLEnvCfg",
    "FlappingBotStraightFlightEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledDeLaurierEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledDeLaurierEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg",
    "FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg",
    "FlappingBotStraightFlightSimpleEnvCfg",
    "FlappingBotStraightFlightCommandedKinematicsDeLaurierEnvCfg",
    "FlappingBotStraightFlightDeLaurierEnvCfg",
    "FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg",
    "FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg",
    "FlappingBotStraightFlightDeLaurierPureRLEnvCfg",
    "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg",
    "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aEnvCfg",
    "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2bEnvCfg",
    "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2cEnvCfg",
    "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg",
    "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuImplicitEnvCfg",
    "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg",
    "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuPhaseMatchedEnvCfg",
    "FlappingBotStraightFlightEnv",
]


def __getattr__(name: str):
    if name == "FlappingBotCfg":
        from .assets import FlappingBotCfg

        return FlappingBotCfg
    if name in ("FlappingBotEnv", "FlappingBotEnvCfg"):
        from .direct.flapping_bot import (
            FlappingBotEnv,
            FlappingBotEnvCfg,
        )

        return {
            "FlappingBotEnv": FlappingBotEnv,
            "FlappingBotEnvCfg": FlappingBotEnvCfg,
        }[name]
    if name in (
        "FlappingBotPathTrackingEnv",
        "FlappingBotPathTrackingEnvCfg",
        "FlappingBotPathTrackingWeakTeacherRLEnvCfg",
        "FlappingBotPathTrackingPureRLEnvCfg",
        "FlappingBotPathTrackingPrimitiveTeacherRLEnvCfg",
        "FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg",
        "FlappingBotPathTrackingPrimitivePureRLEnvCfg",
    ):
        from .direct.flapping_bot import (
            FlappingBotPathTrackingEnv,
            FlappingBotPathTrackingEnvCfg,
            FlappingBotPathTrackingPrimitiveTeacherRLEnvCfg,
            FlappingBotPathTrackingPrimitivePureRLEnvCfg,
            FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg,
            FlappingBotPathTrackingPureRLEnvCfg,
            FlappingBotPathTrackingWeakTeacherRLEnvCfg,
        )

        return {
            "FlappingBotPathTrackingEnv": FlappingBotPathTrackingEnv,
            "FlappingBotPathTrackingEnvCfg": FlappingBotPathTrackingEnvCfg,
            "FlappingBotPathTrackingWeakTeacherRLEnvCfg": FlappingBotPathTrackingWeakTeacherRLEnvCfg,
            "FlappingBotPathTrackingPureRLEnvCfg": FlappingBotPathTrackingPureRLEnvCfg,
            "FlappingBotPathTrackingPrimitiveTeacherRLEnvCfg": FlappingBotPathTrackingPrimitiveTeacherRLEnvCfg,
            "FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg": FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg,
            "FlappingBotPathTrackingPrimitivePureRLEnvCfg": FlappingBotPathTrackingPrimitivePureRLEnvCfg,
        }[name]
    if name in (
        "FlappingBotStraightFlightEnv",
        "FlappingBotStraightFlightEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledDeLaurierEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledDeLaurierEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg",
        "FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg",
        "FlappingBotStraightFlightSimpleEnvCfg",
        "FlappingBotStraightFlightCommandedKinematicsDeLaurierEnvCfg",
        "FlappingBotStraightFlightDeLaurierEnvCfg",
        "FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg",
        "FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg",
        "FlappingBotStraightFlightDeLaurierPureRLEnvCfg",
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg",
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aEnvCfg",
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2bEnvCfg",
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2cEnvCfg",
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg",
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuImplicitEnvCfg",
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg",
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuPhaseMatchedEnvCfg",
    ):
        from .direct.flapping_bot import (
            FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aEnvCfg,
            FlappingBotStraightFlightDeLaurierMeasuredPureRLC2bEnvCfg,
            FlappingBotStraightFlightDeLaurierMeasuredPureRLC2cEnvCfg,
            FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuImplicitEnvCfg,
            FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuPhaseMatchedEnvCfg,
            FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
            FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg,
            FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg,
            FlappingBotStraightFlightDeLaurierPureRLEnvCfg,
            FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg,
            FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg,
            FlappingBotStraightFlightDeLaurierEnvCfg,
            FlappingBotStraightFlightCommandedKinematicsDeLaurierEnvCfg,
            FlappingBotStraightFlightEnv,
            FlappingBotStraightFlightEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledDeLaurierEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledDeLaurierEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg,
            FlappingBotStraightFlightSimpleEnvCfg,
        )

        return {
            "FlappingBotStraightFlightEnv": FlappingBotStraightFlightEnv,
            "FlappingBotStraightFlightEnvCfg": FlappingBotStraightFlightEnvCfg,
            "FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledDeLaurierEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledDeLaurierEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodyIdealTorqueCoupledDeLaurierEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodyPrescribedCoupledDeLaurierEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodySinusoidalPhaseCoupledDeLaurierEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledDeLaurierEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodyIdealInverseDynamicsPhaseCoupledDeLaurierEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicEnvCfg
            ),
            "FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg": (
                FlappingBotStraightFlightMeasuredWingMultibodyNativeHolonomicDeLaurierEnvCfg
            ),
            "FlappingBotStraightFlightSimpleEnvCfg": FlappingBotStraightFlightSimpleEnvCfg,
            "FlappingBotStraightFlightCommandedKinematicsDeLaurierEnvCfg": (
                FlappingBotStraightFlightCommandedKinematicsDeLaurierEnvCfg
            ),
            "FlappingBotStraightFlightDeLaurierEnvCfg": FlappingBotStraightFlightDeLaurierEnvCfg,
            "FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg": FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg,
            "FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg": FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg,
            "FlappingBotStraightFlightDeLaurierPureRLEnvCfg": FlappingBotStraightFlightDeLaurierPureRLEnvCfg,
            "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg": FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
            "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aEnvCfg": FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aEnvCfg,
            "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2bEnvCfg": FlappingBotStraightFlightDeLaurierMeasuredPureRLC2bEnvCfg,
            "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2cEnvCfg": FlappingBotStraightFlightDeLaurierMeasuredPureRLC2cEnvCfg,
            "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg": FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuImplicitEnvCfg,
            "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuImplicitEnvCfg": FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuImplicitEnvCfg,
            "FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg": FlappingBotStraightFlightDeLaurierMeasuredPureRLGpuPhaseMatchedEnvCfg,
            "FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuPhaseMatchedEnvCfg": FlappingBotStraightFlightDeLaurierMeasuredPureRLC2aGpuPhaseMatchedEnvCfg,
        }[name]
    if name == "FlappingRoomSceneCfg":
        from .scenes import FlappingRoomSceneCfg

        return FlappingRoomSceneCfg
    raise AttributeError(name)

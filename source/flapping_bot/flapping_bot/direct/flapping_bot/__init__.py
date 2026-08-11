"""Direct RL environment definitions for the flapping-wing robot.

Imports stay lazy so pure-Python unit tests can import helper modules without
pulling IsaacSim dependencies such as ``carb`` at package import time.
"""

__all__ = [
    "FlappingBotEnv",
    "FlappingBotEnvCfg",
    "FlappingBotPathTrackingEnv",
    "FlappingBotPathTrackingEnvCfg",
    "FlappingBotPathTrackingWeakTeacherRLEnvCfg",
    "FlappingBotPathTrackingPureRLEnvCfg",
    "FlappingBotPathTrackingPrimitiveTeacherRLEnvCfg",
    "FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg",
    "FlappingBotPathTrackingPrimitivePureRLEnvCfg",
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
]


def __getattr__(name: str):
    if name in ("FlappingBotEnv", "FlappingBotEnvCfg"):
        from .flapping_env import FlappingBotEnv, FlappingBotEnvCfg

        return {"FlappingBotEnv": FlappingBotEnv, "FlappingBotEnvCfg": FlappingBotEnvCfg}[name]

    if name in (
        "FlappingBotPathTrackingEnv",
        "FlappingBotPathTrackingEnvCfg",
        "FlappingBotPathTrackingWeakTeacherRLEnvCfg",
        "FlappingBotPathTrackingPureRLEnvCfg",
        "FlappingBotPathTrackingPrimitiveTeacherRLEnvCfg",
        "FlappingBotPathTrackingPrimitiveWeakTeacherRLEnvCfg",
        "FlappingBotPathTrackingPrimitivePureRLEnvCfg",
    ):
        from .path_tracking_env import (
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
        from .straight_flight_env import (
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
            FlappingBotStraightFlightMeasuredWingMultibodyEnvCfg,
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

    raise AttributeError(name)

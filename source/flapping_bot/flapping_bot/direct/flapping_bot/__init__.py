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
    "FlappingBotStraightFlightSimpleEnvCfg",
    "FlappingBotStraightFlightDeLaurierEnvCfg",
    "FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg",
    "FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg",
    "FlappingBotStraightFlightDeLaurierPureRLEnvCfg",
    "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg",
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
        "FlappingBotStraightFlightSimpleEnvCfg",
        "FlappingBotStraightFlightDeLaurierEnvCfg",
        "FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg",
        "FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg",
        "FlappingBotStraightFlightDeLaurierPureRLEnvCfg",
        "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg",
    ):
        from .straight_flight_env import (
            FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
            FlappingBotStraightFlightDeLaurierPureRLEnvCfg,
            FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg,
            FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg,
            FlappingBotStraightFlightDeLaurierEnvCfg,
            FlappingBotStraightFlightEnv,
            FlappingBotStraightFlightEnvCfg,
            FlappingBotStraightFlightMeasuredWingMultibodyIdealCoupledEnvCfg,
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
            "FlappingBotStraightFlightSimpleEnvCfg": FlappingBotStraightFlightSimpleEnvCfg,
            "FlappingBotStraightFlightDeLaurierEnvCfg": FlappingBotStraightFlightDeLaurierEnvCfg,
            "FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg": FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg,
            "FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg": FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg,
            "FlappingBotStraightFlightDeLaurierPureRLEnvCfg": FlappingBotStraightFlightDeLaurierPureRLEnvCfg,
            "FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg": FlappingBotStraightFlightDeLaurierMeasuredPureRLEnvCfg,
        }[name]

    raise AttributeError(name)

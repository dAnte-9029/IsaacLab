"""Direct RL environment definitions for the flapping-wing robot.

Imports stay lazy so pure-Python unit tests can import helper modules without
pulling IsaacSim dependencies such as ``carb`` at package import time.
"""

__all__ = [
    "FlappingBotEnv",
    "FlappingBotEnvCfg",
    "FlappingBotPathTrackingEnv",
    "FlappingBotPathTrackingEnvCfg",
    "FlappingBotStraightFlightEnv",
    "FlappingBotStraightFlightEnvCfg",
    "FlappingBotStraightFlightSimpleEnvCfg",
    "FlappingBotStraightFlightDeLaurierEnvCfg",
    "FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg",
    "FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg",
    "FlappingBotStraightFlightDeLaurierPureRLEnvCfg",
]


def __getattr__(name: str):
    if name in ("FlappingBotEnv", "FlappingBotEnvCfg"):
        from .flapping_env import FlappingBotEnv, FlappingBotEnvCfg

        return {"FlappingBotEnv": FlappingBotEnv, "FlappingBotEnvCfg": FlappingBotEnvCfg}[name]

    if name in ("FlappingBotPathTrackingEnv", "FlappingBotPathTrackingEnvCfg"):
        from .path_tracking_env import FlappingBotPathTrackingEnv, FlappingBotPathTrackingEnvCfg

        return {
            "FlappingBotPathTrackingEnv": FlappingBotPathTrackingEnv,
            "FlappingBotPathTrackingEnvCfg": FlappingBotPathTrackingEnvCfg,
        }[name]

    if name in (
        "FlappingBotStraightFlightEnv",
        "FlappingBotStraightFlightEnvCfg",
        "FlappingBotStraightFlightSimpleEnvCfg",
        "FlappingBotStraightFlightDeLaurierEnvCfg",
        "FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg",
        "FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg",
        "FlappingBotStraightFlightDeLaurierPureRLEnvCfg",
    ):
        from .straight_flight_env import (
            FlappingBotStraightFlightDeLaurierPureRLEnvCfg,
            FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg,
            FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg,
            FlappingBotStraightFlightDeLaurierEnvCfg,
            FlappingBotStraightFlightEnv,
            FlappingBotStraightFlightEnvCfg,
            FlappingBotStraightFlightSimpleEnvCfg,
        )

        return {
            "FlappingBotStraightFlightEnv": FlappingBotStraightFlightEnv,
            "FlappingBotStraightFlightEnvCfg": FlappingBotStraightFlightEnvCfg,
            "FlappingBotStraightFlightSimpleEnvCfg": FlappingBotStraightFlightSimpleEnvCfg,
            "FlappingBotStraightFlightDeLaurierEnvCfg": FlappingBotStraightFlightDeLaurierEnvCfg,
            "FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg": FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg,
            "FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg": FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg,
            "FlappingBotStraightFlightDeLaurierPureRLEnvCfg": FlappingBotStraightFlightDeLaurierPureRLEnvCfg,
        }[name]

    raise AttributeError(name)

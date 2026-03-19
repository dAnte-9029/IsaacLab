"""Direct RL environment definitions for the flapping-wing robot."""

from .flapping_env import FlappingBotEnv, FlappingBotEnvCfg
from .path_tracking_env import FlappingBotPathTrackingEnv, FlappingBotPathTrackingEnvCfg
from .straight_flight_env import (
    FlappingBotStraightFlightDeLaurierPureRLEnvCfg,
    FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg,
    FlappingBotStraightFlightDeLaurierWeakTeacherRLEnvCfg,
    FlappingBotStraightFlightDeLaurierEnvCfg,
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightEnvCfg,
    FlappingBotStraightFlightSimpleEnvCfg,
)

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

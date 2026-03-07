"""Direct RL environment definitions for the flapping-wing robot."""

from .flapping_env import FlappingBotEnv, FlappingBotEnvCfg
from .straight_flight_env import (
    FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg,
    FlappingBotStraightFlightDeLaurierEnvCfg,
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightEnvCfg,
    FlappingBotStraightFlightSimpleEnvCfg,
)

__all__ = [
    "FlappingBotEnv",
    "FlappingBotEnvCfg",
    "FlappingBotStraightFlightEnv",
    "FlappingBotStraightFlightEnvCfg",
    "FlappingBotStraightFlightSimpleEnvCfg",
    "FlappingBotStraightFlightDeLaurierEnvCfg",
    "FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg",
]

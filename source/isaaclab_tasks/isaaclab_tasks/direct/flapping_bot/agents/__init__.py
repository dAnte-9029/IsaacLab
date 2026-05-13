"""Agent configurations for the FlappingBot direct tasks."""

from .rsl_rl_ppo_cfg import FlappingBotPPORunnerCfg
from .rsl_rl_ppo_straightflight_cfg import (
    FlappingBotPathTrackingPPORunnerCfg,
    FlappingBotPathTrackingPrimitivePurePPORunnerCfg,
    FlappingBotStraightFlightPPORunnerCfg,
)

__all__ = [
    "FlappingBotPPORunnerCfg",
    "FlappingBotPathTrackingPPORunnerCfg",
    "FlappingBotPathTrackingPrimitivePurePPORunnerCfg",
    "FlappingBotStraightFlightPPORunnerCfg",
]

"""Agent configurations for the FlappingBot direct tasks."""

from .rsl_rl_ppo_cfg import FlappingBotPPORunnerCfg
from .rsl_rl_ppo_straightflight_cfg import FlappingBotStraightFlightPPORunnerCfg

__all__ = [
    "FlappingBotPPORunnerCfg",
    "FlappingBotStraightFlightPPORunnerCfg",
]

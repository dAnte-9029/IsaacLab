# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Direct workflow environments.
"""

import gymnasium as gym

from .flapping_bot import (
    FlappingBotStraightFlightDeLaurierMeasuredPureRLC3aEnvCfg,
    FlappingBotStraightFlightDeLaurierMeasuredPureRLC3bEnvCfg,
    FlappingBotStraightFlightDeLaurierMeasuredPureRLC3cEnvCfg,
    FlappingBotStraightFlightEnv,
)


_RSL_RL_CFG = (
    "isaaclab_tasks.direct.flapping_bot.agents."
    "rsl_rl_ppo_straightflight_cfg:FlappingBotStraightFlightPPORunnerCfg"
)


gym.register(
    id="Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0",
    entry_point=FlappingBotStraightFlightEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlappingBotStraightFlightDeLaurierMeasuredPureRLC3aEnvCfg,
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
)

gym.register(
    id="Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3b-Direct-v0",
    entry_point=FlappingBotStraightFlightEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlappingBotStraightFlightDeLaurierMeasuredPureRLC3bEnvCfg,
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
)

gym.register(
    id="Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3c-Direct-v0",
    entry_point=FlappingBotStraightFlightEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlappingBotStraightFlightDeLaurierMeasuredPureRLC3cEnvCfg,
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
)

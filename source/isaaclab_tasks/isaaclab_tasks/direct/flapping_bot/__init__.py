"""Flapping-wing robot: Direct RL task registration.

This module registers the flapping-wing environment into Gym so it can be
consumed by Isaac Lab's standard RL training scripts (Hydra + rsl-rl).

It references the environment implemented in the local "flapping_bot"
extension package and provides an RSL-RL default configuration entry point.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gymnasium as gym

# Ensure the custom extension package (flapping_bot) is importable when running
# via training scripts. Fallback to adding the local source path if needed.
try:  # pragma: no cover - import guard
    import flapping_bot  # noqa: F401
except Exception:  # pragma: no cover - best-effort path fix for local dev
    # __file__ may point either inside the repository tree or to a site-packages
    # install of isaaclab_tasks. First, walk upwards from this file looking for
    # a sibling "flapping_bot" extension directory.
    _here = Path(__file__).resolve()
    for _p in _here.parents:
        _candidate = _p / "flapping_bot"
        # prefer the real extension package root: it contains its own flapping_bot/ submodule
        if _candidate.exists() and (_candidate / "flapping_bot").is_dir():
            if str(_candidate) not in sys.path:
                sys.path.insert(0, str(_candidate))
            break
    else:
        # If not found relative to this file (common when isaaclab_tasks is
        # pip-installed), fall back to the ISAACLAB_PATH environment variable.
        _root = os.environ.get("ISAACLAB_PATH")
        if _root:
            for _rel in ("source/flapping_bot", "flapping_bot"):
                _candidate = Path(_root) / _rel
                if _candidate.exists() and (_candidate / "flapping_bot").is_dir():
                    if str(_candidate) not in sys.path:
                        sys.path.insert(0, str(_candidate))
                    break

from .agents.rsl_rl_ppo_cfg import FlappingBotPPORunnerCfg  # noqa: E402
from .agents.rsl_rl_ppo_straightflight_cfg import FlappingBotStraightFlightPPORunnerCfg  # noqa: E402

# Import environment classes directly from the extension's top-level package.
# The flapping_bot extension re-exports these symbols in its __init__.py.
from flapping_bot import (  # noqa: E402
    FlappingBotEnv,
    FlappingBotEnvCfg,
    FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg,
    FlappingBotStraightFlightDeLaurierEnvCfg,
    FlappingBotStraightFlightEnv,
    FlappingBotStraightFlightSimpleEnvCfg,
)


gym.register(
    id="Isaac-FlappingBot-Direct-v0",
    entry_point=FlappingBotEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlappingBotEnvCfg,
        "rsl_rl_cfg_entry_point": FlappingBotPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-FlappingBot-StraightFlight-Simple-Direct-v0",
    entry_point=FlappingBotStraightFlightEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlappingBotStraightFlightSimpleEnvCfg,
        "rsl_rl_cfg_entry_point": FlappingBotStraightFlightPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-FlappingBot-StraightFlight-DeLaurier-Direct-v0",
    entry_point=FlappingBotStraightFlightEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlappingBotStraightFlightDeLaurierEnvCfg,
        "rsl_rl_cfg_entry_point": FlappingBotStraightFlightPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-FlappingBot-StraightFlight-DeLaurier-TeacherRL-Direct-v0",
    entry_point=FlappingBotStraightFlightEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FlappingBotStraightFlightDeLaurierTeacherRLEnvCfg,
        "rsl_rl_cfg_entry_point": FlappingBotStraightFlightPPORunnerCfg,
    },
)

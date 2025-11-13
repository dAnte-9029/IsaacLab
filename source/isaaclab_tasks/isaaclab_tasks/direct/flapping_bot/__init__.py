"""Flapping-wing robot: Direct RL task registration.

This module registers the flapping-wing environment into Gym so it can be
consumed by Isaac Lab's standard RL training scripts (Hydra + rsl-rl).

It references the environment implemented in the local "flapping_bot"
extension package and provides an RSL-RL default configuration entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym

# Ensure the custom extension package (flapping_bot) is importable when running
# via training scripts. Fallback to adding the local source path if needed.
try:  # pragma: no cover - import guard
    import flapping_bot  # noqa: F401
except Exception:  # pragma: no cover - best-effort path fix for local dev
    # __file__ = .../IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py
    # We want to add .../IsaacLab/source/flapping_bot to sys.path. Be robust to path depth.
    _here = Path(__file__).resolve()
    for _p in _here.parents:
        _candidate = _p / "flapping_bot"
        if _candidate.exists():
            if str(_candidate) not in sys.path:
                sys.path.insert(0, str(_candidate))
            break

from .agents.rsl_rl_ppo_cfg import FlappingBotPPORunnerCfg  # noqa: E402

# Import environment classes directly to avoid string-based dynamic imports.
from flapping_bot.flapping_bot.direct.flapping_bot.flapping_env import (  # noqa: E402
    FlappingBotEnv,
    FlappingBotEnvCfg,
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

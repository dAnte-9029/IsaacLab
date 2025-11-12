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
    # We want to add .../IsaacLab/source/flapping_bot to sys.path
    _source_dir = Path(__file__).resolve().parents[4]
    _ext_root = _source_dir / "flapping_bot"
    if str(_ext_root) not in sys.path:
        sys.path.insert(0, str(_ext_root))

from . import agents  # noqa: E402


gym.register(
    id="Isaac-FlappingBot-Direct-v0",
    entry_point=(
        "flapping_bot.flapping_bot.direct.flapping_bot.flapping_env:FlappingBotEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "flapping_bot.flapping_bot.direct.flapping_bot.flapping_env:FlappingBotEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:FlappingBotPPORunnerCfg"
        ),
    },
)

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom flapping-wing robot tasks and assets for Isaac Lab."""

from pathlib import Path

import toml

_EXT_DIR = Path(__file__).resolve().parent.parent
_EXT_METADATA = toml.load(_EXT_DIR / "config" / "extension.toml")
__version__ = _EXT_METADATA["package"]["version"]

# Expose configuration entry points at package import time.
from .assets import FlappingBotCfg  # noqa: E402
from .direct.flapping_bot import FlappingBotEnv, FlappingBotEnvCfg  # noqa: E402
from .scenes import FlappingRoomSceneCfg  # noqa: E402

__all__ = [
    "FlappingBotCfg",
    "FlappingRoomSceneCfg",
    "FlappingBotEnvCfg",
    "FlappingBotEnv",
]

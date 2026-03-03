"""Scene configuration providing a minimal environment for the flapping bot."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg

from ..assets import FlappingBotCfg


class FlappingRoomSceneCfg(InteractiveSceneCfg):
    """Flat test scene with ground, lighting, and a single flapping-wing vehicle."""

    num_envs: int = 1
    env_spacing: float = 4.0
    replicate_physics: bool = True

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(500.0, 500.0)),
    )

    robot = FlappingBotCfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=5000.0, color=(0.9, 0.9, 0.9)),
    )

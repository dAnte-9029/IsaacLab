"""Configuration for the flapping-wing aerial robot."""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


FLAPPING_BOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/FlappingBot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/flapping_bot/robots/robot.urdf",
        usd_dir=f"{ISAACLAB_ASSETS_DATA_DIR}/flapping_bot/robots",
        usd_file_name="flapping_bot.usd",
        fix_base=False,
        merge_fixed_joints=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        make_instanceable=False,
        force_usd_conversion=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_wing": -0.3,
            "right_wing": 0.3,
            "left_tail": 0.0,
            "right_tail": 0.0,
            "mid_tail": 0.0,
        },
    ),
    actuators={
        "wing_servos": ImplicitActuatorCfg(
            joint_names_expr=["left_wing", "right_wing"],
            stiffness=14.0,
            damping=0.8,
        ),
        "tail_servos": ImplicitActuatorCfg(
            joint_names_expr=["left_tail", "right_tail", "mid_tail"],
            stiffness=28.0,
            damping=1.8,
        ),
    },
)
"""URDF-based articulation configuration for the flapping bot."""

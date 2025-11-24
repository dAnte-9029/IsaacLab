"""Configuration for the flapping-wing aerial robot."""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

import os


FLAPPING_BOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/FlappingBot",
    spawn=sim_utils.UrdfFileCfg(
        # Use the v50 flapping robot from the repo (no machine-specific paths).
        asset_path=os.path.join(
            ISAACLAB_ASSETS_DATA_DIR,
            "flapping_bot",
            "robots",
            "flap_robot_v50",
            "urdf",
            "flap_robot_v50.urdf",
        ),
        usd_dir=os.path.join(
            ISAACLAB_ASSETS_DATA_DIR,
            "flapping_bot",
            "robots",
            "flap_robot_v50",
            "urdf",
        ),
        usd_file_name="flap_robot_v50.usd",
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
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 10.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(5.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        joint_pos={
            # Wings start symmetrically; tails start at +20 deg to provide
            # a small initial nose-up trim.
            "left_wing": 0.0,
            "right_wing": 0.0,
            "left_tail": 0.3491,
            "right_tail": 0.3491,
        },
    ),
    actuators={
        "wing_servos": ImplicitActuatorCfg(
            joint_names_expr=["left_wing", "right_wing"],
            stiffness=18.0,
            damping=1.0,
        ),
        # Left/Right tail servos (differential pair)
        "tail_servos": ImplicitActuatorCfg(
            joint_names_expr=["left_tail", "right_tail"],
            stiffness=40.0,
            damping=2.2,
        ),
    },
)

"""URDF-based articulation configuration for the flapping bot."""

"""Utility script to convert the flapping bot URDF into USD."""

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
EXT_ROOT = SRC_ROOT / "flapping_bot"
if str(EXT_ROOT) not in sys.path:
    sys.path.append(str(EXT_ROOT))

from isaaclab.app import AppLauncher


def main():
    app = AppLauncher(headless=True).app
    import isaaclab.sim as sim_utils
    from isaaclab.sim.converters import UrdfConverter
    converter = UrdfConverter(
        sim_utils.UrdfFileCfg(
            asset_path=f"{SRC_ROOT}/isaaclab_assets/data/flapping_bot/robots/robot.urdf",
            usd_dir=f"{SRC_ROOT}/isaaclab_assets/data/flapping_bot/robots",
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
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
        )
    )
    print("Generated:", converter.usd_path)
    app.close()


if __name__ == "__main__":
    main()

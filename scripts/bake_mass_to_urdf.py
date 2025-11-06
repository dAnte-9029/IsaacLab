"""Bake mass/COM/inertia from mass_props.json into a copied URDF and convert to USD.

Outputs:
  - robot_mass6.urdf (no overwrite of original)
  - flapping_bot_mass6.usd

Usage (from IsaacLab root):
  .\isaaclab.bat -p source/flapping_bot/scripts/bake_mass_to_urdf.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


SRC_ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = SRC_ROOT / "flapping_bot"
ROBOTS_DIR = SRC_ROOT / "isaaclab_assets" / "data" / "flapping_bot" / "robots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bake mass props into URDF and convert to USD")
    parser.add_argument("--mass-json", type=str, default=str(EXT_ROOT / "flapping_bot" / "config" / "mass_props.json"), help="Path to mass_props.json")
    parser.add_argument("--input-urdf", type=str, default=str(ROBOTS_DIR / "robot.urdf"), help="Input URDF path")
    parser.add_argument("--output-urdf", type=str, default=str(ROBOTS_DIR / "robot_mass6.urdf"), help="Output URDF path")
    parser.add_argument("--usd-out", type=str, default=str(ROBOTS_DIR / "flapping_bot_mass6.usd"), help="Output USD path")
    return parser.parse_args()


def _ensure_inertial(link_el: ET.Element) -> ET.Element:
    inertial = link_el.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(link_el, "inertial")
    # Ensure children exist
    origin = inertial.find("origin")
    if origin is None:
        origin = ET.SubElement(inertial, "origin")
    mass = inertial.find("mass")
    if mass is None:
        mass = ET.SubElement(inertial, "mass")
    inertia = inertial.find("inertia")
    if inertia is None:
        inertia = ET.SubElement(inertial, "inertia")
    return inertial


def bake_to_urdf(mass_json: Path, input_urdf: Path, output_urdf: Path) -> None:
    import json

    with open(mass_json, "r", encoding="utf-8") as f:
        mass_map = json.load(f)

    tree = ET.parse(input_urdf)
    root = tree.getroot()

    # Build lookup of link elements by name
    name_to_link: dict[str, ET.Element] = {}
    for link_el in root.findall("link"):
        nm = link_el.attrib.get("name")
        if nm:
            name_to_link[nm] = link_el

    edited = []
    for link_name, props in mass_map.items():
        link_el = name_to_link.get(link_name)
        if link_el is None:
            continue
        inertial = _ensure_inertial(link_el)
        origin = inertial.find("origin")
        mass_el = inertial.find("mass")
        inertia_el = inertial.find("inertia")

        m = float(props.get("mass", 0.0))
        cx, cy, cz = [float(v) for v in props.get("com", [0.0, 0.0, 0.0])]
        ixx, iyy, izz = [float(v) for v in props.get("inertia", [0.0, 0.0, 0.0])]

        origin.set("xyz", f"{cx} {cy} {cz}")
        origin.set("rpy", "0 0 0")
        mass_el.set("value", f"{m}")
        inertia_el.set("ixx", f"{ixx}")
        inertia_el.set("ixy", "0.0")
        inertia_el.set("ixz", "0.0")
        inertia_el.set("iyy", f"{iyy}")
        inertia_el.set("iyz", "0.0")
        inertia_el.set("izz", f"{izz}")
        edited.append(link_name)

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print("[OK] Baked mass props for:", edited)
    print("[OK] Wrote:", output_urdf)


def convert_to_usd(urdf_path: Path, usd_out: Path) -> None:
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sim.converters import UrdfConverter
        cfg = sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            usd_dir=str(usd_out.parent),
            usd_file_name=str(usd_out.name),
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
        )
        converter = UrdfConverter(cfg)
        print("[OK] Generated:", converter.usd_path)
    finally:
        app.close()


def main() -> int:
    args = parse_args()
    mass_json = Path(args.mass_json)
    in_urdf = Path(args.input_urdf)
    out_urdf = Path(args.output_urdf)
    usd_out = Path(args.usd_out)

    bake_to_urdf(mass_json, in_urdf, out_urdf)
    convert_to_usd(out_urdf, usd_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())


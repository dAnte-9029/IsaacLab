from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET


CFG_FILE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "isaaclab_assets"
    / "isaaclab_assets"
    / "robots"
    / "flapping_bot.py"
)
URDF_FILE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "isaaclab_assets"
    / "data"
    / "flapping_bot"
    / "robots"
    / "flap_robot_552"
    / "urdf"
    / "flap_robot_552.urdf"
)


def _cfg_text() -> str:
    return CFG_FILE.read_text(encoding="utf-8")


def _joint_limit_deg(joint_name: str) -> tuple[float, float]:
    root = ET.parse(URDF_FILE).getroot()
    for joint in root.findall("joint"):
        if joint.get("name") != joint_name:
            continue
        limit = joint.find("limit")
        assert limit is not None
        lower = math.degrees(float(limit.get("lower", "nan")))
        upper = math.degrees(float(limit.get("upper", "nan")))
        return lower, upper
    raise AssertionError(f"joint {joint_name} not found in {URDF_FILE}")


def test_shared_flapping_asset_points_to_latest_robot() -> None:
    text = _cfg_text()
    assert '"flap_robot_552"' in text
    assert '"flap_robot_552.urdf"' in text


def test_shared_flapping_asset_does_not_force_runtime_urdf_conversion() -> None:
    text = _cfg_text()
    assert "force_usd_conversion=False" in text


def test_shared_flapping_asset_declares_all_expected_joint_defaults() -> None:
    text = _cfg_text()
    for joint_name in ("left_wing", "right_wing", "rudder", "left_tail", "right_tail"):
        assert f'"{joint_name}":' in text


def test_shared_flapping_asset_declares_rudder_actuator() -> None:
    text = _cfg_text()
    assert '"wing_servos"' in text
    assert '"tail_servos"' in text
    assert '"rudder_servo"' in text


def test_latest_tail_and_rudder_joint_limits_support_near_41_deg_authority() -> None:
    for joint_name in ("left_tail", "right_tail", "rudder"):
        lower_deg, upper_deg = _joint_limit_deg(joint_name)
        assert lower_deg <= -41.0
        assert upper_deg >= 41.0

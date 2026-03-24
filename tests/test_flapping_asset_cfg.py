from __future__ import annotations

from pathlib import Path


CFG_FILE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "isaaclab_assets"
    / "isaaclab_assets"
    / "robots"
    / "flapping_bot.py"
)


def _cfg_text() -> str:
    return CFG_FILE.read_text(encoding="utf-8")


def test_shared_flapping_asset_points_to_latest_robot() -> None:
    text = _cfg_text()
    assert '"flap_robot_552"' in text
    assert '"flap_robot_552.urdf"' in text


def test_shared_flapping_asset_declares_all_expected_joint_defaults() -> None:
    text = _cfg_text()
    for joint_name in ("left_wing", "right_wing", "rudder", "left_tail", "right_tail"):
        assert f'"{joint_name}":' in text


def test_shared_flapping_asset_declares_rudder_actuator() -> None:
    text = _cfg_text()
    assert '"wing_servos"' in text
    assert '"tail_servos"' in text
    assert '"rudder_servo"' in text

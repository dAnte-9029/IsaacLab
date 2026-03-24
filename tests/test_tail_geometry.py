from __future__ import annotations

from pathlib import Path

from flapping_bot.flapping_bot.physics.tail_geometry import PlaceholderValue, load_tail_geometry_from_urdf


ASSET_URDF = (
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


def test_tail_geometry_reads_direct_control_surface_fields_from_latest_robot() -> None:
    geometry = load_tail_geometry_from_urdf(ASSET_URDF)

    assert geometry.rudder.link_name == "rudder"
    assert geometry.rudder.joint_name == "rudder"
    assert geometry.rudder.joint_origin_body == (-0.5045, 0.0, -0.00085)
    assert geometry.rudder.hinge_axis_body == (0.0, 0.0, 1.0)
    assert geometry.rudder.joint_limits_rad == (-0.725, 0.725)
    assert geometry.rudder.mass_kg == 0.0531258284570821
    assert geometry.rudder.mesh_path.name == "rudder.STL"

    assert geometry.left_elevon.link_name == "left_tail"
    assert geometry.left_elevon.joint_name == "left_tail"
    assert geometry.left_elevon.joint_origin_body == (-0.5365, 0.15, -0.00335)
    assert geometry.left_elevon.hinge_axis_body == (0.0, 1.0, 0.0)
    assert geometry.left_elevon.joint_limits_rad == (-0.725, 0.725)
    assert geometry.left_elevon.mass_kg == 0.105656479984979
    assert geometry.left_elevon.mesh_path.name == "left_tail.STL"


def test_tail_geometry_keeps_unreadable_tail_parameters_as_placeholders() -> None:
    geometry = load_tail_geometry_from_urdf(ASSET_URDF)

    assert isinstance(geometry.fixed_horizontal.projected_area_m2, PlaceholderValue)
    assert isinstance(geometry.fixed_vertical.projected_area_m2, PlaceholderValue)
    assert isinstance(geometry.left_elevon.mean_chord_m, PlaceholderValue)
    assert isinstance(geometry.right_elevon.span_m, PlaceholderValue)
    assert isinstance(geometry.rudder.aerodynamic_center_body, PlaceholderValue)

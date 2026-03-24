"""Direct tail-geometry metadata loading from URDF files.

This module intentionally separates:
- fields that can be read directly from the exported files
- fields that are required by a future aerodynamic model but are not directly available

Unknown values are represented explicitly with placeholders instead of guesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Inertia6 = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class PlaceholderValue:
    """Explicit marker for a required but currently unreadable parameter."""

    field_name: str
    reason: str


@dataclass(frozen=True)
class TailSurfaceGeometry:
    """Tail-surface metadata loaded directly from files or left as placeholders."""

    name: str
    link_name: str
    joint_name: str | None
    mesh_path: Path | None
    joint_origin_body: Vec3 | PlaceholderValue
    hinge_axis_body: Vec3 | PlaceholderValue
    joint_limits_rad: Vec2 | PlaceholderValue
    com_link_frame: Vec3 | PlaceholderValue
    mass_kg: float | PlaceholderValue
    inertia_link_frame: Inertia6 | PlaceholderValue
    projected_area_m2: float | PlaceholderValue
    span_m: float | PlaceholderValue
    mean_chord_m: float | PlaceholderValue
    aerodynamic_center_body: Vec3 | PlaceholderValue
    incidence_rad: float | PlaceholderValue


@dataclass(frozen=True)
class TailGeometry:
    """Bundle of latest-robot tail surfaces and placeholder-only fixed surfaces."""

    source_urdf: Path
    left_elevon: TailSurfaceGeometry
    right_elevon: TailSurfaceGeometry
    rudder: TailSurfaceGeometry
    fixed_horizontal: TailSurfaceGeometry
    fixed_vertical: TailSurfaceGeometry


def _parse_vec3(text: str) -> Vec3:
    parts = [float(v) for v in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values, got: {text!r}")
    return (parts[0], parts[1], parts[2])


def _placeholder(field_name: str, reason: str) -> PlaceholderValue:
    return PlaceholderValue(field_name=field_name, reason=reason)


def _find_link(root: ET.Element, link_name: str) -> ET.Element:
    for link in root.findall("link"):
        if link.get("name") == link_name:
            return link
    raise KeyError(f"Link not found: {link_name}")


def _find_joint(root: ET.Element, joint_name: str) -> ET.Element:
    for joint in root.findall("joint"):
        if joint.get("name") == joint_name:
            return joint
    raise KeyError(f"Joint not found: {joint_name}")


def _mesh_path(urdf_path: Path, link: ET.Element) -> Path:
    visual = link.find("visual")
    if visual is None:
        raise KeyError(f"Link {link.get('name')} has no visual block.")
    geometry = visual.find("geometry")
    if geometry is None:
        raise KeyError(f"Link {link.get('name')} has no visual geometry block.")
    mesh = geometry.find("mesh")
    if mesh is None or mesh.get("filename") is None:
        raise KeyError(f"Link {link.get('name')} has no mesh filename.")
    return (urdf_path.parent / mesh.get("filename")).absolute()


def _inertial(link: ET.Element) -> tuple[Vec3, float, Inertia6]:
    inertial = link.find("inertial")
    if inertial is None:
        raise KeyError(f"Link {link.get('name')} has no inertial block.")
    origin = inertial.find("origin")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if origin is None or mass is None or inertia is None:
        raise KeyError(f"Link {link.get('name')} inertial block is incomplete.")
    return (
        _parse_vec3(origin.get("xyz", "0 0 0")),
        float(mass.get("value")),
        (
            float(inertia.get("ixx")),
            float(inertia.get("ixy")),
            float(inertia.get("ixz")),
            float(inertia.get("iyy")),
            float(inertia.get("iyz")),
            float(inertia.get("izz")),
        ),
    )


def _surface_from_jointed_link(urdf_path: Path, root: ET.Element, *, link_name: str, joint_name: str, name: str) -> TailSurfaceGeometry:
    link = _find_link(root, link_name)
    joint = _find_joint(root, joint_name)
    origin = joint.find("origin")
    axis = joint.find("axis")
    limit = joint.find("limit")
    if origin is None or axis is None or limit is None:
        raise KeyError(f"Joint {joint_name} is missing origin/axis/limit data.")
    com_link_frame, mass_kg, inertia_link_frame = _inertial(link)

    unreadable_reason = "Not directly encoded in the exported URDF/STL metadata; leave as a placeholder in this pass."

    return TailSurfaceGeometry(
        name=name,
        link_name=link_name,
        joint_name=joint_name,
        mesh_path=_mesh_path(urdf_path, link),
        joint_origin_body=_parse_vec3(origin.get("xyz", "0 0 0")),
        hinge_axis_body=_parse_vec3(axis.get("xyz", "0 0 0")),
        joint_limits_rad=(float(limit.get("lower")), float(limit.get("upper"))),
        com_link_frame=com_link_frame,
        mass_kg=mass_kg,
        inertia_link_frame=inertia_link_frame,
        projected_area_m2=_placeholder("projected_area_m2", unreadable_reason),
        span_m=_placeholder("span_m", unreadable_reason),
        mean_chord_m=_placeholder("mean_chord_m", unreadable_reason),
        aerodynamic_center_body=_placeholder("aerodynamic_center_body", unreadable_reason),
        incidence_rad=_placeholder("incidence_rad", unreadable_reason),
    )


def _fixed_surface_placeholder(urdf_path: Path, root: ET.Element, *, name: str) -> TailSurfaceGeometry:
    base_link = _find_link(root, "base_link")
    base_mesh_path = _mesh_path(urdf_path, base_link)
    reason = "The fixed tail geometry is merged into base_link in this export, so it cannot be isolated directly."
    return TailSurfaceGeometry(
        name=name,
        link_name="base_link",
        joint_name=None,
        mesh_path=base_mesh_path,
        joint_origin_body=_placeholder("joint_origin_body", reason),
        hinge_axis_body=_placeholder("hinge_axis_body", reason),
        joint_limits_rad=_placeholder("joint_limits_rad", reason),
        com_link_frame=_placeholder("com_link_frame", reason),
        mass_kg=_placeholder("mass_kg", reason),
        inertia_link_frame=_placeholder("inertia_link_frame", reason),
        projected_area_m2=_placeholder("projected_area_m2", reason),
        span_m=_placeholder("span_m", reason),
        mean_chord_m=_placeholder("mean_chord_m", reason),
        aerodynamic_center_body=_placeholder("aerodynamic_center_body", reason),
        incidence_rad=_placeholder("incidence_rad", reason),
    )


def load_tail_geometry_from_urdf(urdf_path: str | Path) -> TailGeometry:
    """Load direct tail metadata from a normalized robot URDF."""

    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()
    return TailGeometry(
        source_urdf=urdf_path,
        left_elevon=_surface_from_jointed_link(
            urdf_path, root, link_name="left_tail", joint_name="left_tail", name="left_elevon"
        ),
        right_elevon=_surface_from_jointed_link(
            urdf_path, root, link_name="right_tail", joint_name="right_tail", name="right_elevon"
        ),
        rudder=_surface_from_jointed_link(urdf_path, root, link_name="rudder", joint_name="rudder", name="rudder"),
        fixed_horizontal=_fixed_surface_placeholder(urdf_path, root, name="fixed_horizontal"),
        fixed_vertical=_fixed_surface_placeholder(urdf_path, root, name="fixed_vertical"),
    )

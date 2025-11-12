"""
Compute projected planform areas for wings/tails from URDF + STL and summarize masses.

Usage (from repo root):
  .\\IsaacLab\\isaaclab.bat -p source/flapping_bot/scripts/compute_wing_areas.py \
      --urdf f:/isaac/flap_robot_v50/urdf/flap_robot_v50.urdf \
      --links left_wing right_wing left_tail right_tail \
      [--mesh-scale 1.0]

Notes
- The area is computed by projecting mesh triangles onto the plane perpendicular
  to the joint (hinge) axis obtained from the URDF joint that drives the link.
- The projection uses 0.5 * max(dot(n, a), 0) per triangle, where n is the
  unnormalized face normal (e1 x e2) and a is the unit hinge axis. This counts
  only one side of a thin shell to avoid double counting top/bottom surfaces.
- STL is loaded without external dependencies; both binary and ASCII STL are handled.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET
from typing import Iterable, List, Tuple


Vec3 = Tuple[float, float, float]


def _to_float_tuple(s: str) -> Vec3:
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) != 3:
        raise ValueError(f"Expect 3 numbers, got '{s}'")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_norm(a: Vec3) -> float:
    return (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) ** 0.5


def _v_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _v_normalize(a: Vec3) -> Vec3:
    n = _v_norm(a)
    if n <= 0:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def _stl_load(path: Path) -> List[Tuple[Vec3, Vec3, Vec3]]:
    """Load triangles from STL file (binary or ASCII). Returns list of (v0, v1, v2).

    The function is robust to both encodings without external dependencies.
    """
    data = path.read_bytes()
    # Try binary STL by size heuristic: 80-byte header + 4-byte count + 50 bytes/tri.
    if len(data) >= 84:
        tri_count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + 50 * tri_count
        if expected == len(data):
            tris: List[Tuple[Vec3, Vec3, Vec3]] = []
            off = 84
            for _ in range(tri_count):
                # normal (3f), v0 (3f), v1 (3f), v2 (3f), attr (H)
                # We ignore the stored normal and compute our own from vertices.
                floats = struct.unpack_from("<12fH", data, off)
                v0 = (floats[3], floats[4], floats[5])
                v1 = (floats[6], floats[7], floats[8])
                v2 = (floats[9], floats[10], floats[11])
                tris.append((v0, v1, v2))
                off += 50
            return tris
    # ASCII fallback (simple, line-based)
    text = data.decode(errors="ignore")
    tris = []
    vbuf: List[Vec3] = []
    for line in text.splitlines():
        line = line.strip().lower()
        if line.startswith("vertex"):
            _, *nums = line.split()
            if len(nums) >= 3:
                v = (float(nums[0]), float(nums[1]), float(nums[2]))
                vbuf.append(v)
                if len(vbuf) == 3:
                    tris.append((vbuf[0], vbuf[1], vbuf[2]))
                    vbuf.clear()
        elif line.startswith("endfacet") and vbuf:
            vbuf.clear()
    if not tris:
        raise RuntimeError(f"No triangles parsed from STL: {path}")
    return tris


def _projected_area(tris: Iterable[Tuple[Vec3, Vec3, Vec3]], axis: Vec3, scale: float = 1.0) -> float:
    """Compute projected planform area onto plane perpendicular to `axis`.

    Uses A = 0.5 * sum(max(dot(cross(e1, e2), a_hat), 0)). The scale factor applies
    uniformly to coordinates (i.e., meters if scale=1, millimeters => scale=1e-3).
    """
    a_hat = _v_normalize(axis)
    if _v_norm(a_hat) == 0.0:
        raise ValueError("Hinge axis is zero; cannot compute projection.")
    s = scale
    area = 0.0
    for (v0, v1, v2) in tris:
        # Apply uniform scaling
        v0s = _v_scale(v0, s)
        v1s = _v_scale(v1, s)
        v2s = _v_scale(v2, s)
        e1 = _v_sub(v1s, v0s)
        e2 = _v_sub(v2s, v0s)
        n = _v_cross(e1, e2)
        d = _v_dot(n, a_hat)
        if d > 0.0:
            area += 0.5 * d
    return area


@dataclass
class LinkGeom:
    name: str
    mesh_path: Path
    hinge_axis: Vec3


def _parse_urdf(urdf_path: Path) -> ET.Element:
    tree = ET.parse(urdf_path)
    return tree.getroot()


def _find_link_mesh(root: ET.Element, link_name: str) -> Path | None:
    for link in root.findall("link"):
        if link.get("name") != link_name:
            continue
        # Prefer visual mesh, else collision
        visual = link.find("visual")
        if visual is not None:
            geom = visual.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None and mesh.get("filename"):
                    return Path(mesh.get("filename"))
        collision = link.find("collision")
        if collision is not None:
            geom = collision.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None and mesh.get("filename"):
                    return Path(mesh.get("filename"))
    return None


def _find_joint_axis_for_child(root: ET.Element, child_link: str) -> Vec3 | None:
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None:
            continue
        if child.get("link") == child_link:
            axis = joint.find("axis")
            if axis is not None and axis.get("xyz"):
                return _to_float_tuple(axis.get("xyz"))
            # Default Revolute axis if unspecified in URDF standards is z-axis, but we prefer explicit.
            return None
    return None


def _collect_masses(root: ET.Element) -> Tuple[float, list[tuple[str, float]]]:
    parts: list[tuple[str, float]] = []
    total = 0.0
    for link in root.findall("link"):
        name = link.get("name") or "<unnamed>"
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        if mass is None or mass.get("value") is None:
            continue
        try:
            m = float(mass.get("value"))
        except Exception:
            continue
        parts.append((name, m))
        total += m
    return total, parts


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute flapping robot wing/tail planform areas and mass summary from URDF")
    ap.add_argument("--urdf", type=str, default=str(Path("flap_robot_v50/urdf/flap_robot_v50.urdf").resolve()))
    ap.add_argument("--links", nargs="*", default=["left_wing", "right_wing", "left_tail", "right_tail"], help="Link names to evaluate area for")
    ap.add_argument("--mesh-scale", type=float, default=1.0, help="Uniform mesh coordinate scale to meters (e.g., 0.001 for mm)")
    args = ap.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print(f"[ERR] URDF not found: {urdf_path}")
        return 2

    root = _parse_urdf(urdf_path)
    urdf_dir = urdf_path.parent

    total_mass, parts = _collect_masses(root)
    print("[MASS] per-link (kg):")
    for name, m in parts:
        print(f"  {name:12s} : {m:.6f}")
    print(f"[MASS] total: {total_mass:.6f} kg\n")

    results: list[tuple[str, float]] = []
    for link_name in args.links:
        mesh_rel = _find_link_mesh(root, link_name)
        if mesh_rel is None:
            print(f"[WARN] No mesh found for link '{link_name}'. Skipping.")
            continue
        hinge_axis = _find_joint_axis_for_child(root, link_name)
        if hinge_axis is None:
            print(f"[WARN] No joint axis found for link '{link_name}'. Assuming x-axis (1,0,0).")
            hinge_axis = (1.0, 0.0, 0.0)
        mesh_path = (urdf_dir / mesh_rel).resolve()
        if not mesh_path.exists():
            print(f"[ERR] Mesh not found for link '{link_name}': {mesh_path}")
            continue
        tris = _stl_load(mesh_path)
        area = _projected_area(tris, hinge_axis, scale=float(args.mesh_scale))
        results.append((link_name, area))
        print(f"[AREA] {link_name:12s} (axis={hinge_axis}, scale={args.mesh_scale:g}) -> {area:.6f} m^2")

    if results:
        total_area = sum(a for _, a in results)
        print(f"\n[AREA] sum({len(results)} links): {total_area:.6f} m^2")
    else:
        print("[AREA] No areas computed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


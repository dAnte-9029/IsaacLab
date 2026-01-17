#!/usr/bin/env python3
"""Summarize mean/RMS forces and torques from wind-tunnel CSV.

Input CSV is expected to come from `scripts/isaac_wind_tunnel_flappingbot_v50.py`.

Why this exists:
- The CSV already contains total forces/torques in world frame.
- The CSV also contains `tau_world_*_T_about_base` (net torque about base_link origin).
- For attitude trim you usually want torque about the vehicle COM, not about base_link origin.
  This script can estimate COM from the v50 URDF mass/inertial model and shift the wrench:

    tau_about_COM = tau_about_base - r_COM × F_total

where r_COM is the COM position vector expressed in world coordinates (from base_link origin).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


def _is_finite(x: float) -> bool:
    return math.isfinite(x) and not math.isnan(x)


def _get_float(row: dict[str, str], key: str) -> float:
    v = row.get(key, "")
    if v == "":
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _stats(values: list[float]) -> dict[str, float] | None:
    xs = [v for v in values if _is_finite(v)]
    if not xs:
        return None
    mean = sum(xs) / len(xs)
    rms = math.sqrt(sum(v * v for v in xs) / len(xs))
    std = statistics.pstdev(xs) if len(xs) >= 2 else 0.0
    mn = min(xs)
    mx = max(xs)
    return {"n": float(len(xs)), "mean": mean, "rms": rms, "std": std, "min": mn, "max": mx, "p2p": mx - mn}


def _read_csv_rows(path: Path, *, start_step: int, end_step: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for row in r:
            step_s = row.get("step", "")
            if not step_s:
                continue
            step = int(float(step_s))
            if step < start_step:
                continue
            if end_step is not None and step > end_step:
                continue
            rows.append(row)
    return rows


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _mat_vec(R: list[list[float]], v: list[float]) -> list[float]:
    return [
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    ]


def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Rz(yaw) * Ry(pitch) * Rx(roll)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _R_axis_angle(ax: float, ay: float, az: float, q: float) -> list[list[float]]:
    n = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / n, ay / n, az / n
    c = math.cos(q)
    s = math.sin(q)
    C = 1.0 - c
    return [
        [c + ax * ax * C, ax * ay * C - az * s, ax * az * C + ay * s],
        [ay * ax * C + az * s, c + ay * ay * C, ay * az * C - ax * s],
        [az * ax * C - ay * s, az * ay * C + ax * s, c + az * az * C],
    ]


@dataclass(frozen=True)
class _LinkInertial:
    mass: float
    com_xyz: list[float]


@dataclass(frozen=True)
class _Joint:
    xyz: list[float]
    rpy: list[float]
    axis: list[float]


def _load_urdf_mass_model(urdf_path: Path) -> tuple[dict[str, _LinkInertial], dict[str, _Joint]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links: dict[str, _LinkInertial] = {}
    for link in root.findall("link"):
        name = link.get("name")
        inert = link.find("inertial")
        if name is None or inert is None:
            continue
        mass_el = inert.find("mass")
        origin_el = inert.find("origin")
        if mass_el is None or origin_el is None:
            continue
        mass = float(mass_el.get("value", "nan"))
        xyz = [float(x) for x in origin_el.get("xyz", "0 0 0").split()]
        links[name] = _LinkInertial(mass=mass, com_xyz=xyz)

    joints: dict[str, _Joint] = {}
    for j in root.findall("joint"):
        name = j.get("name")
        origin_el = j.find("origin")
        axis_el = j.find("axis")
        if name is None or origin_el is None or axis_el is None:
            continue
        xyz = [float(x) for x in origin_el.get("xyz", "0 0 0").split()]
        rpy = [float(x) for x in origin_el.get("rpy", "0 0 0").split()]
        axis = [float(x) for x in axis_el.get("xyz", "1 0 0").split()]
        joints[name] = _Joint(xyz=xyz, rpy=rpy, axis=axis)

    return links, joints


def _joint_R_t(j: _Joint, q: float) -> tuple[list[list[float]], list[float]]:
    R0 = _rpy_to_R(*j.rpy)
    Rq = _R_axis_angle(*j.axis, q)
    return _mat_mul(R0, Rq), j.xyz


def _estimate_tau_about_com_world(
    rows: list[dict[str, str]],
    *,
    urdf_path: Path,
    tail_q_rad: float,
    body_pitch_deg: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Returns (Tx, Ty, Tz, mean_com_base_xyz)."""
    links, joints = _load_urdf_mass_model(urdf_path)

    link_names = ["base_link", "left_wing", "right_wing", "left_tail", "right_tail"]
    for nm in link_names:
        if nm not in links:
            raise ValueError(f"URDF missing inertial for link '{nm}': {urdf_path}")

    for jn in ["left_wing", "right_wing", "left_tail", "right_tail"]:
        if jn not in joints:
            raise ValueError(f"URDF missing joint '{jn}': {urdf_path}")

    mass_total = sum(links[nm].mass for nm in link_names)

    # Base -> world rotation matches the wind-tunnel script:
    # it uses pitch = -deg2rad(body_pitch_deg) when creating the fixed-base rig.
    pitch = -math.radians(body_pitch_deg)
    R_bw = _rpy_to_R(0.0, pitch, 0.0)

    # Tails: fixed at tail_q_rad unless you drive them (wind-tunnel script sets them to 0 by default).
    R_Lt, t_Lt = _joint_R_t(joints["left_tail"], tail_q_rad)
    R_Rt, t_Rt = _joint_R_t(joints["right_tail"], tail_q_rad)

    Tx: list[float] = []
    Ty: list[float] = []
    Tz: list[float] = []
    com_sum = [0.0, 0.0, 0.0]
    n_com = 0

    for row in rows:
        qL = _get_float(row, "joint_pos_L")
        qR = _get_float(row, "joint_pos_R")
        if not (_is_finite(qL) and _is_finite(qR)):
            continue

        R_Lw, t_Lw = _joint_R_t(joints["left_wing"], qL)
        R_Rw, t_Rw = _joint_R_t(joints["right_wing"], qR)

        p0 = links["base_link"].com_xyz
        p1 = _vec_add(t_Lw, _mat_vec(R_Lw, links["left_wing"].com_xyz))
        p2 = _vec_add(t_Rw, _mat_vec(R_Rw, links["right_wing"].com_xyz))
        p3 = _vec_add(t_Lt, _mat_vec(R_Lt, links["left_tail"].com_xyz))
        p4 = _vec_add(t_Rt, _mat_vec(R_Rt, links["right_tail"].com_xyz))

        com_b = [0.0, 0.0, 0.0]
        for nm, p in [
            ("base_link", p0),
            ("left_wing", p1),
            ("right_wing", p2),
            ("left_tail", p3),
            ("right_tail", p4),
        ]:
            m = links[nm].mass
            com_b[0] += m * p[0]
            com_b[1] += m * p[1]
            com_b[2] += m * p[2]
        com_b = [com_b[0] / mass_total, com_b[1] / mass_total, com_b[2] / mass_total]

        com_sum[0] += com_b[0]
        com_sum[1] += com_b[1]
        com_sum[2] += com_b[2]
        n_com += 1

        com_w = _mat_vec(R_bw, com_b)

        # About-base net torque (preferred)
        tau_b = [
            _get_float(row, "tau_world_x_T_about_base"),
            _get_float(row, "tau_world_y_T_about_base"),
            _get_float(row, "tau_world_z_T_about_base"),
        ]
        F = [
            _get_float(row, "F_world_x_T"),
            _get_float(row, "F_world_y_T"),
            _get_float(row, "F_world_z_T"),
        ]
        if not all(_is_finite(v) for v in tau_b + F):
            continue

        tau_com = [tau_b[i] - _cross(com_w, F)[i] for i in range(3)]
        Tx.append(tau_com[0])
        Ty.append(tau_com[1])
        Tz.append(tau_com[2])

    if n_com == 0:
        raise ValueError("Could not estimate COM: missing joint_pos_{L,R} in CSV or no finite samples.")
    mean_com_b = [com_sum[0] / n_com, com_sum[1] / n_com, com_sum[2] / n_com]
    return Tx, Ty, Tz, mean_com_b


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize wind-tunnel CSV forces/torques (mean/RMS/min/max).")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--start-step", type=int, default=240)
    ap.add_argument("--end-step", type=int, default=None)
    ap.add_argument("--about-com", action="store_true", help="Also report tau about COM (needs URDF + joint_pos columns).")
    ap.add_argument(
        "--urdf",
        type=Path,
        default=Path("source/isaaclab_assets/data/flapping_bot/robots/flap_robot_v50/urdf/flap_robot_v50.urdf"),
    )
    ap.add_argument("--tail-q-deg", type=float, default=0.0, help="Tail joint angle used in the run (default: 0).")
    ap.add_argument(
        "--body-pitch-deg",
        type=float,
        default=None,
        help="If omitted, read from CSV 'body_pitch_deg' (first selected row).",
    )
    args = ap.parse_args()

    rows = _read_csv_rows(args.csv, start_step=int(args.start_step), end_step=args.end_step)
    if not rows:
        raise ValueError("No rows selected. Check --start-step/--end-step.")

    has_about_base = all(
        k in rows[0] for k in ("tau_world_x_T_about_base", "tau_world_y_T_about_base", "tau_world_z_T_about_base")
    )
    tau_keys = (
        ("tau_world_x_T_about_base", "tau_world_y_T_about_base", "tau_world_z_T_about_base")
        if has_about_base
        else ("tau_world_x_T", "tau_world_y_T", "tau_world_z_T")
    )

    series: dict[str, list[float]] = {
        "F_world_x_T": [],
        "F_world_y_T": [],
        "F_world_z_T": [],
        tau_keys[0]: [],
        tau_keys[1]: [],
        tau_keys[2]: [],
    }
    for row in rows:
        for k in series.keys():
            series[k].append(_get_float(row, k))

    last_step = int(float(rows[-1].get("step", "nan")))
    print(f"[OK] file: {args.csv}")
    print(f"[OK] samples used: {len(rows)}  (steps {int(args.start_step)}..{last_step})")
    for k in ("F_world_x_T", "F_world_y_T", "F_world_z_T", *tau_keys):
        s = _stats(series[k])
        if s is None:
            print(f"[WARN] {k}: no finite samples")
            continue
        print(f"[OK] {k}: mean={s['mean']:+.6f}  rms={s['rms']:.6f}  min={s['min']:+.6f}  max={s['max']:+.6f}")

    if args.about_com:
        body_pitch_deg = args.body_pitch_deg
        if body_pitch_deg is None:
            body_pitch_deg = _get_float(rows[0], "body_pitch_deg")
        if not _is_finite(float(body_pitch_deg)):
            raise ValueError("Body pitch unknown: pass --body-pitch-deg or ensure CSV has 'body_pitch_deg'.")

        Tx, Ty, Tz, mean_com_b = _estimate_tau_about_com_world(
            rows,
            urdf_path=args.urdf,
            tail_q_rad=math.radians(float(args.tail_q_deg)),
            body_pitch_deg=float(body_pitch_deg),
        )
        sx, sy, sz = _stats(Tx), _stats(Ty), _stats(Tz)
        print(
            f"[OK] mean COM in base_link frame (m): ({mean_com_b[0]:+.4f}, {mean_com_b[1]:+.4f}, {mean_com_b[2]:+.4f})"
        )
        if sx and sy and sz:
            print(
                "[OK] tau_world_*_T_about_com: "
                f"mean=({sx['mean']:+.6f}, {sy['mean']:+.6f}, {sz['mean']:+.6f}) N·m  "
                f"rms=({sx['rms']:.6f}, {sy['rms']:.6f}, {sz['rms']:.6f})"
            )


if __name__ == "__main__":
    main()


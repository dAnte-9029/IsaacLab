"""Actual-link kinematics adapter for the shared 3D aerodynamic interface.

Link order right,left; all inputs world FLU except link-local geometry. Rigid
material velocity comes exclusively from actual link COM twist. Prescribed
relative acceleration is used ONLY for restricted added mass, never for rigid
mechanical inertia. No previous-frame body acceleration is used.
"""
import xml.etree.ElementTree as ET
import torch
from .ullt_geometry import load_csv_urdf_geometry, rotation_xyz
from .ullt_state_space import ULLTGeometry, ULLTMotion
from .ullt_twist import PrescribedTwistConfig, twist_wing


def concatenate(items, cls):
    return cls(*(torch.cat([getattr(x, k) for x in items], 1) for k in vars(items[0])))


class LinkKinematics3D:
    def __init__(self, csv, urdf, config, device="cpu", dtype=torch.float32):
        self.config = config
        neutral = load_csv_urdf_geometry(
            csv, urdf, sections_per_wing=config.sections_per_wing, device=device, dtype=dtype
        )
        root = ET.parse(urdf).getroot()
        self.local = []
        self.origins = []
        self.neutral_rot = []
        n = config.sections_per_wing
        for i, name in enumerate(("right", "left")):
            origin = root.find(f"joint[@name='{name}_wing']/origin")
            p = neutral.chord_m.new_tensor([float(x) for x in origin.attrib["xyz"].split()])
            r = rotation_xyz(neutral.chord_m.new_tensor([float(x) for x in origin.attrib["rpy"].split()]))
            sl = slice(i * n, (i + 1) * n)
            self.local.append(
                ULLTGeometry(
                    *[
                        (getattr(neutral, k)[:, sl] - p) @ r
                        for k in ("edge_start_w_m", "edge_end_w_m", "quarter_chord_w_m")
                    ],
                    neutral.chord_forward_w[:, sl] @ r,
                    neutral.chord_m[:, sl],
                )
            )
            self.origins.append(p)
            self.neutral_rot.append(r)
        self.twist = PrescribedTwistConfig(
            config.twist_amplitude_deg,
            config.twist_kappa,
            config.twist_phase_deg,
            config.twist_theta0_root_deg,
            config.twist_theta0_tip_deg,
        )

    def sample(
        self, position, rotation, com, velocity, omega, wind, phase, phase_rate, phase_accel, joint_rate, joint_accel
    ):
        """B,2,3 / B,2,3,3 states; wind B,3 or B,2,N,3 at stations.

        Actual pose includes URDF mounting rotation and actual hinge angle.
        joint_rate/accel are signed right,left +x rotations. Acceleration is
        analytical governor/phase acceleration; body motion excluded from AM.
        """
        geometries = []
        motions = []
        relative = []
        for i, local in enumerate(self.local):
            b = len(phase)
            r = rotation[:, i]

            def rot(x):
                return torch.einsum("bij,bnj->bni", r, x.expand(b, -1, -1))

            g, t = twist_wing(local, phase.new_zeros(3), phase, phase_rate, phase_accel, self.twist)
            points = [
                rot(getattr(g, k)) + position[:, i, None]
                for k in ("edge_start_w_m", "edge_end_w_m", "quarter_chord_w_m")
            ]
            world = ULLTGeometry(*points, rot(g.chord_forward_w), g.chord_m.expand(b, -1))
            om = omega[:, i, None].expand_as(world.quarter_chord_w_m)
            w = wind[:, None] if wind.ndim == 2 else wind[:, i]

            def flow(point, local_flow):
                return w - velocity[:, i, None] - torch.linalg.cross(om, point - com[:, i, None]) + rot(local_flow)

            qaxis = phase.new_zeros((b, 1, 3))
            qaxis[..., 0] = 1
            relom = rot(qaxis * joint_rate[:, i, None, None])
            relalpha = rot(qaxis * joint_accel[:, i, None, None])
            arm = world.three_quarter_chord_w_m - position[:, i, None]
            twist_v = -rot(t.flow_three_quarter_w_mps)
            rv = torch.linalg.cross(relom.expand_as(arm), arm) + twist_v
            ra = (
                torch.linalg.cross(relalpha.expand_as(arm), arm)
                + torch.linalg.cross(relom.expand_as(arm), torch.linalg.cross(relom.expand_as(arm), arm))
                + 2 * torch.linalg.cross(relom.expand_as(arm), twist_v)
                + rot(t.acceleration_w_mps2)
            )
            rom = relom + rot(t.angular_velocity_w_rad_s)
            ralpha = (
                relalpha
                + rot(t.angular_acceleration_w_rad_s2)
                + torch.linalg.cross(relom.expand_as(arm), rot(t.angular_velocity_w_rad_s))
            )
            full = ULLTMotion(
                flow(world.three_quarter_chord_w_m, t.flow_three_quarter_w_mps),
                flow(world.edge_start_w_m, t.flow_edge_start_w_mps),
                flow(world.edge_end_w_m, t.flow_edge_end_w_mps),
                ra,
                om + rot(t.angular_velocity_w_rad_s),
                ralpha,
            )
            rel = ULLTMotion(-rv, torch.zeros_like(rv), torch.zeros_like(rv), ra, rom, ralpha)
            geometries.append(world)
            motions.append(full)
            relative.append(rel)
        return (
            concatenate(geometries, ULLTGeometry),
            concatenate(motions, ULLTMotion),
            concatenate(relative, ULLTMotion),
        )

    def prescribed(self, velocity, omega, reference, phase, frequency, amplitude=0.5233):
        """Frozen body-rate probes, identical geometry to the actual-link adapter."""
        b = len(phase)
        signs = phase.new_tensor([-1, 1])
        rate = 2 * torch.pi * frequency
        q = amplitude * phase.sin()[:, None] * signs
        qd = amplitude * rate * phase.cos()[:, None] * signs
        qdd = -amplitude * rate**2 * phase.sin()[:, None] * signs
        angles = phase.new_zeros((b, 2, 3))
        angles[..., 0] = q
        rotation = rotation_xyz(angles) @ torch.stack(self.neutral_rot)[None]
        position = torch.stack(self.origins)[None].expand(b, -1, -1)
        vel = velocity[:, None] + torch.linalg.cross(omega[:, None].expand_as(position), position - reference[:, None])
        axis = phase.new_zeros((b, 2, 3))
        axis[..., 0] = qd
        om = omega[:, None] + axis
        return self.sample(
            position,
            rotation,
            position,
            vel,
            om,
            torch.zeros_like(velocity),
            phase,
            torch.full_like(phase, rate),
            torch.zeros_like(phase),
            qd,
            qdd,
        )

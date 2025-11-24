"""Simplified quasi-steady aerodynamic model for flapping wings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import math
import torch
import torch.nn.functional as F


@dataclass
class WingQSMCfg:
    """Wing-specific parameters for the quasi-steady model."""

    name: str
    joint_name: str
    hinge_axis_body: Sequence[float]
    lever_arm_body: Sequence[float]
    area: float
    lift_coefficient: float
    drag_coefficient: float
    effective_radius_fraction: float = 0.75
    hinge_damping: float = 0.0


@dataclass
class FlappingQSMCfg:
    """Configuration for the complete flapping robot quasi-steady model."""

    wings: Sequence[WingQSMCfg] = field(default_factory=list)
    air_density: float = 1.225  # kg/m^3 at sea level


class QuasiSteadyWingModel:
    """Compute aerodynamic forces/torques for a set of wings using a quasi-steady assumption."""

    def __init__(self, cfg: FlappingQSMCfg, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)
        self._wing_cfgs = list(cfg.wings)
        if not self._wing_cfgs:
            raise ValueError("QuasiSteadyWingModel requires at least one wing configuration.")

        self._hinge_axes = torch.tensor(
            [wing.hinge_axis_body for wing in self._wing_cfgs], device=self.device, dtype=torch.float32
        )
        self._hinge_axes = F.normalize(self._hinge_axes, dim=1)

        # Approximate chord directions in body frame for AoA calculation.
        # Use cross(hinge_axis, up) as a reference chord; fall back to cross with x if parallel to up.
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float32).expand_as(self._hinge_axes)
        chord = torch.cross(self._hinge_axes, up, dim=1)
        fallback = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32).expand_as(self._hinge_axes)
        mask = chord.norm(dim=1, keepdim=True) < 1e-6
        chord = torch.where(mask, torch.cross(self._hinge_axes, fallback, dim=1), chord)
        self._chord_dirs = F.normalize(chord, dim=1)

        self._lever_arms = torch.tensor(
            [wing.lever_arm_body for wing in self._wing_cfgs], device=self.device, dtype=torch.float32
        )

        self._areas = torch.tensor([wing.area for wing in self._wing_cfgs], device=self.device, dtype=torch.float32)
        self._lift_coeff = torch.tensor(
            [wing.lift_coefficient for wing in self._wing_cfgs], device=self.device, dtype=torch.float32
        )
        self._drag_coeff = torch.tensor(
            [wing.drag_coefficient for wing in self._wing_cfgs], device=self.device, dtype=torch.float32
        )
        self._eff_radius = torch.tensor(
            [wing.effective_radius_fraction for wing in self._wing_cfgs],
            device=self.device,
            dtype=torch.float32,
        )
        self._hinge_damping = torch.tensor(
            [wing.hinge_damping for wing in self._wing_cfgs], device=self.device, dtype=torch.float32
        )
        # Cache joint names to distinguish flapping wings (use virtual twist)
        self._joint_names = [wing.joint_name for wing in self._wing_cfgs]

    def compute_forces(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return body-frame forces, torques, and hinge torques for each wing.

        Args:
            joint_pos: Joint positions of the controlled wings [num_envs, num_wings].
            joint_vel: Joint velocities of the controlled wings [num_envs, num_wings].
            root_lin_vel: Base linear velocity in body/world frame [num_envs, 3].
            root_ang_vel: Base angular velocity in body frame [num_envs, 3].

        Returns:
            Tuple containing tensors with per-wing contributions:
                - forces: body-frame forces per wing [num_envs, num_wings, 3]
                - torques: body-frame torques per wing [num_envs, num_wings, 3]
                - hinge_torques: scalar hinge torque per wing [num_envs, num_wings]
        """
        num_envs = joint_vel.shape[0]
        num_wings = len(self._wing_cfgs)
        if joint_vel.shape[1] != num_wings:
            raise ValueError(f"Expected joint_vel with {num_wings} columns, received {joint_vel.shape[1]}.")

        # Per-wing force/torque contributions in the body frame
        forces = torch.zeros(num_envs, num_wings, 3, device=self.device)
        torques = torch.zeros(num_envs, num_wings, 3, device=self.device)
        hinge_torques = torch.zeros(num_envs, num_wings, device=self.device)

        rho = self.cfg.air_density

        # Reference values for the virtual twist / angle-of-attack model
        # These are chosen for typical flapping conditions (~5 Hz, +/-30 deg).
        alpha_ref = math.radians(25.0)
        sin_2_alpha_ref = math.sin(2.0 * alpha_ref)
        cd_shape_ref = 1.0 - math.cos(2.0 * alpha_ref)
        cd_base_frac = 0.2  # upstroke keeps ~20% of old constant Cd
        cd_gain = 0.8 / cd_shape_ref if cd_shape_ref > 1e-6 else 1.0

        for i in range(num_wings):
            axis = self._hinge_axes[i].expand(num_envs, -1)
            chord = self._chord_dirs[i].expand(num_envs, -1)
            lever = self._lever_arms[i].expand(num_envs, -1)
            effective_lever = lever * self._eff_radius[i]

            # Effective angular velocity at the wing section
            wing_ang_vel = root_ang_vel + joint_vel[:, i].unsqueeze(1) * axis
            rel_linear_vel = torch.cross(wing_ang_vel, effective_lever, dim=1)
            # Account for translational motion of the body
            rel_linear_vel = rel_linear_vel + root_lin_vel

            speed = rel_linear_vel.norm(dim=1, keepdim=True).clamp(min=1e-6)
            velocity_dir = rel_linear_vel / speed
            lift_dir = torch.cross(velocity_dir, axis, dim=1)
            lift_dir = F.normalize(lift_dir, dim=1)
            drag_dir = -velocity_dir

            dynamic_pressure = 0.5 * rho * speed.squeeze(1) ** 2
            area = self._areas[i]

            # ------------------------------------------------------------------
            # Angle-of-attack dependent lift/drag for flapping wings
            # ------------------------------------------------------------------
            if "wing" in self._joint_names[i]:
                # Virtual twist / effective angle-of-attack model:
                # - Use flapping joint speed to distinguish downstroke vs upstroke.
                # - Geometric AoA comes from relative flow vs. chord direction.
                phi_dot = joint_vel[:, i]  # [num_envs]

                # Geometric AoA from local flow in the wing plane.
                # Project relative velocity into plane orthogonal to hinge axis (span),
                # then measure angle between incoming flow and chord direction.
                v_rel = rel_linear_vel
                span = axis
                v_proj = v_rel - torch.sum(v_rel * span, dim=1, keepdim=True) * span
                v_proj_norm = v_proj.norm(dim=1, keepdim=True).clamp(min=1e-6)
                v_in = -v_proj / v_proj_norm  # incoming flow direction in wing plane
                dot = torch.sum(v_in * chord, dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                cross_vec = torch.cross(chord, v_in, dim=1)
                sin = torch.sum(cross_vec * span, dim=1)
                alpha_geom = torch.atan2(sin, dot)  # [num_envs]

                # Virtual twist term: downstroke (phi_dot < 0) -> larger AoA,
                # upstroke (phi_dot > 0)  -> smaller AoA.
                phi_dot_ref = 20.0  # typical peak for 5 Hz, +/-30deg
                eps = 0.1 * phi_dot_ref  # smoothing region for tanh
                sm = torch.tanh(-phi_dot / eps)  # >0 on downstroke, <0 on upstroke

                k_p = 0.25  # rad, ~14 deg difference between up/down
                k_v = 0.005  # s, small directional term
                alpha_eff = alpha_geom + k_p * sm + k_v * phi_dot
                alpha_eff = torch.clamp(alpha_eff, -math.pi / 2.0, math.pi / 2.0)

                # Lift coefficient: scale base CL so that at alpha_ref we recover self._lift_coeff[i]
                if sin_2_alpha_ref > 1e-6:
                    cl_shape = torch.sin(2.0 * alpha_eff) / sin_2_alpha_ref
                else:
                    cl_shape = torch.sin(2.0 * alpha_eff)
                cl_eff = self._lift_coeff[i] * cl_shape

                # Drag coefficient:
                # - Keep a small base drag so upstroke still has some damping.
                # - At alpha_ref, total Cd matches the original constant drag_coefficient.
                cd0 = self._drag_coeff[i]
                cd_base = cd_base_frac * cd0
                cd_shape = 1.0 - torch.cos(2.0 * alpha_eff)
                cd_eff = cd_base + cd_gain * cd0 * cd_shape

                lift = (dynamic_pressure * area * cl_eff).unsqueeze(1)
                drag = (dynamic_pressure * area * cd_eff).unsqueeze(1)
            else:
                # Non-flapping surfaces (e.g., tail): keep simple constant CL/CD for now.
                lift = (dynamic_pressure * area * self._lift_coeff[i]).unsqueeze(1)
                drag = (dynamic_pressure * area * self._drag_coeff[i]).unsqueeze(1)

            wing_force = lift * lift_dir + drag * drag_dir  # [N, 3]
            wing_torque = torch.cross(lever, wing_force, dim=1)  # [N, 3]
            hinge_resist = -self._hinge_damping[i] * joint_vel[:, i]  # [N]
            hinge_from_force = -torch.sum(torch.cross(lever, wing_force, dim=1) * axis, dim=1)  # [N]

            forces[:, i, :] = wing_force
            torques[:, i, :] = wing_torque
            hinge_torques[:, i] = hinge_from_force + hinge_resist

        return forces, torques, hinge_torques

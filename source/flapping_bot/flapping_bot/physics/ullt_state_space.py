"""Izraelevitz 2017 ULLT for prescribed motion (DOI 10.2514/1.J055144).

No Isaac Sim dependency or mutable global state. All vectors use one inertial
frame. Scalar fields are (B,N), vectors (B,N,3), and wake state (B,N,K).
Incident flow is air minus section velocity. Endpoints run along positive
span (right to left for FLU); positive circulation generates upward lift and
has bound-vortex orientation opposite positive span. See the stage-1 ADR for
manuscript errata, the six-state geometry gate and force closure.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, replace
from typing import Callable

from flapping_bot.config.ullt import ULLTConfig

Tensor = torch.Tensor


@dataclass(frozen=True)
class ULLTGeometry:
    """Quarter-chord endpoints/centers (m) and perpendicular chord (m).

    `chord_forward_w` points toward LE. The two endpoint fields are oriented
    in positive span, not necessarily outboard. They include no center-gap
    panels. `normal_w = chord_forward_w cross span_w`.
    """

    edge_start_w_m: Tensor
    edge_end_w_m: Tensor
    quarter_chord_w_m: Tensor
    chord_forward_w: Tensor
    chord_m: Tensor

    @property
    def width_m(self) -> Tensor:
        return torch.linalg.vector_norm(self.edge_end_w_m - self.edge_start_w_m, dim=-1)

    @property
    def span_w(self) -> Tensor:
        return (self.edge_end_w_m - self.edge_start_w_m) / self.width_m.unsqueeze(-1)

    @property
    def normal_w(self) -> Tensor:
        return torch.linalg.cross(self.chord_forward_w, self.span_w)

    @property
    def three_quarter_chord_w_m(self) -> Tensor:
        return self.quarter_chord_w_m - 0.5 * self.chord_m.unsqueeze(-1) * self.chord_forward_w


@dataclass(frozen=True)
class ULLTMotion:
    """Prescribed section kinematics, all (B,N,3) in the inertial frame.

    Flow is air minus body motion, at 3/4 chord and each trailer origin.
    `acceleration_w_mps2` is d(body velocity relative to air)/dt at 3/4 chord,
    expressed in the inertial frame, NOT a derivative in rotating components.
    Angular velocity/acceleration describe each section orientation. This
    acceleration contract is for prescribed inputs, not delayed PhysX feedback.
    """

    flow_three_quarter_w_mps: Tensor
    flow_edge_start_w_mps: Tensor
    flow_edge_end_w_mps: Tensor
    acceleration_w_mps2: Tensor
    angular_velocity_w_rad_s: Tensor
    angular_acceleration_w_rad_s2: Tensor


@dataclass(frozen=True)
class ULLTState:
    """Explicit wake memory (B,N,K), in m/s. No automatic time advancement."""

    values_mps: Tensor

    def reset(self, env_ids: Tensor, replacement: ULLTState | None = None) -> ULLTState:
        """Return a partial reset; untouched environment rows are bitwise retained."""
        if env_ids.dtype != torch.long or env_ids.ndim != 1 or env_ids.device != self.values_mps.device:
            raise ValueError("env_ids must be a device-local 1D int64 tensor")
        if torch.any(env_ids < 0) or torch.any(env_ids >= self.values_mps.shape[0]):
            raise ValueError("env_ids out of range")
        result = self.values_mps.clone()
        if replacement is None:
            result[env_ids] = 0
        else:
            if (
                replacement.values_mps.shape != result.shape
                or replacement.values_mps.dtype != result.dtype
                or replacement.values_mps.device != result.device
            ):
                raise ValueError("replacement must match the full state")
            result[env_ids] = replacement.values_mps[env_ids]
        return ULLTState(result)


@dataclass(frozen=True)
class ULLTLoads:
    """Integrated section loads and resultant in the input inertial frame.

    Force fields (B,N,3) are N, NOT N/m. Added-mass couple is N m about
    3/4 chord. Resultant force/moment are (B,3), in N/N m; moment is about
    the explicitly supplied reference. Positive moments follow the right hand.
    `power_input_w` is (B,) mechanical power against these aerodynamic loads,
    in W, evaluated relative to the air; it is not electrical/motor power.
    """

    circulation_m2_s: Tensor
    effective_normalwash_mps: Tensor
    induced_velocity_w_mps: Tensor
    circulatory_force_w_n: Tensor
    profile_force_w_n: Tensor
    added_mass_force_w_n: Tensor
    added_mass_couple_three_quarter_w_nm: Tensor
    section_force_w_n: Tensor
    section_moment_about_reference_w_nm: Tensor
    force_w_n: Tensor
    moment_about_reference_w_nm: Tensor
    geometric_incidence_rad: Tensor
    correction_fit_rms: Tensor
    power_input_w: Tensor
    effective_incidence_rad: Tensor | None = None
    equivalent_normalwash_mps: Tensor | None = None
    section_cl: Tensor | None = None
    section_cd: Tensor | None = None
    nonlinear_residual_mps: Tensor | None = None
    nonlinear_iterations: Tensor | None = None
    nonlinear_tolerance_mps: Tensor | None = None
    effective_to_incident_speed_ratio: Tensor | None = None


def _same_tensor_contract(reference: Tensor, tensors: tuple[Tensor, ...]) -> None:
    if reference.dtype not in (torch.float32, torch.float64):
        raise ValueError("ULLT requires float32 or float64")
    for value in tensors:
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError("All inputs must share dtype and device; no implicit conversion")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Non-finite ULLT input")


def trailer_influence(geometry: ULLTGeometry, motion: ULLTMotion, cfg: ULLTConfig) -> Tensor:
    """Return vector influence W (B,target,source,3), 1/m, from trailing legs.

    Bound-leg induction is excluded as in Eq. (1). Reject a control point on
    a downstream ray; no vortex core or silent geometric regularization.
    """

    def ray(origin: Tensor, flow: Tensor) -> Tensor:
        speed = torch.linalg.vector_norm(flow, dim=-1)
        if bool((speed <= cfg.minimum_speed_mps).any()):
            raise ValueError("Undefined trailer direction at near-zero speed")
        direction = flow / speed.unsqueeze(-1)
        r = geometry.quarter_chord_w_m[:, :, None, :] - origin[:, None, :, :]
        radius = torch.linalg.vector_norm(r, dim=-1)
        projection = (r * direction[:, None]).sum(-1)
        cross = torch.linalg.cross(direction[:, None].expand_as(r), r)
        transverse = torch.linalg.vector_norm(cross, dim=-1)
        if bool(
            (
                (radius <= cfg.singularity_distance_m)
                | ((projection >= 0) & (transverse <= cfg.singularity_distance_m))
            ).any()
        ):
            raise ValueError("Singular lifting-line control point on a trailer")
        # Stable form of |r|*(|r|-u.r) when cancellation would occur downstream.
        denominator = radius * (radius - projection)
        alternative = radius * transverse.square() / (radius + projection).clamp_min(cfg.singularity_distance_m)
        denominator = torch.where(projection > 0, alternative, denominator)
        return cross / (4 * math.pi * denominator.unsqueeze(-1))

    return ray(geometry.edge_start_w_m, motion.flow_edge_start_w_mps) - ray(
        geometry.edge_end_w_m, motion.flow_edge_end_w_mps
    )


def downwash_correction(geometry: ULLTGeometry, cfg: ULLTConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Eqs. (73)--(77), restricted to a straight coplanar unswept lifting line.

    C1/C2 are least-squares fits of the analytic half-loop kernel, not measured
    aerodynamic parameters. The fit residual is dimensionless (kernel / (2/c)).
    """
    c = geometry.chord_m
    s = geometry.span_w
    axis = s[:, :1]
    offset = geometry.quarter_chord_w_m - geometry.quarter_chord_w_m[:, :1]
    off_axis = offset - (offset * axis).sum(-1, keepdim=True) * axis
    chord_axis = geometry.chord_forward_w
    if (
        bool((torch.linalg.vector_norm(s - axis, dim=-1) > cfg.geometry_tolerance).any())
        or bool((torch.linalg.vector_norm(off_axis, dim=-1) > cfg.geometry_tolerance).any())
        or bool((torch.linalg.vector_norm(chord_axis - chord_axis[:, :1], dim=-1) > cfg.geometry_tolerance).any())
    ):
        raise ValueError("six_state correction requires a straight coplanar unswept lifting line; use four_state")
    # Receiving point minus source edge; the steady half-loop difference must
    # equal the Biot-Savart trailer downwash, including the negative self term.
    d_left = ((geometry.quarter_chord_w_m[:, :, None] - geometry.edge_start_w_m[:, None]) * axis[:, :, None]).sum(-1)
    d_right = ((geometry.quarter_chord_w_m[:, :, None] - geometry.edge_end_w_m[:, None]) * axis[:, :, None]).sum(-1)
    eta = 2 * torch.stack((d_left, d_right), dim=-1) / c[:, None, :, None]
    tau = torch.linspace(0, cfg.correction_fit_end, cfg.correction_fit_samples, device=c.device, dtype=c.dtype)
    poles = c.new_tensor([-0.045, -0.300])
    amplitudes = c.new_tensor([-0.165, -0.335]) * torch.exp(-3 * poles) / (4 * math.pi)
    basis = torch.exp(tau[:, None] * poles) * amplitudes
    length = tau + 1
    # Rationalized form is finite at eta=0 and avoids sqrt(x)-1 cancellation.
    target = eta[..., None] / (4 * math.pi * length * (torch.sqrt(length.square() + eta[..., None].square()) + length))
    coefficients = target @ torch.linalg.pinv(basis).T
    residual = torch.sqrt((target - coefficients @ basis.T).square().mean(dim=(-1, -2, -3, -4)))
    eye = torch.eye(c.shape[1], device=c.device, dtype=c.dtype).unsqueeze(0)
    w1 = coefficients[..., 1, 0] - coefficients[..., 0, 0] - eye
    w2 = coefficients[..., 1, 1] - coefficients[..., 0, 1] - eye
    w0 = (w1 * amplitudes[0] + w2 * amplitudes[1]) * (2 / c[:, None, :])
    return w0, w1, w2, residual


def added_mass_wrench(geometry: ULLTGeometry, motion: ULLTMotion, cfg: ULLTConfig) -> tuple[Tensor, Tensor]:
    """Eqs. (35)--(40): force N and free couple N m at 3/4 chord, (B,N,3)."""
    # Paper's right-handed frame: chord aft, negative span, upward normal.
    rotation = torch.stack((-geometry.chord_forward_w, -geometry.span_w, geometry.normal_w), dim=-1)

    def local(value: Tensor) -> Tensor:
        return (rotation.transpose(-1, -2) @ value.unsqueeze(-1)).squeeze(-1)

    velocity = local(-motion.flow_three_quarter_w_mps)
    omega = local(motion.angular_velocity_w_rad_s)
    velocity_dot = local(motion.acceleration_w_mps2) - torch.linalg.cross(omega, velocity)
    omega_dot = local(motion.angular_acceleration_w_rad_s2)
    chord, width = geometry.chord_m, geometry.width_m
    m33 = cfg.air_density_kg_m3 * math.pi * cfg.added_mass_factor * (chord / 2).square() * width
    m35 = m33 * chord / 4
    m55 = cfg.air_density_kg_m3 * math.pi / 8 * (chord / 2).pow(4) * width + m33 * (chord / 4).square()
    zeros = torch.zeros_like(m33)
    momentum = torch.stack((zeros, zeros, m33 * velocity[..., 2] + m35 * omega[..., 1]), -1)
    angular_momentum = torch.stack((zeros, m35 * velocity[..., 2] + m55 * omega[..., 1], zeros), -1)
    inertia_force = torch.stack((zeros, zeros, m33 * velocity_dot[..., 2] + m35 * omega_dot[..., 1]), -1)
    inertia_couple = torch.stack((zeros, m35 * velocity_dot[..., 2] + m55 * omega_dot[..., 1], zeros), -1)
    force = -inertia_force - torch.linalg.cross(omega, momentum)
    couple = -inertia_couple - torch.linalg.cross(velocity, momentum) - torch.linalg.cross(omega, angular_momentum)
    couple = couple + torch.stack((zeros, -m33 * velocity[..., 0] * velocity[..., 2], zeros), -1)
    return ((rotation @ force.unsqueeze(-1)).squeeze(-1), (rotation @ couple.unsqueeze(-1)).squeeze(-1))


class ULLTBackend:
    """Selectable standalone backend; state is supplied and returned explicitly."""

    def __init__(self, config: ULLTConfig = ULLTConfig()):
        self.config = config
        self.state_count = {"quasi_steady": 0, "one_state": 1, "four_state": 4, "six_state": 6}[config.model]
        self.solver_statistics = {
            "calls": 0,
            "environment_solves": 0,
            "newton_iterations_sum": 0,
            "newton_iterations_max": 0,
            "backtracks_sum": 0,
            "residual_max_mps": 0.0,
        }

    def zero_state(self, geometry: ULLTGeometry) -> ULLTState:
        """Create an impulsive-start state matching geometry device and dtype."""
        return ULLTState(geometry.chord_m.new_zeros((*geometry.chord_m.shape, self.state_count)))

    def _validate(self, geometry: ULLTGeometry, motion: ULLTMotion, state: ULLTState) -> None:
        c = geometry.chord_m
        if c.ndim != 2 or min(c.shape) < 1:
            raise ValueError("chord_m must have shape (B,N), B,N >= 1")
        vectors = (
            geometry.edge_start_w_m,
            geometry.edge_end_w_m,
            geometry.quarter_chord_w_m,
            geometry.chord_forward_w,
            *vars(motion).values(),
        )
        if any(v.shape != (*c.shape, 3) for v in vectors):
            raise ValueError("Geometry and motion vectors must have shape (B,N,3)")
        if state.values_mps.shape != (*c.shape, self.state_count):
            raise ValueError("Wake state shape does not match geometry/model")
        _same_tensor_contract(c, (c, *vectors, state.values_mps))
        if bool((c <= 0).any()) or bool((geometry.width_m <= self.config.singularity_distance_m).any()):
            raise ValueError("Chord and section width must be positive")
        tol = self.config.geometry_tolerance
        if bool(((torch.linalg.vector_norm(geometry.chord_forward_w, dim=-1) - 1).abs() > tol).any()):
            raise ValueError("Chord direction must be unit length")
        if bool(((geometry.chord_forward_w * geometry.span_w).sum(-1).abs() > tol).any()):
            raise ValueError("Chord must be perpendicular to local sweep/span")
        if bool(
            (
                -(motion.flow_three_quarter_w_mps * geometry.chord_forward_w).sum(-1) <= self.config.minimum_speed_mps
            ).any()
        ):
            raise ValueError("Near-zero or reverse chordwise flow is outside stage-1 ULLT")

    def _coefficients(self, reference: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, float, float]:
        if self.state_count == 1:
            return (
                reference.new_tensor([-0.5]),
                reference.new_tensor([-0.25]),
                reference.new_tensor([1.6]),
                reference.new_tensor([-0.8]),
                0.5,
                0.2,
            )
        a_phi = reference.new_tensor([-0.165, -0.335])
        a_gamma = reference.new_tensor([-0.177, -0.609, 0.170, -0.259])
        gain = torch.cat((a_gamma[:2] / a_phi, reference.new_ones(2)))
        return (
            torch.cat((a_phi, a_gamma[2:])),
            reference.new_tensor([-0.045, -0.300, -0.433, -1.419]),
            gain,
            a_gamma,
            0.5,
            1 + float(sum((-0.177, -0.609, 0.170, -0.259))),
        )

    def _solve(self, geometry: ULLTGeometry, motion: ULLTMotion, state: ULLTState) -> dict[str, Tensor]:
        self._validate(geometry, motion, state)
        c = geometry.chord_m
        cfg = self.config
        flow = motion.flow_three_quarter_w_mps
        normal = geometry.normal_w
        perpendicular = flow - (flow * geometry.span_w).sum(-1, keepdim=True) * geometry.span_w
        speed = torch.linalg.vector_norm(perpendicular, dim=-1)
        vn = (flow * normal).sum(-1)
        w_vector = trailer_influence(geometry, motion, cfg)
        wn = (w_vector * normal[:, :, None]).sum(-1)
        correction = torch.zeros_like(vn)
        w0 = torch.zeros_like(wn)
        residual = c.new_zeros(c.shape[0])
        if self.state_count == 6:
            w0, w1, w2, residual = downwash_correction(geometry, cfg)
            correction = (w1 @ state.values_mps[..., 4, None] + w2 @ state.values_mps[..., 5, None]).squeeze(-1)
        if self.state_count == 0:
            gamma0, phi0 = 1.0, 1.0
            history_gamma, history_lift = torch.zeros_like(vn), torch.zeros_like(vn)
        else:
            _, _, gain, _, phi0, gamma0 = self._coefficients(c)
            history_gamma = (state.values_mps[..., : gain.numel()] * gain).sum(-1)
            history_lift = state.values_mps[..., : 1 if self.state_count == 1 else 2].sum(-1)
        matrix = torch.diag_embed(2 / (c * cfg.lift_slope_per_rad)) - gamma0 * (wn + w0)
        gamma = torch.linalg.solve(matrix, (gamma0 * (vn + correction) + history_gamma).unsqueeze(-1)).squeeze(-1)
        nonlinear = {}
        if cfg.section_model == "nonlinear_flat_plate":
            from .ullt_nonlinear import solve_nonlinear_circulation

            tangent = -(w_vector * geometry.chord_forward_w[:, :, None]).sum(-1)
            nonlinear = solve_nonlinear_circulation(
                c,
                -(flow * geometry.chord_forward_w).sum(-1),
                vn,
                tangent,
                wn,
                gamma0,
                history_gamma,
                gamma,
                speed,
                cfg,
            )
            gamma = nonlinear["gamma"]
            if cfg.collect_solver_statistics:
                stats = self.solver_statistics
                stats["calls"] += 1
                stats["environment_solves"] += len(c)
                stats["newton_iterations_sum"] += int(nonlinear["nonlinear_iterations"].sum())
                stats["newton_iterations_max"] = max(
                    stats["newton_iterations_max"], int(nonlinear["nonlinear_iterations"].max())
                )
                stats["backtracks_sum"] += int(nonlinear["nonlinear_backtracks"].sum())
                stats["restarts_sum"] = stats.get("restarts_sum", 0) + int(nonlinear["nonlinear_restarts"].sum())
                stats["residual_max_mps"] = max(
                    stats["residual_max_mps"], float(nonlinear["nonlinear_residual_mps"].max())
                )
        induced = torch.einsum("bijn,bj->bin", w_vector, gamma)
        correction = correction + (w0 @ gamma.unsqueeze(-1)).squeeze(-1)
        ve = vn + (induced * normal).sum(-1) + correction
        if nonlinear:
            ve = nonlinear["equivalent"]
        lift_response = phi0 * ve + history_lift
        return dict(
            **{k: v for k, v in nonlinear.items() if k != "gamma"},
            gamma=gamma,
            ve=ve,
            induced=induced,
            correction=correction,
            response=lift_response,
            speed=speed,
            perpendicular=perpendicular,
            fit_rms=residual,
        )

    def derivative(self, geometry: ULLTGeometry, motion: ULLTMotion, state: ULLTState) -> Tensor:
        """Return wake-state derivatives (B,N,K), m/s², without mutation."""
        solved = self._solve(geometry, motion, state)
        if self.state_count == 0:
            return torch.zeros_like(state.values_mps)
        c = geometry.chord_m
        amplitudes, poles, _, _, _, _ = self._coefficients(c)
        rate = 2 * solved["speed"] / c
        count = amplitudes.numel()
        main = rate.unsqueeze(-1) * poles * (state.values_mps[..., :count] + amplitudes * solved["ve"].unsqueeze(-1))
        if self.state_count != 6:
            return main
        bw = c.new_tensor([-0.045, -0.300])
        aw = c.new_tensor([-0.165, -0.335]) * torch.exp(-3 * bw) / (4 * math.pi)
        wake = rate.unsqueeze(-1) * bw * (state.values_mps[..., 4:] + (2 * solved["gamma"] / c).unsqueeze(-1) * aw)
        return torch.cat((main, wake), -1)

    def steady_state(self, geometry: ULLTGeometry, motion: ULLTMotion) -> ULLTState:
        """Initialize a fully developed steady wake, not an impulsive start."""
        qs = ULLTBackend(replace(self.config, model="quasi_steady"))
        solved = qs._solve(geometry, motion, qs.zero_state(geometry))
        if self.state_count == 0:
            return self.zero_state(geometry)
        c = geometry.chord_m
        amplitudes, _, _, _, _, _ = self._coefficients(c)
        values = -solved["ve"].unsqueeze(-1) * amplitudes
        if self.state_count == 6:
            downwash_correction(geometry, self.config)  # enforce geometry gate even at steady initialization
            bw = c.new_tensor([-0.045, -0.300])
            aw = c.new_tensor([-0.165, -0.335]) * torch.exp(-3 * bw) / (4 * math.pi)
            values = torch.cat((values, -(2 * solved["gamma"] / c).unsqueeze(-1) * aw), -1)
        return ULLTState(values)

    def evaluate(
        self, geometry: ULLTGeometry, motion: ULLTMotion, state: ULLTState, reference_w_m: Tensor
    ) -> ULLTLoads:
        """Evaluate loads about reference (B,3) m, without changing wake state."""
        solved = self._solve(geometry, motion, state)
        c = geometry.chord_m
        if reference_w_m.shape != (c.shape[0], 3):
            raise ValueError("reference_w_m must have shape (B,3)")
        _same_tensor_contract(c, (reference_w_m,))
        flow, speed = motion.flow_three_quarter_w_mps, solved["speed"]
        wake_normal = torch.linalg.cross(geometry.span_w, solved["perpendicular"] / speed.unsqueeze(-1))
        w2d = solved["response"] - solved["ve"]
        total_flow = flow + solved["induced"] + wake_normal * (w2d + solved["correction"]).unsqueeze(-1)
        total_speed = torch.linalg.vector_norm(total_flow, dim=-1)
        if bool((total_speed <= self.config.minimum_speed_mps).any()):
            raise ValueError("Undefined resultant flow direction")
        direction = torch.linalg.cross(geometry.span_w, total_flow / total_speed.unsqueeze(-1))
        lift = (
            0.5
            * self.config.air_density_kg_m3
            * c
            * self.config.lift_slope_per_rad
            * speed
            * solved["response"]
            * geometry.width_m
        )
        circulatory = lift.unsqueeze(-1) * direction
        profile = (
            (
                0.5
                * self.config.air_density_kg_m3
                * c
                * geometry.width_m
                * speed.square()
                * self.config.profile_drag_coefficient
            ).unsqueeze(-1)
            * solved["perpendicular"]
            / speed.unsqueeze(-1)
        )
        if self.config.section_model == "nonlinear_flat_plate":
            # Extra separated profile drag only. Baseline Cd and the induced-flow
            # orientation of circulatory force are retained, without another
            # induced-drag, rotational-lift or added-mass contribution.
            effective_flow = flow + solved["induced"]
            effective_perp = effective_flow - (effective_flow * geometry.span_w).sum(-1, keepdim=True) * geometry.span_w
            separated_cd = solved["section_cd"] - self.config.profile_drag_coefficient
            profile = (
                profile
                + (0.5 * self.config.air_density_kg_m3 * c * geometry.width_m * speed * separated_cd).unsqueeze(-1)
                * effective_perp
            )
        added_force, added_couple = torch.zeros_like(flow), torch.zeros_like(flow)
        if self.config.include_added_mass:
            added_force, added_couple = added_mass_wrench(geometry, motion, self.config)
        force = circulatory + profile + added_force
        moment = (
            torch.linalg.cross(geometry.quarter_chord_w_m - reference_w_m[:, None], circulatory + profile)
            + torch.linalg.cross(geometry.three_quarter_chord_w_m - reference_w_m[:, None], added_force)
            + added_couple
        )
        incidence = torch.atan2((flow * geometry.normal_w).sum(-1), -(flow * geometry.chord_forward_w).sum(-1))
        effective_flow = flow + solved["induced"]
        effective_n = (effective_flow * geometry.normal_w).sum(-1)
        effective_u = -(effective_flow * geometry.chord_forward_w).sum(-1)
        effective_alpha = torch.atan2(effective_n, effective_u)
        effective_speed = torch.sqrt(effective_n.square() + effective_u.square())
        if not bool(torch.isfinite(force).all()) or not bool(torch.isfinite(moment).all()):
            raise FloatingPointError("Non-finite ULLT loads")
        velocity_three_quarter = -flow
        velocity_quarter = velocity_three_quarter + torch.linalg.cross(
            motion.angular_velocity_w_rad_s, geometry.quarter_chord_w_m - geometry.three_quarter_chord_w_m
        )
        # Mechanical power needed against the modeled aerodynamic wrench, W.
        # This is neither electrical power nor a motor/gearbox feasibility claim.
        power = -(
            (circulatory + profile) * velocity_quarter
            + added_force * velocity_three_quarter
            + added_couple * motion.angular_velocity_w_rad_s
        ).sum(dim=(-1, -2))
        if not bool(torch.isfinite(power).all()):
            raise FloatingPointError("Non-finite aerodynamic power")
        return ULLTLoads(
            solved["gamma"],
            solved.get("physical_ve", solved["ve"]),
            solved["induced"],
            circulatory,
            profile,
            added_force,
            added_couple,
            force,
            moment,
            force.sum(1),
            moment.sum(1),
            incidence,
            solved["fit_rms"],
            power,
            effective_alpha,
            solved["ve"],
            solved.get("section_cl", self.config.lift_slope_per_rad * effective_alpha.sin()),
            solved.get("section_cd", torch.full_like(c, self.config.profile_drag_coefficient)),
            solved.get("nonlinear_residual_mps", torch.zeros_like(c[:, 0])),
            solved.get("nonlinear_iterations", torch.zeros_like(c[:, 0], dtype=torch.long)),
            solved.get("nonlinear_tolerance_mps", torch.zeros_like(c[:, 0])),
            effective_speed / speed,
        )

    def advance(
        self, state: ULLTState, sample: Callable[[float], tuple[ULLTGeometry, ULLTMotion]], time_s: float, dt_s: float
    ) -> ULLTState:
        """RK4 with stage-consistent prescribed kinematics and bounded pole steps.

        `sample(t)` must be pure and retain section identities. Local rate bounds
        are sampled at start/mid/end; convergence checks are still required for
        arbitrarily fast prescribed inputs. The input state is never mutated.
        """
        if not math.isfinite(dt_s) or dt_s <= 0 or not math.isfinite(time_s):
            raise ValueError("time must be finite and dt positive")
        rate_bound = 0.0
        for t in (time_s, time_s + dt_s / 2, time_s + dt_s):
            geometry, motion = sample(t)
            self._validate(geometry, motion, state)
            flow = motion.flow_three_quarter_w_mps
            perpendicular = flow - (flow * geometry.span_w).sum(-1, keepdim=True) * geometry.span_w
            rate_bound = max(
                rate_bound, float((2 * torch.linalg.vector_norm(perpendicular, dim=-1) / geometry.chord_m).max())
            )
        if self.state_count == 0:
            return ULLTState(state.values_mps.clone())
        pole = 0.25 if self.state_count == 1 else 1.419
        steps = max(1, math.ceil(dt_s * rate_bound * pole / self.config.maximum_pole_step))
        if steps > self.config.maximum_substeps:
            raise ValueError("Too many ULLT substeps; reduce dt or review near-zero tip chord")
        h = dt_s / steps
        values = state.values_mps

        def rhs(t: float, x: Tensor) -> Tensor:
            return self.derivative(*sample(t), ULLTState(x))

        for index in range(steps):
            t = time_s + index * h
            k1 = rhs(t, values)
            k2 = rhs(t + h / 2, values + h / 2 * k1)
            k3 = rhs(t + h / 2, values + h / 2 * k2)
            k4 = rhs(t + h, values + h * k3)
            values = values + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("Non-finite wake integration")
        return ULLTState(values)

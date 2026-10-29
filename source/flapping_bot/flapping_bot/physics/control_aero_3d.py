"""Control-oriented L3D/I3D. SI, proper FLU vectors, air-minus-material flow.

Stateless batched backends: resetting any subset requires no wake-history reset.
I3D uses straight horseshoes, linear attached circulation and a local separated
load. This is a finite-wing engineering closure, not nonlinear lifting line or
unsteady wake theory. Added mass accepts ONLY relative-motion kinematics.
"""
from dataclasses import dataclass
import math
import torch
from flapping_bot.config.nonlinear_section import FlatPlatePolarConfig
from flapping_bot.config.ullt import ULLTConfig
from .flat_plate_section import flat_plate_polar
from .ullt_state_space import ULLTGeometry, ULLTMotion, added_mass_wrench


@dataclass(frozen=True)
class BodyDrag3DConfig:
    """Geometric engineering priors; projected CdA per station and FLU axis."""

    points_m: tuple = ((-0.03, 0.0, 0.0), (-0.13, 0.0, 0.0), (-0.23, 0.0, 0.0))
    cda_m2: tuple = ((0.0012, 0.006, 0.006),) * 3

    def __post_init__(self):
        if len(self.points_m) != len(self.cda_m2) or not self.points_m:
            raise ValueError("Body drag requires matching nonempty stations and CdA")
        if any(len(x) != 3 or not all(math.isfinite(v) for v in x) for x in self.points_m + self.cda_m2):
            raise ValueError("Body geometry and CdA must be finite 3-vectors")
        if any(v < 0 for row in self.cda_m2 for v in row):
            raise ValueError("Body drag CdA must be dissipative")


@dataclass(frozen=True)
class ControlAero3DConfig:
    body: BodyDrag3DConfig = BodyDrag3DConfig()
    backend: str = "L3D"
    sections_per_wing: int = 10
    rho: float = 1.225
    profile_cd: float = 0.028  # engineering prior, shared; not fitted
    span_friction_cd: float = 0.005  # skin friction only, no lateral derivative gain
    polar: FlatPlatePolarConfig = FlatPlatePolarConfig()
    twist_theta0_root_deg: float = 0.0
    twist_theta0_tip_deg: float = 0.0
    twist_amplitude_deg: float = 0.0
    twist_kappa: float = -0.5
    twist_phase_deg: float = 0.0
    include_added_mass: bool = True
    minimum_speed: float = 0.25
    max_condition: float = 1e6

    def __post_init__(self):
        if not all(
            math.isfinite(v)
            for v in (self.rho, self.profile_cd, self.span_friction_cd, self.minimum_speed, self.max_condition)
        ):
            raise ValueError("Nonfinite aero configuration")
        if self.backend not in ("L3D", "I3D") or self.sections_per_wing < 2:
            raise ValueError("Expected L3D/I3D and >=2 stations/wing")
        if (
            min(self.rho, self.minimum_speed, self.max_condition) <= 0
            or min(self.profile_cd, self.span_friction_cd) < 0
        ):
            raise ValueError("Invalid physical/numerical parameters")


@dataclass
class SectionLoads3D:
    force: torch.Tensor
    moment: torch.Tensor  # per section about requested reference, including free couple
    gamma: torch.Tensor
    alpha: torch.Tensor
    separation: torch.Tensor
    residual: torch.Tensor
    condition: torch.Tensor
    induced: torch.Tensor


def horseshoe_influence(g, targets, direction, *, omit_self_bound=False):
    """Unit physical circulation; bound start->end, end->wake, wake->start.

    No finite-core tuning: reject geometric singularities. At quarter chord the
    own bound segment is its principal value (zero), not regularized circulation.
    """
    a, b = g.edge_start_w_m, g.edge_end_w_m

    def ray(origin):
        r = targets[:, :, None] - origin[:, None]
        d = direction[:, None].expand_as(r)
        rad = r.norm(dim=-1)
        proj = (r * d).sum(-1)
        cross = torch.linalg.cross(d, r)
        den = rad * (rad - proj)
        den = torch.where(proj > 0, rad * cross.square().sum(-1) / (rad + proj).clamp_min(1e-12), den)
        if bool((den < 1e-12).any()):
            raise RuntimeError("I3D singular wake ray")
        return cross / (4 * math.pi * den[..., None])

    ra, rb = targets[:, :, None] - a[:, None], targets[:, :, None] - b[:, None]
    cross = torch.linalg.cross(ra, rb)
    den = cross.square().sum(-1)
    eye = torch.eye(den.shape[-1], device=den.device, dtype=torch.bool)[None]
    on_segment = (ra * rb).sum(-1) <= 0
    if bool(((den < 1e-16) & on_segment & ~(eye if omit_self_bound else torch.zeros_like(eye))).any()):
        raise RuntimeError("I3D singular bound segment")
    factor = (
        (b - a)[:, None]
        * (ra / ra.norm(dim=-1, keepdim=True).clamp_min(1e-12) - rb / rb.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    ).sum(-1)
    bound = cross * (factor / (4 * math.pi * den.clamp_min(1e-16)))[..., None]
    if omit_self_bound:
        bound = bound.masked_fill(eye[..., None], 0.0)
    return bound + ray(b) - ray(a), bound


class ControlAero3D:
    def __init__(self, config=ControlAero3DConfig()):
        self.config = config

    def evaluate(self, g: ULLTGeometry, m: ULLTMotion, reference, relative_motion: ULLTMotion | None = None):
        cfg = self.config
        flow = m.flow_three_quarter_w_mps
        span, normal, chord = g.span_w, g.normal_w, g.chord_forward_w
        perp = flow - (flow * span).sum(-1, keepdim=True) * span
        speed = perp.norm(dim=-1)
        drag = perp / speed.clamp_min(1e-12)[..., None]
        lift = torch.linalg.cross(span, drag)
        alpha = torch.atan2((flow * normal).sum(-1), -(flow * chord).sum(-1))
        polar = flat_plate_polar(alpha, cfg.polar, baseline_cd=cfg.profile_cd)
        sep = polar.separation_weight
        area = g.chord_m * g.width_m
        dynamic = 0.5 * cfg.rho * speed.square() * area
        gamma = torch.zeros_like(speed)
        induced = torch.zeros_like(flow)
        condition = torch.ones_like(speed[:, 0])
        residual = condition * 0
        if cfg.backend == "L3D":
            force = dynamic[..., None] * (polar.cl[..., None] * lift + polar.cd[..., None] * drag)
        else:
            # Low-speed and reverse-flow stations lose attached circulation
            # continuously; separated loads remain bounded and explicitly local.
            forward = -(flow * chord).sum(-1)
            gate = (forward / cfg.minimum_speed).clamp(0, 1)
            gate = gate.square() * (3 - 2 * gate)
            attached = (1 - sep) * gate
            # Wake follows the shared translational downstream direction. At
            # rest choose chord-aft; zero circulation makes this choice immaterial.
            mean = flow.mean(1, keepdim=True)
            norm = mean.norm(dim=-1, keepdim=True)
            if bool(((norm[:, 0, 0] < 1e-9) & (attached.amax(1) > 0)).any()):
                raise RuntimeError("I3D undefined wake direction with active attached circulation")
            fallback = -chord.mean(1, keepdim=True)
            direction = torch.where(norm > 1e-9, mean, fallback)
            direction = (direction / direction.norm(dim=-1, keepdim=True)).expand_as(flow)
            # Lifting-line closure at the bound quarter-chord. The local 2D
            # singular bound contribution is represented by the 2*pi slope;
            # nonlocal bound segments and both trailing legs remain explicit.
            # Incidence is sampled at 3/4 chord, retaining pitch-rate kinematics.
            k, _ = horseshoe_influence(g, g.quarter_chord_w_m, direction, omit_self_bound=True)
            kn = (k * normal[:, :, None]).sum(-1)
            a = math.pi * g.chord_m * attached
            matrix = torch.eye(speed.shape[1], device=speed.device, dtype=speed.dtype)[None] + a[..., None] * kn
            rhs = -a * (flow * normal).sum(-1)
            condition = torch.linalg.cond(matrix)
            if not bool(torch.isfinite(condition).all()) or bool((condition > cfg.max_condition).any()):
                raise RuntimeError("I3D ill-conditioned circulation system")
            gamma = torch.linalg.solve(matrix, rhs[..., None]).squeeze(-1)
            residual = (matrix @ gamma[..., None] - rhs[..., None]).abs().amax((1, 2)) / (1 + rhs.abs().amax(1))
            if bool((residual > (2e-5 if flow.dtype == torch.float32 else 1e-10)).any()):
                raise RuntimeError("I3D circulation residual exceeded tolerance")
            induced = torch.einsum("bijc,bj->bic", k, gamma)
            # Kutta-Joukowski already includes induced drag; only profile and
            # separated drag added below (no Oswald/CL^2 correction).
            circulatory = cfg.rho * gamma[..., None] * g.width_m[..., None] * torch.linalg.cross(flow + induced, span)
            separated_cl = cfg.polar.separated_lift_amplitude * (2 * alpha).sin()
            force = circulatory + dynamic[..., None] * (
                sep[..., None] * separated_cl[..., None] * lift + polar.cd[..., None] * drag
            )
        # Orthogonal span skin friction; baseline profile Cd acts only in the
        # normal plane, hence this term cannot double count it.
        vs = (flow * span).sum(-1)
        force = force + (0.5 * cfg.rho * cfg.span_friction_cd * area * vs.abs() * vs)[..., None] * span
        moment = torch.linalg.cross(g.quarter_chord_w_m - reference[:, None], force)
        if cfg.include_added_mass:
            if relative_motion is None:
                raise ValueError("Relative-only added-mass motion is required explicitly")
            af, ac = added_mass_wrench(g, relative_motion, ULLTConfig())
            af, ac = af * (cfg.rho / 1.225), ac * (cfg.rho / 1.225)
            force = force + af
            moment = moment + torch.linalg.cross(g.three_quarter_chord_w_m - reference[:, None], af) + ac
        if not bool(torch.isfinite(force).all() & torch.isfinite(moment).all()):
            raise FloatingPointError("Nonfinite 3D aerodynamic loads")
        return SectionLoads3D(force, moment, gamma, alpha, sep, residual, condition, induced)


def distributed_body_drag(velocity, omega, reference, rho=1.225, config=BodyDrag3DConfig()):
    """Three FLU stations; CdA engineering priors (m²), not measured damping.

    Approximate 0.30 m long, 0.06 m wide/deep central body envelope. Axial
    frontal area is shared across stations; lateral projected area distributed
    by length. Component-wise quadratic drag guarantees F dot u <= 0.
    """
    points = velocity.new_tensor(config.points_m)
    cda = velocity.new_tensor(config.cda_m2)
    arm = points[None] - reference[:, None]
    u = velocity[:, None] + torch.linalg.cross(omega[:, None].expand_as(arm), arm)
    f = -0.5 * rho * cda * u.abs() * u
    return f.sum(1), torch.linalg.cross(arm, f).sum(1)

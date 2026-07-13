"""DeLaurier (1993) strip-theory aerodynamic model for flapping wings."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from .qsm_wang2016 import WingGeometry

Tensor = torch.Tensor


@dataclass
class DeLaurierParams:
    """Parameter set for the DeLaurier (1993) model."""

    alpha0_rad: float = math.radians(0.5)
    eta_s: float = 0.98
    cd_cf: float = 1.98
    alpha_stall_min_rad: float = -1e9
    alpha_stall_max_rad: float = math.radians(13.0)
    xi: float = 0.0
    c_mac: float = 0.025
    nu: float = 1.5e-5
    cd_f: float | None = None
    stall_smoothing_width_rad: float = 0.0


@dataclass(frozen=True)
class DeLaurierStripLoads:
    """Raw DeLaurier strip loads in the co-rotating Wang frame.

    The Wang axes are right-handed: ``x`` is spanwise and coincides with the
    pitching axis, ``y`` is surface-normal, and ``z`` points chordwise toward
    the leading edge. Positive ``theta`` and ``thetad`` follow the right-hand
    rotation about Wang ``+x``. A positive ``dM_ac`` or ``dM_a`` is likewise a
    positive axial couple about Wang ``+x``. Every load tensor has shape
    ``(batch, num_strips)`` and already includes the strip width ``strip_width``.

    ``d_hat`` is the normalized leading-edge-to-pitch-axis offset. Therefore a
    force applied at a chord fraction ``x_over_c`` from the leading edge is at
    ``(d_hat - x_over_c) * chord`` along the positive Wang chord axis from the
    pitching axis.

    ``resultant_normal_force`` and ``resultant_chordwise_force`` retain the
    legacy separation-selected resultant for the compatibility wrapper only.
    The strip-wrench integrator deliberately uses the attached-flow raw
    components and never distributes separated-flow loads.
    """

    dN_c: Tensor
    dN_a: Tensor
    dT_s: Tensor
    dD_camber: Tensor
    dD_f: Tensor
    dM_ac: Tensor
    dM_a: Tensor
    span: Tensor
    chord: Tensor
    strip_width: Tensor
    d_hat: Tensor
    resultant_normal_force: Tensor
    resultant_chordwise_force: Tensor
    power_input: Tensor
    separation_weight: Tensor
    alpha: Tensor | None = None
    alpha_prime: Tensor | None = None
    alpha_le: Tensor | None = None
    alpha_stall: Tensor | None = None
    reduced_frequency: Tensor | None = None

    @property
    def normal_force_total(self) -> Tensor:
        """Attached-flow normal force per strip, shape ``(B, N)`` in N."""

        return self.dN_c + self.dN_a

    @property
    def chordwise_force_total(self) -> Tensor:
        """Attached-flow chordwise force per strip, shape ``(B, N)`` in N."""

        return self.dT_s - self.dD_camber - self.dD_f

    @property
    def free_pitching_moment_total(self) -> Tensor:
        """Free pitching couple per strip, shape ``(B, N)`` in N m."""

        return self.dM_ac + self.dM_a


@dataclass(frozen=True)
class DeLaurierStripWrench:
    """Strip-integrated DeLaurier wrench in the Wang frame.

    ``force_wang`` is the resultant force of one wing, shape ``(B, 3)`` in N.
    ``moment_wang_about_wing_origin`` is about the wing-root pitching-axis
    origin, shape ``(B, 3)`` in N m. Component moments use that same reference
    and are exposed for diagnostics and later strip-level corrections.
    """

    force_wang: Tensor
    moment_wang_about_wing_origin: Tensor
    force_normal_wang: Tensor
    force_chordwise_wang: Tensor
    moment_from_dN_c_wang: Tensor
    moment_from_dN_a_wang: Tensor
    moment_from_dT_s_wang: Tensor
    moment_from_dD_camber_wang: Tensor
    moment_from_dD_f_wang: Tensor
    moment_from_dM_ac_wang: Tensor
    moment_from_dM_a_wang: Tensor

    @property
    def moment_from_chordwise_wang(self) -> Tensor:
        """Combined chordwise-force moment, shape ``(B, 3)`` in N m."""

        return self.moment_from_dT_s_wang + self.moment_from_dD_camber_wang + self.moment_from_dD_f_wang

    @property
    def component_moments_wang(self) -> dict[str, Tensor]:
        """Named component moments about the wing-root pitching-axis origin."""

        return {
            "dN_c": self.moment_from_dN_c_wang,
            "dN_a": self.moment_from_dN_a_wang,
            "dT_s": self.moment_from_dT_s_wang,
            "dD_camber": self.moment_from_dD_camber_wang,
            "dD_f": self.moment_from_dD_f_wang,
            "dM_ac": self.moment_from_dM_ac_wang,
            "dM_a": self.moment_from_dM_a_wang,
        }


def _as_tensor(x: Any, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _broadcast_strips(x: Tensor, *, B: int, N: int) -> Tensor:
    if x.ndim == 2 and x.shape[-1] == N:
        return x
    if x.ndim == 1 and x.shape[0] == N:
        return x.view(1, N).expand(B, N)
    if x.ndim == 0:
        return x.view(1, 1).expand(B, N)
    raise ValueError("Invalid strip tensor shape; expected (B,N), (N,), or scalar.")


def _f_g_prime(k: Tensor, AR: float) -> tuple[Tensor, Tensor]:
    c1 = (0.5 * AR) / (2.32 + AR)
    c2 = 0.181 + (0.772 / AR)
    k2 = k * k
    denom = k2 + (c2 * c2)
    f_prime = 1.0 - (c1 * k2) / denom
    g_prime = -(c1 * c2 * k) / denom
    return f_prime, g_prime


def _cd_f_from_re(re: Tensor) -> Tensor:
    log_re = torch.log10(torch.clamp(re, min=2.0))
    return 0.89 / torch.clamp(log_re, min=1e-3) ** 2.58


def _force_vector_wang(*, normal: Tensor | None = None, chordwise: Tensor | None = None) -> Tensor:
    """Build a Wang-frame strip-force vector, shape ``(B, N, 3)`` in N."""

    scalar = normal if normal is not None else chordwise
    if scalar is None:
        raise ValueError("At least one Wang force component must be provided.")
    zeros = torch.zeros_like(scalar)
    return torch.stack(
        (
            zeros,
            normal if normal is not None else zeros,
            chordwise if chordwise is not None else zeros,
        ),
        dim=-1,
    )


def _integrate_strip_vector(strip_vector_wang: Tensor) -> Tensor:
    """Sum a Wang-frame strip vector of shape ``(B, N, 3)`` over span."""

    return strip_vector_wang.sum(dim=1)


def integrate_delaurier_strip_wrench(
    strip_loads: DeLaurierStripLoads,
    *,
    include_aerodynamic_center_moment: bool = True,
    include_apparent_mass_moment: bool = True,
    chordwise_application_chord_fraction_from_le: float | None = None,
) -> DeLaurierStripWrench:
    """Integrate attached-flow DeLaurier strip loads about the wing-root origin.

    No body/world transform or vehicle-CG translation happens here. The returned
    force and moment remain in the Wang frame and use the wing-root pitching-axis
    point as their reference. ``dM_ac`` and ``dM_a`` are independent free
    couples along positive Wang ``x``; their power convention is
    ``P_moment = -(dM_ac + dM_a) * thetad`` as used by the source equations.

    Args:
        strip_loads: Raw attached-flow components and geometry, each scalar
            field shaped ``(B, N)``.
        include_aerodynamic_center_moment: Include the ``dM_ac`` free couple.
        include_apparent_mass_moment: Include the ``dM_a`` free couple.
        chordwise_application_chord_fraction_from_le: Optional common point for
            the three chordwise components, measured from LE. ``None`` uses
            the pitching axis. This argument exists for invariance tests; all
            choices on the chordwise force line produce the same moment.
    """

    # Shape: (B, N, 3). The wing-root pitching-axis origin is at (0, 0, 0).
    chordwise_offset = torch.zeros_like(strip_loads.chord)
    if chordwise_application_chord_fraction_from_le is not None:
        chordwise_offset = (strip_loads.d_hat - float(chordwise_application_chord_fraction_from_le)) * strip_loads.chord
    strip_chordwise_force_point_wang = torch.stack(
        (strip_loads.span, torch.zeros_like(strip_loads.span), chordwise_offset),
        dim=-1,
    )
    strip_normal_circulatory_point_wang = torch.stack(
        (
            strip_loads.span,
            torch.zeros_like(strip_loads.span),
            (strip_loads.d_hat - 0.25) * strip_loads.chord,
        ),
        dim=-1,
    )
    strip_normal_apparent_mass_point_wang = torch.stack(
        (
            strip_loads.span,
            torch.zeros_like(strip_loads.span),
            (strip_loads.d_hat - 0.50) * strip_loads.chord,
        ),
        dim=-1,
    )

    # Normal components act along Wang +y. Chordwise components act along Wang
    # +z (toward LE); drag signs are carried by their individual force vectors.
    force_dN_c_wang = _force_vector_wang(normal=strip_loads.dN_c)
    force_dN_a_wang = _force_vector_wang(normal=strip_loads.dN_a)
    force_dT_s_wang = _force_vector_wang(chordwise=strip_loads.dT_s)
    force_dD_camber_wang = _force_vector_wang(chordwise=-strip_loads.dD_camber)
    force_dD_f_wang = _force_vector_wang(chordwise=-strip_loads.dD_f)

    moment_from_dN_c_wang = _integrate_strip_vector(
        torch.linalg.cross(strip_normal_circulatory_point_wang, force_dN_c_wang)
    )
    moment_from_dN_a_wang = _integrate_strip_vector(
        torch.linalg.cross(strip_normal_apparent_mass_point_wang, force_dN_a_wang)
    )
    # Moving a chordwise force along its own line of action does not change this
    # cross product. It still contributes through its spanwise lever arm.
    moment_from_dT_s_wang = _integrate_strip_vector(
        torch.linalg.cross(strip_chordwise_force_point_wang, force_dT_s_wang)
    )
    moment_from_dD_camber_wang = _integrate_strip_vector(
        torch.linalg.cross(strip_chordwise_force_point_wang, force_dD_camber_wang)
    )
    moment_from_dD_f_wang = _integrate_strip_vector(
        torch.linalg.cross(strip_chordwise_force_point_wang, force_dD_f_wang)
    )

    # Free couples are axial vectors about Wang +x, not forces at an assumed
    # application point. Shape: (B, 3).
    moment_from_dM_ac_wang = torch.stack(
        (
            strip_loads.dM_ac.sum(dim=1),
            torch.zeros_like(strip_loads.dM_ac.sum(dim=1)),
            torch.zeros_like(strip_loads.dM_ac.sum(dim=1)),
        ),
        dim=-1,
    )
    moment_from_dM_a_wang = torch.stack(
        (
            strip_loads.dM_a.sum(dim=1),
            torch.zeros_like(strip_loads.dM_a.sum(dim=1)),
            torch.zeros_like(strip_loads.dM_a.sum(dim=1)),
        ),
        dim=-1,
    )
    if not include_aerodynamic_center_moment:
        moment_from_dM_ac_wang = torch.zeros_like(moment_from_dM_ac_wang)
    if not include_apparent_mass_moment:
        moment_from_dM_a_wang = torch.zeros_like(moment_from_dM_a_wang)

    force_normal_wang = _integrate_strip_vector(force_dN_c_wang + force_dN_a_wang)
    force_chordwise_wang = _integrate_strip_vector(
        force_dT_s_wang + force_dD_camber_wang + force_dD_f_wang
    )
    force_wang = force_normal_wang + force_chordwise_wang
    moment_wang_about_wing_origin = (
        moment_from_dN_c_wang
        + moment_from_dN_a_wang
        + moment_from_dT_s_wang
        + moment_from_dD_camber_wang
        + moment_from_dD_f_wang
        + moment_from_dM_ac_wang
        + moment_from_dM_a_wang
    )
    return DeLaurierStripWrench(
        force_wang=force_wang,
        moment_wang_about_wing_origin=moment_wang_about_wing_origin,
        force_normal_wang=force_normal_wang,
        force_chordwise_wang=force_chordwise_wang,
        moment_from_dN_c_wang=moment_from_dN_c_wang,
        moment_from_dN_a_wang=moment_from_dN_a_wang,
        moment_from_dT_s_wang=moment_from_dT_s_wang,
        moment_from_dD_camber_wang=moment_from_dD_camber_wang,
        moment_from_dD_f_wang=moment_from_dD_f_wang,
        moment_from_dM_ac_wang=moment_from_dM_ac_wang,
        moment_from_dM_a_wang=moment_from_dM_a_wang,
    )


def transform_wang_wrench_to_link(
    force_wang: Tensor,
    moment_wang: Tensor,
    wang_to_link: Tensor,
) -> tuple[Tensor, Tensor]:
    """Transform a Wang wrench to a wing-link frame with polar/axial parity.

    ``wang_to_link`` maps polar vectors and has shape ``(B, 3, 3)``. The
    right-wing map is a reflection, so force uses ``A F`` whereas an axial
    moment uses ``det(A) A M``. This is equivalent to explicitly transforming
    both vectors in every ``r x F`` term before taking their cross product.
    """

    force_link = torch.bmm(wang_to_link, force_wang.unsqueeze(-1)).squeeze(-1)
    polar_moment_link = torch.bmm(wang_to_link, moment_wang.unsqueeze(-1)).squeeze(-1)
    determinant = torch.linalg.det(wang_to_link).unsqueeze(-1)
    return force_link, determinant * polar_moment_link


def translate_wrench_moment(
    force: Tensor,
    moment_about_origin: Tensor,
    origin_position: Tensor,
    reference_position: Tensor,
) -> Tensor:
    """Translate a moment to another point in one proper Cartesian frame.

    All inputs have shape ``(B, 3)``. ``origin_position`` and
    ``reference_position`` are positions in metres in the same frame as
    ``force``; the result is the moment about ``reference_position`` in N m.
    """

    return moment_about_origin + torch.linalg.cross(origin_position - reference_position, force)


def compute_delaurier_strip_loads(
    h: Tensor,
    hdot: Tensor,
    hddot: Tensor,
    theta: Tensor,
    thetad: Tensor,
    thetadd: Tensor,
    wing_geom: Mapping[str, Any] | WingGeometry,
    rho: float,
    U: float,
    *,
    theta_a: float,
    theta_bar: float,
    omega_ref: float,
    params: DeLaurierParams,
    enable_separation: bool = True,
) -> DeLaurierStripLoads:
    """Compute raw DeLaurier (1993) loads without span integration.

    All raw components remain attached-flow components even when
    ``enable_separation`` is true. The separation-selected resultant is stored
    only for the legacy aggregate compatibility API.
    """
    if not isinstance(h, torch.Tensor):
        raise TypeError("h must be a torch tensor.")

    device, dtype = h.device, h.dtype
    geom = wing_geom if isinstance(wing_geom, WingGeometry) else WingGeometry.from_input(wing_geom, device=device, dtype=dtype)
    N = int(geom.x_mid.numel())
    B = int(h.shape[0])

    y = geom.x_mid.view(1, N).expand(B, N)
    dx = geom.dx.view(1, N).expand(B, N)
    c = geom.c.view(1, N).expand(B, N)
    d_hat = geom.d_hat.view(1, N).expand(B, N)

    h = _broadcast_strips(h, B=B, N=N)
    hdot = _broadcast_strips(hdot, B=B, N=N)
    hddot = _broadcast_strips(hddot, B=B, N=N)
    theta = _broadcast_strips(theta, B=B, N=N)
    thetad = _broadcast_strips(thetad, B=B, N=N)
    thetadd = _broadcast_strips(thetadd, B=B, N=N)

    rho_t = _as_tensor(rho, device=device, dtype=dtype)
    U_t = _as_tensor(U, device=device, dtype=dtype)
    U_safe = torch.clamp(U_t, min=1e-6)

    theta_a_t = _as_tensor(theta_a, device=device, dtype=dtype)
    theta_bar_t = _as_tensor(theta_bar, device=device, dtype=dtype)
    alpha0 = _as_tensor(params.alpha0_rad, device=device, dtype=dtype)

    omega_t = _as_tensor(omega_ref, device=device, dtype=dtype)
    omega_t = _broadcast_strips(omega_t, B=B, N=N)
    k = (c * omega_t) / (2.0 * U_safe)

    f_prime, g_prime = _f_g_prime(k, float(geom.aspect_ratio))
    g_over_k = torch.where(k.abs() > 1e-8, g_prime / k, torch.zeros_like(k))

    theta_minus = theta - theta_a_t
    alpha = (hdot * torch.cos(theta_minus) + 0.75 * c * thetad + U_t * (theta - theta_bar_t)) / U_safe
    alpha_dot = (
        hddot * torch.cos(theta_minus)
        - hdot * torch.sin(theta_minus) * thetad
        + 0.75 * c * thetadd
        + U_t * thetad
    ) / U_safe

    w0_over_u = 2.0 * (alpha0 + theta_bar_t) / (2.0 + float(geom.aspect_ratio))
    alpha_prime = (float(geom.aspect_ratio) / (2.0 + float(geom.aspect_ratio))) * (
        f_prime * alpha + (c / (2.0 * U_safe)) * g_over_k * alpha_dot
    ) - w0_over_u

    Vx = U_t * torch.cos(theta) - hdot * torch.sin(theta_minus)
    Vy = U_t * (alpha_prime + theta_bar_t) - 0.5 * c * thetad
    V = torch.sqrt(Vx * Vx + Vy * Vy)

    Cn = 2.0 * math.pi * (alpha_prime + alpha0 + theta_bar_t)
    qn = 0.5 * rho_t * U_t * V
    dN_c = qn * Cn * c * dx

    v2_dot = U_t * alpha_dot - 0.25 * c * thetadd
    dN_a = (rho_t * math.pi * c * c / 4.0) * v2_dot * dx

    dN_att = dN_c + dN_a

    dD_camber = -2.0 * math.pi * alpha0 * (alpha_prime + theta_bar_t) * qn * c * dx
    dT_s = (
        params.eta_s
        * 2.0
        * math.pi
        * (alpha_prime + theta_bar_t - 0.25 * (c * thetad / U_safe)) ** 2
        * qn
        * c
        * dx
    )

    if params.cd_f is None:
        re = (U_safe * c) / max(params.nu, 1e-9)
        cd_f = _cd_f_from_re(re)
    else:
        cd_f = torch.zeros_like(c) + float(params.cd_f)
    dD_f = cd_f * (0.5 * rho_t * Vx * Vx) * c * dx
    dF_x_att = dT_s - dD_camber - dD_f

    alpha_stall = params.alpha_stall_max_rad + params.xi * torch.sqrt(
        torch.clamp(c * alpha_dot.abs() / (2.0 * U_safe), min=0.0)
    )
    alpha_le = alpha_prime + theta_bar_t - 0.75 * (c * thetad / U_safe)
    attached = (alpha_le >= params.alpha_stall_min_rad) & (alpha_le <= alpha_stall)

    Vn = hdot * torch.cos(theta_minus) + 0.5 * c * thetad + U_t * torch.sin(theta)
    Vhat = torch.sqrt(Vx * Vx + Vn * Vn)
    dN_c_sep = params.cd_cf * (0.5 * rho_t * Vhat * Vn) * c * dx
    dN_sep = dN_c_sep + 0.5 * dN_a

    smoothing_width = float(getattr(params, "stall_smoothing_width_rad", 0.0))
    if enable_separation and smoothing_width > 0.0:
        delta = torch.as_tensor(smoothing_width, device=device, dtype=dtype)
        w_hi = torch.sigmoid((alpha_le - alpha_stall) / delta)
        w_lo = torch.sigmoid((_as_tensor(params.alpha_stall_min_rad, device=device, dtype=dtype) - alpha_le) / delta)
        sep_weight = torch.clamp(w_lo + w_hi - w_lo * w_hi, min=0.0, max=1.0)
        att_weight = 1.0 - sep_weight
        dN = att_weight * dN_att + sep_weight * dN_sep
        dF_x = att_weight * dF_x_att
    elif enable_separation:
        sep_weight = torch.where(attached, torch.zeros_like(dN_att), torch.ones_like(dN_att))
        dN = torch.where(attached, dN_att, dN_sep)
        dF_x = torch.where(attached, dF_x_att, torch.zeros_like(dF_x_att))
    else:
        sep_weight = torch.zeros_like(dN_att)
        dN = dN_att
        dF_x = dF_x_att

    # Input power (Eq. 35/37) with C_mac approximation.
    dM_ac = params.c_mac * (0.5 * rho_t * U_t * V) * (c * c) * dx
    dM_a = -((rho_t * math.pi * c**3) * (thetad * U_t) / 16.0 + (rho_t * math.pi * c**4) * thetadd / 128.0) * dx
    dP_att = (
        dF_x * hdot * torch.sin(theta_minus)
        + dN * (hdot * torch.cos(theta_minus) + 0.25 * c * thetad)
        + dN_a * (0.25 * c * thetad)
        - dM_ac * thetad
        - dM_a * thetad
    )
    dP_sep = dN_sep * (hdot * torch.cos(theta_minus) + 0.5 * c * thetad)
    if enable_separation and smoothing_width > 0.0:
        dP_in = (1.0 - sep_weight) * dP_att + sep_weight * dP_sep
    else:
        dP_in = torch.where(attached, dP_att, dP_sep) if enable_separation else dP_att

    area = (c * dx)
    area_sum = torch.clamp(area.sum(dim=1), min=1e-12)
    sep_ratio = (sep_weight * area).sum(dim=1) / area_sum if enable_separation else torch.zeros_like(area_sum)
    return DeLaurierStripLoads(
        dN_c=dN_c,
        dN_a=dN_a,
        dT_s=dT_s,
        dD_camber=dD_camber,
        dD_f=dD_f,
        dM_ac=dM_ac,
        dM_a=dM_a,
        span=y,
        chord=c,
        strip_width=dx,
        d_hat=d_hat,
        resultant_normal_force=dN,
        resultant_chordwise_force=dF_x,
        power_input=dP_in,
        separation_weight=sep_weight,
        alpha=alpha,
        alpha_prime=alpha_prime,
        alpha_le=alpha_le,
        alpha_stall=alpha_stall,
        reduced_frequency=k,
    )


def compute_aero_wrench_delaurier1993(
    h: Tensor,
    hdot: Tensor,
    hddot: Tensor,
    theta: Tensor,
    thetad: Tensor,
    thetadd: Tensor,
    wing_geom: Mapping[str, Any] | WingGeometry,
    rho: float,
    U: float,
    *,
    theta_a: float,
    theta_bar: float,
    omega_ref: float,
    params: DeLaurierParams,
    enable_separation: bool = True,
    return_terms: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Compatibility wrapper returning the legacy span-integrated resultant.

    This API retains the historical separation-selected force and a zero
    intrinsic moment. New callers that need strip-level forces or moments must
    use :func:`compute_delaurier_strip_loads` and
    :func:`integrate_delaurier_strip_wrench`.
    """

    strip_loads = compute_delaurier_strip_loads(
        h,
        hdot,
        hddot,
        theta,
        thetad,
        thetadd,
        wing_geom,
        rho,
        U,
        theta_a=theta_a,
        theta_bar=theta_bar,
        omega_ref=omega_ref,
        params=params,
        enable_separation=enable_separation,
    )
    force_normal = strip_loads.resultant_normal_force.sum(dim=1)
    force_chordwise = strip_loads.resultant_chordwise_force.sum(dim=1)
    force_wang = torch.stack((torch.zeros_like(force_normal), force_normal, force_chordwise), dim=-1)
    zero_moment_wang = torch.zeros_like(force_wang)
    power_input = strip_loads.power_input.sum(dim=1)
    area = strip_loads.chord * strip_loads.strip_width
    area_sum = torch.clamp(area.sum(dim=1), min=1e-12)
    separation_ratio = (strip_loads.separation_weight * area).sum(dim=1) / area_sum

    if not return_terms:
        return force_wang, zero_moment_wang, power_input, separation_ratio

    def _area_weighted_mean(value: Tensor) -> Tensor:
        return (value * area).sum(dim=1) / area_sum

    assert strip_loads.alpha_prime is not None
    assert strip_loads.alpha_le is not None
    assert strip_loads.alpha_stall is not None
    assert strip_loads.reduced_frequency is not None
    smoothing_width = float(getattr(params, "stall_smoothing_width_rad", 0.0))
    if smoothing_width > 0.0:
        lower_transition = (
            strip_loads.alpha_le - _as_tensor(params.alpha_stall_min_rad, device=area.device, dtype=area.dtype)
        ).abs() < 3.0 * smoothing_width
        upper_transition = (strip_loads.alpha_le - strip_loads.alpha_stall).abs() < 3.0 * smoothing_width
        transition_mask = lower_transition | upper_transition
    else:
        transition_mask = torch.zeros_like(area, dtype=torch.bool)

    terms = {
        "N_c": strip_loads.dN_c.sum(dim=1),
        "N_a": strip_loads.dN_a.sum(dim=1),
        "Fx_suction": strip_loads.dT_s.sum(dim=1),
        "Fx_camber": (-strip_loads.dD_camber).sum(dim=1),
        "Fx_friction": (-strip_loads.dD_f).sum(dim=1),
        "Fx_total_att": strip_loads.chordwise_force_total.sum(dim=1),
        "Fx_total": strip_loads.resultant_chordwise_force.sum(dim=1),
        "k_mean": _area_weighted_mean(strip_loads.reduced_frequency),
        "alpha_prime_mean": _area_weighted_mean(strip_loads.alpha_prime),
        "alpha_le_mean": _area_weighted_mean(strip_loads.alpha_le),
        "alpha_stall_mean": _area_weighted_mean(strip_loads.alpha_stall),
        "sep_weight_mean": _area_weighted_mean(strip_loads.separation_weight),
        "sep_weight_mid_area_ratio": (
            torch.where(
                (strip_loads.separation_weight > 0.1) & (strip_loads.separation_weight < 0.9),
                area,
                torch.zeros_like(area),
            ).sum(dim=1)
            / area_sum
        ),
        "stall_transition_3delta_area_ratio": (
            torch.where(transition_mask, area, torch.zeros_like(area)).sum(dim=1) / area_sum
        ),
        "alpha_tip": strip_loads.alpha[:, -1] if strip_loads.alpha is not None else torch.zeros_like(power_input),
        "alpha_prime_tip": strip_loads.alpha_prime[:, -1],
        "alpha_le_tip": strip_loads.alpha_le[:, -1],
    }
    return force_wang, zero_moment_wang, power_input, separation_ratio, terms

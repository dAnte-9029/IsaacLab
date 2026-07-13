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
    """Compute DeLaurier (1993) strip-theory loads.

    Returns (F_c, tau_c, power_in, sep_ratio). F_c is in the co-rotating frame
    with x along span, y normal, z chordwise (toward LE as in Wang2016 axes).
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

    F_y = dN.sum(dim=1)
    F_z = dF_x.sum(dim=1)
    F_c = torch.stack((torch.zeros_like(F_y), F_y, F_z), dim=-1)
    tau_c = torch.zeros_like(F_c)

    area = (c * dx)
    area_sum = torch.clamp(area.sum(dim=1), min=1e-12)
    sep_ratio = (sep_weight * area).sum(dim=1) / area_sum if enable_separation else torch.zeros_like(area_sum)
    power_in = dP_in.sum(dim=1)

    if return_terms:
        def _wmean(x: Tensor) -> Tensor:
            return (x * area).sum(dim=1) / area_sum

        if smoothing_width > 0.0:
            lower_transition = (
                alpha_le - _as_tensor(params.alpha_stall_min_rad, device=device, dtype=dtype)
            ).abs() < 3.0 * smoothing_width
            upper_transition = (alpha_le - alpha_stall).abs() < 3.0 * smoothing_width
            transition_mask = lower_transition | upper_transition
        else:
            transition_mask = torch.zeros_like(area, dtype=torch.bool)

        terms = {
            "N_c": dN_c.sum(dim=1),
            "N_a": dN_a.sum(dim=1),
            "Fx_suction": dT_s.sum(dim=1),
            "Fx_camber": (-dD_camber).sum(dim=1),
            "Fx_friction": (-dD_f).sum(dim=1),
            "Fx_total_att": dF_x_att.sum(dim=1),
            "Fx_total": dF_x.sum(dim=1),
            "k_mean": _wmean(k),
            "alpha_prime_mean": _wmean(alpha_prime),
            "alpha_le_mean": _wmean(alpha_le),
            "alpha_stall_mean": _wmean(alpha_stall),
            "sep_weight_mean": _wmean(sep_weight),
            "sep_weight_mid_area_ratio": (
                torch.where((sep_weight > 0.1) & (sep_weight < 0.9), area, torch.zeros_like(area)).sum(dim=1)
            )
            / area_sum,
            "stall_transition_3delta_area_ratio": (
                torch.where(transition_mask, area, torch.zeros_like(area)).sum(dim=1)
            )
            / area_sum,
            "alpha_tip": alpha[:, -1],
            "alpha_prime_tip": alpha_prime[:, -1],
            "alpha_le_tip": alpha_le[:, -1],
        }
        return F_c, tau_c, power_in, sep_ratio, terms

    return F_c, tau_c, power_in, sep_ratio

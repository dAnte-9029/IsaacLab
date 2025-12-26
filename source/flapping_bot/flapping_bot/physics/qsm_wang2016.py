"""Quasi-steady aerodynamic model for flapping wings (Wang et al., JFM 2016).

This module implements the analytical quasi-steady model described in:

    Q. Wang, J. F. L. Goosen, F. van Keulen,
    "A predictive quasi-steady model of aerodynamic loads on flapping wings",
    Journal of Fluid Mechanics, 2016.

The model decomposes the aerodynamic loading into four terms:
translation-induced, rotation-induced, coupling, and added-mass loads.

All forces/torques are returned in the co-rotating frame (x_c, y_c, z_c) as defined
in the paper (§2.1). The x_c-axis coincides with the pitching axis (spanwise),
z_c lies in the wing plane (chordwise), and y_c is normal to the wing surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import math

import torch

Tensor = torch.Tensor


def _as_tensor(x: Any, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _sgn(x: Tensor) -> Tensor:
    # Matches the usual signum: {-1, 0, +1}
    return torch.sign(x)


def _safe_acos(x: Tensor) -> Tensor:
    return torch.acos(torch.clamp(x, -1.0, 1.0))


def _safe_div(numer: Tensor, denom: Tensor, eps: float = 1e-8) -> Tensor:
    return numer / torch.clamp(denom, min=eps)


def _quat_apply_wxyz(q_wxyz: Tensor, v: Tensor) -> Tensor:
    """Rotate vectors by quaternion (w, x, y, z)."""
    # q * (0,v) * q_conj, vectorized.
    w, x, y, z = q_wxyz.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w*t + cross(q_xyz, t)
    vpx = vx + w * tx + (y * tz - z * ty)
    vpy = vy + w * ty + (z * tx - x * tz)
    vpz = vz + w * tz + (x * ty - y * tx)
    return torch.stack((vpx, vpy, vpz), dim=-1)


def rotation_matrix_i_from_c(phi: Tensor, theta: Tensor, eta: Tensor) -> Tensor:
    """Return rotation matrix R_i_c = R_phi R_theta R_eta (paper eq. (2.1a-c))."""
    cph, sph = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    cet, set_ = torch.cos(eta), torch.sin(eta)

    # R_phi (about z_i)
    rphi = torch.stack(
        (
            torch.stack((cph, -sph, torch.zeros_like(cph)), dim=-1),
            torch.stack((sph, cph, torch.zeros_like(cph)), dim=-1),
            torch.stack((torch.zeros_like(cph), torch.zeros_like(cph), torch.ones_like(cph)), dim=-1),
        ),
        dim=-2,
    )
    # R_theta (about y_theta)
    rth = torch.stack(
        (
            torch.stack((cth, torch.zeros_like(cth), sth), dim=-1),
            torch.stack((torch.zeros_like(cth), torch.ones_like(cth), torch.zeros_like(cth)), dim=-1),
            torch.stack((-sth, torch.zeros_like(cth), cth), dim=-1),
        ),
        dim=-2,
    )
    # R_eta (about x_eta)
    ret = torch.stack(
        (
            torch.stack((torch.ones_like(cet), torch.zeros_like(cet), torch.zeros_like(cet)), dim=-1),
            torch.stack((torch.zeros_like(cet), cet, -set_), dim=-1),
            torch.stack((torch.zeros_like(cet), set_, cet), dim=-1),
        ),
        dim=-2,
    )
    return rphi @ rth @ ret


def eta_shape_piecewise(
    x_c: Tensor,
    *,
    x0: float,
    x1: float,
    kind: str = "smoothstep",
) -> Tensor:
    """Build a spanwise twist shape g(x) in [0,1] for distributed twist models.

    This is an *extension* for flexible/twisting wings. Wang2016 assumes a rigid wing
    with a single pitching angle η(t). Here we allow η(x,t) = g(x) * η_tip(t) as a
    practical approximation when only the outer wing twists significantly.

    Args:
        x_c: Spanwise coordinate samples (e.g., `WingGeometry.x_mid`), shape (N,).
        x0: Start of twist region (g=0 for x<=x0).
        x1: End of twist region (g=1 for x>=x1). Must satisfy x1 > x0.
        kind: "linear" or "smoothstep" (C1 continuous).
    """
    if x1 <= x0:
        raise ValueError("eta_shape_piecewise requires x1 > x0.")
    t = (x_c - x0) / (x1 - x0)
    t = torch.clamp(t, 0.0, 1.0)
    if kind == "linear":
        return t
    if kind == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    raise ValueError(f"Unknown kind: {kind}")


def cp_location_rotation(d_hat: Tensor) -> Tensor:
    """Chord-normalized CP location for pure rotation (paper eq. (2.19)).

    Args:
        d_hat: Pitch-axis offset d̂ from LE, normalized by chord (0 <= d̂ < 0.5 in the paper).

    Returns:
        d̂_cp^rot (normalized by chord, measured from LE).
    """
    # d̂_cp^rot = - [3(d̂-1)^4 + d̂^4] / [4(d̂-1)^3 + d̂^3] + d̂
    num = 3.0 * (d_hat - 1.0) ** 4 + d_hat**4
    den = 4.0 * (d_hat - 1.0) ** 3 + d_hat**3
    return -_safe_div(num, den) + d_hat


@dataclass(frozen=True)
class WingGeometry:
    """Spanwise-discretized wing geometry for Wang2016 QSM.

    Attributes:
        R: Wing span/half-span (integration limit in the paper).
        x_mid: Midpoint x_c positions along span, shape (N,).
        dx: Strip widths, shape (N,).
        c: Chord length c(x_c) at midpoints, shape (N,).
        d_hat: Normalized LE-to-pitch-axis offset d̂(x_c) at midpoints, shape (N,).
        c_bar: Mean chord length c̄ (= area/R).
        aspect_ratio: Effective aspect ratio used in eq. (2.8) and (2.18).
        gyradius_rhat2: r̂2 from Appendix A, eq. (A3).
        enable_*: Term toggles (default: True for 4 loads).
        wagner_t_star: Optional non-dimensional distance t* (eq. (2.26)); only used if include_wagner=True.

    Notes:
        - d̂ is defined in the paper as the chord-normalized distance from LE to pitching axis.
        - If you need an "effective" aspect ratio (e.g. validation §3.1 uses A_eff=2.83),
          provide it explicitly via `aspect_ratio` in `from_input()`.
    """

    R: float
    x_mid: Tensor
    dx: Tensor
    c: Tensor
    d_hat: Tensor
    c_bar: float
    aspect_ratio: float
    gyradius_rhat2: float

    enable_translation: bool = True
    enable_rotation: bool = True
    enable_coupling: bool = True
    enable_added_mass: bool = True

    wagner_t_star: float | None = None

    # Optional distributed twist shape g(x) in [0,1], used as η(x,t)=g(x)*η_tip(t)
    # when the caller supplies scalar η(t) but wants spanwise-varying pitch.
    eta_shape: Tensor | None = None

    # Precomputed integrals used by multiple terms (scalars on same device/dtype as x_mid).
    I_x2_c: Tensor | None = None
    I_x3_c: Tensor | None = None
    I_x2_c2: Tensor | None = None
    I_d_x2_c2: Tensor | None = None

    I_c2_x: Tensor | None = None
    I_d_c2_x: Tensor | None = None
    I_c2_x2: Tensor | None = None
    I_d_c2_x2: Tensor | None = None

    I_c3_x: Tensor | None = None
    I_d_c3_x: Tensor | None = None
    I_d2_c3_x: Tensor | None = None

    I_rot_F: Tensor | None = None
    I_rot_tau_x: Tensor | None = None
    I_rot_tau_z: Tensor | None = None

    def subspan(
        self,
        x_min: float,
        x_max: float,
        *,
        recompute_aspect_ratio: bool = False,
    ) -> "WingGeometry":
        """Return a geometry view restricted to x_c in [x_min, x_max].

        This is useful for:
          - computing aerodynamic loads only on an outboard segment (e.g., 0.42-0.65 m),
          - driving a virtual passive twist DOF from the outboard aerodynamic moment.

        Notes:
            - x_c remains the *absolute* spanwise coordinate used in Wang2016 integrals.
            - By default we keep `aspect_ratio` unchanged. Set `recompute_aspect_ratio=True`
              to recompute it from the subspan chord average.
        """
        if x_max <= x_min:
            raise ValueError("subspan requires x_max > x_min.")
        mask = (self.x_mid >= float(x_min)) & (self.x_mid <= float(x_max))
        if not bool(mask.any().item()):
            raise ValueError("subspan selection is empty; check x_min/x_max against geometry span.")

        x_mid = self.x_mid[mask]
        dx = self.dx[mask]
        c = self.c[mask]
        d_hat = self.d_hat[mask]
        eta_shape = self.eta_shape[mask] if self.eta_shape is not None else None

        # Keep R unchanged by default; it's not used directly in integration, but keep for reference.
        R = self.R

        c_bar = self.c_bar
        aspect_ratio = self.aspect_ratio
        gyradius_rhat2 = self.gyradius_rhat2
        if recompute_aspect_ratio:
            area = torch.sum(c * dx)
            span = float((x_mid.max() - x_mid.min()).item())
            if span > 0:
                c_bar = float((area / span).item())
                aspect_ratio = float((span / c_bar) if c_bar > 0 else aspect_ratio)
            # gyradius for subspan is not used by core equations; keep original to avoid ambiguity.
            gyradius_rhat2 = self.gyradius_rhat2

        out = WingGeometry(
            R=R,
            x_mid=x_mid,
            dx=dx,
            c=c,
            d_hat=d_hat,
            c_bar=c_bar,
            aspect_ratio=aspect_ratio,
            gyradius_rhat2=gyradius_rhat2,
            enable_translation=self.enable_translation,
            enable_rotation=self.enable_rotation,
            enable_coupling=self.enable_coupling,
            enable_added_mass=self.enable_added_mass,
            wagner_t_star=self.wagner_t_star,
            eta_shape=eta_shape,
        )
        return out._with_precomputed_integrals()

    @staticmethod
    def _sample_span_distribution(
        values: float | Sequence[float] | Tensor | Callable[[Tensor], Tensor],
        x_mid: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> Tensor:
        if callable(values):
            out = values(x_mid)
            return _as_tensor(out, device=device, dtype=dtype)
        if isinstance(values, (float, int)):
            return torch.full_like(x_mid, float(values), device=device, dtype=dtype)
        t = _as_tensor(values, device=device, dtype=dtype).flatten()
        if t.numel() == 1:
            return torch.full_like(x_mid, float(t.item()), device=device, dtype=dtype)
        if t.numel() == x_mid.numel():
            return t
        raise ValueError(f"`{name}` must be scalar, callable, or length-N array; got length {t.numel()}.")

    @classmethod
    def from_input(
        cls,
        wing_geom: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "WingGeometry":
        device = torch.device(device)
        if "R" not in wing_geom:
            raise KeyError("`wing_geom` must contain `R`.")
        R = float(wing_geom["R"])
        if R <= 0:
            raise ValueError("`wing_geom['R']` must be positive.")
        N = int(wing_geom.get("N", 20))
        if N <= 0:
            raise ValueError("`wing_geom['N']` must be positive.")

        # Midpoint discretization over [0, R].
        x_edges = torch.linspace(0.0, R, N + 1, device=device, dtype=dtype)
        x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
        dx = x_edges[1:] - x_edges[:-1]

        chord_src = wing_geom.get("chord_func", wing_geom.get("c", None))
        if chord_src is None:
            raise KeyError("`wing_geom` must contain `chord_func` or `c`.")
        c = cls._sample_span_distribution(
            chord_src,
            x_mid,
            device=device,
            dtype=dtype,
            name="chord_func/c",
        )
        if torch.any(c <= 0):
            raise ValueError("Chord length must be positive over the span.")

        dhat_src = wing_geom.get("dhat_func", wing_geom.get("dhat", None))
        if dhat_src is None:
            raise KeyError("`wing_geom` must contain `dhat_func` or `dhat`.")
        d_hat = cls._sample_span_distribution(
            dhat_src,
            x_mid,
            device=device,
            dtype=dtype,
            name="dhat_func/dhat",
        )

        eta_shape_src = wing_geom.get("eta_shape_func", wing_geom.get("eta_shape", None))
        eta_shape = None
        if eta_shape_src is not None:
            eta_shape = cls._sample_span_distribution(
                eta_shape_src,
                x_mid,
                device=device,
                dtype=dtype,
                name="eta_shape_func/eta_shape",
            )
            eta_shape = torch.clamp(eta_shape, 0.0, 1.0)

        # Default toggles
        enable_translation = bool(wing_geom.get("enable_translation", True))
        enable_rotation = bool(wing_geom.get("enable_rotation", True))
        enable_coupling = bool(wing_geom.get("enable_coupling", True))
        enable_added_mass = bool(wing_geom.get("enable_added_mass", True))

        # Mean chord and area.
        area = torch.sum(c * dx)
        c_bar = float((area / R).item())
        aspect_ratio_geom = float((R / c_bar) if c_bar > 0 else 0.0)
        aspect_ratio = float(wing_geom.get("aspect_ratio", aspect_ratio_geom))

        # Appendix A, eq. (A3): r̂2 = sqrt( 1/(S R^2) * ∫ x^2 c dx )
        gyradius_rhat2 = wing_geom.get("gyradius_rhat2", None)
        if gyradius_rhat2 is None:
            I_x2_c = torch.sum((x_mid**2) * c * dx)
            gyradius_rhat2 = float(torch.sqrt(_safe_div(I_x2_c, area * (R**2))).item())
        else:
            gyradius_rhat2 = float(gyradius_rhat2)

        out = cls(
            R=R,
            x_mid=x_mid,
            dx=dx,
            c=c,
            d_hat=d_hat,
            c_bar=c_bar,
            aspect_ratio=aspect_ratio,
            gyradius_rhat2=gyradius_rhat2,
            enable_translation=enable_translation,
            enable_rotation=enable_rotation,
            enable_coupling=enable_coupling,
            enable_added_mass=enable_added_mass,
            wagner_t_star=wing_geom.get("wagner_t_star", None),
            eta_shape=eta_shape,
        )
        return out._with_precomputed_integrals()

    def _with_precomputed_integrals(self) -> "WingGeometry":
        # Precompute common spanwise integrals using midpoint rule.
        x = self.x_mid
        dx = self.dx
        c = self.c
        d = self.d_hat

        I_x2_c = torch.sum((x**2) * c * dx)
        I_x3_c = torch.sum((x**3) * c * dx)
        I_x2_c2 = torch.sum((x**2) * (c**2) * dx)
        I_d_x2_c2 = torch.sum(d * (x**2) * (c**2) * dx)

        I_c2_x = torch.sum((c**2) * x * dx)
        I_d_c2_x = torch.sum(d * (c**2) * x * dx)
        I_c2_x2 = torch.sum((c**2) * (x**2) * dx)
        I_d_c2_x2 = torch.sum(d * (c**2) * (x**2) * dx)

        I_c3_x = torch.sum((c**3) * x * dx)
        I_d_c3_x = torch.sum(d * (c**3) * x * dx)
        I_d2_c3_x = torch.sum((d**2) * (c**3) * x * dx)

        # Rotation-induced chordwise integrals collapsed analytically (paper eq. (2.15)-(2.17)).
        poly3 = ((d - 1.0) ** 3 + d**3) / 3.0
        poly4 = ((d - 1.0) ** 4 + d**4) / 4.0
        I_rot_F = torch.sum((c**3) * poly3 * dx)
        I_rot_tau_z = torch.sum(x * (c**3) * poly3 * dx)
        I_rot_tau_x = torch.sum((c**4) * poly4 * dx)

        return WingGeometry(
            **{
                **self.__dict__,
                "I_x2_c": I_x2_c,
                "I_x3_c": I_x3_c,
                "I_x2_c2": I_x2_c2,
                "I_d_x2_c2": I_d_x2_c2,
                "I_c2_x": I_c2_x,
                "I_d_c2_x": I_d_c2_x,
                "I_c2_x2": I_c2_x2,
                "I_d_c2_x2": I_d_c2_x2,
                "I_c3_x": I_c3_x,
                "I_d_c3_x": I_d_c3_x,
                "I_d2_c3_x": I_d2_c3_x,
                "I_rot_F": I_rot_F,
                "I_rot_tau_x": I_rot_tau_x,
                "I_rot_tau_z": I_rot_tau_z,
            }
        )

    def wagner_multiplier(self) -> float:
        """Approximate Wagner function φ(t*) from Jones (1940), paper eq. (2.26)."""
        if self.wagner_t_star is None:
            # TODO(Wagner): compute/track t* (semi-chords travelled) from motion history.
            return 1.0
        t_star = float(self.wagner_t_star)
        return 1.0 - 0.165 * math.exp(-0.0455 * t_star) - 0.335 * math.exp(-0.3 * t_star)


def _wang2016_omega_alpha(
    phi: Tensor,
    theta: Tensor,
    eta: Tensor,
    phid: Tensor,
    thetad: Tensor,
    etad: Tensor,
    phidd: Tensor,
    thetadd: Tensor,
    etadd: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute ω_c and α_c (paper eq. (2.2)-(2.3))."""
    sin_th = torch.sin(theta)
    cos_th = torch.cos(theta)
    sin_et = torch.sin(eta)
    cos_et = torch.cos(eta)

    # ω_c = [ η̇ − φ̇ sinθ,
    #         θ̇ cosη + φ̇ cosθ sinη,
    #         φ̇ cosη cosθ − θ̇ sinη ]
    wx = etad - phid * sin_th
    wy = thetad * cos_et + phid * cos_th * sin_et
    wz = phid * cos_et * cos_th - thetad * sin_et
    omega = torch.stack((wx, wy, wz), dim=-1)

    # α_c = ω̇_c (paper eq. (2.3)).
    ax = etadd - phidd * sin_th - phid * thetad * cos_th
    ay = (
        phidd * cos_th * sin_et
        + thetadd * cos_et
        - etad * thetad * sin_et
        + phid * (etad * cos_et * cos_th - thetad * sin_et * sin_th)
    )
    az = (
        phidd * cos_et * cos_th
        - thetadd * sin_et
        - etad * thetad * cos_et
        - phid * (etad * cos_th * sin_et + thetad * cos_et * sin_th)
    )
    alpha = torch.stack((ax, ay, az), dim=-1)
    return omega, alpha


def _broadcast_strip(
    x: Tensor,
    *,
    N: int,
    eta_shape: Tensor | None,
) -> Tensor:
    """Broadcast scalar/batch angles to per-strip angles.

    - If x has last-dim == N, it's treated as already per-strip.
    - Else it is expanded to (..., N). If eta_shape is provided, applies it (η=η_tip*g).
    """
    if x.ndim >= 1 and x.shape[-1] == N:
        return x
    x_exp = x.unsqueeze(-1).expand(*x.shape, N)
    if eta_shape is None:
        return x_exp
    return x_exp * eta_shape


def compute_aero_wrench(
    phi: float | Tensor,
    theta: float | Tensor,
    eta: float | Tensor,
    phid: float | Tensor,
    thetad: float | Tensor,
    etad: float | Tensor,
    phidd: float | Tensor,
    thetadd: float | Tensor,
    etadd: float | Tensor,
    wing_geom: Mapping[str, Any] | WingGeometry,
    rho: float,
    include_wagner: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute total aerodynamic wrench in the co-rotating frame (x_c,y_c,z_c).

    Args:
        phi, theta, eta: Euler angles (sweep/yaw, heave/roll, pitch) in radians.
        phid, thetad, etad: First derivatives (rad/s).
        phidd, thetadd, etadd: Second derivatives (rad/s^2).
        wing_geom: `WingGeometry` or a mapping used to construct it (see `WingGeometry.from_input`).
        rho: Fluid density ρ_f (kg/m^3).
        include_wagner: If True, multiply circulatory loads by Wagner function φ(t*) (paper §2.2.5, eq. (2.26)).

    Returns:
        (F_c, tau_c) where each is a tensor of shape (..., 3) in the co-rotating frame.

    Notes:
        - Wang2016 assumes a rigid wing with a single pitching angle η(t).
        - This implementation also supports an optional distributed twist approximation:
          if `WingGeometry.eta_shape` is provided and η/η̇/η̈ are passed as scalars,
          then η(x,t) = g(x)*η_tip(t) is used per spanwise strip.
    """
    # Decide device/dtype from input angles.
    base = phi if isinstance(phi, torch.Tensor) else theta if isinstance(theta, torch.Tensor) else eta
    if isinstance(base, torch.Tensor):
        device, dtype = base.device, base.dtype
    else:
        device, dtype = torch.device("cpu"), torch.float32

    phi = _as_tensor(phi, device=device, dtype=dtype)
    theta = _as_tensor(theta, device=device, dtype=dtype)
    eta = _as_tensor(eta, device=device, dtype=dtype)
    phid = _as_tensor(phid, device=device, dtype=dtype)
    thetad = _as_tensor(thetad, device=device, dtype=dtype)
    etad = _as_tensor(etad, device=device, dtype=dtype)
    phidd = _as_tensor(phidd, device=device, dtype=dtype)
    thetadd = _as_tensor(thetadd, device=device, dtype=dtype)
    etadd = _as_tensor(etadd, device=device, dtype=dtype)
    rho_t = _as_tensor(rho, device=device, dtype=dtype)

    geom = wing_geom if isinstance(wing_geom, WingGeometry) else WingGeometry.from_input(wing_geom, device=device, dtype=dtype)
    if geom.I_x2_c is None:
        geom = geom._with_precomputed_integrals()

    # Decide whether to compute per-strip (distributed twist) or rigid.
    N = int(geom.x_mid.numel())
    eta_shape = geom.eta_shape
    phi_s = phi
    theta_s = theta
    phid_s = phid
    thetad_s = thetad
    phidd_s = phidd
    thetadd_s = thetadd

    # Broadcast to strips. For η and derivatives, apply eta_shape if provided.
    eta_strip = _broadcast_strip(eta, N=N, eta_shape=eta_shape)
    etad_strip = _broadcast_strip(etad, N=N, eta_shape=eta_shape)
    etadd_strip = _broadcast_strip(etadd, N=N, eta_shape=eta_shape)

    # φ/θ are shared across span (expand only if needed for broadcasting with η(x)).
    phi_strip = phi_s.unsqueeze(-1).expand_as(eta_strip)
    theta_strip = theta_s.unsqueeze(-1).expand_as(eta_strip)
    phid_strip = phid_s.unsqueeze(-1).expand_as(eta_strip)
    thetad_strip = thetad_s.unsqueeze(-1).expand_as(eta_strip)
    phidd_strip = phidd_s.unsqueeze(-1).expand_as(eta_strip)
    thetadd_strip = thetadd_s.unsqueeze(-1).expand_as(eta_strip)

    omega, alpha = _wang2016_omega_alpha(
        phi_strip,
        theta_strip,
        eta_strip,
        phid_strip,
        thetad_strip,
        etad_strip,
        phidd_strip,
        thetadd_strip,
        etadd_strip,
    )
    wx, wy, wz = omega.unbind(-1)  # (...,N)
    ax, _, az = alpha.unbind(-1)  # (...,N)

    # AOA α̃ (paper eq. (2.7)). Note: uses |v_zc/v_c| => depends on |ω_yc| only.
    denom = torch.sqrt(wy**2 + wz**2)
    aoa_raw = _safe_acos(_safe_div(torch.abs(wy), denom))
    aoa = torch.where(denom > 1e-8, aoa_raw, torch.zeros_like(aoa_raw))

    # ------------------------------------------------------------------
    # Coefficients (paper eq. (2.6), (2.8)-(2.10), (2.12), (2.18))
    # ------------------------------------------------------------------
    AR = float(geom.aspect_ratio)
    A_lift_max = math.pi * AR / (2.0 + math.sqrt(AR * AR + 4.0))  # paper eq. (2.8)
    CD_rot = 2.0 * math.pi * AR / (2.0 + math.sqrt(AR * AR + 4.0))  # paper eq. (2.18)

    # paper eq. (2.6)
    CL_trans = A_lift_max * torch.sin(2.0 * aoa)
    sin_aoa = torch.sin(aoa)
    # paper eq. (2.10): CF_y = CL / cos(α).
    # Using CL=A*sin(2α)=2A*sinα*cosα => CF_y = 2A*sinα, which avoids the 0/0 indeterminate at α=π/2.
    CF_y_trans = 2.0 * _as_tensor(A_lift_max, device=device, dtype=dtype) * sin_aoa

    d_hat_cp_trans = _safe_div(aoa, _as_tensor(math.pi, device=device, dtype=dtype))  # paper eq. (2.12)

    # ------------------------------------------------------------------
    # Per-strip geometry tensors
    # ------------------------------------------------------------------
    x = geom.x_mid.view(*([1] * (aoa.ndim - 1)), N)  # broadcast to (...,N)
    dx = geom.dx.view(*([1] * (aoa.ndim - 1)), N)
    c = geom.c.view(*([1] * (aoa.ndim - 1)), N)
    d_hat = geom.d_hat.view(*([1] * (aoa.ndim - 1)), N)

    # ------------------------------------------------------------------
    # 1) Translation-induced load (paper eq. (2.11)-(2.14))
    # ------------------------------------------------------------------
    Fy_trans = torch.zeros_like(phi_strip[..., 0])
    tau_x_trans = torch.zeros_like(phi_strip[..., 0])
    tau_z_trans = torch.zeros_like(phi_strip[..., 0])
    if geom.enable_translation:
        sgn_wz = _sgn(wz)
        w2 = wy**2 + wz**2
        base_trans = -sgn_wz * 0.5 * rho_t * w2 * CF_y_trans
        dFy = base_trans * (x**2) * c * dx
        Fy_trans = torch.sum(dFy, dim=-1)

        tau_z_trans = torch.sum(base_trans * (x**3) * c * dx, dim=-1)

        # paper eq. (2.13): LE/TE swap when ω_yc > 0 (per strip for distributed twist).
        k = torch.where(wy > 0.0, 1.0 - d_hat_cp_trans, d_hat_cp_trans)
        tau_x_trans = torch.sum(base_trans * (k - d_hat) * (x**2) * (c**2) * dx, dim=-1)

    # ------------------------------------------------------------------
    # 2) Rotation-induced load (paper eq. (2.15)-(2.19))
    # ------------------------------------------------------------------
    Fy_rot = torch.zeros_like(Fy_trans)
    tau_x_rot = torch.zeros_like(Fy_trans)
    tau_z_rot = torch.zeros_like(Fy_trans)
    if geom.enable_rotation:
        base_rot = 0.5 * rho_t * wx * torch.abs(wx) * _as_tensor(CD_rot, device=device, dtype=dtype)
        poly3 = ((d_hat - 1.0) ** 3 + d_hat**3) / 3.0
        poly4 = ((d_hat - 1.0) ** 4 + d_hat**4) / 4.0
        Fy_rot = torch.sum(base_rot * (c**3) * poly3 * dx, dim=-1)
        tau_x_rot = torch.sum(-base_rot * (c**4) * poly4 * dx, dim=-1)
        tau_z_rot = torch.sum(base_rot * x * (c**3) * poly3 * dx, dim=-1)

    # ------------------------------------------------------------------
    # 3) Coupling load (paper eq. (2.21)-(2.23))
    # ------------------------------------------------------------------
    Fy_coup = torch.zeros_like(Fy_trans)
    tau_x_coup = torch.zeros_like(Fy_trans)
    tau_z_coup = torch.zeros_like(Fy_trans)
    if geom.enable_coupling:
        base_coup = _as_tensor(math.pi, device=device, dtype=dtype) * rho_t * wx * wy
        main = torch.where(wy > 0.0, d_hat - 0.25, 0.75 - d_hat)
        Fy_coup = torch.sum(base_coup * (main + 0.25) * (c**2) * x * dx, dim=-1)
        tau_z_coup = torch.sum(base_coup * (main + 0.25) * (c**2) * (x**2) * dx, dim=-1)

        coefA_le = (0.75 - d_hat) * (0.25 - d_hat)
        coefB_le = 0.25 * (0.75 - d_hat)
        coefA_te = (d_hat - 0.25) * (0.75 - d_hat)
        coefB_te = 0.25 * (0.25 - d_hat)
        coef_sum = torch.where(wy > 0.0, coefA_te + coefB_te, coefA_le + coefB_le)
        tau_x_coup = torch.sum(base_coup * coef_sum * (c**3) * x * dx, dim=-1)

    # ------------------------------------------------------------------
    # 4) Added-mass load (paper eq. (2.24)-(2.25))
    # ------------------------------------------------------------------
    Fy_am = torch.zeros_like(Fy_trans)
    tau_x_am = torch.zeros_like(Fy_trans)
    tau_z_am = torch.zeros_like(Fy_trans)
    if geom.enable_added_mass:
        # Local spanwise acceleration along y_c for points on pitching axis (paper eq. (2.5), y-component).
        # a_yc(x_c) = x_c * (α_zc + ω_xc ω_yc)
        a_y = x * (az + wx * wy)
        alpha_x = ax

        # Added-mass matrix M per unit span (paper eq. (2.24)).
        pi = _as_tensor(math.pi, device=device, dtype=dtype)
        factor = (pi / 4.0) * rho_t
        h = 0.5 - d_hat  # (1/2 - d̂0)

        m22 = factor * (c**2)
        m24 = factor * (c**3) * h
        m42 = m24
        m44 = factor * ((c**4) / 32.0 + (c**4) * (h**2))

        # Integrate over span (eq. (2.25)).
        # Force density along y_c:
        fy_strip = -(m22 * a_y + m24 * alpha_x)
        tx_strip = -(m42 * a_y + m44 * alpha_x)
        Fy_am = torch.sum(fy_strip * dx, dim=-1)
        tau_x_am = torch.sum(tx_strip * dx, dim=-1)
        # Torque about z_c from spanwise moment arm: τ_z = ∫ x_c dF_y
        tau_z_am = torch.sum((x * fy_strip) * dx, dim=-1)

    # ------------------------------------------------------------------
    # Wagner multiplier (paper §2.2.5, eq. (2.26)) applied to circulatory loads.
    # ------------------------------------------------------------------
    wagner = 1.0
    if include_wagner:
        wagner = geom.wagner_multiplier()
    wagner_t = _as_tensor(wagner, device=device, dtype=dtype)

    Fy_circ = (Fy_trans + Fy_rot + Fy_coup) * wagner_t
    tau_x_circ = (tau_x_trans + tau_x_rot + tau_x_coup) * wagner_t
    tau_z_circ = (tau_z_trans + tau_z_rot + tau_z_coup) * wagner_t

    Fy = Fy_circ + Fy_am
    tau_x = tau_x_circ + tau_x_am
    tau_z = tau_z_circ + tau_z_am

    F_c = torch.stack((torch.zeros_like(Fy), Fy, torch.zeros_like(Fy)), dim=-1)
    tau_c = torch.stack((tau_x, torch.zeros_like(tau_x), tau_z), dim=-1)
    return F_c, tau_c


def transform_wrench_c_to_world(
    F_c: Tensor,
    tau_c: Tensor,
    quat_w_c: Tensor,
) -> tuple[Tensor, Tensor]:
    """Transform a wrench from co-rotating frame to world frame.

    Args:
        F_c: Force in co-rotating frame, shape (..., 3).
        tau_c: Torque in co-rotating frame, shape (..., 3).
        quat_w_c: Quaternion (w,x,y,z) of co-rotating frame in world, shape (..., 4).

    Returns:
        (F_w, tau_w) in world frame.
    """
    F_w = _quat_apply_wxyz(quat_w_c, F_c)
    tau_w = _quat_apply_wxyz(quat_w_c, tau_c)
    return F_w, tau_w

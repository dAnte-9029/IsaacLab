"""Virtual passive twist DOF for flexible wings.

Wang2016 QSM assumes a rigid wing (single pitching angle η(t)). If your URDF model
is rigid but the physical wing twists passively (e.g., only the outboard membrane
section twists), you can introduce a *virtual* twist angle η_tip(t) and use a spanwise
shape g(x) to approximate:

    η(x,t) = g(x) * η_tip(t)

The structural dynamics of η_tip(t) is *not* specified by Wang2016. Therefore this
module provides switchable defaults and explicit TODOs for parameters that must be
identified from your hardware or from calibration experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class VirtualTwistCfg:
    """Configuration for a virtual torsional DOF η_tip(t).

    All parameters below are TODOs (not in Wang2016) and must be identified.

    Args:
        stiffness: torsional stiffness k (N·m/rad). If you use quasi-static mode,
            η_tip ≈ τ / k.
        damping: torsional damping c (N·m·s/rad).
        inertia: torsional inertia I (kg·m²) for dynamic mode.
        eta_limit: optional absolute clamp |η| <= eta_limit (rad).
        tau_limit: optional absolute clamp |τ| <= tau_limit (N·m) before integration.
        mode: "quasi_static" or "dynamic".
        sign: Direction convention. In quasi-static mode, the update uses
            η_tip = η0 + sign * τ_x / k. Use sign=-1 if the passive twist direction
            is flipped relative to your desired definition of positive η.
    """

    stiffness: float = 0.3  # TODO: identify from wing torsion test
    damping: float = 0.01  # TODO: identify from decay test
    inertia: float = 1e-4  # TODO: identify/estimate effective inertia
    eta_limit: float | None = None
    tau_limit: float | None = None
    mode: str = "dynamic"
    sign: float = 1.0


class VirtualTwistState:
    """State for η_tip(t) integrated in time."""

    def __init__(self, eta: Tensor, etad: Tensor):
        self.eta = eta
        self.etad = etad

    @classmethod
    def zeros(cls, batch_shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> "VirtualTwistState":
        # torch.zeros(()) requires passing the shape as a tuple, not splatting an empty tuple.
        eta = torch.zeros(batch_shape, device=device, dtype=dtype)
        etad = torch.zeros(batch_shape, device=device, dtype=dtype)
        return cls(eta=eta, etad=etad)

    def reset_(self, eta: float = 0.0, etad: float = 0.0):
        self.eta.fill_(float(eta))
        self.etad.fill_(float(etad))


def step_virtual_twist(
    state: VirtualTwistState,
    *,
    tau_x: Tensor,
    dt: float,
    cfg: VirtualTwistCfg,
    eta_rest: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Advance η_tip(t) by one step.

    The driving torque `tau_x` is typically the aerodynamic pitching moment about
    the pitching axis for the outboard segment (e.g., x in [0.42, 0.65] m).

    Returns:
        (eta, etad, etadd) after the step.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    eta = state.eta
    etad = state.etad
    tau = tau_x

    if cfg.tau_limit is not None:
        tau = torch.clamp(tau, -float(cfg.tau_limit), float(cfg.tau_limit))

    if cfg.mode == "quasi_static":
        # η = η0 + τ/k, with optional low-pass behavior via damping not modeled here.
        k = float(cfg.stiffness)
        if k <= 0:
            raise ValueError("stiffness must be positive in quasi_static mode.")
        eta_new = float(eta_rest) + float(cfg.sign) * tau / k
        etad_new = torch.zeros_like(etad)
        etadd_new = torch.zeros_like(etad)
    elif cfg.mode == "dynamic":
        # I η̈ + c η̇ + k (η-η0) = τ
        I = float(cfg.inertia)
        c = float(cfg.damping)
        k = float(cfg.stiffness)
        if I <= 0 or k <= 0:
            raise ValueError("inertia and stiffness must be positive in dynamic mode.")
        etadd = (tau - c * etad - k * (eta - float(eta_rest))) / I
        # Semi-implicit Euler for better stability: v_{n+1}=v_n+a*dt; x_{n+1}=x_n+v_{n+1}*dt
        etad_new = etad + etadd * float(dt)
        eta_new = eta + etad_new * float(dt)
        etadd_new = etadd
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    if cfg.eta_limit is not None:
        lim = float(cfg.eta_limit)
        eta_new = torch.clamp(eta_new, -lim, lim)
        # If clamped, prevent runaway velocities (simple projection).
        etad_new = torch.where((eta_new.abs() >= lim), torch.zeros_like(etad_new), etad_new)

    state.eta.copy_(eta_new)
    state.etad.copy_(etad_new)
    return state.eta, state.etad, etadd_new


def solve_quasi_static_eta_tip(
    eta_init: Tensor,
    *,
    tau_of_eta,
    cfg: VirtualTwistCfg,
    eta_rest: float = 0.0,
    max_iters: int = 3,
    relax: float = 1.0,
) -> Tensor:
    """Fixed-point solve for quasi-static equilibrium when τ depends on η.

    We solve:   k(η-η0) = sign * τ(η)   =>   η = η0 + sign*τ(η)/k

    This is useful because the Wang2016 aerodynamic moment typically depends on η.

    Args:
        eta_init: Initial guess for η (tensor).
        tau_of_eta: Callable tau_of_eta(eta)->tau_x (same shape).
        cfg: Must have cfg.mode == "quasi_static" and cfg.stiffness > 0.
        eta_rest: η0.
        max_iters: Small number of iterations (2-5 is typical).
        relax: Relaxation factor in (0,1]. Use <1 if it oscillates.
    """
    if cfg.mode != "quasi_static":
        raise ValueError("solve_quasi_static_eta_tip requires cfg.mode == 'quasi_static'.")
    k = float(cfg.stiffness)
    if k <= 0:
        raise ValueError("stiffness must be positive.")
    if max_iters <= 0:
        return eta_init
    if not (0.0 < relax <= 1.0):
        raise ValueError("relax must be in (0, 1].")

    eta = eta_init
    for _ in range(max_iters):
        tau = tau_of_eta(eta)
        if cfg.tau_limit is not None:
            tau = torch.clamp(tau, -float(cfg.tau_limit), float(cfg.tau_limit))
        eta_target = float(eta_rest) + float(cfg.sign) * tau / k
        eta = (1.0 - float(relax)) * eta + float(relax) * eta_target
        if cfg.eta_limit is not None:
            lim = float(cfg.eta_limit)
            eta = torch.clamp(eta, -lim, lim)
    return eta

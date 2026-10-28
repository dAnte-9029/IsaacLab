"""Batched analytic-Newton circulation closure; no fallback to linear loads."""

from __future__ import annotations

import torch

from flapping_bot.config.ullt import ULLTConfig
from .flat_plate_section import flat_plate_polar


class NonlinearConvergenceError(ValueError):
    """Explicit failed solve with per-environment velocity residuals."""

    def __init__(self, reason: str, residual: torch.Tensor, iterations: torch.Tensor):
        self.residual_mps = residual.detach().cpu().tolist()
        self.iterations = iterations.detach().cpu().tolist()
        super().__init__(f"Nonlinear ULLT {reason}; residual_mps={self.residual_mps}; iterations={self.iterations}")


def nonlinear_normalwash(u: torch.Tensor, n: torch.Tensor, cfg: ULLTConfig) -> dict[str, torch.Tensor]:
    """Map physical effective flow to equivalent attached-flow input, in m/s.

    u=-Veff.chord_forward, n=Veff.normal. ue=|Veff|*Cl(alpha)/Cla.
    Low-angle ue=n, due/du=0, due/dn=1 exactly, including induced velocities.
    """
    magnitude = torch.sqrt(u.square() + n.square()).clamp_min(cfg.minimum_speed_mps)
    alpha = torch.atan2(n, u)
    polar = flat_plate_polar(alpha, cfg.nonlinear_polar, cfg.lift_slope_per_rad, cfg.profile_drag_coefficient)
    low = polar.separation_weight == 0
    equivalent = torch.where(low, n, magnitude * polar.cl / cfg.lift_slope_per_rad)
    du = torch.where(
        low, torch.zeros_like(n), (u * polar.cl - n * polar.dcl_dalpha) / (cfg.lift_slope_per_rad * magnitude)
    )
    dn = torch.where(
        low, torch.ones_like(n), (n * polar.cl + u * polar.dcl_dalpha) / (cfg.lift_slope_per_rad * magnitude)
    )
    return dict(
        equivalent=equivalent,
        du=du,
        dn=dn,
        alpha_effective=alpha,
        effective_speed=magnitude,
        section_cl=polar.cl,
        section_cd=polar.cd,
        separation_weight=polar.separation_weight,
    )


def solve_nonlinear_circulation(
    chord_m: torch.Tensor,
    incident_u: torch.Tensor,
    incident_n: torch.Tensor,
    tangent_influence: torch.Tensor,
    normal_influence: torch.Tensor,
    gamma0: float,
    history_gamma_mps: torch.Tensor,
    initial_gamma: torch.Tensor,
    speed_mps: torch.Tensor,
    cfg: ULLTConfig,
) -> dict[str, torch.Tensor]:
    """Solve 2*Gamma/(c*Cla) - gamma0*ue(Gamma) - history_gamma = 0.

    Gamma is (B,N), influence matrices (B,N,N); residuals have velocity units.
    Predict from the linear solve or supplied wake history, with bounded fixed
    restarts and no hidden cache. Damped Newton uses per-batch backtracking.
    Post-stall roots need not be globally unique, even at an accepted residual.
    """
    diagonal = 2 / (chord_m * cfg.lift_slope_per_rad)
    scale = speed_mps.amax(1)
    tolerance = torch.maximum(
        cfg.nonlinear_absolute_tolerance_mps + cfg.nonlinear_relative_tolerance * scale,
        32 * torch.finfo(chord_m.dtype).eps * scale,
    )
    iterations = torch.zeros(len(chord_m), device=chord_m.device, dtype=torch.long)
    backtracks = torch.zeros_like(iterations)

    def residual(gamma: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        u = incident_u + (tangent_influence @ gamma.unsqueeze(-1)).squeeze(-1)
        n = incident_n + (normal_influence @ gamma.unsqueeze(-1)).squeeze(-1)
        fields = nonlinear_normalwash(u, n, cfg)
        fields["physical_ve"] = n
        fields["effective_chordwise_speed"] = u
        r = diagonal * gamma - gamma0 * fields["equivalent"] - history_gamma_mps
        return r, fields

    # Fixed, data-independent predictors. Restarts change numerical initialization,
    # never the polar, residual, wake state or accepted tolerance.
    incident = nonlinear_normalwash(incident_u, incident_n, cfg)
    predictors = (
        initial_gamma,
        (gamma0 * incident["equivalent"] + history_gamma_mps) / diagonal,
        0.5 * initial_gamma,
        torch.zeros_like(initial_gamma),
    )
    gamma = initial_gamma
    if gamma0 < 1:
        # Recover the steady circulation implied by the supplied wake state.
        # This preserves a previously initialized steady root when roots multiply.
        memory_predictor = history_gamma_mps / ((1 - gamma0) * diagonal)
        initial_r, _ = residual(gamma)
        memory_r, memory_fields = residual(memory_predictor)
        prefer_memory = (memory_r.abs().amax(1) < initial_r.abs().amax(1)) & (
            memory_fields["effective_chordwise_speed"] > cfg.minimum_speed_mps
        ).all(1)
        gamma = torch.where(prefer_memory[:, None], memory_predictor, gamma)
    restarts = torch.zeros_like(iterations)
    attempt_iterations = torch.zeros_like(iterations)

    def restart(mask: torch.Tensor, current: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        if bool((mask & (restarts >= cfg.nonlinear_max_restarts)).any()):
            raise NonlinearConvergenceError("iteration/backtracking limit after bounded restarts", norm, iterations)
        restarts.add_(mask.long())
        attempt_iterations.masked_fill_(mask, 0)
        for index in range(1, len(predictors)):
            current = torch.where((mask & (restarts == index))[:, None], predictors[index], current)
        return current

    limit = (cfg.nonlinear_max_iterations + 2) * (cfg.nonlinear_max_restarts + 1)
    for _ in range(limit):
        r, fields = residual(gamma)
        norm = r.abs().amax(1)
        physical = (fields["effective_chordwise_speed"] > cfg.minimum_speed_mps).all(1)
        active = (norm > tolerance) | ~physical
        if not bool(torch.isfinite(norm).all()):
            raise NonlinearConvergenceError("non-finite residual", norm, iterations)
        if not bool(active.any()):
            return {
                **fields,
                "gamma": gamma,
                "nonlinear_residual_mps": norm,
                "nonlinear_tolerance_mps": tolerance,
                "nonlinear_iterations": iterations,
                "nonlinear_backtracks": backtracks,
                "nonlinear_restarts": restarts,
            }
        exhausted = active & ((attempt_iterations >= cfg.nonlinear_max_iterations) | ((norm <= tolerance) & ~physical))
        if bool(exhausted.any()):
            gamma = restart(exhausted, gamma, norm)
            continue
        jacobian = torch.diag_embed(diagonal) - gamma0 * (
            fields["du"].unsqueeze(-1) * tangent_influence + fields["dn"].unsqueeze(-1) * normal_influence
        )
        update, info = torch.linalg.solve_ex(jacobian, -r.unsqueeze(-1))
        bad = active & ((info != 0) | ~torch.isfinite(update).all(dim=(1, 2)))
        if bool(bad.any()):
            gamma = restart(bad, gamma, norm)
            continue
        update = update.squeeze(-1)
        damping = torch.ones_like(norm)
        accepted = ~active
        next_gamma = gamma.clone()
        # L2 merit is smooth even where the largest residual changes section.
        merit = r.square().sum(1)
        for step in range(cfg.nonlinear_max_backtracks):
            trial = gamma + damping[:, None] * update
            trial_residual, trial_fields = residual(trial)
            trial_norm = trial_residual.abs().amax(1)
            okay = (
                (~accepted)
                & torch.isfinite(trial_norm)
                & (trial_fields["effective_chordwise_speed"] > cfg.minimum_speed_mps).all(1)
                & ((trial_residual.square().sum(1) <= merit * (1 - 1e-4 * damping)) | (trial_norm <= tolerance))
            )
            next_gamma = torch.where(okay[:, None], trial, next_gamma)
            backtracks += okay.long() * step
            accepted |= okay
            if bool(accepted.all()):
                break
            damping = torch.where(accepted, damping, 0.5 * damping)
        iterations += active.long()
        attempt_iterations += active.long()
        gamma = next_gamma
        if not bool(accepted.all()):
            gamma = restart(~accepted, gamma, norm)
    raise NonlinearConvergenceError("total iteration limit", norm, iterations)

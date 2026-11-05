"""Arc-length cubic B-spline geometry; no reference speed or legacy task classes.

World frame is z-up. kappa is d(heading)/ds, NOT Frenet curvature.
The lateral/up normal frame remains defined at zero curvature.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class TrajectorySpec:
    knots: tuple[float, ...]
    kappa_coeff: tuple[float, ...]
    gamma_coeff: tuple[float, ...]
    length_m: float
    origin: tuple[float, float, float] = (0., 0., 20.)
    heading_rad: float = 0.
    seed: int = 0

    def to_dict(self) -> dict:
        return dict(knots=[float(x) for x in self.knots],
                    kappa_coeff=[float(x) for x in self.kappa_coeff],
                    gamma_coeff=[float(x) for x in self.gamma_coeff],
                    length_m=float(self.length_m), origin=[float(x) for x in self.origin],
                    heading_rad=float(self.heading_rad), seed=int(self.seed))


def clamped_knots(length_m: float, spacing_m: float) -> tuple[float, ...]:
    if not np.isfinite([length_m, spacing_m]).all() or min(length_m, spacing_m) <= 0:
        raise ValueError("Length and knot spacing must be finite and positive")
    return tuple([0.] * 4 + list(np.arange(spacing_m, length_m, spacing_m)) + [length_m] * 4)


class Trajectory:
    """Float64 spline evaluation and 8-point Gauss integration on a <= ds grid."""

    def __init__(self, spec: TrajectorySpec, ds: float = .2):
        t = np.asarray(spec.knots)
        n = len(t) - 4
        if (n < 4 or len(spec.kappa_coeff) != n or len(spec.gamma_coeff) != n
                or not np.isfinite(t).all() or not np.isfinite(spec.kappa_coeff).all()
                or not np.isfinite(spec.gamma_coeff).all() or not np.isfinite(spec.origin).all()
                or not np.isfinite(spec.heading_rad) or not np.isfinite(ds) or ds <= 0
                or not np.isfinite(spec.length_m) or spec.length_m <= 0
                or not np.all(t[:4] == 0) or not np.all(t[-4:] == spec.length_m)
                or np.any(np.diff(t[3:-3]) <= 0)):
            raise ValueError("Require finite cubic coefficients and clamped knots with simple interior knots")
        self.spec = spec
        self.k = BSpline(t, spec.kappa_coeff, 3, extrapolate=False)
        self.g = BSpline(t, spec.gamma_coeff, 3, extrapolate=False)
        self.ki = self.k.antiderivative()
        self.nodes, self.weights = np.polynomial.legendre.leggauss(8)
        self.s = np.unique(np.r_[np.linspace(0, spec.length_m, int(np.ceil(spec.length_m / ds)) + 1), t])
        increments = self._integrate(self.s[:-1], self.s[1:])
        self.positions = np.vstack([np.zeros(3), np.cumsum(increments, axis=0)]) + spec.origin

    def _angles(self, s):
        return self.spec.heading_rad + self.ki(s) - self.ki(0), self.g(s)

    def _tangent(self, s):
        psi, gamma = self._angles(s)
        return np.stack([np.cos(gamma)*np.cos(psi), np.cos(gamma)*np.sin(psi), np.sin(gamma)], axis=-1)

    def _integrate(self, lo, hi):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        q = mid[..., None] + half[..., None] * self.nodes
        return half[..., None] * np.sum(self._tangent(q) * self.weights[:, None], axis=-2)

    def query(self, s) -> dict[str, np.ndarray]:
        """Clamp previews at the endpoint; all derivatives are analytic in metres."""
        s = np.clip(np.asarray(s, dtype=float), 0, self.spec.length_m)
        i = np.minimum(np.searchsorted(self.s, s, side="right") - 1, len(self.s)-2)
        psi, gamma = self._angles(s)
        lateral = np.stack([-np.sin(psi), np.cos(psi), np.zeros_like(s)], axis=-1)
        tangent = self._tangent(s)
        up = np.cross(tangent, lateral)
        k, gprime = self.k(s), self.g.derivative()(s)
        tangent_prime = (np.cos(gamma)*k)[..., None]*lateral + gprime[..., None]*up
        return dict(position=self.positions[i] + self._integrate(self.s[i], s), tangent=tangent,
                    normal=lateral, up=up, tangent_prime=tangent_prime, heading=psi,
                    kappa=k, gamma=gamma, kappa_prime=self.k.derivative()(s), gamma_prime=gprime,
                    curvature=np.linalg.norm(tangent_prime, axis=-1))

    def project(self, point, previous_s: float, window_m: float) -> float:
        """Local segment projection prevents jumping to a distant path branch."""
        lo, hi = max(0., previous_s-window_m), min(self.spec.length_m, previous_s+window_m)
        s = np.unique(np.r_[lo, self.s[(self.s > lo) & (self.s < hi)], hi])
        p = self.query(s)["position"]
        delta = np.diff(p, axis=0)
        alpha = np.clip(np.sum((point-p[:-1])*delta, axis=1) / np.maximum(np.sum(delta*delta, axis=1), 1e-20), 0, 1)
        distance = np.sum((p[:-1]+alpha[:, None]*delta-point)**2, axis=1)
        i = np.argmin(distance)
        return float(s[i] + alpha[i]*(s[i+1]-s[i]))


def generate_batch(specs: list[TrajectorySpec], ds: float = .2) -> list[Trajectory]:
    """Reset-time CPU batch, independent lengths/knots supported."""
    return [Trajectory(spec, ds) for spec in specs]

"""Arc-length cubic B-spline geometry; no reference speed or legacy task classes.

World frame is z-up. kappa is d(heading)/ds, NOT Frenet curvature.
The lateral/up normal frame remains defined at zero curvature.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import BSpline, PPoly
from scipy.spatial import cKDTree


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


@dataclass(frozen=True)
class GeometryLimits:
    kappa: float = .06  # rad/m
    gamma: float = .4  # rad
    kappa_prime: float = .04  # rad/m^2
    gamma_prime: float = .2  # rad/m
    min_height: float = 1.
    max_height: float = 60.
    intersection_distance: float = .5
    nonlocal_separation_m: float = 8.


def _peak(spline: BSpline, length: float) -> float:
    roots = PPoly.from_spline(spline.derivative()).roots(extrapolate=False)
    candidates = np.r_[0., length, spline.t, roots[np.isfinite(roots) & (roots >= 0) & (roots <= length)]]
    return float(np.max(np.abs(spline(candidates))))


def screen(path: Trajectory, limits: GeometryLimits) -> list[str]:
    """Exact spline extrema; conservative grid height and nonlocal proximity checks.

    Simple interior knots certify C2 kappa/gamma and continuous position/tangent.
    Nonlocal proximity is a geometric rejection, not a dynamic feasibility claim.
    """
    if (not np.isfinite(list(asdict(limits).values())).all()
            or min(limits.kappa, limits.gamma, limits.kappa_prime, limits.gamma_prime) < 0
            or limits.min_height >= limits.max_height or limits.intersection_distance <= 0
            or limits.nonlocal_separation_m <= limits.intersection_distance):
        raise ValueError("Invalid geometry limits")
    errors = []
    for name, spline in (("kappa", path.k), ("gamma", path.g),
                         ("kappa_prime", path.k.derivative()), ("gamma_prime", path.g.derivative())):
        if _peak(spline, path.spec.length_m) > getattr(limits, name) + 1e-12:
            errors.append(name)
    q = path.query(path.s)
    if not all(np.isfinite(value).all() for value in q.values()):
        return errors + ["nonfinite"]
    ds = float(np.max(np.diff(path.s)))
    # |dz/ds| <= 1: no between-sample height excursion can exceed ds/2.
    z = path.positions[:, 2]
    if z.min()-ds/2 < limits.min_height or z.max()+ds/2 > limits.max_height:
        errors.append("height")
    if np.any(np.linalg.norm(np.diff(path.positions, axis=0), axis=1) > np.diff(path.s)*(1+1e-9)):
        errors.append("continuity")
    pairs = cKDTree(path.positions).query_pairs(limits.intersection_distance + ds, output_type="ndarray")
    if len(pairs) and np.any(np.abs(path.s[pairs[:, 1]]-path.s[pairs[:, 0]]) >= limits.nonlocal_separation_m):
        errors.append("nonlocal_intersection")
    return errors


@dataclass(frozen=True)
class SamplingConfig:
    length_m: float = 60.
    knot_spacing_m: float = 10.
    kappa_amplitude: float = .025
    gamma_amplitude: float = .15
    transition_strength: float = 1.  # 0: constant coefficients; 1: independent coefficients
    coupling: float = .5  # [-1,1], signed lateral/vertical coefficient coupling
    required_reversals: tuple[str, ...] = ("kappa", "gamma")
    transition_interval: tuple[float, float] = (.15, .85)  # fractions of the completed path
    reversal_fraction: float = .08  # both signs must exceed this fraction of amplitude
    max_attempts: int = 256

    def validate(self):
        values = [self.length_m, self.knot_spacing_m, self.kappa_amplitude, self.gamma_amplitude,
                  self.transition_strength, self.coupling, *self.transition_interval, self.reversal_fraction]
        if (not np.isfinite(values).all() or min(values[:2]) <= 0 or min(values[2:4]) < 0
                or not 0 <= self.transition_strength <= 1 or not -1 <= self.coupling <= 1
                or not 0 < self.transition_interval[0] < self.transition_interval[1] < 1
                or not 0 < self.reversal_fraction < 1 or self.max_attempts < 1
                or any(x not in ("kappa", "gamma") for x in self.required_reversals)):
            raise ValueError("Invalid continuous sampling configuration")
        if self.required_reversals and (self.transition_strength == 0 or
                any(getattr(self, name+"_amplitude") == 0 for name in self.required_reversals)):
            raise ValueError("Required reversals need nonzero amplitude and transition strength")


def reversal_locations(path: Trajectory, cfg: SamplingConfig) -> dict[str, list[float]]:
    lo, hi = np.asarray(cfg.transition_interval)*cfg.length_m
    result = {}
    for name, spline in (("kappa", path.k), ("gamma", path.g)):
        roots = PPoly.from_spline(spline).roots(extrapolate=False)
        roots = roots[np.isfinite(roots) & (roots > lo) & (roots < hi)]
        s = np.unique(np.r_[np.linspace(lo, hi, 257), roots])
        values = spline(s)
        threshold = cfg.reversal_fraction*getattr(cfg, name+"_amplitude")
        result[name] = roots.tolist() if values.min() < -threshold and values.max() > threshold else []
    return result


def sample_trajectory(seed: int, cfg: SamplingConfig = SamplingConfig(),
                      limits: GeometryLimits = GeometryLimits(), origin=(0., 0., 20.), heading=0.) -> Trajectory:
    """Sample directly in coefficient space, with bounded rejection and saved seed."""
    cfg.validate()
    rng = np.random.default_rng(seed)
    knots = clamped_knots(cfg.length_m, cfg.knot_spacing_m)
    n = len(knots)-4
    reasons = []
    for _ in range(cfg.max_attempts):
        x, y = rng.uniform(-1, 1, (2, n))
        y = cfg.coupling*x + (1-abs(cfg.coupling))*y
        baseline = rng.uniform(-1, 1, 2)
        a = cfg.kappa_amplitude*((1-cfg.transition_strength)*baseline[0]+cfg.transition_strength*x)
        b = cfg.gamma_amplitude*((1-cfg.transition_strength)*baseline[1]+cfg.transition_strength*y)
        path = Trajectory(TrajectorySpec(knots, tuple(a), tuple(b), cfg.length_m, tuple(origin), heading, seed))
        reasons = screen(path, limits)
        events = reversal_locations(path, cfg)
        reasons += ["missing_"+name+"_reversal" for name in cfg.required_reversals if not events[name]]
        if not reasons:
            return path
    raise RuntimeError(f"No admissible trajectory in {cfg.max_attempts} draws; last rejection: {reasons}")

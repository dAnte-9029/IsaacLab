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
    initial_straight_m: float = 5.
    axis_active_probability: float = .75  # exact-zero mass per complete channel
    nonzero_density: float = .8  # nonzero probability at each coefficient refresh
    coefficient_change_probability: float = .35  # otherwise repeat: long plateaus
    transition_strength: float = .8  # adjacent update gain; 0 keeps the first active coefficient
    coupling: float = .5  # [-1,1], signed lateral/vertical coefficient coupling
    required_reversals: tuple[str, ...] = ()
    transition_interval: tuple[float, float] = (.15, .85)  # fractions of the completed path
    reversal_fraction: float = .08  # both signs must exceed this fraction of amplitude
    max_attempts: int = 256

    def validate(self):
        values = [self.length_m, self.knot_spacing_m, self.kappa_amplitude, self.gamma_amplitude,
                  self.transition_strength, self.coupling, *self.transition_interval, self.reversal_fraction,
                  self.initial_straight_m, self.axis_active_probability, self.nonzero_density,
                  self.coefficient_change_probability]
        if (not np.isfinite(values).all() or min(values[:2]) <= 0 or min(values[2:4]) < 0
                or not 0 <= self.initial_straight_m < self.length_m
                or any(not 0 <= p <= 1 for p in (self.axis_active_probability, self.nonzero_density,
                                                self.coefficient_change_probability))
                or not 0 <= self.transition_strength <= 1 or not -1 <= self.coupling <= 1
                or not 0 < self.transition_interval[0] < self.transition_interval[1] < 1
                or not 0 < self.reversal_fraction < 1 or self.max_attempts < 1
                or any(x not in ("kappa", "gamma") for x in self.required_reversals)):
            raise ValueError("Invalid continuous sampling configuration")
        if self.required_reversals and (self.transition_strength == 0 or self.axis_active_probability == 0
                or self.nonzero_density == 0 or self.coefficient_change_probability == 0 or
                any(getattr(self, name+"_amplitude") == 0 for name in self.required_reversals)):
            raise ValueError("Required reversals need active, nonzero, changing coefficients")
        # Reject impossible entry/transition layouts before drawing tasks.
        active_start = self.initial_straight_m
        first_free = 4 if active_start > 0 else 3
        knots = sampling_knots(self)
        if self.required_reversals and (len(knots)-4-first_free < 2 or
                active_start >= self.transition_interval[1]*self.length_m):
            raise ValueError("Not enough maneuver span for required reversals after the straight entry")


def sampling_knots(cfg: SamplingConfig) -> tuple[float, ...]:
    """Entry endpoint is an actual knot; zero supported coefficients give exact entry.

    Four initial zero coefficients give zero value, slope and second derivative
    through the entry endpoint; the first free basis turns on cubically there.
    """
    interior = np.arange(cfg.initial_straight_m, cfg.length_m, cfg.knot_spacing_m)
    interior = interior[interior > 0]
    return tuple([0.]*4 + interior.tolist() + [cfg.length_m]*4)


def reversal_locations(path: Trajectory, cfg: SamplingConfig) -> dict[str, list[float]]:
    lo, hi = np.asarray(cfg.transition_interval)*cfg.length_m
    result = {}
    for name, spline in (("kappa", path.k), ("gamma", path.g)):
        roots = PPoly.from_spline(spline).roots(extrapolate=False)
        roots = roots[np.isfinite(roots) & (roots > lo) & (roots < hi)]
        s = np.unique(np.r_[np.linspace(lo, hi, 257), roots])
        values = spline(s)
        threshold = cfg.reversal_fraction*getattr(cfg, name+"_amplitude")
        significant = np.flatnonzero(np.abs(values) > threshold)
        events = []
        for left, right in zip(significant[:-1], significant[1:]):
            if values[left]*values[right] < 0:
                between = roots[(roots > s[left]) & (roots < s[right])]
                if len(between):
                    # Zero plateaus may have several boundary roots. Record one
                    # event only when significant values on its two sides reverse.
                    events.append(float(between[len(between)//2]))
        result[name] = events
    return result


def sample_trajectory(seed: int, cfg: SamplingConfig = SamplingConfig(),
                      limits: GeometryLimits = GeometryLimits(), origin=(0., 0., 20.), heading=0.) -> Trajectory:
    """Sample directly in coefficient space, with bounded rejection and saved seed."""
    cfg.validate()
    rng = np.random.default_rng(seed)
    knots = sampling_knots(cfg)
    n = len(knots)-4
    first_free = 4 if cfg.initial_straight_m > 0 else 3
    reasons = []
    for attempt in range(1, cfg.max_attempts+1):
        # No task family is selected. A spike-and-continuous coefficient process
        # supplies exact zeros, held plateaus, slow updates and sign changes.
        active = rng.random(2) < cfg.axis_active_probability
        for axis, name in enumerate(("kappa", "gamma")):
            if name in cfg.required_reversals:
                active[axis] = True  # condition only on explicitly requested nonzero axes
        values = np.zeros((2, n))
        targets = rng.uniform(-1., 1., (2, n-first_free))
        targets[1] = cfg.coupling*targets[0] + (1-abs(cfg.coupling))*targets[1]
        nonzero = rng.random(targets.shape) < cfg.nonzero_density
        required = np.array([name in cfg.required_reversals for name in ("kappa", "gamma")])
        # Explicit reversals condition the proposal on active, refreshing axes;
        # do not repeatedly draw zero/held channels that cannot reverse.
        nonzero[required] = True
        targets *= nonzero
        for j in range(first_free, n):
            if j == first_free:
                values[:, j] = targets[:, j-first_free]
            else:
                update = (rng.random(2) < cfg.coefficient_change_probability) | required
                target = targets[:, j-first_free]
                proposal = values[:, j-1] + cfg.transition_strength*(target-values[:, j-1])
                # Preserve an exact zero atom in coefficient space, rather than
                # asymptotically decaying toward zero under the smoothing gain.
                if cfg.transition_strength > 0:
                    proposal = np.where(target == 0, 0., proposal)
                values[:, j] = np.where(update, proposal, values[:, j-1])
        values *= active[:, None]
        a, b = values * np.array([cfg.kappa_amplitude, cfg.gamma_amplitude])[:, None]
        path = Trajectory(TrajectorySpec(knots, tuple(a), tuple(b), cfg.length_m, tuple(origin), heading, seed))
        reasons = screen(path, limits)
        events = reversal_locations(path, cfg)
        reasons += ["missing_"+name+"_reversal" for name in cfg.required_reversals if not events[name]]
        if not reasons:
            path.sampling_attempts = attempt
            return path
    raise RuntimeError(f"No admissible trajectory in {cfg.max_attempts} draws; last rejection: {reasons}")

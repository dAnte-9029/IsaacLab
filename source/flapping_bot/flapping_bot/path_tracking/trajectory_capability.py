"""Current-policy empirical capability. No physical feasibility inference."""
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
import numpy as np
from .unified_trajectory import Trajectory, TrajectorySpec
from .trajectory_evidence_geometry import features
from .trajectory_evidence import canonical

# Symmetric distances, not componentwise difficulty ordering. Scales are hypotheses.
FEATURE_KEYS = ('kappa', 'gamma', 'kappa_prime', 'gamma_prime', 'length_m',
                'kappa_max_active_length_m', 'gamma_max_active_length_m', 'net_climb_m',
                'climb_excursion_m', 'coupled_length_m', 'coupling_abs_integral')
DEFAULT_SCALES = (.025, .15, .01, .05, 150., 60., 60., 20., 20., 60., .1)


@lru_cache(maxsize=4096)
def _vector_cached(serialized):
    f = features(Trajectory(TrajectorySpec(**json.loads(serialized))))
    return tuple(f[k] for k in FEATURE_KEYS)


def vector(spec):
    return np.array(_vector_cached(canonical(spec.to_dict())))


@dataclass(frozen=True)
class OnlineConfig:
    base_weight: float = .4
    retention_weight: float = .25
    exploration_weight: float = .35
    base_floor: float = .2
    update_interval: int = 8
    update_rate: float = .2
    neighborhood_radius: float = 1.0
    scales: tuple = DEFAULT_SCALES
    candidate_count: int = 16
    history_window: int = 32
    exploration_floor: float = .15
    offline_bonus: float = 1.

    def __post_init__(self):
        weights = (self.base_weight, self.retention_weight, self.exploration_weight)
        if (not np.isfinite(weights + (self.base_floor, self.update_rate, self.neighborhood_radius,
                                       self.exploration_floor, self.offline_bonus) + tuple(self.scales)).all()
                or min(weights) < 0 or not np.isclose(sum(weights), 1)
                or not 0 < self.base_floor <= self.base_weight or self.retention_weight <= 0
                or self.exploration_weight <= 0 or not 0 < self.update_rate <= 1
                or self.neighborhood_radius <= 0 or len(self.scales) != len(FEATURE_KEYS)
                or min(self.scales) <= 0 or self.update_interval < 1 or self.candidate_count < 2
                or self.history_window < 4 or self.exploration_floor <= 0 or self.offline_bonus < 0):
            raise ValueError('Invalid online sampling configuration')


class PolicyCapability:
    """Finite-neighborhood empirical policy statistics; never feasibility odds."""
    def __init__(self, cfg=OnlineConfig()):
        self.cfg = cfg
        self.rows = []

    def add(self, record):
        self.rows.append((vector(TrajectorySpec(**record['trajectory'])), deepcopy(record)))

    def estimate(self, spec, plant, contract, version):
        x = vector(spec)
        neighbors = [r for v, r in self.rows if r['plant'] == plant and r['contract'] == contract
                     and np.linalg.norm((x-v)/self.cfg.scales) <= self.cfg.neighborhood_radius]
        current = [r for r in neighbors if r['policy_version'] == version][-self.cfg.history_window:]
        outcomes = [r['outcome'] == 'completed' for r in current]
        half = len(outcomes)//2
        progress = float(np.mean(outcomes[half:])-np.mean(outcomes[:half])) if half else 0.
        return dict(count=len(outcomes), successes=sum(outcomes), failures=sum(r['outcome']=='failed' for r in current),
                    timeouts=sum(r['outcome']=='timeout' for r in current),
                    rate=float(np.mean(outcomes)) if outcomes else None, progress=progress,
                    mastered=any(r['outcome']=='completed' for r in neighbors))

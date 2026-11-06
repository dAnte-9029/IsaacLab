"""Current-policy empirical capability. No physical feasibility inference."""
from collections import deque, OrderedDict
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
    region_count: int = 64
    capacity: int = 2048
    frozen_versions: int = 2
    exploration_kernel_weight: float = .65
    historical_retention_weight: float = .2

    def __post_init__(self):
        weights = (self.base_weight, self.retention_weight, self.exploration_weight)
        if (not np.isfinite(weights + (self.base_floor, self.update_rate, self.neighborhood_radius,
                                       self.exploration_floor, self.offline_bonus) + tuple(self.scales)).all()
                or min(weights) < 0 or not np.isclose(sum(weights), 1)
                or not 0 < self.base_floor <= self.base_weight or self.retention_weight <= 0
                or self.exploration_weight <= 0 or not 0 < self.update_rate <= 1
                or self.neighborhood_radius <= 0 or len(self.scales) != len(FEATURE_KEYS)
                or min(self.scales) <= 0 or self.update_interval < 1 or self.candidate_count < 2
                or self.history_window < 4 or self.exploration_floor <= 0 or self.offline_bonus < 0
                or self.region_count < 4 or self.capacity < self.history_window or self.frozen_versions < 2
                or not 0 <= self.exploration_kernel_weight <= 1
                or not 0 <= self.historical_retention_weight <= 1):
            raise ValueError('Invalid online sampling configuration')


class PolicyCapability:
    """Bounded training window and separate explicitly frozen checkpoint windows.

    Mixed or unknown execution never enters checkpoint statistics. Version changes
    at the runner do not select a frozen evaluation and cannot create forgetting.
    """
    def __init__(self, cfg=OnlineConfig()):
        self.cfg = cfg
        self.training = deque(maxlen=cfg.capacity)
        self.frozen = OrderedDict()
        self.latest_frozen = None

    def add(self, record):
        if record['outcome'] not in ('completed', 'failed', 'timeout'): return
        row = (vector(TrajectorySpec(**record['trajectory'])), record['outcome'],
               canonical([record['plant'], record['contract']]))
        checkpoint = record.get('frozen_checkpoint')
        if checkpoint is not None:
            if checkpoint not in self.frozen:
                self.frozen[checkpoint] = deque(maxlen=self.cfg.capacity)
                if len(self.frozen) > self.cfg.frozen_versions: self.frozen.popitem(last=False)
            self.frozen[checkpoint].append(row)
            self.latest_frozen = checkpoint
        else:
            self.training.append(row)

    def estimate(self, spec, plant, contract, version=None, *, stream='frozen'):
        rows = list(self.training if stream=='training' else self.frozen.get(version, ()))
        if rows:
            x = vector(spec); context = canonical([plant, contract])
            distances = np.linalg.norm((np.array([r[0] for r in rows])-x)/self.cfg.scales, axis=1)
            outcomes = [r[1] for r, d in zip(rows, distances)
                        if d <= self.cfg.neighborhood_radius and r[2]==context][-self.cfg.history_window:]
        else: outcomes = []
        values = [x=='completed' for x in outcomes]; half = len(values)//2
        progress = float(np.mean(values[half:])-np.mean(values[:half])) if half else 0.
        return dict(count=len(values), successes=sum(values), failures=outcomes.count('failed'),
                    timeouts=outcomes.count('timeout'), rate=float(np.mean(values)) if values else None,
                    progress=progress, stream=stream, checkpoint=version if stream=='frozen' else None)

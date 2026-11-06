"""Continuous mixture sampling, independent of PPO and legacy curricula.

Public convenience imports keep the evidence, feedback and capability modules
independently usable. No physics, reward or actor implementation lives here.
"""
from copy import deepcopy
from dataclasses import replace
import numpy as np
from .unified_trajectory import Trajectory, TrajectorySpec, SamplingConfig, GeometryLimits, sample_trajectory
from .trajectory_evidence import OfflineEvidence
from .trajectory_feedback import FeedbackStore, RolloutEvaluator
from .trajectory_capability import OnlineConfig, PolicyCapability, FEATURE_KEYS, vector

class OnlineSampler:
    """q0 + retention kernels + exploration importance-resampled q0 candidates.

    Finite candidate proposals are refreshed every request; no fixed task IDs.
    Retention replays historical coefficient shapes with continuous amplitude jitter.
    All routes pass the original geometric screen. Failed proposals fall back to q0.
    """
    def __init__(self, offline, store, plant, contract, policy_version, cfg=OnlineConfig(),
                 sampling=SamplingConfig(), limits=GeometryLimits()):
        self.offline, self.store, self.plant, self.contract = offline, store, deepcopy(plant), deepcopy(contract)
        self.policy_version, self.cfg, self.sampling, self.limits = policy_version, cfg, sampling, limits
        self.capability = PolicyCapability(cfg)
        self._pending = []
        self._seen_ids = set()
        self._score_rows = []
        for r in store.records:
            self.feedback(r)
        self._offline = [(np.array([r['features'][k] for k in FEATURE_KEYS]), r['backend'])
                         for r in offline.records if r['status'] == 'VERIFIED FEASIBLE']

    def feedback(self, record):
        if record['simulated'] != self.store.simulated:
            raise ValueError('Cannot mix synthetic and actual rollout feedback')
        if self.store._by_id.get(record.get('instance_id')) != record:
            raise ValueError('Feedback must be an unchanged evaluator journal record')
        if record['instance_id'] in self._seen_ids:
            raise ValueError('Duplicate sampler feedback')
        self._seen_ids.add(record['instance_id'])
        self.capability.add(record)
        self._pending.append(deepcopy(record))
        if len(self._pending) >= self.cfg.update_interval:
            self._score_rows.extend(self._pending)
            self._pending.clear()

    def statistics(self, spec):
        return self.capability.estimate(spec, self.plant, self.contract, self.policy_version)

    def _components(self, specs):
        # On each fixed candidate set both conditional distributions move by at
        # most update_rate in total variation per feedback batch.
        prior = np.ones(len(specs))
        for j, spec in enumerate(specs):
            x = vector(spec)
            distances = [np.linalg.norm((x-v)/self.cfg.scales) for v, backend in self._offline
                         if backend == self.plant.get('backend')]
            if distances:
                prior[j] += self.cfg.offline_bonus*np.exp(-.5*min(distances)**2)
        e = prior/prior.sum()
        r = np.ones(len(specs))/len(specs)
        cap = PolicyCapability(self.cfg)
        for start in range(0, len(self._score_rows), self.cfg.update_interval):
            for row in self._score_rows[start:start+self.cfg.update_interval]:
                cap.add(row)
            # Freeze each update's policy scope to its terminal batch. Changing
            # the active policy version alone must not rewrite past distributions.
            batch_version = self._score_rows[min(start+self.cfg.update_interval, len(self._score_rows))-1]['policy_version']
            stats = [cap.estimate(s, self.plant, self.contract, batch_version) for s in specs]
            targets = []
            for stat in stats:
                rate = stat['rate'] if stat['rate'] is not None else .5
                targets.append(self.cfg.exploration_floor + 1/np.sqrt(1+stat['count'])
                               + 2*max(0, stat['progress']) + (1-rate)*stat['mastered'])
            target = np.array(targets) + prior-1
            target /= target.sum()
            e += self.cfg.update_rate*(target-e)
            retained = np.array([stat['mastered'] for stat in stats], float)
            target_r = retained/retained.sum() if retained.sum() else np.ones(len(specs))/len(specs)
            r += self.cfg.update_rate*(target_r-r)
        return e, r

    def scores(self, specs):
        return self._components(specs)[0]

    def distribution(self, specs):
        """Conditional probe masses; retention kernels in actual sampling differ.

        This diagnostic assumes the supplied probes partition candidate support;
        it is not the probability of drawing an exact continuous trajectory.
        """
        e, r = self._components(specs)
        return self.cfg.base_weight/len(specs) + self.cfg.retention_weight*r + self.cfg.exploration_weight*e

    def sample(self, seed, origin=(0., 0., 20.), heading=0.):
        from .unified_trajectory import screen, reversal_locations
        rng = np.random.default_rng(seed)
        source = rng.choice(['base', 'retention', 'exploration'], p=[self.cfg.base_weight,
                         self.cfg.retention_weight, self.cfg.exploration_weight])
        draw = lambda: sample_trajectory(int(rng.integers(2**32)), self.sampling, self.limits, origin, heading)
        if source == 'retention':
            # Convex mixture of historical anchor distributions: new batches
            # receive update_rate mass, old distributions retain the remainder.
            batches = len(self._score_rows)//self.cfg.update_interval
            epoch = batches-int(rng.geometric(self.cfg.update_rate))+1
            rows = self._score_rows[:max(0, epoch)*self.cfg.update_interval]
            unique = {r['identity']: r for r in rows if r['outcome']=='completed'
                         and r['plant']==self.plant and r['contract']==self.contract
                         and r['trajectory']['length_m']==self.sampling.length_m}
            successes = list(unique.values())
            if successes:
                # Uniform historical anchors guarantee continued rehearsal after forgetting.
                spec = TrajectorySpec(**successes[int(rng.integers(len(successes)))]['trajectory'])
                path = Trajectory(replace(spec, origin=tuple(origin), heading_rad=heading, seed=seed,
                    kappa_coeff=tuple(np.array(spec.kappa_coeff)*rng.uniform(.9, 1.1)),
                    gamma_coeff=tuple(np.array(spec.gamma_coeff)*rng.uniform(.9, 1.1))))
                events = reversal_locations(path, self.sampling)
                if not screen(path, self.limits) and all(events[k] for k in self.sampling.required_reversals):
                    return path.spec, source
                # Failed jitter must not silently erase an admissible mastered anchor.
                anchor = Trajectory(replace(spec, origin=tuple(origin), heading_rad=heading, seed=seed))
                if not screen(anchor, self.limits) and all(reversal_locations(anchor, self.sampling)[k]
                                                          for k in self.sampling.required_reversals):
                    return anchor.spec, 'retention_anchor'
            source = 'retention_fallback_base'
        if source == 'exploration':
            paths = [draw() for _ in range(self.cfg.candidate_count)]
            scores = self.scores([p.spec for p in paths]); scores /= scores.sum()
            return paths[int(rng.choice(len(paths), p=scores))].spec, source
        return draw().spec, source

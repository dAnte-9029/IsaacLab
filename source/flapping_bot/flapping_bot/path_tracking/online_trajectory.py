"""Incremental continuous mixture; bounded regional state, permanent disk witnesses."""
from copy import deepcopy
from .trajectory_evidence import canonical
from dataclasses import replace
import numpy as np
from .unified_trajectory import Trajectory, TrajectorySpec, SamplingConfig, GeometryLimits, sample_trajectory, screen, reversal_locations
from .trajectory_evidence import OfflineEvidence
from .trajectory_feedback import FeedbackStore, RolloutEvaluator
from .trajectory_capability import OnlineConfig, PolicyCapability, FEATURE_KEYS, vector


class OnlineSampler:
    def __init__(self, offline, store, plant, contract, policy_version=None, cfg=OnlineConfig(),
                 sampling=SamplingConfig(), limits=GeometryLimits()):
        self.offline, self.store, self.plant, self.contract = offline, store, deepcopy(plant), deepcopy(contract)
        self.policy_version, self.cfg, self.sampling, self.limits = policy_version, cfg, sampling, limits
        self.capability = PolicyCapability(cfg)
        # Finite quadrature/proposal bank in the continuous coefficient space, no curriculum IDs.
        self.regions = [sample_trajectory(i, sampling, limits).spec for i in range(cfg.region_count)]
        self.vectors = np.array([vector(s) for s in self.regions])
        self.anchors = [None]*cfg.region_count
        self.best_frozen = np.zeros(cfg.region_count)
        self.forgetting = np.zeros(cfg.region_count)
        self._offline = np.array([[r['features'][k] for k in FEATURE_KEYS] for r in offline.records
                                 if r['status']=='VERIFIED FEASIBLE' and r['backend']==plant.get('backend')])
        self.prior = self._prior(self.vectors)
        self.exploration = self.prior/self.prior.sum()
        # Last entry is q0 fallback; update distributions by convex mixing.
        self.retention = np.r_[np.zeros(cfg.region_count), 1.]
        self._evaluation_receipts = set()
        self._evaluated_version = None
        self._pending = 0; self._cursor = 0; self.witness_count = 0
        # One streaming recovery at construction only. Steps never loaded here.
        for record in store.iter_records(): self._ingest(record)
        self.witness_count = store.witness_count(plant, contract)

    def state_dict(self):
        excluded = {'offline','store','plant','contract','cfg','sampling','limits'}
        return deepcopy({k:v for k,v in vars(self).items() if k not in excluded})

    def load_state_dict(self, state):
        for key, value in deepcopy(state).items():
            if key in ('offline','store','plant','contract','cfg','sampling','limits'):
                raise ValueError('Cannot overwrite sampler contract or evidence store')
            setattr(self,key,value)

    def feedback_evaluation(self, record, source_store):
        if (source_store.simulated or record.get('simulated') or not record.get('frozen_checkpoint')
                or source_store.get(record['instance_id']) != record):
            raise ValueError('Requires actual independently frozen evaluator evidence')
        if record['plant'] != self.plant or record['contract'] != self.contract:
            raise ValueError('Frozen evaluator plant/contract differs from training')
        receipt = canonical([record['frozen_checkpoint'], record['trajectory'], record['initial'], record['plant'], record['contract']])
        if receipt in self._evaluation_receipts: return False
        cursor = self._cursor
        self._ingest(record)
        self._cursor = cursor
        self._evaluation_receipts.add(receipt)
        return True

    def set_policy_version(self, version):
        """Metadata only. Frozen capability is selected by actual frozen evaluations."""
        self.policy_version = version

    def _prior(self, vectors):
        if not len(self._offline): return np.ones(len(vectors))
        d = np.linalg.norm((vectors[:,None,:]-self._offline[None,:,:])/self.cfg.scales, axis=-1)
        return 1+self.cfg.offline_bonus*np.exp(-.5*d.min(axis=1)**2)

    def feedback(self, record):
        if record.get('simulated') != self.store.simulated or self.store.get(record.get('instance_id')) != record:
            raise ValueError('Feedback must be an unchanged evaluator journal record')
        if record['sequence'] <= self._cursor: raise ValueError('Duplicate or out-of-order feedback')
        self._ingest(record)
        self.witness_count = self.store.witness_count(self.plant, self.contract)

    def _ingest(self, record):
        self._cursor = record['sequence']
        if record['plant'] != self.plant or record['contract'] != self.contract: return
        if record['outcome'] not in ('completed', 'failed', 'timeout'): return
        spec = TrajectorySpec(**record['trajectory']); v = vector(spec)
        distances = np.linalg.norm((self.vectors-v)/self.cfg.scales, axis=1)
        index = int(np.argmin(distances))
        if distances[index] > self.cfg.neighborhood_radius:
            free = [i for i, anchor in enumerate(self.anchors) if anchor is None]
            if free:
                index = free[int(np.argmin(self.exploration[free]))]
                self.regions[index] = spec; self.vectors[index] = v
                self.prior[index] = self._prior(v[None])[0]
                self.best_frozen[index] = self.forgetting[index] = 0.
        if record['outcome']=='completed': self.anchors[index] = spec
        self.capability.add(record)
        self._pending += 1
        if self._pending >= self.cfg.update_interval:
            self._update(); self._pending = 0

    def _update(self):
        if self.capability.latest_frozen != self._evaluated_version:
            if self._evaluated_version is not None:
                for i, spec in enumerate(self.regions):
                    previous = self.capability.estimate(spec,self.plant,self.contract,self._evaluated_version)
                    if previous['count']: self.best_frozen[i] = max(self.best_frozen[i],previous['rate'])
            self._evaluated_version = self.capability.latest_frozen
        scores = []; retain = []
        for i, spec in enumerate(self.regions):
            train = self.capability.estimate(spec, self.plant, self.contract, stream='training')
            frozen = self.capability.estimate(spec, self.plant, self.contract, self.capability.latest_frozen)
            if frozen['count']:
                self.forgetting[i] = max(0., self.best_frozen[i]-frozen['rate'])
            scores.append(self.cfg.exploration_floor + 1/np.sqrt(1+train['count'])
                          + 2*max(0, train['progress']) + 2*self.forgetting[i] + self.prior[i]-1)
            retain.append((.25+3*self.forgetting[i]) if self.anchors[i] is not None else 0.)
        e = np.array(scores); e /= e.sum()
        r = np.r_[retain, 0. if any(retain) else 1.]; r /= r.sum()
        self.exploration += self.cfg.update_rate*(e-self.exploration)
        self.retention += self.cfg.update_rate*(r-self.retention)

    def statistics(self, spec, version=None, *, stream='frozen'):
        return self.capability.estimate(spec, self.plant, self.contract,
                                       version or self.capability.latest_frozen, stream=stream)

    def scores(self, specs):
        # Continuous finite-neighborhood interpolation of incrementally stored density ratios.
        vs = np.array([vector(s) for s in specs])
        d = np.linalg.norm((vs[:,None,:]-self.vectors[None,:,:])/self.cfg.scales, axis=-1)
        nearest = np.argmin(d, axis=1)
        weights = self._prior(vs)*self.exploration[nearest]/(self.prior[nearest]/self.prior.sum())
        return weights/weights.sum()

    def distribution(self, specs):
        """Probe-only score; NOT actual region frequency or physical feasibility."""
        vs = np.array([vector(s) for s in specs])
        nearest = np.argmin(np.linalg.norm((vs[:,None,:]-self.vectors[None,:,:])/self.cfg.scales, axis=-1), axis=1)
        r = self.retention[nearest]+self.retention[-1]/len(specs)
        r = r/r.sum() if r.sum() else np.ones(len(specs))/len(specs)
        return self.cfg.base_weight/len(specs)+self.cfg.retention_weight*r+self.cfg.exploration_weight*self.scores(specs)

    def _kernel(self, spec, rng, origin, heading):
        if spec.length_m != self.sampling.length_m: return None
        path = Trajectory(replace(spec, origin=tuple(origin), heading_rad=heading, seed=int(rng.integers(2**32)),
            kappa_coeff=tuple(np.array(spec.kappa_coeff)*rng.uniform(.9,1.1)),
            gamma_coeff=tuple(np.array(spec.gamma_coeff)*rng.uniform(.9,1.1))))
        # Both experiment modes share the configured coefficient domain, not only
        # the looser geometric safety limits. Jitter outside q0's amplitude box falls back.
        if (max(abs(x) for x in path.spec.kappa_coeff)>self.sampling.kappa_amplitude+1e-12
                or max(abs(x) for x in path.spec.gamma_coeff)>self.sampling.gamma_amplitude+1e-12):
            return None
        if not screen(path,self.limits) and all(reversal_locations(path,self.sampling)[k] for k in self.sampling.required_reversals):
            return path.spec
        return None

    def sample(self, seed, origin=(0.,0.,20.), heading=0.):
        rng = np.random.default_rng(seed)
        source = rng.choice(['base','retention','exploration'],p=[self.cfg.base_weight,self.cfg.retention_weight,self.cfg.exploration_weight])
        draw = lambda: sample_trajectory(int(rng.integers(2**32)),self.sampling,self.limits,origin,heading)
        if source=='retention':
            spec = None
            if self.witness_count and rng.random()<self.cfg.historical_retention_weight:
                witness = self.store.witness_at(self.plant,self.contract,int(rng.integers(self.witness_count)))
                spec = TrajectorySpec(**witness['trajectory'])
            else:
                i = int(rng.choice(len(self.retention),p=self.retention))
                if i<len(self.anchors): spec = self.anchors[i]
            if spec is not None:
                result = self._kernel(spec,rng,origin,heading)
                if result is not None: return result,source
            source = 'retention_fallback_base'
        if source=='exploration':
            if rng.random()<self.cfg.exploration_kernel_weight:
                i = int(rng.choice(len(self.regions),p=self.exploration))
                result = self._kernel(self.regions[i],rng,origin,heading)
                if result is not None: return result,source+'_kernel'
                return draw().spec,'exploration_fallback_base'
            paths = [draw() for _ in range(self.cfg.candidate_count)]
            return paths[int(rng.choice(len(paths),p=self.scores([p.spec for p in paths])))].spec,source+'_candidates'
        return draw().spec,source

    def sample_batch(self, seeds, origin=(0.,0.,20.), heading=0.):
        return [self.sample(int(seed),origin,heading) for seed in seeds]

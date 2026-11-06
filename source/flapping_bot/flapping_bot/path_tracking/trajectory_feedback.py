"""Append-only rollout journal and trusted terminal evaluator."""
from copy import deepcopy
import json
import os
from pathlib import Path
import uuid
import numpy as np
from .trajectory_evidence import canonical, identity

class FeedbackStore:
    """Single-writer append-only journal; restart replays outcomes, never migration.

    This is an application trust boundary, not protection against malicious Python
    code or filesystem tampering. Only RolloutEvaluator calls _append_evaluated.
    """
    def __init__(self, directory, *, simulated=False):
        self.simulated = simulated
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / ('simulation_feedback.jsonl' if simulated else 'rollout_feedback.jsonl')
        self._records = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                record = json.loads(line)
                if record['simulated'] != simulated or record['evaluator'] != 'unified_terminal_v1':
                    raise ValueError('Journal provenance mismatch')
                self._records.append(record)
        self._ids = {r['instance_id'] for r in self._records}
        self._by_id = {r['instance_id']: r for r in self._records}

    @property
    def records(self):
        return deepcopy(self._records)

    def _append_evaluated(self, record, evaluator):
        if not isinstance(evaluator, RolloutEvaluator) or evaluator.store is not self:
            raise ValueError('Trusted evaluator required')
        if record['instance_id'] in self._ids:
            raise ValueError('Episode already recorded')
        payload = canonical(record) + '\n'
        with self.path.open('a') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self._records.append(deepcopy(record))
        self._ids.add(record['instance_id'])
        self._by_id[record['instance_id']] = deepcopy(record)

    def query(self, spec, plant, initial, contract):
        key = identity(spec, plant, initial, contract)
        matches = [r for r in self._records if r['identity'] == key and r['outcome'] == 'completed']
        return dict(status=('SIMULATED SUCCESS / NOT PHYSICAL EVIDENCE' if self.simulated else
                            'VERIFIED FEASIBLE') if matches else 'TARGET UNVERIFIED',
                    witnesses=deepcopy(matches))


class RolloutEvaluator:
    """Trusted environment entry: derives outcome from measured terminal predicates.

    Every control step is collected, including command and plant actuator state.
    Manual reset discards unfinished episodes, never manufactures a failure/success.
    """
    def __init__(self, store):
        self.store = store
        self.active = {}

    def begin(self, env_id, spec, plant, initial, contract, policy_version, source):
        if contract.get('kind') != 'unified_terminal_v1' or not policy_version or not plant or not initial:
            raise ValueError('Explicit plant, initial state, policy version and supported contract required')
        required = ('completion_tolerance_m', 'max_path_error_m', 'min_height_m', 'max_height_m',
                    'min_tilt_cos', 'max_body_rate_rad_s', 'max_steps')
        if any(k not in contract for k in required):
            raise ValueError('Incomplete evaluation contract')
        if (not np.isfinite([contract[k] for k in required]).all()
                or contract['completion_tolerance_m'] <= 0 or contract['max_path_error_m'] <= 0
                or contract['min_height_m'] >= contract['max_height_m']
                or contract['max_body_rate_rad_s'] <= 0 or contract['max_steps'] < 1):
            raise ValueError('Invalid terminal thresholds')
        canonical(dict(plant=plant, initial=initial, contract=contract))
        self.active[env_id] = dict(instance_id=uuid.uuid4().hex, trajectory=spec.to_dict(),
            plant=deepcopy(plant), initial=deepcopy(initial), contract=deepcopy(contract),
            policy_version=policy_version, source=source, identity=identity(spec, plant, initial, contract),
            simulated=self.store.simulated, evaluator='unified_terminal_v1', trace=[])

    def observe(self, env_id, *, progress_m, path_error_m, height_m, tilt_cos, body_rate_rad_s,
                finite, step, control):
        r = self.active.get(env_id)
        if r is None:
            return None
        c = r['contract']
        values = [progress_m, path_error_m, height_m, tilt_cos, body_rate_rad_s]
        valid = bool(finite and np.isfinite(values).all())
        failed = (not valid or height_m < c['min_height_m'] or height_m > c['max_height_m']
                  or path_error_m > c['max_path_error_m'] or tilt_cos < c['min_tilt_cos']
                  or body_rate_rad_s > c['max_body_rate_rad_s'])
        completed = (not failed and progress_m >= r['trajectory']['length_m']-c['completion_tolerance_m']
                     and path_error_m <= c['completion_tolerance_m'])
        timeout = step >= c['max_steps']
        expected_step = r['trace'][-1]['step']+1 if r['trace'] else 1
        if not control or step != expected_step:
            raise ValueError('Missing control witness or non-contiguous rollout step')
        sample = dict(zip(('progress_m', 'path_error_m', 'height_m', 'tilt_cos', 'body_rate_rad_s'),
                          [float(v) if np.isfinite(v) else None for v in values]))
        sample.update(finite=valid, step=int(step), control=deepcopy(control))
        r['trace'].append(sample)
        if not (failed or completed or timeout):
            return None
        r.update(outcome='failed' if failed else 'completed' if completed else 'timeout',
                 effective_distance_m=max((x['progress_m'] or 0) for x in r['trace']),
                 tracking_error_m=sample['path_error_m'])
        self.store._append_evaluated(r, self)
        del self.active[env_id]
        return deepcopy(r)

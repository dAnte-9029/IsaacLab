"""Append-only SQLite summaries and separate step evidence, bounded process memory."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import uuid
import numpy as np
from .trajectory_evidence import canonical, identity


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(finite_tree(v) for v in value)
    return isinstance(value, (float, int, np.number)) and bool(np.isfinite(value))


def lossless_json(value):
    """Nonfinite data stays explicitly invalid, never replaced with a finite zero."""
    if isinstance(value, dict): return {k: lossless_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [lossless_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return str(value)
    return value


class FeedbackStore:
    """Single writer; WAL + FULL durability. No traces or full history cached in RAM.

    Legacy JSONL is left untouched; explicit migration is needed to import it.
    Separate episode, steps, terminal and witness tables are insert-only.
    """
    def __init__(self, directory, *, simulated=False):
        self.simulated = simulated
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        legacy = directory / ('simulation_feedback.jsonl' if simulated else 'rollout_feedback.jsonl')
        if legacy.exists():
            raise ValueError('Legacy v1 JSONL present: preserve and explicitly migrate/re-evaluate it; '
                             'v1 sanitized actuator traces cannot be silently certified as v2 evidence')
        self.path = directory / ('simulation_feedback.sqlite3' if simulated else 'rollout_feedback.sqlite3')
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS steps(episode TEXT, step INTEGER, payload TEXT NOT NULL,
                PRIMARY KEY(episode,step));
            CREATE TABLE IF NOT EXISTS terminals(seq INTEGER PRIMARY KEY, id TEXT UNIQUE,
                identity TEXT, outcome TEXT, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS terminal_identity ON terminals(identity,outcome);
            CREATE TABLE IF NOT EXISTS witnesses(context TEXT, rank INTEGER, identity TEXT UNIQUE,
                episode TEXT, PRIMARY KEY(context,rank));
        ''')

    def flush(self):
        self.db.commit()

    def close(self):
        self.flush(); self.db.close()

    def iter_records(self):
        for seq, payload in self.db.execute('SELECT seq,payload FROM terminals ORDER BY seq'):
            yield dict(json.loads(payload), sequence=seq)

    @property
    def records(self):
        # Explicit inspection only; sampler uses streaming recovery.
        return list(self.iter_records())

    def get(self, instance_id):
        row = self.db.execute('SELECT seq,payload FROM terminals WHERE id=?', (instance_id,)).fetchone()
        return dict(json.loads(row[1]), sequence=row[0]) if row else None

    def trace(self, instance_id):
        for payload, in self.db.execute('SELECT payload FROM steps WHERE episode=? ORDER BY step', (instance_id,)):
            yield json.loads(payload)

    def _begin(self, record):
        self.db.execute('INSERT INTO episodes VALUES (?,?)', (record['instance_id'], canonical(record)))

    def _step(self, instance_id, sample):
        self.db.execute('INSERT INTO steps VALUES (?,?,?)', (instance_id, sample['step'], canonical(sample)))

    def _append_evaluated(self, record, evaluator):
        if not isinstance(evaluator, RolloutEvaluator) or evaluator.store is not self:
            raise ValueError('Trusted evaluator required')
        if self.get(record['instance_id']) is not None:
            raise ValueError('Episode already recorded')
        record = deepcopy(record)
        record['trace_ref'] = dict(database=self.path.name, table='steps', episode=record['instance_id'])
        cur = self.db.execute('INSERT INTO terminals(id,identity,outcome,payload) VALUES (?,?,?,?)',
            (record['instance_id'], record['identity'], record['outcome'], canonical(record)))
        if record['outcome'] == 'completed':
            context = canonical([record['plant'], record['contract']])
            rank = self.witness_count(record['plant'], record['contract'])
            self.db.execute('INSERT OR IGNORE INTO witnesses VALUES (?,?,?,?)',
                            (context, rank, record['identity'], record['instance_id']))
        self.flush()
        return dict(record, sequence=cur.lastrowid)

    def witness_count(self, plant, contract):
        return self.db.execute('SELECT COALESCE(MAX(rank)+1,0) FROM witnesses WHERE context=?',
                               (canonical([plant, contract]),)).fetchone()[0]

    def witness_at(self, plant, contract, rank):
        row = self.db.execute('SELECT episode FROM witnesses WHERE context=? AND rank=?',
                             (canonical([plant, contract]), rank)).fetchone()
        return self.get(row[0]) if row else None

    def query(self, spec, plant, initial, contract):
        key = identity(spec, plant, initial, contract)
        matches = [json.loads(r[0]) for r in self.db.execute(
            "SELECT payload FROM terminals WHERE identity=? AND outcome='completed'", (key,))]
        return dict(status=('SIMULATED SUCCESS / NOT PHYSICAL EVIDENCE' if self.simulated else
                            'VERIFIED FEASIBLE') if matches else 'TARGET UNVERIFIED', witnesses=matches)


class RolloutEvaluator:
    """Environment-owned terminal predicates and per-action version provenance.

    State is sampled AFTER the command's decimated physics interval. Each trace
    row binds that command, its actual applied actuator state and resulting body /
    joint state. No checkpoint attribution without an explicit frozen evaluation.
    """
    CONTROL_FIELDS = ('command', 'applied', 'joint_pos', 'joint_vel', 'root_state', 'phase')

    def __init__(self, store):
        self.store = store
        self.active = {}

    def begin(self, env_id, spec, plant, initial, contract, policy_version, source,
              *, evaluation_mode='training', instance_id=None):
        if env_id in self.active:
            raise ValueError('Finish or explicitly abort previous episode before reset')
        if contract.get('kind') != 'unified_terminal_v1' or not plant or not initial:
            raise ValueError('Explicit plant, initial state and supported contract required')
        required = ('completion_tolerance_m', 'max_path_error_m', 'min_height_m', 'max_height_m',
                    'min_tilt_cos', 'max_body_rate_rad_s', 'max_steps')
        if (any(k not in contract for k in required) or not np.isfinite([contract[k] for k in required]).all()
                or contract['completion_tolerance_m'] <= 0 or contract['max_steps'] < 1
                or contract['min_height_m'] >= contract['max_height_m']):
            raise ValueError('Invalid terminal contract')
        if evaluation_mode not in ('training', 'frozen'): raise ValueError('Invalid evaluation mode')
        r = dict(instance_id=instance_id or uuid.uuid4().hex, trajectory=spec.to_dict(),
            plant=deepcopy(plant), initial=deepcopy(initial), contract=deepcopy(contract),
            reset_policy_version=policy_version, evaluation_mode=evaluation_mode, source=source,
            identity=identity(spec, plant, initial, contract), simulated=self.store.simulated,
            evaluator='unified_terminal_v2', timing='command -> decimated physics -> applied/joint/body sample')
        self.store._begin(r)
        self.active[env_id] = dict(record=r, step=0, versions=set(), modes=set(), valid=True,
                                   distance=0., error=None)

    def _finish(self, env_id, outcome):
        state = self.active[env_id]; r = state['record']
        versions = state['versions']; known = versions-{None}
        provenance = 'unknown' if None in versions or not versions else 'single' if len(known)==1 else 'mixed'
        frozen = (provenance == 'single' and state['modes']=={'frozen'} and r['evaluation_mode']=='frozen')
        r.update(outcome=outcome, steps=state['step'], policy_versions=sorted(known),
                 unknown_policy_steps=None in versions, policy_execution=provenance,
                 frozen_checkpoint=next(iter(known)) if frozen else None,
                 effective_distance_m=state['distance'], tracking_error_m=state['error'],
                 control_data_finite=state['valid'])
        result = self.store._append_evaluated(r, self)
        del self.active[env_id]
        return result

    def abort(self, env_id):
        return self._finish(env_id, 'manual_reset') if env_id in self.active else None

    def observe(self, env_id, *, progress_m, path_error_m, height_m, tilt_cos, body_rate_rad_s,
                finite, step, control, policy_version=None, execution_mode='training'):
        state = self.active.get(env_id)
        if state is None: return None
        if step != state['step']+1: raise ValueError('Non-contiguous control interval')
        if execution_mode not in ('training', 'frozen'): raise ValueError('Invalid execution mode')
        c = state['record']['contract']
        values = [progress_m, path_error_m, height_m, tilt_cos, body_rate_rad_s]
        data_valid = all(k in control and finite_tree(control[k]) for k in self.CONTROL_FIELDS)
        valid = bool(finite and np.isfinite(values).all() and data_valid)
        state['valid'] &= valid
        failed = (height_m < c['min_height_m'] or height_m > c['max_height_m']
                  or path_error_m > c['max_path_error_m'] or tilt_cos < c['min_tilt_cos']
                  or body_rate_rad_s > c['max_body_rate_rad_s'])
        completed = (valid and not failed and progress_m >= state['record']['trajectory']['length_m']-c['completion_tolerance_m']
                     and path_error_m <= c['completion_tolerance_m'])
        sample = dict(zip(('progress_m', 'path_error_m', 'height_m', 'tilt_cos', 'body_rate_rad_s'), values))
        sample.update(finite=valid, step=int(step), control=control, policy_version=policy_version,
                      execution_mode=execution_mode)
        self.store._step(state['record']['instance_id'], lossless_json(sample))
        state['step'] = step; state['versions'].add(policy_version); state['modes'].add(execution_mode)
        state['distance'] = max(state['distance'], progress_m if np.isfinite(progress_m) else 0.)
        state['error'] = float(path_error_m) if np.isfinite(path_error_m) else None
        if not valid: return self._finish(env_id, 'invalid_data')
        if failed: return self._finish(env_id, 'failed')
        if completed: return self._finish(env_id, 'completed')
        if step >= c['max_steps']: return self._finish(env_id, 'timeout')
        return None

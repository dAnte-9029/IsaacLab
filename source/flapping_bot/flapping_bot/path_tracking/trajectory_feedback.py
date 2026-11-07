"""Append-only SQLite summaries and separate step evidence, bounded process memory."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3
import uuid
from time import perf_counter
import numpy as np
from .trajectory_evidence import canonical, identity

# Raw actor output is retained before clipping. Remaining fields are sampled
# after the decimated physics interval, including actuator internal state.
CONTROL_FIELDS = ('raw_command', 'command', 'applied', 'actuator_state',
                  'joint_pos', 'joint_vel', 'root_state', 'phase',
                  'path_error_xyz', 'reference', 'projection')


def valid_interval(finite, values, control):
    return bool(finite and finite_tree(values) and
                all(k in control and finite_tree(control[k]) for k in CONTROL_FIELDS))


def terminal_outcome(contract, length, step, values, valid, goal_distance=None):
    progress, error, height, tilt, rate = values
    kind = contract.get('kind', 'unified_terminal_v1')
    if kind not in ('unified_terminal_v1', 'unified_terminal_v2'):
        raise ValueError('Unsupported terminal contract')
    if kind == 'unified_terminal_v2' and (goal_distance is None or not np.isfinite(goal_distance)):
        return 'invalid_data'
    if not valid: return 'invalid_data'
    if (height < contract['min_height_m'] or height > contract['max_height_m']
            or error > contract['max_path_error_m'] or tilt < contract['min_tilt_cos']
            or rate > contract['max_body_rate_rad_s']): return 'failed'
    if (progress >= length-contract['completion_tolerance_m'] and error <= contract['completion_tolerance_m']
            and (kind == 'unified_terminal_v1' or goal_distance <= contract['completion_tolerance_m'])):
        return 'completed'
    if step >= contract['max_steps']: return 'timeout'
    return None


def finite_tree(value):
    if isinstance(value, dict):
        return bool(value) and all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(finite_tree(v) for v in value)
    if isinstance(value, float): return math.isfinite(value)
    return isinstance(value, (int, np.number)) and bool(np.isfinite(value))


def lossless_json(value):
    """Nonfinite data stays explicitly invalid, never replaced with a finite zero."""
    if isinstance(value, dict): return {k: lossless_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [lossless_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    if isinstance(value, np.floating) and not np.isfinite(value): return str(value)
    return value


class FeedbackStore:
    """Single writer; WAL + FULL durability. No traces or full history cached in RAM.

    Legacy JSONL is left untouched; explicit migration is needed to import it.
    Separate episode, steps, terminal and witness tables are insert-only.
    """
    def __init__(self, directory, *, simulated=False, readonly=False):
        self.simulated = simulated
        self.readonly = readonly
        self.flush_seconds = 0.
        self.flush_calls = 0
        directory = Path(directory)
        if not readonly: directory.mkdir(parents=True, exist_ok=True)
        legacy = directory / ('simulation_feedback.jsonl' if simulated else 'rollout_feedback.jsonl')
        if legacy.exists():
            raise ValueError('Legacy v1 JSONL present: preserve and explicitly migrate/re-evaluate it; '
                             'v1 sanitized actuator traces cannot be silently certified as v2 evidence')
        self.path = directory / ('simulation_feedback.sqlite3' if simulated else 'rollout_feedback.sqlite3')
        if readonly:
            self.db = sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro', uri=True)
            self.db.execute('BEGIN')  # one snapshot for validation and sampler receipts
            return
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
        started=perf_counter()
        self.db.commit()
        self.flush_seconds+=perf_counter()-started
        self.flush_calls+=1

    @staticmethod
    def restore_prefix(source, directory, sequence=None):
        """Fork only immutable completed records at a checkpoint boundary.

        Source remains read-only. In-flight physical episodes restart and never
        acquire fabricated terminal outcomes. Later source episodes cannot leak
        into the restored sampler/checkpoint's evidence prefix.
        """
        source = Path(source).resolve()
        target_path = Path(directory)/'rollout_feedback.sqlite3'
        if target_path.exists(): raise FileExistsError(target_path)
        src = sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
        provenance = src.execute('SELECT metadata FROM episodes LIMIT 1').fetchone()
        if (source.name.startswith('simulation_') or
                (provenance is not None and json.loads(provenance[0]).get('simulated'))):
            src.close()
            raise ValueError('Cannot restore synthetic evidence into a real journal')
        target = FeedbackStore(directory)
        try:
            latest = src.execute('SELECT COALESCE(MAX(seq),0) FROM terminals').fetchone()[0]
            cutoff = latest if sequence is None else int(sequence)
            if not 0 <= cutoff <= latest: raise ValueError('Unavailable evidence prefix')
            target.db.executemany('INSERT INTO episodes VALUES (?,?)', src.execute(
                'SELECT e.id,e.metadata FROM episodes e JOIN terminals t ON e.id=t.id WHERE t.seq<=?',(cutoff,)))
            target.db.executemany('INSERT INTO steps VALUES (?,?,?)', src.execute(
                'SELECT s.episode,s.step,s.payload FROM steps s JOIN terminals t ON s.episode=t.id WHERE t.seq<=?',(cutoff,)))
            target.db.executemany('INSERT INTO terminals VALUES (?,?,?,?,?)', src.execute(
                'SELECT seq,id,identity,outcome,payload FROM terminals WHERE seq<=?',(cutoff,)))
            target.db.executemany('INSERT INTO witnesses VALUES (?,?,?,?)', src.execute(
                'SELECT w.context,w.rank,w.identity,w.episode FROM witnesses w JOIN terminals t ON w.episode=t.id WHERE t.seq<=?',(cutoff,)))
            target.flush()
        finally:
            src.close(); target.close()
        return target_path

    def backup(self, destination):
        self.flush()
        destination = Path(destination)
        if destination.exists(): raise FileExistsError(destination)
        target = sqlite3.connect(destination)
        try: self.db.backup(target)
        finally: target.close()

    def close(self):
        if not self.readonly: self.flush()
        self.db.close()

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
    CONTROL_FIELDS = CONTROL_FIELDS

    def __init__(self, store):
        self.store = store
        self.active = {}

    def begin(self, env_id, spec, plant, initial, contract, policy_version, source,
              *, evaluation_mode='training', instance_id=None, smoke_fixture=False):
        if env_id in self.active:
            raise ValueError('Finish or explicitly abort previous episode before reset')
        if contract.get('kind') not in ('unified_terminal_v1', 'unified_terminal_v2') or not plant or not initial:
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
            evaluator=('unified_terminal_v4' if contract['kind']=='unified_terminal_v2' else 'unified_terminal_v3'), timing='command -> decimated physics -> applied/joint/body sample')
        if smoke_fixture: r['smoke_fixture'] = True
        from .unified_trajectory import Trajectory
        endpoint = Trajectory(spec).query(spec.length_m)['position'] if contract['kind']=='unified_terminal_v2' else None
        self.store._begin(r)
        self.active[env_id] = dict(record=r, endpoint=endpoint, step=0, versions=set(), modes=set(), valid=True,
                                   distance=0., error=None, error_squares=[0.,0.], error_max=[0.,0.])

    def _finish(self, env_id, outcome):
        state = self.active[env_id]; r = state['record']
        versions = state['versions']; known = versions-{None}
        provenance = 'unknown' if None in versions or not versions else 'single' if len(known)==1 else 'mixed'
        frozen = (provenance == 'single' and state['modes']=={'frozen'} and r['evaluation_mode']=='frozen')
        r.update(outcome=outcome, steps=state['step'], policy_versions=sorted(known),
                 unknown_policy_steps=None in versions, policy_execution=provenance,
                 frozen_checkpoint=next(iter(known)) if frozen else None,
                 effective_distance_m=state['distance'], tracking_error_m=state['error'],
                 control_data_finite=state['valid'],
                 horizontal_rmse_m=float(np.sqrt(state['error_squares'][0]/max(1,state['step']))),
                 vertical_rmse_m=float(np.sqrt(state['error_squares'][1]/max(1,state['step']))),
                 horizontal_max_m=state['error_max'][0], vertical_max_m=state['error_max'][1])
        result = self.store._append_evaluated(r, self)
        del self.active[env_id]
        return result

    def abort(self, env_id):
        return self._finish(env_id, 'manual_reset') if env_id in self.active else None

    def observe(self, env_id, *, progress_m, path_error_m, height_m, tilt_cos, body_rate_rad_s,
                finite, step, control, policy_version=None, execution_mode='training'):
        if env_id not in self.active: return None
        values = [progress_m, path_error_m, height_m, tilt_cos, body_rate_rad_s]
        valid = valid_interval(finite, values, control)
        return self._observe_interval(env_id, values, valid, step, control, policy_version, execution_mode)

    def observe_intervals(self, values, controls, steps, versions, modes, contract, length, goal_distances=None):
        """Validate original interval data once, then share decisions with the environment.

        This entry accepts raw data, never caller-certified success flags. Strict
        SQLite replay continues to use observe() and independently revalidates it.
        """
        if not all(len(x) == len(values) for x in (controls, steps, versions, modes)):
            raise ValueError("Mismatched interval batch lengths")
        outcomes, records = [], []
        for i, (value, control, step) in enumerate(zip(values, controls, steps)):
            valid = valid_interval(True, value, control)
            outcome = terminal_outcome(contract, length, step, value, valid,
                None if goal_distances is None else goal_distances[i])
            record = self._observe_interval(i, value, valid, step, control, versions[i], modes[i])
            if record is not None:
                assert record['outcome'] == outcome
                records.append((i, record))
            outcomes.append(outcome)
        return outcomes, records

    def _observe_interval(self, env_id, values, valid, step, control, policy_version, execution_mode):
        progress_m, path_error_m, height_m, tilt_cos, body_rate_rad_s = values
        state = self.active.get(env_id)
        if state is None: return None
        if step != state['step']+1: raise ValueError('Non-contiguous control interval')
        if execution_mode not in ('training', 'frozen'): raise ValueError('Invalid execution mode')
        c = state['record']['contract']
        values = [progress_m, path_error_m, height_m, tilt_cos, body_rate_rad_s]
        state['valid'] &= valid
        # Independently derive the fixed goal from the saved trajectory and raw body state.
        # Never trust a logged distance, moving reference, or completed flag.
        goal_distance = (float(np.linalg.norm(np.asarray(control['root_state'][:3])-state['endpoint']))
                         if c['kind']=='unified_terminal_v2' else None)
        outcome = terminal_outcome(c, state['record']['trajectory']['length_m'], step, values, valid, goal_distance)
        sample = dict(zip(('progress_m', 'path_error_m', 'height_m', 'tilt_cos', 'body_rate_rad_s'), values))
        sample.update(finite=valid, step=int(step), control=control, policy_version=policy_version,
                      execution_mode=execution_mode)
        self.store._step(state['record']['instance_id'], lossless_json(sample))
        error = control.get('path_error_xyz')
        if error is not None and np.isfinite(error).all():
            for j, value in enumerate([float(np.linalg.norm(error[:2])), abs(float(error[2]))]):
                state['error_squares'][j] += value*value
                state['error_max'][j] = max(state['error_max'][j], value)
        state['step'] = step; state['versions'].add(policy_version); state['modes'].add(execution_mode)
        state['distance'] = max(state['distance'], progress_m if np.isfinite(progress_m) else 0.)
        state['error'] = float(path_error_m) if np.isfinite(path_error_m) else None
        if outcome: return self._finish(env_id, outcome)
        return None

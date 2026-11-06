"""Common experiment settings; no legacy curriculum or physics implementation."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
from copy import deepcopy
import numpy as np
from flapping_bot.path_tracking.unified_trajectory import SamplingConfig, Trajectory, sample_trajectory
from flapping_bot.path_tracking.trajectory_evidence_geometry import features

# Saved best finite-horizon candidates; speed,nose-up pitch,freq,symmetric,rudder,differential.
CANDIDATES = {
    'L3D': (9.656337493598675,13.971707797921283,3.043205868980297,.873482079293165,1.6310188600911404,-2.2054931298788647),
    'I3D': (9.250000002999998,8.555604979240297,3.743224887001552,-3.5762117485716587,.24606506496647979,-1.9969305732982099),
}
POLICY = dict(actor_hidden_dims=[256,256,256],critic_hidden_dims=[256,256,256],activation='elu',init_noise_std=1.)
PPO_CONFIG = dict(num_learning_epochs=5,num_mini_batches=4,clip_param=.2,gamma=.99,lam=.95,
    value_loss_coef=1.,entropy_coef=.01,learning_rate=3e-4,max_grad_norm=1.,use_clipped_value_loss=True,
    schedule='fixed',desired_kl=None,normalize_advantage_per_mini_batch=False)

EVALUATION_PROTOCOL = 'frozen_full_batch_v3'
EVALUATION_LAYOUT = dict(kind='isaac_grid_z_up', spacing_m=10.)


def evaluation_origin(count, index):
    """Pinned default GridCloner layout; runtime origins must agree."""
    rows = int(np.ceil(count/int(np.sqrt(count))))
    cols = int(np.ceil(count/rows))
    spacing = EVALUATION_LAYOUT['spacing_m']
    return [.5*spacing*(rows-1)-(index//cols)*spacing,
            (index % cols)*spacing-.5*spacing*(cols-1), 0.]


def normalized_initial(initial):
    """Only scene translation/index may differ between training and evaluation."""
    result = deepcopy(initial)
    origin = result.pop('env_origin')
    result.pop('scene')
    result['root_state'][:3] = (np.asarray(result['root_state'][:3])-origin).tolist()
    return result


def require_initial(actual, expected):
    from flapping_bot.path_tracking.trajectory_feedback import finite_tree
    if actual.keys() != expected.keys() or not finite_tree(actual):
        raise ValueError('Evaluation initial state fields/finite mismatch')
    for key in expected:
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        if a.shape != b.shape or not np.allclose(a, b, atol=2e-6, rtol=0):
            raise ValueError('Evaluation initial state mismatch: '+key)


def require_evaluation_checkpoint(loaded, *, backend, sampling, suite, budget):
    c = loaded.get('contract', {})
    if (c.get('evaluation_protocol') != EVALUATION_PROTOCOL or c.get('evaluation_layout') != EVALUATION_LAYOUT
            or not all(k in c for k in ('eval_tasks','final_tasks','eval_step_budget','sampling','backend'))
            or not all(k in loaded for k in ('evaluation_initial','evaluation_contract','plant','evaluation_tasks'))
            or loaded['evaluation_contract'].get('validity') != 'unified_interval_v3'):
        raise ValueError('Legacy/incomplete evaluation contract; migration refused')
    if c['backend'] != backend or c['sampling'] != sampling:
        raise ValueError('Checkpoint backend/sampling mismatch')
    if c['eval_step_budget'] != budget:
        raise ValueError('Checkpoint evaluation budget mismatch')
    # Both predeclared suites are immutable even when only one is executed.
    if (c['eval_tasks'] != evaluation_tasks() or c['final_tasks'] != evaluation_tasks('final')
            or loaded['evaluation_tasks'] != c['eval_tasks']):
        raise ValueError('Checkpoint complete task geometry mismatch')
    required_initial = {'root_state','joint_pos','joint_vel','phase','frequency','action','applied_action','wind'}
    if set(loaded['evaluation_initial']) != required_initial:
        raise ValueError('Legacy/incomplete evaluation initial state; migration refused')
    require_initial(loaded['evaluation_initial'], loaded['evaluation_initial'])
    return evaluation_tasks(suite)


def validate_evaluation_batch(store, tasks, plant, contract, initial, version):
    """Read and replay the entire immutable journal before any sampler mutation."""
    from flapping_bot.path_tracking.trajectory_feedback import RolloutEvaluator
    from flapping_bot.path_tracking.trajectory_evidence import canonical
    if store.simulated: raise ValueError('Synthetic journal is not frozen evaluation evidence')
    records = list(store.iter_records())
    if len(records) != len(tasks) or store.db.execute('SELECT COUNT(*) FROM episodes').fetchone()[0] != len(tasks):
        raise ValueError('Incomplete/extra frozen evaluation records')
    seen = set()
    class ReplayStore:
        simulated = False
        def _begin(self, r): pass
        def _step(self, ident, sample): pass
        def _append_evaluated(self, r, evaluator): return deepcopy(r)
    for r in records:
        if (r.get('simulated') is not False or r.get('frozen_checkpoint') != version
                or r.get('policy_execution') != 'single' or r.get('policy_versions') != [version]
                or r.get('reset_policy_version') != version or r.get('evaluation_mode') != 'frozen'
                or r.get('source') != 'fixed_evaluation' or r.get('evaluator') != 'unified_terminal_v3'
                or r['plant'] != plant or r['contract'] != contract):
            raise ValueError('Invalid frozen evidence identity')
        require_initial(normalized_initial(r['initial']), initial)
        scene = r['initial']['scene']; index = scene['env_index']
        if (type(index) is not int or not 0 <= index < len(tasks) or index in seen
                or scene['num_envs'] != len(tasks) or scene['env_spacing_m'] != 10.):
            raise ValueError('Invalid/duplicate evaluation task assignment')
        if r['initial']['env_origin'] != evaluation_origin(len(tasks), index):
            raise ValueError('Evaluation scene origin mismatch')
        expected = deepcopy(tasks[index]['trajectory'])
        expected['origin'] = (np.asarray(expected['origin'])+r['initial']['env_origin']).tolist()
        if r['trajectory'] != expected:
            raise ValueError('Actual complete trajectory differs from assigned task')
        seen.add(index)
        metadata = store.db.execute('SELECT metadata FROM episodes WHERE id=?', (r['instance_id'],)).fetchone()
        if metadata is None or any(r.get(k) != v for k, v in json.loads(metadata[0]).items()):
            raise ValueError('Episode metadata/terminal mismatch')
        replay = RolloutEvaluator(ReplayStore())
        from flapping_bot.path_tracking.unified_trajectory import TrajectorySpec
        replay.begin(0, TrajectorySpec(**r['trajectory']), plant, r['initial'], contract, version,
                     'fixed_evaluation', evaluation_mode='frozen', instance_id=r['instance_id'])
        result = None
        def decode(v):
            if isinstance(v, dict): return {k: decode(x) for k,x in v.items()}
            if isinstance(v, list): return [decode(x) for x in v]
            if isinstance(v, str) and v in ('nan','inf','-inf'): return float(v)
            return v
        for raw in store.trace(r['instance_id']):
            if result is not None: raise ValueError('Actions after terminal outcome')
            sample = decode(raw)
            if sample['policy_version'] != version or sample['execution_mode'] != 'frozen':
                raise ValueError('Non-frozen action provenance')
            result = replay.observe(0, **sample)
        if result is None or any(canonical(r.get(k)) != canonical(v) for k,v in result.items()):
            raise ValueError('Terminal outcome/trace mismatch')
    return records


def evaluation_tasks(suite="development"):
    if suite == "final":
        return final_evaluation_tasks()
    if suite != "development":
        raise ValueError("Unknown evaluation suite")
    """Fixed development evaluation suite, not a sealed generalization holdout."""
    simple=sample_trajectory(401,SamplingConfig(axis_active_probability=0)).spec
    sustained=sample_trajectory(402,SamplingConfig(axis_active_probability=1,nonzero_density=1,
        coefficient_change_probability=0,gamma_amplitude=0)).spec
    reverse=sample_trajectory(403,SamplingConfig(required_reversals=('kappa',),gamma_amplitude=0)).spec
    opposite=replace(reverse,kappa_coeff=tuple(-x for x in reverse.kappa_coeff),seed=404)
    coupled=sample_trajectory(405,SamplingConfig(required_reversals=('kappa','gamma'))).spec
    return [dict(name=n,trajectory=s.to_dict()) for n,s in zip(
        ('simple','sustained','reverse_right_first','reverse_left_first','coupled_3d'),
        (simple,sustained,reverse,opposite,coupled))]


def categories(spec):
    f=features(Trajectory(spec))
    path=Trajectory(spec); k=path.k(path.s); g=path.g(path.s)
    reversal=lambda a: bool(a.min() < -.08*max(abs(a)) and a.max() > .08*max(abs(a)))
    return dict(simple=f['kappa']<=.00125 and f['gamma']<=.0075,
        sustained=max(f['kappa_max_active_length_m'],f['gamma_max_active_length_m'])>=30.,
        reverse=reversal(k) or reversal(g),coupled_3d=f['coupled_length_m']>=6.)


def source_identity(root):
    root=Path(root)
    files=['source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py',
           'source/flapping_bot/flapping_bot/physics/control_aero_3d.py',
           'source/flapping_bot/flapping_bot/physics/control_aero_kinematics.py']
    return dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        physical_sources={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
        workspace_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root)))


def final_evaluation_tasks():
    """Predeclared six complete routes; never curriculum feedback."""
    configs = [SamplingConfig(axis_active_probability=0),
        SamplingConfig(axis_active_probability=1, nonzero_density=1, coefficient_change_probability=0, gamma_amplitude=0),
        SamplingConfig(axis_active_probability=1, nonzero_density=1, coefficient_change_probability=0, kappa_amplitude=0),
        SamplingConfig(required_reversals=('kappa',), gamma_amplitude=0),
        SamplingConfig(required_reversals=('gamma',), kappa_amplitude=0),
        SamplingConfig(required_reversals=('kappa','gamma'))]
    names = ['straight','sustained_turn','sustained_climb','lateral_reversal','vertical_reversal','coupled_3d']
    result=[]
    for seed,name,cfg in zip(range(1401,1407),names,configs):
        spec=sample_trajectory(seed,replace(cfg,length_m=80.)).spec
        if name=='sustained_climb': spec=replace(spec,gamma_coeff=tuple(abs(x) for x in spec.gamma_coeff))
        result.append(dict(name=name,trajectory=spec.to_dict()))
    return result


def physical_source_identity(root, assets=None):
    root=Path(root)
    files=[root/'source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py']
    files += sorted((root/'source/flapping_bot/flapping_bot/physics').glob('*.py'))
    native=Path(assets or root)/'source/flapping_bot/native_extensions'
    files += sorted(p for p in native.rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.py','.toml','.so'))
    return dict(identity_schema='physical_v2', physical_sources={str(p.relative_to(root)) if p.is_relative_to(root) else 'external_native/'+str(p.relative_to(native)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def require_compatible_plant(actual, saved):
    if saved.get('identity',{}).get('identity_schema') != 'physical_v2':
        raise ValueError('Legacy physical identity lacks complete actuator/native dependency fingerprints; equivalence cannot be verified. Migration refused.')
    if actual != saved:
        raise ValueError('Physical compatibility identity mismatch')

"""Common experiment settings; no legacy curriculum or physics implementation."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
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

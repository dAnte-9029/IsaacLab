"""Synthetic mechanism experiments. Nothing here certifies a physical flight."""
import ast
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flapping_bot.path_tracking.online_trajectory import (
    OfflineEvidence, FeedbackStore, RolloutEvaluator, OnlineSampler, OnlineConfig, vector)
from flapping_bot.path_tracking.unified_trajectory import (
    Trajectory, TrajectorySpec, SamplingConfig, GeometryLimits, sample_trajectory)
from flapping_bot.path_tracking.unified_trajectory_cache import TrajectoryCache

ROOT = Path(__file__).resolve().parents[1]
RULES = Path(os.environ.get('UNIFIED_RULES', ROOT.parent/'control-oriented-3d-wing-aero/logs/unified_trajectory_phase2_v1/screening_rules_v2.json'))
CONTRACT = dict(kind='unified_terminal_v1', completion_tolerance_m=.5, max_path_error_m=8.,
    min_height_m=1., max_height_m=60., min_tilt_cos=.1, max_body_rate_rad_s=20., max_steps=99)
PLANT = dict(backend='L3D', twist_deg=35., scope='synthetic fixture, NOT simulated plant dynamics')
INITIAL = dict(position=[0., 0., 20.], speed=8.)


def make(tmp_path, **kwargs):
    offline = OfflineEvidence(RULES)
    assert offline.count == 88
    store = FeedbackStore(tmp_path, simulated=True)
    sampler = OnlineSampler(offline, store, PLANT, CONTRACT, 'A', **kwargs)
    return sampler, RolloutEvaluator(store)


def episode(sampler, evaluator, spec, success=False, timeout=False, env_id=0):
    evaluator.begin(env_id, spec, sampler.plant, INITIAL, sampler.contract, sampler.policy_version, 'test')
    if timeout:
        for step in range(1, 99):
            assert evaluator.observe(env_id, progress_m=10., path_error_m=0., height_m=20.,
                tilt_cos=1., body_rate_rad_s=0., finite=True, step=step,
                control=dict(command=[0.]*4, applied=[0.]*4)) is None
    result = evaluator.observe(env_id, progress_m=spec.length_m if success else 10.,
        path_error_m=0. if success or timeout else 9., height_m=20., tilt_cos=1., body_rate_rad_s=0.,
        finite=True, step=99 if timeout else 1, control=dict(command=[0.]*4, applied=[0.]*4))
    sampler.feedback(result)
    return result


def tasks():
    simple = sample_trajectory(1, SamplingConfig(axis_active_probability=0)).spec
    hard = sample_trajectory(7, SamplingConfig(required_reversals=('kappa', 'gamma'))).spec
    unknown = sample_trajectory(12, SamplingConfig(kappa_amplitude=.05, gamma_amplitude=.3)).spec
    return [simple, hard, unknown]


def test_four_scenarios(tmp_path):
    # Narrow finite neighborhoods intentionally separate these deterministic probes.
    cfg = OnlineConfig(neighborhood_radius=.08)
    sampler, evaluator = make(tmp_path, cfg=cfg)
    specs = tasks(); simple, hard, unknown = specs
    report = {'synthetic_only': True, 'offline_count': sampler.offline.count,
              'config': cfg.__dict__, 'columns': ['simple', 'hard', 'unknown']}
    before = sampler.distribution(specs)
    for _ in range(32): episode(sampler, evaluator, hard)
    down = sampler.distribution(specs)
    for _ in range(8): episode(sampler, evaluator, hard, success=True)
    recovered = sampler.distribution(specs)
    assert down[1] < before[1] and recovered[1] > down[1]
    assert sampler.store.query(hard, PLANT, INITIAL, CONTRACT)['witnesses']
    report['failure_then_success'] = [before.tolist(), down.tolist(), recovered.tolist()]
    for _ in range(24): episode(sampler, evaluator, hard, success=True)
    assert sampler.statistics(hard)['rate'] == 1
    stable = sampler.distribution(specs)
    sampler.policy_version = 'B'
    np.testing.assert_array_equal(stable, sampler.distribution(specs))
    for _ in range(32): episode(sampler, evaluator, hard)
    assert sampler.statistics(hard)['rate'] == 0 and sampler.statistics(hard)['mastered']
    assert sampler.store.query(hard, PLANT, INITIAL, CONTRACT)['witnesses']
    forgotten = sampler.distribution(specs)
    assert forgotten[1] > cfg.base_weight/3 + .15
    report['forgetting'] = dict(current_rate=sampler.statistics(hard)['rate'],
        historical_witnesses=len(sampler.store.query(hard, PLANT, INITIAL, CONTRACT)['witnesses']),
        probabilities=forgotten.tolist())
    for _ in range(64): episode(sampler, evaluator, unknown)
    persistent = sampler.distribution(specs)
    assert persistent[2] > cfg.base_weight/3 and sampler.scores(specs)[2] > 0
    report['long_unknown'] = persistent.tolist()
    for _ in range(8): episode(sampler, evaluator, simple, success=True)
    start = sampler.distribution(specs)
    for _ in range(4):
        for _ in range(16): episode(sampler, evaluator, hard)
        for _ in range(16): episode(sampler, evaluator, hard, success=True)
    end = sampler.distribution(specs)
    assert end[0] > cfg.base_weight/3 + .1
    report['complex_pressure'] = [start.tolist(), end.tolist()]
    report['coverage'] = dict(base_component=cfg.base_weight, retention_component=cfg.retention_weight,
                              exploration_component=cfg.exploration_weight,
                              candidate_base_floor=cfg.base_weight/3)
    print('\nMECHANISM_REPORT=' + json.dumps(report, sort_keys=True))
    output = os.environ.get('UNIFIED_REPORT')
    if output:
        with Path(output).open('x') as stream: json.dump(report, stream, indent=2)


def test_offline_contract_and_plant_boundaries(tmp_path):
    offline = OfflineEvidence(RULES)
    source = RULES.read_bytes()
    success = next(r for r in offline.records if r['status']=='VERIFIED FEASIBLE')
    spec = TrajectorySpec(**success['trajectory'])
    args = [spec, success['backend'], success['physical_conditions'], success['evaluation_contract']]
    assert offline.query(*args)['status'] == 'VERIFIED FEASIBLE'
    args[-1] = CONTRACT
    assert offline.query(*args)['status'] == 'RELATED EVIDENCE / TARGET UNVERIFIED'
    args[1] = 'DeLaurier'
    assert not offline.query(*args)['witnesses']
    args[1] = success['backend']; args[2] = dict(success['physical_conditions'], twist_deg=0)
    assert not offline.query(*args)['witnesses']
    args[2] = success['physical_conditions']; args[0] = tasks()[0]
    assert not offline.query(*args)['witnesses']
    copy = offline.records; copy.clear()
    assert offline.count == len(offline.records) == 88
    assert RULES.read_bytes() == source
    # Offline heuristic can bias same-backend geometry even with incompatible target length/contract.
    sampler, _ = make(tmp_path)
    on = sampler.scores(tasks())
    sampler.cfg = replace(sampler.cfg, offline_bonus=0)
    assert np.any(on > sampler.scores(tasks()))


def test_journal_provenance_restart_and_terminal_rules(tmp_path):
    sampler, evaluator = make(tmp_path)
    spec = tasks()[1]
    record = episode(sampler, evaluator, spec, success=True)
    episode(sampler, evaluator, spec)
    episode(sampler, evaluator, spec, timeout=True)
    assert [r['outcome'] for r in sampler.store.records] == ['completed', 'failed', 'timeout']
    real = FeedbackStore(tmp_path)
    assert not real.query(spec, PLANT, INITIAL, CONTRACT)['witnesses']
    with pytest.raises(ValueError): sampler.store._append_evaluated(record, None)
    with pytest.raises(TypeError): evaluator.observe(0, status='VERIFIED FEASIBLE')
    with pytest.raises(ValueError): sampler.feedback(dict(record, simulated=False))
    restarted, _ = make(tmp_path)
    assert len(restarted.store.records) == 3
    OfflineEvidence(RULES); OfflineEvidence(RULES)
    assert len(FeedbackStore(tmp_path, simulated=True).records) == 3
    assert not sampler.store.query(spec, dict(PLANT, twist_deg=0), INITIAL, CONTRACT)['witnesses']
    assert not sampler.store.query(spec, PLANT, INITIAL, dict(CONTRACT, max_steps=100))['witnesses']
    evaluator.begin(0, spec, PLANT, INITIAL, CONTRACT, 'A', 'test')
    result = evaluator.observe(0, progress_m=60, path_error_m=0, height_m=0, tilt_cos=1,
        body_rate_rad_s=0, finite=True, step=1, control={'command':[0]*4})
    assert result['outcome'] == 'failed'  # failure wins over completion/timeout


def test_sampling_is_continuous_and_partial_episode_reset(tmp_path):
    sampler, evaluator = make(tmp_path, cfg=OnlineConfig(candidate_count=3))
    simple, hard, _ = tasks()
    for _ in range(32): episode(sampler, evaluator, hard, success=True)
    draws = [sampler.sample(i) for i in range(80)]
    sources = [source for _, source in draws]
    assert all(name in sources for name in ['base', 'retention', 'exploration'])
    assert len({json.dumps(spec.to_dict(), sort_keys=True) for spec, _ in draws}) == 80
    for i in [0, 1]: evaluator.begin(i, hard, PLANT, INITIAL, CONTRACT, 'A', 'test')
    protected = deepcopy(evaluator.active[1])
    evaluator.begin(0, simple, PLANT, INITIAL, CONTRACT, 'B', 'base')
    assert evaluator.active[1] == protected
    assert evaluator.active[0]['policy_version'] == 'B'


def test_features_do_not_resolve_maneuver_order():
    # Signed curvature reversal: equal peaks/duration/net altitude/coupling,
    # opposite turn ordering. Never merge their exact evidence identities.
    spec = tasks()[1]
    spec = replace(spec, gamma_coeff=tuple(0. for _ in spec.gamma_coeff))
    reversed_sign = replace(spec, kappa_coeff=tuple(-x for x in spec.kappa_coeff))
    np.testing.assert_allclose(vector(spec), vector(reversed_sign), atol=1e-12)
    assert spec.to_dict() != reversed_sign.to_dict()


def test_real_environment_methods_without_physx(tmp_path):
    # Execute the actual reset/done method bodies with a tensor plant fixture.
    # This checks wiring, not dynamics or physical flight success.
    source = ROOT/'source/flapping_bot/flapping_bot/direct/flapping_bot/unified_trajectory_env.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='UnifiedTrajectoryEnv')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_reset_idx', '_get_dones')]
    cls.bases = [ast.Name(id='Parent', ctx=ast.Load())]
    class Parent:
        def _reset_idx(self, ids): pass
    ns = dict(Parent=Parent, np=np, torch=torch, Trajectory=Trajectory, replace=replace, sample_trajectory=sample_trajectory)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), 'exec'), ns)
    env = ns['UnifiedTrajectoryEnv']()
    env.cfg = SimpleNamespace(seed=7, sampling=SamplingConfig(), geometry_limits=GeometryLimits(),
        terminate_ground_height=1., max_path_error_m=8., terminate_tilt_deg=80., max_body_rate_rad_s=20.,
        completion_tolerance_m=.5)
    env.device='cpu'; env.num_envs=2
    data = SimpleNamespace(root_pos_w=torch.tensor([[0.,0.,20.],[30.,0.,20.]]),
        root_state_w=torch.zeros(2,13), root_ang_vel_b=torch.zeros(2,3),
        projected_gravity_b=torch.tensor([[0.,0.,-1.]]*2), joint_pos=torch.zeros(2,5), joint_vel=torch.zeros(2,5))
    env._robot=SimpleNamespace(data=data); env.scene=SimpleNamespace(env_origins=torch.zeros(2,3))
    env._straight_line_heading_rad=torch.zeros(2); env._phase=torch.zeros(2); env._freq=torch.ones(2)*4
    env._actions=torch.zeros(2,4); env._act_cmd=torch.zeros(2,4); env._wind_w=torch.zeros(2,3)
    env.task_sampling_sources=[None]*2
    env.paths=[None]*2; env.task_records=[None]*2; env.reset_counts=np.zeros(2,dtype=int)
    env.path_cache=TrajectoryCache(2,60.)
    for key in ['progress','high_water','previous_high_water']: setattr(env,key,torch.zeros(2))
    env.previous_action=torch.zeros(2,4); env.previous_position=torch.zeros(2,3)
    env.episode_length_buf=torch.ones(2,dtype=torch.long); env.max_episode_length=100
    env.extras={}; env._update_reference=lambda: None
    env.online_sampler=None; env.rollout_evaluator=None
    env._reset_idx(torch.tensor([0,1]))
    seed=int(np.random.SeedSequence([7,0,0]).generate_state(1)[0])
    assert env.paths[0].spec.to_dict()==sample_trajectory(seed).spec.to_dict()
    env.online_sampler, env.rollout_evaluator=make(tmp_path)
    env._reset_idx(torch.tensor([0,1]))
    protected=deepcopy(env.rollout_evaluator.active[1]); old=env.path_cache.values[1].clone()
    env._reset_idx(torch.tensor([0]))
    assert env.rollout_evaluator.active[1]==protected
    torch.testing.assert_close(env.path_cache.values[1],old)
    env.reference={'error':torch.zeros(2,3)}; env.progress[0]=60
    terminated, timeout=env._get_dones()
    assert terminated.tolist()==[True,False] and not timeout.any()
    assert len(env.online_sampler.store.records)==1
    assert env.online_sampler.store.records[0]['outcome']=='completed'
    assert 1 in env.rollout_evaluator.active


def test_feedback_batches_rate_limit_and_no_forgery(tmp_path):
    sampler, evaluator = make(tmp_path, cfg=OnlineConfig(neighborhood_radius=.08))
    specs = tasks()
    before = sampler.distribution(specs)
    record = None
    for _ in range(7): record = episode(sampler, evaluator, specs[1], success=True)
    np.testing.assert_array_equal(before, sampler.distribution(specs))
    with pytest.raises(ValueError): sampler.feedback(record)
    with pytest.raises(ValueError): sampler.feedback(dict(record, instance_id='fabricated'))
    episode(sampler, evaluator, specs[1], success=True)
    after = sampler.distribution(specs)
    assert .5*np.abs(after-before).sum() <= (.25+.35)*.2
    evaluator.begin(3, specs[0], PLANT, INITIAL, CONTRACT, 'A', 'test')
    with pytest.raises(ValueError):
        evaluator.observe(3, progress_m=60., path_error_m=0., height_m=20., tilt_cos=1.,
                          body_rate_rad_s=0., finite=True, step=2, control={'command':[0]*4})

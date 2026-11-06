"""Isolated orchestration and tensor fixtures: never PPO, PhysX or real evaluation."""
import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from flapping_bot.analysis.unified_ppo_contract import (
    evaluation_tasks, EVALUATION_PROTOCOL, EVALUATION_LAYOUT, evaluation_origin, require_evaluation_checkpoint,
    normalized_initial, validate_evaluation_batch)
from flapping_bot.path_tracking.trajectory_feedback import (
    FeedbackStore, RolloutEvaluator, valid_interval, terminal_outcome)
from flapping_bot.path_tracking.trajectory_evidence import canonical
from flapping_bot.path_tracking.unified_trajectory import TrajectorySpec
from test_online_trajectory import CONTROL, CONTRACT
from test_unified_pretraining import function, ROOT


def checkpoint():
    return dict(contract=dict(evaluation_protocol=EVALUATION_PROTOCOL,evaluation_layout=EVALUATION_LAYOUT,backend='L3D',sampling='fixed',
        eval_tasks=evaluation_tasks(),final_tasks=evaluation_tasks('final'),eval_step_budget=10),
        evaluation_initial=dict(root_state=[0.]*13,joint_pos=[0.]*5,joint_vel=[0.]*5,
            phase=0.,frequency=3.,action=[0.]*4,applied_action=[0.]*4,wind=[0.]*3),evaluation_contract=dict(CONTRACT,validity='unified_interval_v3'),
        plant={},evaluation_tasks=evaluation_tasks())


@pytest.mark.parametrize('suite', ['development','final'])
@pytest.mark.parametrize('change', ['sampling','geometry','termination','budget','legacy',None])
def test_checkpoint_binding(suite, change):
    loaded=checkpoint()
    if change=='sampling': loaded['contract']['sampling']='adaptive'
    if change=='geometry': loaded['contract']['final_tasks'][0]['trajectory']['length_m']+=1
    if change=='termination': loaded['evaluation_contract'].pop('validity')
    if change=='budget': loaded['contract']['eval_step_budget']+=1
    if change=='legacy': loaded['contract'].pop('evaluation_protocol')
    kwargs=dict(backend='L3D',sampling='fixed',suite=suite,budget=10)
    if change:
        with pytest.raises(ValueError): require_evaluation_checkpoint(loaded,**kwargs)
    else: assert require_evaluation_checkpoint(loaded,**kwargs)==evaluation_tasks(suite)


def make_batch(tmp_path, change=None):
    # Temporary test-only SQLite, no production journal is opened for writing.
    store=FeedbackStore(tmp_path/'fixture')
    evaluator=RolloutEvaluator(store)
    tasks=evaluation_tasks(); initial=dict(root_state=[0.,0.,20.]+[0.]*10,
        joint_pos=[0.]*5,joint_vel=[0.]*5,phase=0.,frequency=3.,action=[0.]*4,
        applied_action=[0.]*4,wind=[0.]*3,env_origin=[0.,0.,0.],
        scene=dict(num_envs=5,env_index=0,env_spacing_m=10.))
    template=normalized_initial(initial)
    count=4 if change=='missing' else 6 if change=='extra' else 5
    for i in range(count):
        idx=min(i,4); spec=deepcopy(tasks[idx]['trajectory']); start=deepcopy(initial)
        start['scene']['env_index']=idx
        # Include world translation, not just zero-origin fixtures.
        start['env_origin']=evaluation_origin(5,idx)
        start['root_state'][:3]=(np.asarray(start['root_state'][:3])+start['env_origin']).tolist()
        spec['origin']=(np.asarray(spec['origin'])+start['env_origin']).tolist()
        if i==4:
            if change=='replacement': spec['kappa_coeff'][0]+=.001
            if change=='duplicate':
                spec=deepcopy(tasks[3]['trajectory'])
                spec['origin']=(np.asarray(spec['origin'])+start['env_origin']).tolist()
            if change=='initial': start['frequency']=4.
            if change=='origin':
                start['env_origin'][0]+=1.; start['root_state'][0]+=1.; spec['origin'][0]+=1.
        version='other' if change=='checkpoint' and i==4 else 'v3'
        evaluator.begin(i,TrajectorySpec(**spec),{'backend':'L3D'},start,CONTRACT,version,
            'fixed_evaluation',evaluation_mode='frozen')
        evaluator.observe(i,progress_m=1.,path_error_m=9.,height_m=20.,tilt_cos=1.,body_rate_rad_s=0.,
            finite=True,step=1,control=CONTROL,policy_version=version,execution_mode='frozen')
    if change=='outcome':
        r=store.records[-1]; r.pop('sequence'); r['outcome']='completed'
        store.db.execute('UPDATE terminals SET payload=? WHERE id=?',(canonical(r),r['instance_id']))
    store.flush(); store.close()
    return template


@pytest.mark.parametrize('change', ['replacement','duplicate','missing','extra','checkpoint','initial','origin','outcome',None])
def test_complete_batch_atomic_receipt(tmp_path,change):
    template=make_batch(tmp_path,change)
    calls=[]
    class Sampler:
        def state_dict(self): return deepcopy(calls)
        def load_state_dict(self,s): calls[:]=s
        def feedback_evaluation(self,r,s): calls.append(r['identity'])
        def _update(self): calls.append('update')
    env=NS(online_sampler=Sampler(),plant_identity={'backend':'L3D'},evaluation_contract=CONTRACT,
        evaluation_initial=template)
    state=dict(eval_env_steps=0,eval_active_steps=0,evaluations=[],pending_evaluation=True)
    before=deepcopy(state)
    summary=dict(passed=True,eval_suite='development',version='v3',environment_steps=50,
        trajectories=evaluation_tasks(),active_episode_steps=5,eval_step_budget=10,
        policy_unchanged=True,optimizer_updates=0)
    # Route the receiver's read-only request to the temporary fixture directory.
    def open_store(path,**kw):
        assert kw=={'readonly':True}
        return FeedbackStore(tmp_path/'fixture',**kw)
    receive=function('receive_evaluation',FeedbackStore=open_store,validate_evaluation_batch=validate_evaluation_batch)
    if change:
        with pytest.raises(ValueError): receive(env,state,tmp_path,summary,'v3',10)
        assert state==before and calls==[]
    else:
        receive(env,state,tmp_path,summary,'v3',10)
        assert len(calls)==6 and not state['pending_evaluation']
        with pytest.raises(ValueError): receive(env,state,tmp_path,summary,'v3',10)
        assert len(calls)==6


def env_methods():
    tree=ast.parse((ROOT/'source/flapping_bot/flapping_bot/direct/flapping_bot/unified_trajectory_env.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UnifiedTrajectoryEnv')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_get_dones','_interval_control','_update_reference')]
    ns=dict(torch=torch,np=np,valid_interval=valid_interval,terminal_outcome=terminal_outcome)
    exec(compile(ast.Module(body=methods,type_ignores=[]),'<tensor environment>','exec'),ns)
    return ns


def tensor_env(tmp_path):
    data=NS(root_state_w=torch.zeros(1,13),root_pos_w=torch.tensor([[0.,0.,20.]]),
        projected_gravity_b=torch.tensor([[0.,0.,-1.]]),root_ang_vel_b=torch.zeros(1,3),
        joint_pos=torch.zeros(1,5),joint_vel=torch.zeros(1,5))
    store=FeedbackStore(tmp_path,simulated=True); evaluator=RolloutEvaluator(store)
    evaluator.begin(0,TrajectorySpec(**evaluation_tasks()[0]['trajectory']),{'test':1},{'test':1},
        CONTRACT,'v3','test')
    env=NS(_robot=NS(data=data),cfg=NS(sampling=NS(length_m=60.),completion_tolerance_m=.5,
        terminate_ground_height=1.,geometry_limits=NS(max_height=60.),max_path_error_m=8.,
        terminate_tilt_deg=np.rad2deg(np.arccos(.1)),max_body_rate_rad_s=20.,evaluation_tasks=None),
        num_envs=1,device='cpu',scene=NS(env_origins=torch.zeros(1,3)),
        progress=torch.tensor([60.]),high_water=torch.tensor([60.]),previous_position=torch.tensor([[0.,0.,20.]]),
        reference={'error':torch.zeros(1,3),'position':torch.tensor([[0.,0.,20.]])},
        _raw_actions=torch.zeros(1,4),_actions=torch.zeros(1,4),_act_cmd=torch.zeros(1,4),
        _act_lpf=torch.zeros(1,4),_freq=torch.ones(1)*3,_phase=torch.zeros(1),
        max_episode_length=5,episode_length_buf=torch.ones(1,dtype=torch.long),extras={},
        rollout_evaluator=evaluator,_action_versions=['v3'],_action_modes=['training'])
    methods=env_methods(); env._update_reference=lambda:None
    env._interval_control=lambda i: methods['_interval_control'](env,i)
    env.records=[]; env.accept_terminal=env.records.append
    return env,methods


@pytest.mark.parametrize('field', ['raw','command','applied','filter','frequency','joint_pos','joint_vel',
    'root','phase','progress','reference','error','previous_position','tilt','rate','height','safe_failure',None])
@pytest.mark.parametrize('bad', [float('nan'),float('inf')])
def test_tensor_and_evidence_end_on_same_interval(tmp_path,field,bad):
    env,methods=tensor_env(tmp_path); d=env._robot.data
    mapping=dict(raw=env._raw_actions,command=env._actions,applied=env._act_cmd,filter=env._act_lpf,
        frequency=env._freq,joint_pos=d.joint_pos,joint_vel=d.joint_vel,root=d.root_state_w,phase=env._phase,
        progress=env.progress,reference=env.reference['position'],error=env.reference['error'],
        previous_position=env.previous_position,tilt=d.projected_gravity_b,rate=d.root_ang_vel_b,height=d.root_pos_w)
    if field in mapping: mapping[field].fill_(bad)
    if field=='safe_failure': d.root_pos_w[:,2]=0.
    done,timeout=methods['_get_dones'](env)
    expected='completed' if field is None else 'failed' if field=='safe_failure' else 'invalid_data'
    assert done.item() and not timeout.item() and env.records[0]['outcome']==expected
    assert env.invalid_data.item()==(expected=='invalid_data')
    assert env.failed.item()==(expected=='failed')
    methods['_get_dones'](env)  # evaluator cannot append a second terminal
    assert len(env.records)==1 and len(env.rollout_evaluator.store.records)==1
    assert len(list(env.rollout_evaluator.store.trace(env.records[0]['instance_id'])))==1
    env.rollout_evaluator.store.close()


@pytest.mark.parametrize('change', ['plant','termination'])
def test_evaluate_rejects_runtime_contract_before_policy(tmp_path,change):
    from flapping_bot.analysis.unified_ppo_contract import require_compatible_plant
    loaded=checkpoint(); loaded['schema']='test'
    loaded['plant']={'identity':{'identity_schema':'physical_v2'}}
    env=NS(plant_identity=deepcopy(loaded['plant']),evaluation_contract=deepcopy(loaded['evaluation_contract']),
        rollout_evaluator=NS(store=NS(close=lambda:None)),close=lambda:None)
    if change=='plant': env.plant_identity['extra']='drift'
    else: env.evaluation_contract['max_steps']+=1
    args=NS(checkpoint=tmp_path/'test.pt',backend='L3D',sampling='fixed',eval_suite='final',
        eval_step_budget=10,threads=1,seed=0,output=tmp_path)
    fake_torch=NS(load=lambda *a,**kw:loaded,set_num_threads=lambda _:None,manual_seed=lambda _:None)
    evaluate=function('evaluate',torch=fake_torch,build_env=lambda *a,**kw:env,
        require_evaluation_checkpoint=require_evaluation_checkpoint,
        require_compatible_plant=require_compatible_plant,json=json)
    with pytest.raises(ValueError,match='mismatch|differs'): evaluate(args,ROOT)


def test_receiver_rolls_back_unexpected_ingestion_error(tmp_path):
    template=make_batch(tmp_path)
    calls=[]
    class Sampler:
        def state_dict(self): return deepcopy(calls)
        def load_state_dict(self,s): calls[:]=s
        def feedback_evaluation(self,r,s):
            calls.append(r['identity'])
            if len(calls)==3: raise RuntimeError('injected interruption')
        def _update(self): raise AssertionError('unreachable')
    env=NS(online_sampler=Sampler(),plant_identity={'backend':'L3D'},evaluation_contract=CONTRACT,
        evaluation_initial=template)
    state=dict(eval_env_steps=0,eval_active_steps=0,evaluations=[],pending_evaluation=True)
    before=deepcopy(state)
    summary=dict(passed=True,eval_suite='development',version='v3',environment_steps=50,
        trajectories=evaluation_tasks(),active_episode_steps=5,eval_step_budget=10,
        policy_unchanged=True,optimizer_updates=0)
    receive=function('receive_evaluation',FeedbackStore=lambda p,**kw:FeedbackStore(tmp_path/'fixture',**kw),
        validate_evaluation_batch=validate_evaluation_batch)
    with pytest.raises(RuntimeError): receive(env,state,tmp_path,summary,'v3',10)
    assert state==before and calls==[]


@pytest.mark.parametrize('field', ['position','previous_progress','previous_position','cache'])
def test_projection_invalid_data_is_not_hidden(tmp_path,field):
    from flapping_bot.path_tracking.unified_trajectory_cache import TrajectoryCache
    from flapping_bot.path_tracking.unified_trajectory import Trajectory
    env,methods=tensor_env(tmp_path)
    env.path_cache=TrajectoryCache(1,60.)
    env.path_cache.update_rows(torch.tensor([0]),[Trajectory(TrajectorySpec(**evaluation_tasks()[0]['trajectory']))])
    env._reference_valid=False; env.preview_distances=torch.tensor([2.,5.,10.])
    env.cfg.projection_margin_m=.5; env.progress[:]=0.
    if field=='position': env._robot.data.root_pos_w[0,0]=float('nan')
    if field=='previous_progress': env.progress[:]=float('nan')
    if field=='previous_position': env.previous_position[0,0]=float('inf')
    if field=='cache': env.path_cache.values[:]=float('nan')
    env._update_reference=lambda:methods['_update_reference'](env)
    done,timeout=methods['_get_dones'](env)
    assert done.item() and not timeout.item() and env.records[0]['outcome']=='invalid_data'
    env.rollout_evaluator.store.close()


def test_invalid_second_interval_has_complete_action_witness(tmp_path):
    env,methods=tensor_env(tmp_path); env.progress[:]=1.
    done,_=methods['_get_dones'](env)
    assert not done.item() and not env.records
    env.episode_length_buf[:]=2; env._act_lpf[:]=float('inf')
    done,_=methods['_get_dones'](env)
    assert done.item() and env.records[0]['outcome']=='invalid_data'
    trace=list(env.rollout_evaluator.store.trace(env.records[0]['instance_id']))
    assert [r['step'] for r in trace]==[1,2]
    assert trace[1]['control']['actuator_state']['_act_lpf']==['inf']*4
    env.rollout_evaluator.store.close()

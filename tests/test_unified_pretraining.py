"""No PPO construction, optimization or PhysX execution."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import os
import random
import numpy as np
import pytest
import torch
from flapping_bot.analysis.unified_ppo_contract import evaluation_tasks,physical_source_identity,require_compatible_plant
from flapping_bot.analysis.unified_ppo_contract import require_checkpoint_use, SMOKE_SCHEMA
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec,screen,GeometryLimits

ROOT=Path(__file__).resolve().parents[1]

def function(name, **globals_):
    tree=ast.parse((ROOT/'source/flapping_bot/flapping_bot/analysis/unified_ppo_training.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    ns=dict(deepcopy=deepcopy,torch=torch,np=np,random=random,os=os,Path=Path,
            SCHEMA='test',SMOKE_SCHEMA=SMOKE_SCHEMA,require_checkpoint_use=require_checkpoint_use,POLICY={},PPO_CONFIG={},evaluation_tasks=evaluation_tasks,
            source_identity=lambda _: {},__file__=str(ROOT/'source/flapping_bot/flapping_bot/analysis/unified_ppo_training.py'))
    if name != 'validate_evaluation_accounting':
        ns['validate_evaluation_accounting'] = function('validate_evaluation_accounting')
    ns.update(globals_)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated orchestration>','exec'),ns)
    return ns[name]

def test_final_suite():
    final=evaluation_tasks('final')
    assert final==evaluation_tasks('final') and len(final)==6
    assert {t['trajectory']['seed'] for t in final}.isdisjoint(t['trajectory']['seed'] for t in evaluation_tasks())
    for t in final:
        assert not screen(Trajectory(TrajectorySpec(**t['trajectory'])),GeometryLimits())
    paths=[Trajectory(TrajectorySpec(**t['trajectory'])) for t in final]
    assert np.max(paths[2].g(paths[2].s))>0 and np.min(paths[2].g(paths[2].s))>=0
    for i,axis in [(3,'k'),(4,'g'),(5,'k'),(5,'g')]:
        v=getattr(paths[i],axis)(paths[i].s); assert v.min()<0<v.max()

def test_physical_identity_ignores_docs_tracks_physics(tmp_path):
    base=tmp_path/'source/flapping_bot/flapping_bot'
    (base/'direct/flapping_bot').mkdir(parents=True); (base/'physics').mkdir()
    (base/'direct/flapping_bot/straight_flight_env.py').write_text('actuator=1')
    p=base/'physics/control_aero_3d.py'; p.write_text('aero=1')
    before=physical_source_identity(tmp_path)
    (tmp_path/'README.md').write_text('new documentation')
    assert physical_source_identity(tmp_path)==before
    p.write_text('aero=2'); assert physical_source_identity(tmp_path)!=before
    with pytest.raises(ValueError,match='Legacy'): require_compatible_plant({}, {'identity':{}})
    with pytest.raises(ValueError,match='mismatch'):
        require_compatible_plant({'identity':before,'dt':2},{'identity':before,'dt':1})

def test_checkpoint_contains_post_feedback_and_prefix(tmp_path):
    store=SimpleNamespace(path=tmp_path/'evidence.sqlite3',flush=lambda:None,
        db=SimpleNamespace(execute=lambda _:SimpleNamespace(fetchone=lambda:(7,))))
    sampler=SimpleNamespace(state_dict=lambda:{'receipts':['evaluation-A']})
    env=SimpleNamespace(rollout_evaluator=SimpleNamespace(store=store),online_sampler=sampler,
        reset_counts=np.array([3]),plant_identity={},evaluation_contract={},evaluation_initial={})
    alg=SimpleNamespace(policy=SimpleNamespace(state_dict=lambda:{'weight':torch.tensor([1.])}),
        optimizer=SimpleNamespace(state_dict=lambda:{'state':{'step':12}}))
    state={'updates':3,'pending_evaluation':False,'evaluations':['evaluation-A']}
    p=function('save_checkpoint')(env,alg,None,state,{},'v3',tmp_path,tag='3_evaluated')
    saved=torch.load(p,weights_only=False)
    assert saved['sampler']['receipts']==['evaluation-A'] and saved['evidence_sequence']==7
    assert saved['optimizer']['state']['step']==12 and saved['state']==state
    assert Path((tmp_path/'latest_checkpoint.txt').read_text().strip())==p

def test_feedback_batch_validation_and_final_isolation(tmp_path):
    records=[dict(steps=1,frozen_checkpoint='v3',policy_execution='single',plant={},contract={}) for _ in range(5)]
    class Store:
        def __init__(self,*args,**kwargs): pass
        def iter_records(self): return iter(records)
        def close(self): pass
    calls=[]
    sampler=SimpleNamespace(feedback_evaluation=lambda r,s:calls.append(r),_update=lambda:calls.append('update'),state_dict=lambda:{})
    env=SimpleNamespace(online_sampler=sampler,plant_identity={},evaluation_contract={},evaluation_initial={})
    state=dict(eval_env_steps=0,eval_active_steps=0,evaluations=[],pending_evaluation=True)
    summary=dict(trajectories=evaluation_tasks(),passed=True,eval_suite='final',version='v3',environment_steps=5,executed_environment_steps=5,reserved_step_budget=10,padding_steps=0,early_stopped=True,active_episode_steps=5,eval_step_budget=10,policy_unchanged=True,optimizer_updates=0)
    receive=function('receive_evaluation',FeedbackStore=Store,validate_evaluation_batch=lambda *a:records)
    with pytest.raises(ValueError): receive(env,state,tmp_path,summary,'v3',10)
    assert not calls and state['pending_evaluation']
    summary['eval_suite']='development'
    receive(env,state,tmp_path,summary,'v3',10)
    assert len(calls)==6 and not state['pending_evaluation'] and state['eval_env_steps']==5

@pytest.mark.parametrize('failure', ['before','during','after',None])
def test_evaluation_transaction_interruptions(failure,tmp_path):
    state=dict(pending_evaluation=True,updates=4)
    saved=[]
    def run(*args):
        if failure in ('before','during'): raise RuntimeError('interrupted evaluation')
        return tmp_path,{}
    def receive(*args): state['pending_evaluation']=False
    def save(*args,**kwargs):
        if failure=='after': raise RuntimeError('interrupted save')
        saved.append(deepcopy(state))
    complete=function('complete_pending_evaluation',run_evaluation_subprocess=run,receive_evaluation=receive,
        save_checkpoint=save,append_json=lambda *a:None)
    args=SimpleNamespace(output=tmp_path,eval_step_budget=10)
    if failure:
        with pytest.raises(RuntimeError): complete(None,None,args,None,state,{},'v4',tmp_path/'pending.pt')
        assert not saved  # on-disk pre-evaluation checkpoint remains pending for replay
    else:
        complete(None,None,args,None,state,{},'v4',tmp_path/'pending.pt')
        assert saved==[dict(pending_evaluation=False,updates=4)]


def test_actual_journal_prefix_is_read_only(tmp_path):
    from flapping_bot.path_tracking.trajectory_feedback import FeedbackStore
    p=ROOT/'logs/unified_ppo_smoke/L3D_adaptive_v2/model_000002_final.pt'
    if not p.exists(): pytest.skip('Historical checkpoint unavailable in this worktree')
    old=torch.load(p,map_location='cpu',weights_only=False)
    source=Path(old['evidence'])
    if not source.exists(): pytest.skip('Historical source journal unavailable')
    # SQLite mode=ro still writes reader locks/read marks to an existing SHM.
    # Historical evidence is a static fixture: read its bytes into a private copy,
    # including committed WAL data, before opening any SQLite connection.
    import shutil
    historical=source
    protected=[p for p in (historical,Path(str(historical)+'-wal'),Path(str(historical)+'-shm')) if p.exists()]
    before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in protected}
    source=tmp_path/historical.name
    shutil.copyfile(historical,source)
    wal=Path(str(historical)+'-wal')
    if wal.exists(): shutil.copyfile(wal,Path(str(source)+'-wal'))
    import sqlite3
    with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as db:
        count=db.execute('SELECT COUNT(*) FROM terminals').fetchone()[0]
        cutoff=db.execute('SELECT MIN(seq) FROM terminals').fetchone()[0]
    target=FeedbackStore.restore_prefix(source,tmp_path/'recovered',cutoff)
    with sqlite3.connect(target) as db:
        assert db.execute('SELECT MAX(seq) FROM terminals').fetchone()[0]==cutoff
    with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as db:
        assert db.execute('SELECT COUNT(*) FROM terminals').fetchone()[0]==count
    assert before=={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in protected}


def test_same_frozen_task_new_instance_is_not_new_evidence(tmp_path):
    from test_online_trajectory import make,episode
    sampler,evaluator=make(tmp_path/'synthetic')
    record=deepcopy(episode(sampler,evaluator,sampler.regions[0],True))
    record.update(simulated=False,frozen_checkpoint='frozen-A',policy_execution='single')
    # In-memory interface fixture only; never write synthetic rows to a real store.
    source=SimpleNamespace(simulated=False,get=lambda ident: record)
    assert sampler.feedback_evaluation(record,source)
    record['instance_id']='new-process-id'
    assert not sampler.feedback_evaluation(record,source)
    saved=sampler.state_dict(); sampler.load_state_dict(saved)
    assert not sampler.feedback_evaluation(record,source)

"""Accounting and single-validation evidence regression gates; no simulator."""
from copy import deepcopy
import pytest
from test_unified_pretraining import function
from test_unified_final_repairs import tensor_env
from flapping_bot.path_tracking.trajectory_feedback import valid_interval, terminal_outcome, CONTROL_FIELDS

@pytest.mark.parametrize('steps,budget', [([126,125,126,124,125],7200), ([10]*5,10), ([2]*6,7200)])
def test_actual_evaluation_accounting(steps,budget):
    validate=function('validate_evaluation_accounting')
    records=[dict(steps=s) for s in steps]
    actual=len(steps)*max(steps)
    summary=dict(environment_steps=actual,executed_environment_steps=actual,
        reserved_step_budget=budget,eval_step_budget=budget,active_episode_steps=sum(steps),
        padding_steps=actual-sum(steps),early_stopped=max(steps)<budget)
    validate(summary,records,budget)
    for key in summary:
        bad=deepcopy(summary)
        bad[key] = not bad[key] if type(bad[key]) is bool else bad[key]+1
        with pytest.raises(ValueError): validate(bad,records,budget)

@pytest.mark.parametrize('field', CONTROL_FIELDS)
@pytest.mark.parametrize('value', [float('nan'),float('inf'),float('-inf')])
def test_batch_does_not_certify_invalid_raw_data(tmp_path,field,value):
    env,methods=tensor_env(tmp_path)
    controls=env._interval_controls()
    controls[0][field]=value
    ev=env.rollout_evaluator
    state=ev.active[0]
    c=state['record']['contract']; length=state['record']['trajectory']['length_m']
    values=[[length,0.,1.,1.,0.]]
    assert not valid_interval(True, values[0],controls[0])
    outcomes,records=ev.observe_intervals(values,controls,[1],['v3'],['frozen'],c,length)
    assert outcomes==['invalid_data'] and records[0][1]['outcome']=='invalid_data'
    assert not records[0][1]['control_data_finite']
    ev.store.close()

def test_batch_size_mismatch_rejected(tmp_path):
    env,_=tensor_env(tmp_path)
    with pytest.raises(ValueError,match='batch'):
        env.rollout_evaluator.observe_intervals([[0]*5],[],[1],['v'],['frozen'],{},60.)
    env.rollout_evaluator.store.close()

@pytest.mark.parametrize('value', [{},[]])
def test_empty_containers_remain_invalid(value):
    from flapping_bot.path_tracking.trajectory_feedback import finite_tree
    assert not finite_tree(value)

@pytest.mark.parametrize('sampling', ['fixed','adaptive'])
def test_pending_evaluated_checkpoint_accounting_roundtrip(tmp_path,sampling):
    import numpy as np
    import torch
    from types import SimpleNamespace as NS
    from flapping_bot.analysis.unified_ppo_contract import evaluation_tasks
    records=[dict(steps=s) for s in [2,3,3,2,3]]
    receipt=dict(passed=True,eval_suite='development',version='v3',trajectories=evaluation_tasks(),
        environment_steps=15,executed_environment_steps=15,active_episode_steps=13,padding_steps=2,
        reserved_step_budget=7200,eval_step_budget=7200,early_stopped=True,policy_unchanged=True,optimizer_updates=0)
    class Store:
        path=tmp_path/'feedback.sqlite3'
        db=NS(execute=lambda _:NS(fetchone=lambda:(5,)))
        def __init__(self,*a,**kw): pass
        def flush(self): pass
        def close(self): pass
    feedback=[]
    sampler=NS(state_dict=lambda:list(feedback),load_state_dict=lambda x:feedback.__setitem__(slice(None),x),
        feedback_evaluation=lambda r,s:feedback.append(r),_update=lambda:None) if sampling=='adaptive' else None
    env=NS(rollout_evaluator=NS(store=Store()),online_sampler=sampler,plant_identity={},
        evaluation_contract={},evaluation_initial={},reset_counts=np.zeros(5,dtype=int))
    alg=NS(policy=NS(state_dict=lambda:{}),optimizer=NS(state_dict=lambda:{}))
    state=dict(updates=3,pending_evaluation=True,evaluations=[],eval_env_steps=50,
        eval_active_steps=40,eval_reserved_env_steps=36000,eval_padding_steps=10)
    save=function('save_checkpoint')
    pending=save(env,alg,None,state,{},'v3',tmp_path,tag='pending')
    resumed=torch.load(pending,weights_only=False)['state']
    receive=function('receive_evaluation',FeedbackStore=Store,validate_evaluation_batch=lambda *a:records)
    complete=function('complete_pending_evaluation',run_evaluation_subprocess=lambda *a:(tmp_path,receipt),
        receive_evaluation=receive,save_checkpoint=save,append_json=lambda *a:None)
    complete(env,alg,NS(output=tmp_path,eval_step_budget=7200),None,resumed,{},'v3',pending)
    evaluated=torch.load(tmp_path/'model_000003_evaluated.pt',weights_only=False)['state']
    assert evaluated['eval_env_steps']==65 and evaluated['eval_active_steps']==53
    assert evaluated['eval_reserved_env_steps']==72000 and evaluated['eval_padding_steps']==12
    assert not evaluated['pending_evaluation'] and evaluated['evaluations']==[str(tmp_path)]
    assert torch.load(pending,weights_only=False)['state']==state
    with pytest.raises(ValueError,match='duplicate'): receive(env,evaluated,tmp_path,receipt,'v3',7200)

@pytest.mark.parametrize('suite', ['development','final'])
@pytest.mark.parametrize('budget', [1,4,7200])
def test_evaluation_loop_stops_only_after_last_terminal(tmp_path,suite,budget):
    from collections import Counter
    from time import perf_counter
    from types import SimpleNamespace as NS
    import numpy as np
    import torch
    from flapping_bot.analysis.unified_ppo_contract import evaluation_tasks
    tasks=evaluation_tasks(suite);n=len(tasks);schedule=[2]*(n-1)+[4]
    records=[];closed=[]
    env=NS(num_envs=n,plant_identity={},evaluation_contract={},evaluation_initial={},
        online_sampler=None,max_episode_length=7200,_evaluation_finished=np.zeros(n,dtype=bool),extras={},
        rollout_evaluator=NS(store=NS(close=lambda:closed.append('store'))),close=lambda:closed.append('env'),
        reset=lambda:({},{}),set_policy_version=lambda *a,**kw:None,steps=0)
    def step(action):
        env.steps+=1
        done=np.array([s==env.steps for s in schedule])
        env._evaluation_finished |= done
        env.extras['unified']={'failed':done}
        for i in np.flatnonzero(done):
            records.append(dict(initial={'scene':{'env_index':int(i)}},outcome='failed',steps=env.steps,
                trajectory=tasks[i]['trajectory'],policy_execution='single',frozen_checkpoint='v3',
                effective_distance_m=0,horizontal_rmse_m=0,vertical_rmse_m=0,horizontal_max_m=0,vertical_max_m=0))
        return {},None,done,np.zeros(n,dtype=bool),{}
    env.step=step
    loaded=dict(schema='test',plant={},evaluation_contract={},evaluation_initial={},version='v3',policy={})
    policy=NS(load_state_dict=lambda _:None,requires_grad_=lambda _:None,eval=lambda:None,act_inference=lambda _:None)
    evaluate=function('evaluate',torch=NS(load=lambda *a,**kw:loaded,set_num_threads=lambda _:None,
        manual_seed=lambda _:None,inference_mode=torch.inference_mode),
        require_checkpoint_use=lambda *a,**kw:None,require_evaluation_checkpoint=lambda *a,**kw:tasks,
        build_env=lambda *a,**kw:env,require_compatible_plant=lambda *a:None,require_initial=lambda *a:None,
        check_initial=lambda *a:{},inspect_evaluation_geometry=lambda *a:{},tensor_obs=lambda *a:{},
        make_policy=lambda *a:policy,policy_digest=lambda _: 'same',perf_counter=perf_counter,
        collect_logs=lambda *a:([],[r for r in records if r['steps']==env.steps]),
        validate_evaluation_batch=lambda *a,**kw:records,Counter=Counter,
        TrajectorySpec=lambda **kw:kw,categories=lambda _:[])
    args=NS(checkpoint=tmp_path/'frozen.pt',backend='L3D',sampling='fixed',eval_suite=suite,
        eval_step_budget=budget,threads=1,seed=0,output=tmp_path)
    if budget<4:
        with pytest.raises(RuntimeError,match='budget exhausted'): evaluate(args,tmp_path)
        assert env.steps==budget
    else:
        summary=evaluate(args,tmp_path)
        assert env.steps==4 and len(records)==n
        function('validate_evaluation_accounting')(summary,records,budget)
    assert closed==['store','env']

def test_finite_tree_scalar_rules_match_original():
    import numpy as np
    from flapping_bot.path_tracking.trajectory_feedback import finite_tree
    values=[True,False,0,1.,-1e300,float('nan'),float('inf'),None,'nan',
        np.float32('nan'),np.float64(1.),np.longdouble('1e400'),np.complex128(1+2j)]
    for v in values:
        expected=isinstance(v,(float,int,np.number)) and bool(np.isfinite(v))
        assert finite_tree(v)==expected
    with pytest.raises(TypeError): finite_tree(10**40)

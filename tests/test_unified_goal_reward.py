"""Strict-goal boundaries and independent journal certification, no PhysX required."""
import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from flapping_bot.path_tracking.trajectory_feedback import FeedbackStore, RolloutEvaluator, terminal_outcome
from flapping_bot.path_tracking.unified_trajectory import Trajectory, sample_trajectory, SamplingConfig
from flapping_bot.analysis.unified_goal_reward import goal_potential, shaping_terms

C=dict(kind='unified_terminal_v2', completion_tolerance_m=.5,max_path_error_m=8.,
       min_height_m=1.,max_height_m=100.,min_tilt_cos=.1,max_body_rate_rad_s=20.,max_steps=2)

@pytest.mark.parametrize('s,e,d,step,height,valid,expected',[
 (59.524,.399,.620187,1,20,True,None),(59.5,.5,.5,1,20,True,'completed'),
 (60,0,.50000001,1,20,True,None),(59.49,0,0,1,20,True,None),
 (60,0,0,2,0,True,'failed'),(60,0,0,2,20,True,'completed'),
 (59,1,1,2,20,True,'timeout'),(60,0,float('nan'),1,20,True,'invalid_data'),
 (60,0,0,1,20,False,'invalid_data')])
def test_terminal(s,e,d,step,height,valid,expected):
 assert terminal_outcome(C,60,step,[s,e,height,1,0],valid,d)==expected
 assert terminal_outcome(C,60,step,[s,e,height,1,0],valid,d)==expected

def test_legacy_dispatch():
 assert terminal_outcome(dict(C,kind='unified_terminal_v1'),60,1,[59.524,.399,20,1,0],True)=='completed'

def test_recovery_and_potential():
 e=torch.tensor([0.,2.,4.,8.],dtype=torch.float64)
 d=torch.tensor([0.,.5,5.,1e12],dtype=torch.float64)
 p=goal_potential(d)
 assert torch.isfinite(p).all() and ((p>=-1)&(p<=0)).all()
 rec,g=shaping_terms(e,p,d,torch.zeros(4,dtype=torch.bool),1/60,.99)
 np.testing.assert_allclose(rec.numpy(),[-.5/60*(np.hypot(x,2)-2)/2 for x in e.numpy()],rtol=1e-15)
 torch.testing.assert_close(g,.99*p-p,atol=0,rtol=0)
 _,g=shaping_terms(e,p,d,torch.ones(4,dtype=torch.bool),1/60,.99)
 torch.testing.assert_close(g,-p,atol=0,rtol=0)

@pytest.mark.parametrize('heading,origin',[(0,(0,0,20)),(1.2,(25,-40,35)),(-2.,(-100,50,20))])
def test_raw_endpoint_witness_and_replay(tmp_path,heading,origin):
 spec=replace(sample_trajectory(1,SamplingConfig(axis_active_probability=0)).spec,heading_rad=heading,origin=origin)
 endpoint=Trajectory(spec).query(60)['position']
 store=FeedbackStore(tmp_path);ev=RolloutEvaluator(store)
 ev.begin(0,spec,{'backend':'L3D'},{'position':origin},C,'frozen','fixed_evaluation',evaluation_mode='frozen')
 control={k:[0.] for k in ev.CONTROL_FIELDS}
 # False logged goal and moving reference must not confer success.
 control.update(root_state=(endpoint+np.array([1,0,0])).tolist()+[1,0,0,0,0,0,0,0,0,0],goal_distance=0.,path_error_xyz=[0,0,0])
 kwargs=dict(progress_m=60.,path_error_m=0.,height_m=20.,tilt_cos=1.,body_rate_rad_s=0.,finite=True,control=control,policy_version='frozen',execution_mode='frozen')
 assert ev.observe(0,step=1,**kwargs) is None
 assert store.witness_count({'backend':'L3D'},C)==0
 control=deepcopy(control);control['root_state'][:3]=endpoint.tolist();kwargs['control']=control
 result=ev.observe(0,step=2,**kwargs)
 assert result['outcome']=='completed' and store.witness_count({'backend':'L3D'},C)==1
 trace=list(store.trace(result['instance_id']))
 replay=RolloutEvaluator(FeedbackStore(tmp_path/'replay'))
 replay.begin(0,spec,{'backend':'L3D'},{'position':origin},C,'frozen','fixed_evaluation',evaluation_mode='frozen')
 assert replay.observe(0,**trace[0]) is None
 assert replay.observe(0,**trace[1])['outcome']=='completed'
 store.close();replay.store.close()

def env_methods():
 path=Path(__file__).resolve().parents[1]/'source/flapping_bot/flapping_bot/direct/flapping_bot/unified_trajectory_env.py'
 node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='UnifiedTrajectoryEnv')
 methods=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='_get_rewards']
 ns=dict(torch=torch,shaping_terms=shaping_terms)
 exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),ns)
 return ns['_get_rewards']

def test_actual_reward_method_baseline_exact_and_shaping():
 f=env_methods();e=torch.tensor([[0.,0,0],[2.,0,0],[0,4.,0],[0,0,8.]])
 x=SimpleNamespace(_update_reference=lambda:None,reference={'error':e},
  cfg=SimpleNamespace(tracking_scale_m=2.,progress_weight=1.,tracking_weight=1.,action_change_weight=.01,
    completion_bonus=10.,failure_penalty=5.,reward_variant='strict_goal',goal_gamma=.99),
  high_water=torch.ones(4),previous_high_water=torch.zeros(4),_actions=torch.ones(4,4)*.1,
  previous_action=torch.zeros(4,4),completed=torch.zeros(4,dtype=torch.bool),failed=torch.zeros(4,dtype=torch.bool),
  invalid_data=torch.zeros(4,dtype=torch.bool),step_dt=1/60,extras={},
  previous_goal_potential=torch.tensor([-.8]*4,dtype=torch.float64),goal_position=torch.zeros(4,3,dtype=torch.float64),
  _robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=e)))
 tracking=torch.exp(-e.square().sum(-1)/4)
 old=tracking+(tracking-1)*(1/60)-.01*torch.mean(x._actions.square(),-1)
 torch.testing.assert_close(f(x),old,rtol=0,atol=0)
 x.cfg.reward_variant='recovery_goal';actual=f(x)
 expected=old+x.reward_terms['recovery'].float()+x.reward_terms['goal'].float()
 assert torch.equal(actual,expected) and torch.equal(actual,f(x))


def test_nonfinite_terminal_shaping_stays_finite():
 rec,goal=shaping_terms(torch.tensor([float('inf'),float('nan')]),torch.tensor([-.5,-.5]),
     torch.tensor([float('inf'),float('nan')]),torch.ones(2,dtype=torch.bool),1/60,.99)
 assert torch.equal(rec,torch.zeros(2)) and torch.equal(goal,torch.tensor([.5,.5]))


def test_existing_ppo_timeout_interface_and_true_done():
 from tensordict import TensorDict
 from rsl_rl.modules import ActorCritic
 from rsl_rl.algorithms import PPO
 from flapping_bot.analysis.unified_ppo_contract import PPO_CONFIG
 obs=TensorDict({'policy':torch.zeros(2,39)},batch_size=[2])
 policy=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},4,actor_hidden_dims=[8],critic_hidden_dims=[8])
 alg=PPO(policy,device='cpu',**PPO_CONFIG);alg.init_storage('rl',2,1,obs,[4])
 with torch.inference_mode():
  alg.act(obs);before=alg.transition.values.clone().squeeze(1)
  reward=torch.tensor([1.,2.]);alg.process_env_step(obs,reward,torch.ones(2,dtype=torch.bool),{'time_outs':torch.tensor([False,True])})
 assert alg.storage.rewards[0,0,0]==reward[0]
 torch.testing.assert_close(alg.storage.rewards[0,1,0],reward[1]+.99*before[1],rtol=0,atol=0)
 assert alg.storage.dones[0].all()


def test_raw_nonfinite_state_cannot_create_witness(tmp_path):
 spec=sample_trajectory(1,SamplingConfig(axis_active_probability=0)).spec
 store=FeedbackStore(tmp_path);ev=RolloutEvaluator(store)
 ev.begin(0,spec,{'backend':'L3D'},{'position':[0,0,20]},C,'frozen','fixed_evaluation',evaluation_mode='frozen')
 control={k:[0.] for k in ev.CONTROL_FIELDS};control['root_state']=[float('nan'),0,20];control['path_error_xyz']=[0,0,0]
 result=ev.observe(0,progress_m=60.,path_error_m=0.,height_m=20.,tilt_cos=1.,body_rate_rad_s=0.,finite=True,
     step=1,control=control,policy_version='frozen',execution_mode='frozen')
 assert result['outcome']=='invalid_data' and store.witness_count({'backend':'L3D'},C)==0
 store.close()


def test_old_checkpoint_evaluation_contract_refused():
 from flapping_bot.analysis.unified_ppo_contract import require_evaluation_checkpoint
 with pytest.raises(ValueError,match='Legacy/incomplete'):
  require_evaluation_checkpoint({'schema':'unified_li_ppo_v1','contract':{'evaluation_protocol':'frozen_full_batch_v3'}},
      backend='L3D',sampling='fixed',suite='development',budget=7200)

"""Bounded first-rollout B replay plus reset/state-boundary validation; no PPO updates."""
import os,json,sqlite3
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from isaaclab.app import AppLauncher
root=Path(__file__).resolve().parents[2]
base=Path(os.environ.get('GOAL_OUTPUT_ROOT',root/'logs/unified_goal_consistent_reward_v1'))
assets=root.parent/'control-oriented-3d-wing-aero'
app=AppLauncher(headless=True,device='cpu',kit_args=f'--ext-folder {assets}/source/flapping_bot/native_extensions --enable omni.flapping_bot.holonomic_constraint').app
from flapping_bot.analysis import unified_ppo_training as runner
from flapping_bot.analysis.unified_goal_reward import goal_potential
old=root.parent/'unified-trim-centered-pilot-v1/logs/unified_trim_centered_pilot_v1/B'
args=SimpleNamespace(backend='L3D',seed=0,num_envs=256,sampling='fixed',eval_suite='development',reward_variant='strict_goal',
 output=base/'physx_check',rules=assets/'logs/unified_trajectory_phase2_v1/screening_rules_v2.json',plant_assets=assets,initialization='trim_local')
args.output.mkdir(exist_ok=False)
torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
env=runner.build_env(args,root)
try:
 obs,_=env.reset();runner.check_initial(env,'L3D')
 snap=torch.load(old/'initialization.pt',weights_only=False)
 assert torch.equal(obs['policy'],snap['observations'])
 obs=runner.tensor_obs(obs,256);states=[]
 for variant in ['strict_goal','recovery_goal']:
  args.reward_variant=variant;dest=args.output/variant;dest.mkdir();args.output=dest
  torch.manual_seed(0);policy=runner.make_policy(obs);runner.initialize_policy(policy,obs,env,args,root)
  states.append(({k:v.clone() for k,v in policy.state_dict().items()},torch.get_rng_state().clone()))
  args.output=dest.parent
 assert all(torch.equal(v,states[1][0][k]) and torch.equal(v,snap['after'][k]) for k,v in states[0][0].items())
 assert torch.equal(states[0][1],states[1][1])
 db=sqlite3.connect((old/'evidence/rollout_feedback.sqlite3').as_uri()+'?mode=ro',uri=True)
 rows=[json.loads(x) for x, in db.execute('select payload from steps order by rowid limit 12288')]
 # Preserve identical command tape; compare actual native PhysX, every actuator field and reference.
 for step in range(48):
  chunk=rows[step*256:(step+1)*256]
  actions=torch.tensor([s['control']['raw_command'] for s in chunk])
  _,reward,done,timeout,_=env.step(actions)
  actual=env._interval_controls()
  for i,s in enumerate(chunk):
   assert actual[i]==s['control'],(step,i,'raw/actuator/PhysX/reference mismatch')
  e2=env.reference['error'].square().sum(-1);track=torch.exp(-e2/4.)
  expected=(env.high_water-env.previous_high_water).clamp_min(0)*track+(track-1)*env.step_dt-.01*(env._actions-env.previous_action).square().mean(-1)+10*env.completed-5*(env.failed|env.invalid_data)
  assert torch.equal(reward,expected)
  assert not done.any() and not timeout.any()
 protected=env.previous_goal_potential[1:].clone();goals=env.goal_position[1:].clone()
 env.previous_goal_potential[0]=1234
 env._reset_idx(torch.tensor([0]))
 assert torch.equal(protected,env.previous_goal_potential[1:]) and torch.equal(goals,env.goal_position[1:])
 expected=goal_potential(torch.linalg.vector_norm(env._robot.data.root_pos_w[0].double()-env.goal_position[0]))
 assert torch.equal(expected,env.previous_goal_potential[0])
 env.cfg.reward_variant='recovery_goal'
 pos=env._robot.data.root_pos_w.clone();goal=env.goal_position.clone()
 env.step(torch.zeros(256,4))
 assert torch.equal(env.previous_goal_potential,goal_potential(torch.linalg.vector_norm(pos.double()-goal,dim=-1)))
 for i in list(env.rollout_evaluator.active):env.rollout_evaluator.abort(i)
 (base/'pretraining_gate.json').write_text(json.dumps(dict(passed=True,physics_actuator_reference_exact_steps=12288,baseline_reward_exact=True,initial_actor_critic_std_rng_exact=True,initial_matches_B=True,partial_reset_no_potential_leak=True,pre_action_potential_exact=True,ppo_updates=0),indent=2))
 print('GOAL_PRETRAINING_PASSED',flush=True)
finally:
 env.rollout_evaluator.store.close();env.close();app.close()

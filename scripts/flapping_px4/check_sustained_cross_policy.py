"""CPU-only arithmetic and deterministic Actor acceptance; no PhysX or PPO."""
import importlib.util
from pathlib import Path
import json
import torch
import numpy as np
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict
from flapping_bot.analysis.unified_ppo_contract import POLICY

root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('intervention',root/'scripts/flapping_px4/run_sustained_cross_policy.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
h=root.parent/'unified-goal-consistent-reward-v1/logs'
paths={25:h/'unified_goal_consistent_reward_v1/G/model_000025.pt',150:h/'unified_longrun_l3d_G_seed0_u026_150/model_000150.pt'}
obs=TensorDict({'policy':torch.from_numpy(np.load(root/'docs/results/unified_g150_diagnostics/fixed_observations.npz')['observations'][:5])},batch_size=[5])
policies={};digests={}
import hashlib
for u,path in paths.items():
 c=torch.load(path,map_location='cpu',weights_only=False)
 p=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},4,**POLICY)
 p.load_state_dict(c['policy']);p.requires_grad_(False);p.eval();policies[u]=p
 digest=hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in p.parameters())).hexdigest()
 assert digest==c['policy_digest'];digests[u]=digest
before=m.rng_state()
with torch.inference_mode():
 a=policies[25].act_inference(obs);d=policies[150].act_inference(obs)
 for g,(f,t) in m.GROUPS.items():
  out=m.select_actions(a,d,g,1)
  assert torch.equal(out[1,:1],{25:a,150:d}[f][1,:1])
  assert torch.equal(out[1,1:],{25:a,150:d}[t][1,1:])
  assert torch.equal(out[[0,2,3,4]],d[[0,2,3,4]])
 assert torch.equal(m.select_actions(a,d,'A',1)[1],a[1])
 assert torch.equal(m.select_actions(a,d,'D',1),d)
m.require_rng(before)
for exact in (False,True):
 try:m.numeric_compare({'root_state':[0.]},{'root_state':[1.]},{},exact=exact)
 except ValueError:pass
 else:raise AssertionError('comparison did not fail closed')
print(json.dumps(dict(passed=True,checkpoint_digests=digests,groups_exact=True,inference_rng_unchanged=True,tolerance_failure_detected=True)))

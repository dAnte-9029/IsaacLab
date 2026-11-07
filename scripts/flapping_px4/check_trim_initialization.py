import os
"""Check paired initialization on captured actual 256-env reset observations, without physics or PPO updates."""
from pathlib import Path
from types import SimpleNamespace
import json
import torch
from tensordict import TensorDict
from flapping_bot.analysis import unified_ppo_training as runner
root=Path(__file__).resolve().parents[2]
out=Path(os.environ.get('TRIM_OUTPUT_ROOT',root/'logs/unified_trim_centered_pilot_v1'))
ck=torch.load(out/'A_reproduction/model_000000.pt',weights_only=False)
snap=torch.load(out/'A_reproduction/initialization.pt',weights_only=False)
c=json.loads((out/'A_reproduction/contract.json').read_text())
obs=TensorDict({'policy':snap['observations']},batch_size=[256])
# Physical reset was checked in the live A reproduction; use its captured contract here.
runner.check_initial=lambda env,backend:c['initial']
env=SimpleNamespace(plant_identity=ck['plant'],num_envs=256,_actions=torch.tensor(ck['evaluation_initial']['action']).repeat(256,1))
policies={}
for mode in ['trim_mean','trim_local']:
 dest=out/('check_'+mode);dest.mkdir()
 args=SimpleNamespace(initialization=mode,backend='L3D',seed=0,output=dest)
 torch.manual_seed(0);policy=runner.make_policy(obs)
 runner.initialize_policy(policy,obs,env,args,root)
 policies[mode]=policy
m,b=policies.values()
assert all(torch.equal(v,b.state_dict()[k]) for k,v in m.state_dict().items() if k!='std')
for policy in policies.values():
 torch.manual_seed(123)
 action=policy.act(obs)
 logp=policy.get_actions_log_prob(action)
 expected=torch.distributions.Normal(policy.action_mean,policy.action_std).log_prob(action).sum(-1)
 torch.testing.assert_close(logp,expected,rtol=0,atol=0)
 assert policy.std.requires_grad
(out/'paired_initialization_check.json').write_text(json.dumps(dict(passed=True,shared_weights_and_bias=True,gaussian_logprob_exact=True,std_trainable=True),indent=2))
print('PAIRED_INITIALIZATION_PASSED')

import os
"""Restore policy/Adam/RNG and evidence prefix without adding a PPO update."""
from pathlib import Path
import json,random
import numpy as np
import torch
from tensordict import TensorDict
from flapping_bot.analysis.unified_ppo_training import make_policy,validate_saved_policy,PPO,PPO_CONFIG
from flapping_bot.path_tracking.trajectory_feedback import FeedbackStore
root=Path(__file__).resolve().parents[2];base=Path(os.environ.get('GOAL_OUTPUT_ROOT',root/'logs/unified_goal_consistent_reward_v1'))
torch.set_num_threads(1)
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
results={}
for group in ['G','R']:
 path=base/group;ck=torch.load(path/'model_000025_final.pt',weights_only=False)
 ini=torch.load(path/'initialization.pt',weights_only=False)
 obs=TensorDict({'policy':ini['observations']},batch_size=[256])
 policy=make_policy(obs);policy.load_state_dict(ck['policy']);validate_saved_policy(ck,policy)
 alg=PPO(policy,device='cpu',**PPO_CONFIG);alg.optimizer.load_state_dict(ck['optimizer'])
 assert equal(alg.optimizer.state_dict(),ck['optimizer'])
 samples=[]
 for _ in range(2):
  torch.set_rng_state(ck['rng']['torch']);np.random.set_state(ck['rng']['numpy']);random.setstate(ck['rng']['python'])
  samples.append((policy.act(obs).detach(),np.random.random(),random.random()))
 assert equal(samples[0],samples[1])
 restored=base/f'{group}_recovery_evidence'
 FeedbackStore.restore_prefix(ck['evidence'],restored,ck['evidence_sequence'])
 store=FeedbackStore(restored,readonly=True)
 assert store.db.execute('pragma integrity_check').fetchone()[0]=='ok'
 assert store.db.execute('select max(seq) from terminals').fetchone()[0]==ck['evidence_sequence']
 store.close()
 for u in [0,5,10,15,20,25]:
  x=torch.load(path/f'model_{u:06d}.pt',weights_only=False)
  assert x['initialization_contract']==ck['initialization_contract']
  policy.load_state_dict(x['policy']);validate_saved_policy(x,policy)
 assert ck['state']['updates']==25 and ck['state']['train_transitions']==307200
 results[group]=dict(policy_and_optimizer_exact=True,rng_sample_exact=True,evidence_prefix_integrity=True,initialization_contract_all_checkpoints=True,updates=25,transitions=307200,adam_steps=max(float(v['step']) for v in alg.optimizer.state.values()),checkpoint=str(path/'model_000025_final.pt'),limitation='Offline restore check; physical episodes reset on resume; no extra PPO update executed')
m=torch.load(base/'G/initialization.pt',weights_only=False);b=torch.load(base/'R/initialization.pt',weights_only=False)
assert equal(m['observations'],b['observations'])
assert all(equal(v,b['after'][k]) for k,v in m['after'].items())
assert m['contract']['bias_offset']==b['contract']['bias_offset']
g=torch.load(base/'G/model_000000.pt',weights_only=False);r=torch.load(base/'R/model_000000.pt',weights_only=False)
assert equal(g['rng']['torch'],r['rng']['torch'])
assert np.array_equal(g['rng']['numpy'][1],r['rng']['numpy'][1]) and g['rng']['numpy'][2:]==r['rng']['numpy'][2:]
assert g['rng']['python']==r['rng']['python']
assert g['plant']==r['plant'] and g['evaluation_contract']==r['evaluation_contract']
cg=dict(g['contract']);cr=dict(r['contract']);assert cg.pop('reward_variant')=='strict_goal' and cr.pop('reward_variant')=='recovery_goal';assert cg==cr
old=root.parent/'unified-trim-centered-pilot-v1/logs/unified_trim_centered_pilot_v1/B'
oldplant=json.loads((old/'contract.json').read_text())['plant']
diffs=[]
def compare(a,b,path=''):
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:compare(a[k],b[k],path+'/'+k)
 elif a!=b:diffs.append(path)
compare(g['plant'],oldplant)
assert set(diffs)=={'/configuration/scene/robot/spawn/asset_path','/configuration/scene/robot/spawn/usd_dir','/configuration/robot/spawn/asset_path','/identity/urdf'}
assert g['plant']['identity']['physical_sources']==oldplant['identity']['physical_sources']
results['historical_B_physics_source_exact']=True
results['historical_B_path_only_configuration_differences']=diffs
results['actual_G_R_shared_initialization']=True
(base/'recovery_check.json').write_text(json.dumps(results,indent=2))
print('RECOVERY_CHECK_PASSED')

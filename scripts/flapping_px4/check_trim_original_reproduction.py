import os
import json,sqlite3
from pathlib import Path
root=Path(__file__).resolve().parents[2]
r=Path(os.environ.get('TRIM_OUTPUT_ROOT',root/'logs/unified_trim_centered_pilot_v1'))
a=root.parent/'unified-rl-performance-phase4/logs/unified_pilot_l3d_fixed_seed0_25u'
def rows(p):
 d=sqlite3.connect(f'file:{p}/evidence/rollout_feedback.sqlite3?mode=ro',uri=True)
 for payload, in d.execute('select payload from steps order by rowid limit 12288'):
  x=json.loads(payload);x.pop('policy_version');yield x
x=list(rows(a));y=list(rows(r/'A_reproduction'))
xreport=json.loads((r/'A_reproduction/training.jsonl').read_text().splitlines()[0]); yreport=json.loads((a/'training.jsonl').read_text().splitlines()[0])
g={k:xreport[k]==yreport[k] for k in ['parameter_delta_l2','losses','reward_mean','episode_outcomes','reset_sources','reset_geometry_counts']}
g['initial_digest']=json.loads((r/'A_reproduction/contract.json').read_text())['initial_policy_digest']==json.loads((a/'contract.json').read_text())['initial_policy_digest']
assert all(g.values())
g['all_12288_step_payloads_exact']=x==y;g['steps']=len(y)
(r/'A_gate.json').write_text(json.dumps(g,indent=2));print(g)
assert x==y and len(y)==12288

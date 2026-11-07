"""Inspect the fixed candidate reward scale on all historical B training rows, read-only."""
import os,json,sqlite3
import numpy as np
from pathlib import Path
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec
root=Path(__file__).resolve().parents[2]
base=Path(os.environ.get('GOAL_OUTPUT_ROOT',root/'logs/unified_goal_consistent_reward_v1'));base.mkdir(parents=True,exist_ok=True)
old=root.parent/'unified-trim-centered-pilot-v1/logs/unified_trim_centered_pilot_v1/B'
db=sqlite3.connect((old/'evidence/rollout_feedback.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
rec=[];goal=[];timeouts=0
for eid,raw in db.execute('select id,metadata from episodes'):
 m=json.loads(raw);path=Trajectory(TrajectorySpec(**m['trajectory']));g=path.query(path.spec.length_m)['position']
 d=np.linalg.norm(np.array(m['initial']['root_state'][:3])-g);prev=-d/(d+5)
 for raw, in db.execute('select payload from steps where episode=? order by step',(eid,)):
  s=json.loads(raw);e=s['path_error_m'];d=np.linalg.norm(np.array(s['control']['root_state'][:3])-g);psi=-d/(d+5)
  rec.append(-.5/60*(np.sqrt(e*e+4)-2)/2);goal.append(.99*psi-prev);prev=psi
  timeouts+=s['step']==m['contract']['max_steps']
def stat(a):return dict(zip(['min','p50','p95','max'],map(float,np.percentile(a,[0,50,95,100]))))
assert len(rec)==307200 and np.isfinite(rec).all() and np.isfinite(goal).all()
r=dict(source=str(old.resolve()),readonly=True,steps=len(rec),recovery=stat(rec),goal_nonabsorbing=stat(goal),
 recovery_at_0_2_4_8=[-.5/60*(np.sqrt(e*e+4)-2)/2 for e in [0,2,4,8]],absorbing_goal_bound=[0,1],timeout_count=int(timeouts))
(base/'reward_scale_preflight.json').write_text(json.dumps(r,indent=2));db.close();print(json.dumps(r,indent=2))

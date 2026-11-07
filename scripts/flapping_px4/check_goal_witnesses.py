"""Independently verify every new training success/witness from raw terminal state."""
import os,json
from pathlib import Path
import numpy as np
from flapping_bot.path_tracking.trajectory_feedback import FeedbackStore,terminal_outcome,valid_interval
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec
root=Path(__file__).resolve().parents[2]
base=Path(os.environ.get('GOAL_OUTPUT_ROOT',root/'logs/unified_goal_consistent_reward_v1'))
result={}
for group in ['G','R']:
 store=FeedbackStore(base/group/'evidence',readonly=True);completed=[]
 for r in store.iter_records():
  if r['outcome']!='completed':continue
  raw=store.db.execute('select payload from steps where episode=? order by step desc limit 1',(r['instance_id'],)).fetchone()[0]
  s=json.loads(raw);path=Trajectory(TrajectorySpec(**r['trajectory']))
  dg=float(np.linalg.norm(path.query(path.spec.length_m)['position']-np.asarray(s['control']['root_state'][:3])))
  v=[s[k] for k in ['progress_m','path_error_m','height_m','tilt_cos','body_rate_rad_s']]
  assert terminal_outcome(r['contract'],path.spec.length_m,s['step'],v,valid_interval(s['finite'],v,s['control']),dg)=='completed'
  completed.append(dict(id=r['instance_id'],distance_m=dg,policy_execution=r['policy_execution']))
 success_ids={r['id'] for r in completed}
 witnesses=[x[0] for x in store.db.execute('select episode from witnesses')]
 assert set(witnesses)<=success_ids
 result[group]=dict(completed_episodes=len(completed),witnesses=len(witnesses),raw_goal_checked=True,episodes=completed)
 store.close()
(base/'training_witness_check.json').write_text(json.dumps(result,indent=2))
print({g:{k:v for k,v in r.items() if k!='episodes'} for g,r in result.items()})

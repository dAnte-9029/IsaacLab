"""Separate existing protocol completion from actual geometric endpoint distance."""
import os,json,sqlite3
from pathlib import Path
import numpy as np
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec
root=Path(__file__).resolve().parents[2];base=Path(os.environ.get('TRIM_OUTPUT_ROOT',root/'logs/unified_trim_centered_pilot_v1'))
manifest=json.loads((base/'verified_evaluations.json').read_text())['evaluations'];results=[]
for run in manifest:
 db=sqlite3.connect(Path(run['evaluation'],'evidence/rollout_feedback.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
 for eid,metadata in db.execute('select id,metadata from episodes'):
  m=json.loads(metadata);trace=[json.loads(p) for p, in db.execute('select payload from steps where episode=? order by step',(eid,))]
  terminal=json.loads(db.execute('select payload from terminals where id=?',(eid,)).fetchone()[0])
  endpoint=Trajectory(TrajectorySpec(**m['trajectory'])).query(m['trajectory']['length_m'])['position']
  preview_distance=m['plant']['configuration']['preview_distances_m'][-1]
  candidates=[s for s in trace if s['progress_m']+preview_distance>=m['trajectory']['length_m']]
  if candidates:
   actual=np.array(candidates[-1]['control']['reference']['preview'][-1]);np.testing.assert_allclose(actual,endpoint,rtol=0,atol=2e-5);endpoint=actual
  distances=[float(np.linalg.norm(endpoint-np.array(s['control']['root_state'][:3]))) for s in trace]
  first_projection=next((i for i,s in enumerate(trace) if s['progress_m']>=m['trajectory']['length_m']-.001),None)
  first_actual=next((i for i,d in enumerate(distances) if d<=.5),None)
  results.append(dict(group=run['group'],update=run['update'],episode=eid,slot=m['initial']['scene']['env_index'],protocol_completed=terminal['outcome']=='completed',endpoint=endpoint.tolist(),terminal_endpoint_error=distances[-1],min_endpoint_error=min(distances),first_projection_endpoint_error=distances[first_projection] if first_projection is not None else None,first_true_endpoint_step=trace[first_actual]['step'] if first_actual is not None else None,true_endpoint_observed=first_actual is not None,strict_completed=terminal['outcome']=='completed' and distances[-1]<=.5))
 db.close()
(base/'endpoint_audit.json').write_text(json.dumps(dict(protocol='progress >= length-0.5 AND error to current projected path point <= 0.5; safety failure precedes completion',strict='Euclidean distance to actual fixed endpoint <= 0.5 m',results=results),indent=2))
print('protocol completions',sum(r['protocol_completed'] for r in results),'strict completions',sum(r['strict_completed'] for r in results),'observed true endpoint entries',sum(r['true_endpoint_observed'] for r in results))

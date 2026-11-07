"""Read-only legacy batch replay and actual B25 counterexample under strict goal."""
import os,json
from pathlib import Path
import numpy as np
from flapping_bot.analysis.unified_ppo_contract import validate_evaluation_batch,evaluation_tasks
from flapping_bot.path_tracking.trajectory_feedback import FeedbackStore,terminal_outcome
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec
root=Path(__file__).resolve().parents[2];base=Path(os.environ.get('GOAL_OUTPUT_ROOT',root/'logs/unified_goal_consistent_reward_v1'))
old=root.parent/'unified-trim-centered-pilot-v1/logs/unified_trim_centered_pilot_v1/B'
s=FeedbackStore(old/'evaluation_000025/evidence',readonly=True)
r=s.records[0]
from flapping_bot.analysis.unified_ppo_contract import normalized_initial
records=validate_evaluation_batch(s,evaluation_tasks(),r['plant'],r['contract'],normalized_initial(r['initial']),r['frozen_checkpoint'])
r=next(r for r in records if r['outcome']=='completed');last=list(s.trace(r['instance_id']))[-1]
p=Trajectory(TrajectorySpec(**r['trajectory']));dg=float(np.linalg.norm(p.query(60)['position']-last['control']['root_state'][:3]))
values=[last[k] for k in ['progress_m','path_error_m','height_m','tilt_cos','body_rate_rad_s']]
new=terminal_outcome(dict(r['contract'],kind='unified_terminal_v2'),60,last['step'],values,True,dg)
assert .6201<dg<.6203 and new is None
(base/'historical_replay_check.json').write_text(json.dumps(dict(passed=True,legacy_batch_replayed=len(records),old_completed=1,new_completed=0,endpoint_distance_m=dg,strict_outcome=new,steps=last['step'],unobserved_after_legacy_termination=True),indent=2))
s.close();print('LEGACY_REPLAY_AND_B25_COUNTEREXAMPLE_PASSED')

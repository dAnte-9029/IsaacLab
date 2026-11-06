"""Summarize measured PPO, evaluation and deterministic replay comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_unified_phase2 import compare

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory
base=ROOT.parent/'unified-rl-performance-phase1/docs/results/unified_rl_performance_phase1'
def load(p):return json.loads(p.read_text())
def metric(p):
    s=load(p/'run/summary.json');t=load(p/'timings.json')['rows']
    def total(suffix):return sum(r['inclusive_s'] for r in t if r['path'].endswith(suffix) and r['path'].startswith('train/env_step_active'))
    return dict(rollout_s=s['rollout_seconds'],update_s=s['update_seconds'],iteration_s=s['train_seconds'],
        transitions_per_s=s['measured_training_transitions_per_s'],get_dones_s=total('/_get_dones'),
        conversion_s=total('/_interval_control')+total('/_interval_controls'),validity_s=total('valid_interval'),
        process_s=load(p/'completion.json')['end_to_end_s'])
summary=dict(training={},evaluation={},replay={})
for b in ('L3D','I3D'):
    for sampling in ('fixed','adaptive'):
        original=base/f'{b}_{sampling}_n256_t1'
        if b=='L3D' and sampling=='fixed': original=base/'L3D_fixed_n256_t1_v2'
        new=out/f'{b}_{sampling}'
        if not (new/'completion.json').exists() or not load(new/'completion.json')['passed']:
            new=out/f'{b}_{sampling}_v2'
        before,after=metric(original),metric(new)
        summary['training'][f'{b}_{sampling}']=dict(baseline=before,optimized=after,
            speedup=before['iteration_s']/after['iteration_s'],get_dones_reduction=1-after['get_dones_s']/before['get_dones_s'],
            consistency=compare(original,new))
    before,after=[out/f'{b}_eval_{k}7200' for k in ('baseline','optimized')]
    def evaluation(p):
        s=load(p/'run/summary.json')
        return dict(loop_and_validation_s=s['wall_s'],process_s=load(p/'completion.json')['end_to_end_s'],
            executed_environment_steps=s['environment_steps'],active_episode_steps=s['active_episode_steps'],
            padding_steps=s['padding_steps'],reserved_step_budget=s.get('reserved_step_budget',s['eval_step_budget']),
            early_stopped=s.get('early_stopped',False))
    eb,ea=evaluation(before),evaluation(after)
    summary['evaluation'][b]=dict(baseline=eb,optimized=ea,wall_saved_s=eb['process_s']-ea['process_s'],
        speedup=eb['process_s']/ea['process_s'],consistency=compare(before,after))
    rb,ro,rp=[out/f'{b}_replay_{k}' for k in ('baseline','optimized','profile')]
    arrays=[np.load(p/'replay.npz') for p in (rb,ro,rp)]
    checks={}
    for key in arrays[0].files:
        checks[key]=dict(shape=list(arrays[0][key].shape),baseline_equal=np.array_equal(arrays[0][key],arrays[1][key]),
            profile_equal=np.array_equal(arrays[1][key],arrays[2][key]),
            max_abs=float(np.max(np.abs(arrays[0][key].astype(float)-arrays[1][key].astype(float)))))
    times=[load(p/'completion.json')['env_step_s'] for p in (rb,ro,rp)]
    summary['replay'][b]=dict(arrays=checks,evidence=compare(rb,ro),plant_identity_equal=load(rb/'identity.json')==load(ro/'identity.json'),
        env_step_s=dict(zip(('baseline','optimized','profile'),times)),instrumentation_difference_percent=100*(times[2]/times[1]-1))
    for data in arrays:data.close()
summary['passed'] = (
    all(v['consistency']['evidence_equal'] and v['consistency']['losses_equal'] and v['consistency']['reward_mean_equal']
        and all(v['consistency']['ppo_summary_equal'].values()) for v in summary['training'].values())
    and all(v['consistency']['evidence_equal'] for v in summary['evaluation'].values())
    and all(v['plant_identity_equal'] and v['evidence']['evidence_equal']
        and all(x['baseline_equal'] and x['profile_equal'] for x in v['arrays'].values()) for v in summary['replay'].values()))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:{name:{key:v for key,v in entry.items() if key in ('speedup','wall_saved_s','get_dones_reduction','instrumentation_difference_percent')} for name,entry in section.items()} for k,section in summary.items() if isinstance(section,dict)},indent=2))
if not summary['passed']: raise SystemExit("Consistency comparison failed; inspect summary.json")

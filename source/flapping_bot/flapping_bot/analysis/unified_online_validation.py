"""Bounded synthetic experiments using actual sample(), never flight evidence."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from flapping_bot.path_tracking.online_trajectory import OfflineEvidence,FeedbackStore,RolloutEvaluator,OnlineSampler,OnlineConfig,vector
from flapping_bot.path_tracking.unified_trajectory import Trajectory,reversal_locations
from flapping_bot.path_tracking.trajectory_evidence_geometry import features

PLANT=dict(backend='L3D',twist_deg=35.,scope='synthetic mechanism fixture only')
INITIAL=dict(position=[0.,0.,20.],speed=8.)
CONTRACT=dict(kind='unified_terminal_v1',completion_tolerance_m=.5,max_path_error_m=8.,min_height_m=1.,
              max_height_m=60.,min_tilt_cos=.1,max_body_rate_rad_s=20.,max_steps=4)
CONTROL=dict(command=[0.]*4,applied=[0.]*4,joint_pos=[0.]*5,joint_vel=[0.]*5,root_state=[0.]*13,phase=0.)


def make(rules,directory,cfg):
    store=FeedbackStore(directory,simulated=True)
    sampler=OnlineSampler(OfflineEvidence(rules),store,PLANT,CONTRACT,'A',cfg)
    return sampler,RolloutEvaluator(store)


def feed(sampler,evaluator,spec,success,version='A',mode='training',unique_initial=None):
    initial=dict(INITIAL,fixture_index=unique_initial) if unique_initial is not None else INITIAL
    evaluator.begin(0,spec,PLANT,initial,CONTRACT,version,'synthetic',evaluation_mode=mode)
    result=evaluator.observe(0,progress_m=60. if success else 10.,path_error_m=0. if success else 9.,
        height_m=20.,tilt_cos=1.,body_rate_rad_s=0.,finite=True,step=1,control=CONTROL,
        policy_version=version,execution_mode=mode)
    sampler.feedback(result)


def coverage(sampler,seeds,hard,known_vectors):
    counts=Counter(); sources=Counter(); vals=[]
    for spec,source in sampler.sample_batch(seeds):
        f=features(Trajectory(spec)); v=vector(spec); vals.append(v.tolist()); sources[source]+=1
        rev=reversal_locations(Trajectory(spec),sampler.sampling)
        counts['simple']+=f['kappa']<=.00125 and f['gamma']<=.0075
        counts['sustained']+=max(f['kappa_max_active_length_m'],f['gamma_max_active_length_m'])>=30.
        counts['complex_reverse']+=bool(rev['kappa'] or rev['gamma'])
        counts['coupled_3d']+=f['coupled_length_m']>=6.
        d=np.linalg.norm((sampler.vectors-v)/sampler.cfg.scales,axis=1)
        counts['hard_region']+=int(np.argmin(d)) in hard
        counts['unknown_region']+=not len(known_vectors) or np.min(np.linalg.norm((known_vectors-v)/sampler.cfg.scales,axis=1))>sampler.cfg.neighborhood_radius
    n=len(seeds)
    return dict(counts={k:int(v) for k,v in counts.items()},frequencies={k:float(v/n) for k,v in counts.items()},
        sources=dict(sources),mean_features=np.mean(vals,axis=0).tolist(),
        p95_abs_features=np.quantile(np.abs(vals),.95,axis=0).tolist(),budget=n,
        probe_scores=sampler.distribution([sampler.regions[i] for i in sorted(hard)[:3]]).tolist())


def run(rules,output,budget=1024,sizes=(0,256,4096)):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    cfg=OnlineConfig(neighborhood_radius=.3)
    sampler,evaluator=make(rules,output/'scenarios',cfg)
    scores=[]
    for i,s in enumerate(sampler.regions):
        rev=reversal_locations(Trajectory(s),sampler.sampling)
        f=features(Trajectory(s))
        scores.append((10*bool(rev['kappa'] or rev['gamma'])+f['coupled_length_m']/60+f['kappa']/.025,i))
    hard={i for _,i in sorted(scores,reverse=True)[:8]}
    simple={i for i,s in enumerate(sampler.regions) if np.max(np.abs(s.kappa_coeff))+np.max(np.abs(s.gamma_coeff))==0}
    seeds=np.arange(10000,10000+budget)
    result=dict(synthetic_only=True,seed_start=10000,config=asdict(cfg),offline_count=sampler.offline.count,
                hard_region_indices=sorted(hard),scenarios={})
    result['scenarios']['initial']=coverage(sampler,seeds,hard,np.empty((0,11)))
    for _ in range(8):
        for i in sorted(hard): feed(sampler,evaluator,sampler.regions[i],False)
    known=sampler.vectors[sorted(hard)]
    result['scenarios']['failed']=coverage(sampler,seeds,hard,known)
    for _ in range(8):
        for i in sorted(hard|simple): feed(sampler,evaluator,sampler.regions[i],True)
    for _ in range(4):
        for i in sorted(hard|simple): feed(sampler,evaluator,sampler.regions[i],True,mode='frozen')
    known=sampler.vectors[sorted(hard|simple)]
    result['scenarios']['recovered']=coverage(sampler,seeds,hard,known)
    sampler.set_policy_version('B')
    # Explicit frozen B evaluation, not mixed training episodes, supplies forgetting.
    for _ in range(8):
        for i in sorted(hard): feed(sampler,evaluator,sampler.regions[i],False,'B','frozen')
        for i in sorted(simple): feed(sampler,evaluator,sampler.regions[i],True,'B','frozen')
    result['scenarios']['forgotten']=coverage(sampler,seeds,hard,known)
    sampler.store.close()
    perf,ev=make(rules,output/'benchmark',cfg); rows=[]; count=0
    for size in sizes:
        while count<size:
            feed(perf,ev,perf.regions[count%cfg.region_count],count%3==0,unique_initial=count)
            count+=1
        single=[]
        for seed in range(20):
            t=perf_counter(); perf.sample(seed); single.append(perf_counter()-t)
        batches=[]
        for _ in range(3):
            t=perf_counter(); perf.sample_batch(range(256)); batches.append(perf_counter()-t)
        writes=[]
        for j in range(32):
            t=perf_counter(); feed(perf,ev,perf.regions[j%cfg.region_count],j%3==0,unique_initial=count)
            writes.append(perf_counter()-t); count+=1
        rows.append(dict(accumulated_before_timing=size,actual_summary_count=perf._cursor,
            sample_median_ms=1000*float(np.median(single)),sample_p95_ms=1000*float(np.quantile(single,.95)),
            batch256_median_ms=1000*float(np.median(batches)),
            feedback_median_ms=1000*float(np.median(writes)),feedback_p95_ms=1000*float(np.quantile(writes,.95)),
            training_window_records=len(perf.capability.training),permanent_unique_witnesses=perf.witness_count))
    result['performance']=rows
    perf.store.close()
    stages=result['scenarios']; values=[stages[s]['frequencies']['hard_region'] for s in stages]
    result['checks']=dict(failure_reduces=values[1]<=values[0],success_restores=values[2]>values[1],
        forgetting_increases=values[3]>values[2],unknown_retained=all(s['counts']['unknown_region']>0 for s in stages.values()),
        simple_retained=all(s['counts']['simple']>0 for s in stages.values()))
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if not all(result['checks'].values()): raise AssertionError(result['checks'])
    return result

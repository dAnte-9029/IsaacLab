"""Exact numerical/evidence acceptance and measured phase-three costs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
from compare_unified_phase3 import compare
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/flapping_bot'))
from flapping_bot.path_tracking.trajectory_feedback import RolloutEvaluator
from flapping_bot.path_tracking.unified_trajectory import TrajectorySpec
from flapping_bot.path_tracking.trajectory_evidence import canonical

def load(p):return json.loads(p.read_text())
def normalize(v):
    if isinstance(v,dict):return {k:normalize(x) for k,x in v.items()}
    if isinstance(v,list):return [normalize(x) for x in v]
    if isinstance(v,str):
        for name in ('unified-rl-performance-phase3','unified-rl-performance-phase2','unified-trajectory-ppo-v1'):
            v=v.replace('/home/zn/IsaacLab/.worktrees/'+name,'<checkout>')
    return v

def identity(left,right):
    def read(p):return load(p/'runtime.json')['plant'] if (p/'runtime.json').exists() else load(p/'identity.json')
    x,y=read(left),read(right)
    sources=[]
    for p,v in ((left,x),(right,y)):
        base=p/'run' if (p/'run').exists() else p
        for name,digest in v['identity']['physical_sources'].items():
            assert hashlib.sha256((base/'source_snapshot/physical'/name).read_bytes()).hexdigest()==digest,name
        sources.append(v['identity']['physical_sources'])
    assert sources[0].keys()==sources[1].keys()
    changed={k:[sources[0][k],sources[1][k]] for k in sources[0] if sources[0][k]!=sources[1][k]}
    allowed={'source/flapping_bot/flapping_bot/physics/'+n for n in ('control_aero_3d.py','control_aero_kinematics.py')}
    assert set(changed)==allowed,changed
    xx,yy=deepcopy(x),deepcopy(y)
    for v in (xx,yy):v['identity'].pop('physical_sources')
    assert normalize(xx)==normalize(yy),'Non-source plant contract changed'
    return dict(source_identity_equal=False,changed_sources=changed,snapshots_verified=True,remaining_plant_contract_equal=True)

def journal(p):
    dbpath=p/('run/evidence/rollout_feedback.sqlite3' if (p/'run').exists() else 'evidence/rollout_feedback.sqlite3')
    db=sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    class Store:
        simulated=False
        def _begin(self,r):pass
        def _step(self,*args):pass
        def _append_evaluated(self,r,e):return deepcopy(r)
    count=steps=0
    for ident,raw in db.execute('SELECT id,metadata FROM episodes ORDER BY rowid'):
        m=json.loads(raw); ev=RolloutEvaluator(Store())
        ev.begin(0,TrajectorySpec(**m['trajectory']),m['plant'],m['initial'],m['contract'],m['reset_policy_version'],m['source'],
                 evaluation_mode=m['evaluation_mode'],instance_id=ident,smoke_fixture=m.get('smoke_fixture',False))
        result=None
        for (payload,) in db.execute('SELECT payload FROM steps WHERE episode=? ORDER BY step',(ident,)):
            assert result is None,'Post-terminal action'
            result=ev.observe(0,**json.loads(payload));steps+=1
        if result is None:result=ev.abort(0)
        terminal=db.execute('SELECT payload FROM terminals WHERE id=?',(ident,)).fetchone()
        assert terminal is not None,'Unterminated episode'
        recorded=json.loads(terminal[0])
        assert all(canonical(recorded.get(k))==canonical(v) for k,v in result.items()),ident
        count+=1
    witnesses=0
    for context,rank,ident,episode in db.execute('SELECT context,rank,identity,episode FROM witnesses'):
        t=json.loads(db.execute('SELECT payload FROM terminals WHERE id=?',(episode,)).fetchone()[0])
        assert t['outcome']=='completed' and t['identity']==ident and context==canonical([t['plant'],t['contract']])
        witnesses+=1
    db.close();return dict(passed=True,episodes=count,steps=steps,witnesses=witnesses)

def metric(p):
    s=load(p/'run/summary.json');rows=load(p/'timings.json')['rows']
    def total(suffix):return sum(r['inclusive_s'] for r in rows if r['path'].startswith('train/env_step_active') and r['path'].endswith(suffix))
    resources=[json.loads(line) for line in (p/'resources.jsonl').read_text().splitlines()]
    cpu=[sum(q['cpu_percent'] for q in r['processes']) for r in resources]
    rss=[sum(q['rss'] for q in r['processes'])/2**20 for r in resources]
    return dict(rollout_s=s['rollout_seconds'],update_s=s['update_seconds'],iteration_s=s['train_seconds'],
        transitions_per_s=s['measured_training_transitions_per_s'],process_s=load(p/'completion.json')['end_to_end_s'],
        aero_s=total('/_compute_control_3d_wrench'),horseshoe_s=total('aero.horseshoe_influence'),
        kinematics_s=total('aero.kinematics'),condition_s=total('linalg.cond'),get_dones_s=total('/_get_dones'),
        peak_rss_mib=s['peak_rss_mib'],sampled_tree_peak_rss_mib=max(rss),sampled_tree_mean_cpu_percent=float(np.mean(cpu)),
        sampled_tree_peak_cpu_percent=max(cpu),kit_start_s=sum(r['inclusive_s'] for r in rows if r['path']=='kit_start'),
        build_env_s=sum(r['inclusive_s'] for r in rows if r['path']=='train/build_env'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory
    old=ROOT.parent/'unified-rl-performance-phase2/docs/results/unified_rl_performance_phase2'
    result=dict(training={},replay={})
    for name in ('I3D_fixed','I3D_adaptive','L3D_fixed'):
        before=old/('L3D_fixed_v2' if name=='L3D_fixed' else name);after=out/name
        c=compare(before,after)
        assert c['evidence_equal'] and c['losses_equal'] and c['reward_mean_equal'] and all(c['ppo_summary_equal'].values()),c
        b,n=metric(before),metric(after)
        result['training'][name]=dict(baseline=b,optimized=n,speedup=b['iteration_s']/n['iteration_s'],
            consistency=c,identity=identity(before,after),journal=journal(after))
    for backend in ('I3D','L3D'):
        before,after=[out/f'{backend}_replay_{kind}' for kind in ('baseline','optimized')]
        checks={}
        with np.load(before/'replay.npz') as b,np.load(after/'replay.npz') as n:
            for k in b.files:
                assert b[k].dtype==n[k].dtype and b[k].shape==n[k].shape and b[k].tobytes()==n[k].tobytes(),k
                checks[k]=dict(shape=list(b[k].shape),equal=True,bitwise_equal=True,max_abs=0.,max_rel=0.,error_quantiles=[0.,0.,0.,0.,0.])
        c=compare(before,after);assert c['evidence_equal'],c
        internal=load(after/'internal_parity.json');assert internal['passed']
        result['replay'][backend]=dict(arrays=checks,internal=internal,evidence=c,identity=identity(before,after),
            baseline_journal=journal(before),optimized_journal=journal(after))
    result['passed']=True
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v['speedup'] for k,v in result['training'].items()},indent=2))

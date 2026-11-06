"""Compare bounded runs without modifying either evidence database."""
import argparse
import json
from pathlib import Path
import sqlite3
from itertools import zip_longest
import numpy as np


def compare(left, right):
    report = dict(left=str(left), right=str(right))
    def load(path): return json.loads(path.read_text())
    ls = load(left/'run/summary.json') if (left/'run/summary.json').exists() else {}
    rs = load(right/'run/summary.json') if (right/'run/summary.json').exists() else {}
    if 'initial_policy_digest' in ls:
        keys = ['initial_policy_digest','final_policy_digest','optimizer_step_before','optimizer_step_after']
        report['ppo_summary_equal'] = {k:ls[k]==rs[k] for k in keys}
        report['losses_equal'] = ls['iterations'][0]['losses']==rs['iterations'][0]['losses']
        report['reward_mean_equal'] = ls['iterations'][0]['reward_mean']==rs['iterations'][0]['reward_mean']
    errors = {}; differences=[]
    def equal(a,b,path):
        if isinstance(a, dict) and isinstance(b,dict):
            if a.keys()!=b.keys(): differences.append(path+':keys'); return
            for k in a: equal(a[k],b[k],path+'.'+k)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b): differences.append(path+':length'); return
            for x,y in zip(a,b): equal(x,y,path+'[]')
        elif a!=b:
            if isinstance(a,(int,float)) and isinstance(b,(int,float)):
                errors[path]=max(errors.get(path,0),abs(a-b))
            elif len(differences)<30: differences.append(path)
    def normalize_paths(value):
        if isinstance(value,dict): return {k:normalize_paths(v) for k,v in value.items()}
        if isinstance(value,list): return [normalize_paths(v) for v in value]
        if isinstance(value,str):
            for name in ('unified-rl-performance-phase1','unified-rl-performance-phase2','unified-trajectory-ppo-v1'):
                value=value.replace('/home/zn/IsaacLab/.worktrees/'+name,'<physical-checkout>')
        return value
    if (left/'runtime.json').exists() and (right/'runtime.json').exists():
        equal(normalize_paths(load(left/'runtime.json')['plant']),normalize_paths(load(right/'runtime.json')['plant']),'plant')
    def database(root):
        p=root/'run/evidence/rollout_feedback.sqlite3'
        return p if p.exists() else root/'evidence/rollout_feedback.sqlite3'
    dbs=[]
    for root in (left,right):
        p=database(root)
        dbs.append(sqlite3.connect(p.resolve().as_uri()+'?mode=ro&immutable=1',uri=True))
    counts=[]
    episodes=[]
    for db in dbs:
        counts.append({table:db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in ('episodes','steps','terminals','witnesses')})
        episodes.append(db.execute('SELECT id,metadata FROM episodes ORDER BY rowid').fetchall())
    report['counts']=counts
    for (li,lm),(ri,rm) in zip(*episodes):
        lm,rm=json.loads(lm),json.loads(rm)
        for k in ('trajectory','initial','contract','source','evaluation_mode','simulated','evaluator','timing'):
            equal(lm[k],rm[k],'metadata.'+k)
        for lrow,rrow in zip_longest(dbs[0].execute('SELECT payload FROM steps WHERE episode=? ORDER BY step',(li,)),dbs[1].execute('SELECT payload FROM steps WHERE episode=? ORDER BY step',(ri,))):
            if lrow is None or rrow is None:
                differences.append('episode trace length');continue
            a,b=json.loads(lrow[0]),json.loads(rrow[0])
            a.pop('policy_version'); b.pop('policy_version')
            equal(a,b,'step')
        lt=dbs[0].execute('SELECT seq,payload FROM terminals WHERE id=?',(li,)).fetchone()
        rt=dbs[1].execute('SELECT seq,payload FROM terminals WHERE id=?',(ri,)).fetchone()
        if (lt is None)!=(rt is None): differences.append('missing terminal');continue
        if lt:
            equal(lt[0],rt[0],'terminal.sequence')
            a,b=json.loads(lt[1]),json.loads(rt[1])
            for k in ('outcome','steps','effective_distance_m','tracking_error_m','control_data_finite','horizontal_rmse_m','vertical_rmse_m','horizontal_max_m','vertical_max_m','policy_execution','unknown_policy_steps'):
                equal(a[k],b[k],'terminal.'+k)
    report.update(numeric_max_abs=errors, nonnumeric_differences=differences,
        evidence_equal=counts[0]==counts[1] and not errors and not differences,
        database_bytes=[database(p).stat().st_size for p in (left,right)],
        excluded_provenance=['run-specific UUID/policy version','absolute checkout and output paths','identity derived from those paths'])
    for db in dbs: db.close()
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare(a.left,a.right);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

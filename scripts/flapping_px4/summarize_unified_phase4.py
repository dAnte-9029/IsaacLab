"""Read-only acceptance and separate bounded throughput denominators."""
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
from summarize_unified_phase3 import journal
from compare_unified_phase3 import compare

def load(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output
    result={'cases':{},'evidence':{},'replay':{}};paths=set();ids=set()
    for casefile in sorted(out.glob('*/case.json')):
        case=load(casefile);assert case['passed'],casefile
        rows=[]
        for w in sorted(casefile.parent.glob('w[0-9]*')):
            if not w.is_dir():continue
            db=w/('run/evidence/rollout_feedback.sqlite3' if (w/'run').exists() else 'evidence/rollout_feedback.sqlite3')
            assert str(db.resolve()) not in paths;paths.add(str(db.resolve()))
            result['evidence'][str(w.relative_to(out))]=journal(w)
            conn=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
            assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            instances={x[0] for x in conn.execute('select id from episodes')};assert not ids.intersection(instances);ids.update(instances)
            conn.close()
            if case['replay']:continue
            runtime=load(w/'runtime.json');summary=load(w/'run/summary.json')
            assert result['evidence'][str(w.relative_to(out))]['steps']==summary['local_train_transitions']
            resources=[json.loads(s) for s in (w/'resources.jsonl').read_text().splitlines()];timings=load(w/'timings.json')['rows']
            perpid={}
            for r in resources:
                for q in r['processes']:perpid[q['pid']]=q
            cpu=sum(q['cpu_times']['user']+q['cpu_times']['system'] for q in perpid.values())
            row=dict(seed=load(w/'parameters.json')['seed'],sampling=summary['sampling'],
                torch_threads=runtime['torch_threads'],torch_interop_threads=runtime['torch_interop_threads'],settings=runtime['settings'],affinity=runtime['affinity'],
                train_s=summary['train_seconds'],rollout_s=summary['rollout_seconds'],update_s=summary['update_seconds'],
                process_s=load(w/'completion.json')['end_to_end_s'],
                child_preclose_s=load(w/'benchmark_only.json')['child_wall_before_kit_close_s'],
                close_wrapper_poll_residual_s=load(w/'completion.json')['end_to_end_s']-load(w/'benchmark_only.json')['child_wall_before_kit_close_s'],cpu_s_sampled_lower_bound=cpu,cpu_ms_per_transition_lower_bound=cpu*1000/summary['local_train_transitions'],
                peak_tree_rss_mib=max(sum(q['rss'] for q in r['processes']) for r in resources)/2**20,
                peak_threads=max(sum(q['threads'] for q in r['processes']) for r in resources),
                io_write_mib=sum(q['io']['write_bytes'] for q in perpid.values())/2**20,
                database_mib=db.stat().st_size/2**20,
                kit_start_s=sum(x['inclusive_s'] for x in timings if x['path']=='kit_start'),
                build_env_s=sum(x['inclusive_s'] for x in timings if x['path'].endswith('/build_env')),
                sqlite_commit_s=sum(x['inclusive_s'] for x in timings if x['path'].endswith('sqlite.commit_including_sync')),
                per_core_mean_percent=np.mean([r['cpu_per_core'] for r in resources],axis=0).tolist(),
                frequencies_mhz_mean=np.mean([[f['current'] for f in r['frequency']] for r in resources],axis=0).tolist(),
                max_cpu_temperature_c=max([t['current'] for r in resources for t in r['temperatures'].get('coretemp',[])] or [0]),
                gpu_samples=[r['gpu']['output'] for r in resources if 'output' in r['gpu']])
            rows.append(row)
        case['resource_workers']=rows;result['cases'][casefile.parent.name]=case
    baseline=out/'replay1/w0'
    if baseline.exists():
        for peer in (out/'replay2').glob('w[0-9]'):
            checks={}
            with np.load(baseline/'replay.npz') as x,np.load(peer/'replay.npz') as y:
                for key in x.files:
                    assert x[key].tobytes()==y[key].tobytes(),(peer,key)
                    checks[key]={'shape':list(x[key].shape),'bitwise_equal':True,'max_abs':0}
            comparison=compare(baseline,peer);assert comparison['evidence_equal'];result['replay'][peer.name]={'arrays':checks,'evidence':comparison}
    # Source identity must be identical, path differs only by worktree; backend identity stays separate.
    baseline_runtime=load(out/'i1/w0/runtime.json')
    for runtimefile in out.glob('*/w*/runtime.json'):
        runtime=load(runtimefile)
        assert runtime['plant']['identity']['physical_sources']==baseline_runtime['plant']['identity']['physical_sources']
        assert runtime['physics_dt']==1/480 and runtime['decimation']==8 and runtime['device']=='cpu'
    result['paired_independent']={}
    for name,left,right in [('fixed_seed0',out/'i1/w0',out/'i2/w0'),('fixed_seed1',out/'seed1/w0',out/'i2/w1'),('adaptive_seed1',out/'adaptive1/w0',out/'mixed2/w1')]:
        if not (left/'run/summary.json').exists() or not (right/'run/summary.json').exists():continue
        x,y=load(left/'run/summary.json'),load(right/'run/summary.json')
        same=all(x[k]==y[k] for k in ('initial_policy_digest','final_policy_digest','optimizer_step_after'))
        loss=all(p['losses']==q['losses'] and p['reward_mean']==q['reward_mean'] for p,q in zip(x['iterations'],y['iterations']))
        result['paired_independent'][name]=dict(policy_equal=same,all_iteration_losses_rewards_equal=loss)
        assert same and loss,name
    if (out/'ipc.json').exists():
        result['ipc']=load(out/'ipc.json');assert result['ipc']['passed']
    result['passed']=True
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{field:v[field] for field in ('wall_s','sum_worker_tps','worker_train_s') if field in v} for k,v in result['cases'].items()},indent=2))
if __name__=='__main__':main()

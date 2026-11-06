"""Summarize completed real benchmarks; no simulator, no source database writes."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def read(p): return json.loads(p.read_text())


def summarize(directory):
    s = read(directory/'run/summary.json')
    p = read(directory/'parameters.json')
    rows = read(directory/'timings.json')['rows']
    by_path = {r['path']: r for r in rows}
    inc = lambda name: by_path.get(name, {}).get('inclusive_s', 0.)
    out = dict(name=directory.name, backend=p['backend'], sampling=p['sampling'], num_envs=p['num_envs'],
               threads=p['threads'], affinity=p['affinity'], profiled=not p['no_profile'],
               end_to_end_s=read(directory/'completion.json')['end_to_end_s'],
               process_wall_s=s['process_wall_s'], kit_start_s=inc('kit_start'))
    mode = 'evaluate' if p.get('evaluate_checkpoint') else 'train'
    out.update(build_env_s=inc(mode+'/build_env'), reset_s=inc(mode+'/reset'),
               checkpoint_s=inc('train/save_checkpoint'),
               checkpoint_bytes=sum(x['bytes'] for x in read(directory/'benchmark_only.json')['checkpoints_deleted']))
    if mode == 'train':
        out.update(iteration_s=s['train_seconds'], rollout_s=s['rollout_seconds'], update_s=s['update_seconds'],
                   env_s=s['environment_step_s'], transitions=s['local_train_transitions'],
                   transitions_per_s=s['measured_training_transitions_per_s'],
                   env_transitions_per_s=s['local_train_transitions']/s['environment_step_s'],
                   update_sample_equivalent_per_s=s['local_train_transitions']/s['update_seconds'],
                   sqlite_bytes=s['sqlite_bytes'], sqlite_flush_all_s=s['sqlite_flush_s'],
                   ppo_update_share=s['update_seconds']/s['train_seconds'],
                   zero_update_speedup_ceiling=s['train_seconds']/(s['train_seconds']-s['update_seconds']))
        prefix = 'train/env_step_active'
        # Immediate children partition environment time; grandchildren are NEVER added again.
        out['environment_partition'] = {r['path'].removeprefix(prefix+'/'): r['inclusive_s'] for r in rows
                                      if r['path'].startswith(prefix+'/') and r['path'].count('/') == 2}
        out['environment_partition']['other'] = s['environment_step_s']-sum(out['environment_partition'].values())
        for name, suffix in [('aero', '/_apply_action/_compute_control_3d_wrench'),
                             ('aero_formula', '/_apply_action/_compute_control_3d_wrench/aero.evaluate'),
                             ('control_to_python', '/_get_dones/_interval_control'),
                             ('evidence_observe', '/_get_dones/evidence.observe')]:
            out[name+'_s'] = inc(prefix+suffix)
        out['sqlite_commit_rollout_s'] = sum(r['inclusive_s'] for r in rows if r['path'].startswith(prefix+'/') and r['path'].endswith('/sqlite.commit_including_sync'))
        out['sqlite_execute_rollout_s'] = sum(r['inclusive_s'] for r in rows if r['path'].startswith(prefix+'/') and r['path'].endswith('/sqlite.execute'))
        out['json_encode_rollout_s'] = sum(r['inclusive_s'] for r in rows if r['path'].startswith(prefix+'/') and r['path'].endswith('/evidence.json_encode'))
        out['dones_excluding_reference_and_sql_s'] = (inc(prefix+'/_get_dones')-inc(prefix+'/_get_dones/_update_reference')
                -out['sqlite_commit_rollout_s']-out['sqlite_execute_rollout_s'])
        out['zero_dones_python_speedup_ceiling'] = s['train_seconds']/(s['train_seconds']-out['dones_excluding_reference_and_sql_s'])
        out['zero_sql_speedup_ceiling'] = s['train_seconds']/(s['train_seconds']-out['sqlite_commit_rollout_s']-out['sqlite_execute_rollout_s'])
        out['physics_calls'] = by_path.get(prefix+'/sim.step', {}).get('calls')
    else:
        last = max(t['steps'] for t in s['tasks'])
        out.update(num_envs=len(s['tasks']), eval_budget=s['eval_step_budget'], num_eval_envs=len(s['tasks']),
                   active_episode_steps=s['active_episode_steps'], total_env_steps=s['environment_steps'],
                   all_finished_cycle=last, total_padding_env_steps=s['padding_steps'],
                   all_finished_padding_cycles=s['eval_step_budget']-last,
                   all_finished_padding_env_steps=(s['eval_step_budget']-last)*len(s['tasks']),
                   active_cycle_env_s=inc('evaluate/env_step_active'),
                   all_finished_padding_env_s=inc('evaluate/env_step_padding'),
                   validation_s=inc('evaluate/validate_evaluation_batch'), evaluation_wall_s=s['wall_s'],
                   timestep_limit_unchanged=True)
    trace = [json.loads(x) for x in (directory/'resources.jsonl').read_text().splitlines()]
    # Main Kit process is the descendant with largest observed RSS (exclude shell).
    candidates = [p for row in trace for p in row['processes'] if 'python' in p['name'].lower()]
    if candidates:
        pid = max(candidates, key=lambda x:x['rss'])['pid']
        samples = [(r['elapsed_s'], p) for r in trace for p in r['processes'] if p['pid'] == pid]
        first, last = samples[0][1], samples[-1][1]
        duration = samples[-1][0]-samples[0][0]
        cpu_s = sum(last['cpu_times'][k]-first['cpu_times'][k] for k in ('user','system'))
        initial_threads={t['id']: t['user_time']+t['system_time'] for t in first['thread_cpu']}
        thread_work=sorted([(t['id'],t['user_time']+t['system_time']-initial_threads.get(t['id'],0)) for t in last['thread_cpu']], key=lambda x:-x[1])
        out['resources'] = dict(samples=len(samples), sampled_duration_s=duration,
            peak_rss_gib=max(p['rss'] for _,p in samples)/2**30,
            peak_threads=max(p['threads'] for _,p in samples), affinity=last['affinity'],
            mean_process_cpu_percent=100*cpu_s/duration if duration else None,
            peak_process_cpu_percent=max(p['cpu_percent'] for _,p in samples),
            top_thread_cpu_s=thread_work[:10], main_thread_cpu_s=next((v for tid,v in thread_work if tid==pid),None),
            sampled_process_cpu_s=cpu_s, io_delta={k:last['io'][k]-first['io'][k] for k in last['io']},
            available_memory_min_gib=min(r['memory']['available'] for r in trace)/2**30,
            per_core_mean_percent=[statistics.mean(r['cpu_per_core'][i] for r in trace) for i in range(len(trace[0]['cpu_per_core']))])
        out['resources']['mean_process_read_mib_s'] = out['resources']['io_delta']['read_bytes']/2**20/duration if duration else None
        out['resources']['mean_process_write_mib_s'] = out['resources']['io_delta']['write_bytes']/2**20/duration if duration else None
        intervals = [(b['io']['write_bytes']-a['io']['write_bytes'])/2**20/(tb-ta)
                     for (ta,a),(tb,b) in zip(samples,samples[1:]) if tb>ta]
        out['resources']['peak_sampled_process_write_mib_s'] = max(intervals,default=None)
    gpu = {}
    for row in trace:
        for line in row['gpu'].get('output','').splitlines():
            try:
                i,u,m,capacity,w = map(float,line.split(','))
                gpu.setdefault(int(i), []).append((u,m,capacity,w))
            except ValueError: pass
    out['gpu'] = {i:dict(util_mean=statistics.mean(x[0] for x in v),util_peak=max(x[0] for x in v),
                        memory_peak_mib=max(x[1] for x in v),capacity_mib=v[0][2]) for i,v in gpu.items()}
    if not out['profiled']:
        for key in ('kit_start_s', 'build_env_s', 'reset_s', 'checkpoint_s', 'aero_s', 'aero_formula_s',
                    'control_to_python_s', 'evidence_observe_s', 'sqlite_commit_rollout_s', 'sqlite_execute_rollout_s',
                    'json_encode_rollout_s', 'dones_excluding_reference_and_sql_s',
                    'zero_dones_python_speedup_ceiling', 'zero_sql_speedup_ceiling'):
            if key in out: out[key] = None
        out['environment_partition'] = {}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    results=[]; excluded=[]
    for d in sorted(a.directory.iterdir()):
        if not d.is_dir(): continue
        if (d/'excluded.json').exists(): excluded.append(dict(name=d.name,**read(d/'excluded.json'))); continue
        if not (d/'completion.json').exists() or not read(d/'completion.json')['passed']: continue
        results.append(summarize(d))
    (a.directory/'summary.json').write_text(json.dumps(dict(runs=results,excluded=excluded),indent=2))
    scalar=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in results]
    if scalar:
        keys=list(dict.fromkeys(k for r in scalar for k in r))
        with (a.directory/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(scalar)
    for r in results: print(r['name'], round(r.get('iteration_s',r.get('evaluation_wall_s',0)),3))


if __name__=='__main__': main()

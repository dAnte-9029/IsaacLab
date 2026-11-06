"""Bounded independent Kit processes; no distributed optimizer. New output only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import signal
ROOT = Path(__file__).resolve().parents[2]

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2))

def affinity_sets(count):
    # Group SMT siblings by sysfs physical core; classify hybrid cores by sibling count.
    groups = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        d = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = (int((d/'physical_package_id').read_text()), int((d/'core_id').read_text()))
        groups.setdefault(key, []).append(cpu)
    p = [v for v in groups.values() if len(v)>1]
    e = [v for v in groups.values() if len(v)==1]
    bins = [[] for _ in range(count)]
    for group in (p,e):
        for i, cpus in enumerate(group): bins[i%count].extend(cpus)
    return [','.join(map(str,sorted(x))) for x in bins]

def run_case(out, name, backend, n, envs, affinity=False, modes=None, seeds=None, export=False, replay=False):
    dest=out/name; dest.mkdir()
    masks=affinity_sets(n) if affinity else [None]*n
    jobs=[]; start=time.perf_counter()
    try:
        for i in range(n):
            worker=dest/f'w{i}'
            args=['bash',str(ROOT/'isaaclab.sh'),'-p',str(ROOT/'scripts/flapping_px4'/('replay_unified_phase3.py' if replay else 'benchmark_unified_performance.py')),
                  '--backend',backend,'--output',str(worker)]
            if not replay:
                args+=['--sampling',(modes or ['fixed']*n)[i],'--seed',str((seeds or list(range(n)))[i]),'--num-envs',str(envs),'--iterations','2','--timeout','180']
                if masks[i]: args+=['--affinity',masks[i]]
                if export: args+=['--export-rollout']
            elif masks[i]: args=['taskset','-c',masks[i]]+args
            log=(dest/f'w{i}.log').open('w')
            proc=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,TERM='xterm'),stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log,worker))
        finished={}
        while len(finished)<n:
            if time.perf_counter()-start>240: raise TimeoutError(name)
            for i,(proc,log,worker) in enumerate(jobs):
                if i not in finished and proc.poll() is not None:
                    finished[i]=time.perf_counter()-start
                    completion=worker/'completion.json'
                    if not completion.exists() or not json.loads(completion.read_text())['passed']:
                        raise RuntimeError(f'Worker failed: {worker}; stop configuration, no retry')
            time.sleep(.1)
        result=dict(passed=True,backend=backend,workers=n,envs_per_worker=envs,affinities=masks,
                    wall_s=time.perf_counter()-start,worker_finish_s=finished,replay=replay)
        if not replay:
            summaries=[json.loads((w/'run/summary.json').read_text()) for _,_,w in jobs]
            result['worker_train_s']=[s['train_seconds'] for s in summaries]
            result['worker_rollout_s']=[s['rollout_seconds'] for s in summaries]
            result['worker_tps']=[s['local_train_transitions']/s['train_seconds'] for s in summaries]
            total=sum(s['local_train_transitions'] for s in summaries)
            result.update(transitions=total,sum_worker_tps=sum(result['worker_tps']),
                          budget_over_slowest_train_tps=total/max(result['worker_train_s']),end_to_end_tps=total/result['wall_s'])
        write(dest/'case.json',result); print(name,json.dumps(result),flush=True)
        return result
    except BaseException as error:
        write(dest/'case.json',dict(passed=False,error=str(error),wall_s=time.perf_counter()-start))
        # Harness creates a session per Kit child. Terminate descendants including these sessions.
        import psutil
        for proc,_,_ in jobs:
            try:
                children=psutil.Process(proc.pid).children(recursive=True)
                for child in reversed(children): child.terminate()
                proc.terminate()
                _,alive=psutil.wait_procs(children,timeout=5)
                for child in alive: child.kill()
            except psutil.NoSuchProcess: pass
        raise
    finally:
        for _,log,_ in jobs: log.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--continue-completed', action='store_true', help='Skip completed cases only; never retry a failed case')
    p.add_argument('--case',default='all',choices=['all','i1','i2','i4','b2','b4','l1','l2','aff2','seed1','adaptive1','mixed2','ipc_source','replay1','replay2'])
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if not (out/'topology.json').exists():
        import psutil
        power={}
        for f in Path('/sys/class/powercap/intel-rapl:0').glob('constraint_*'):
            try:power[str(f)]=f.read_text().strip()
            except OSError:power[str(f)]='unavailable'
        write(out/'topology.json',dict(lscpu=subprocess.check_output(['lscpu','-e'],text=True),
            affinity=sorted(os.sched_getaffinity(0)),load=os.getloadavg(),power_limits=power,
            split2=affinity_sets(2),split4=affinity_sets(4),
            memory=psutil.virtual_memory()._asdict(),gpu=subprocess.getoutput('nvidia-smi'),
            processes=subprocess.getoutput('ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu | head -16')))
    configs={'i1':('I3D',1,256),'i2':('I3D',2,256),'i4':('I3D',4,256),'b2':('I3D',2,128),'b4':('I3D',4,64),
             'l1':('L3D',1,256),'l2':('L3D',2,256),'aff2':('I3D',2,256),'seed1':('I3D',1,256),
             'adaptive1':('I3D',1,256),'mixed2':('I3D',2,256),'ipc_source':('I3D',1,256),
             'replay1':('I3D',1,256),'replay2':('I3D',2,256)}
    for name in (configs if a.case=='all' else [a.case]):
        if a.continue_completed and (out/name).exists():
            assert json.loads((out/name/'case.json').read_text())['passed'], 'Do not retry failed configuration'
            continue
        if name=='i4' and a.case=='all':
            base=json.loads((out/'i1/case.json').read_text());dual=json.loads((out/'i2/case.json').read_text())
            if dual['budget_over_slowest_train_tps']<base['budget_over_slowest_train_tps']*1.15:
                write(out/'i4_skipped.json',{'reason':'dual speedup below 1.15; avoid larger contention'});continue
        run_case(out,name,*configs[name],affinity=name=='aff2',
                 modes=['fixed','adaptive'] if name=='mixed2' else ['adaptive'] if name=='adaptive1' else None,
                 seeds=[1] if name in ('seed1','adaptive1') else None,export=name=='ipc_source',replay=name.startswith('replay'))

    if a.case=='all':
        for script, args, expected in [
            ('benchmark_unified_phase4_ipc.py',['--source',out/'ipc_source/w0/real_rollout.pt','--output',out/'ipc.json'],out/'ipc.json'),
            ('summarize_unified_phase4.py',[out],out/'summary.json')]:
            if a.continue_completed and expected.exists():
                assert json.loads(expected.read_text())['passed'];continue
            subprocess.run(['bash',str(ROOT/'isaaclab.sh'),'-p',str(ROOT/'scripts/flapping_px4'/script),*map(str,args)],cwd=ROOT,timeout=180,check=True)
            assert json.loads(expected.read_text())['passed']

if __name__=='__main__': main()

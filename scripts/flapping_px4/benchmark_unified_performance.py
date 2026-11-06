"""Bounded real-plant performance harness. No physics/PPO changes; one or two updates only.

Run with ./isaaclab.sh -p scripts/flapping_px4/benchmark_unified_performance.py.
Each invocation owns a new output directory; profiling checkpoints are deleted
only after the original runner has verified its save/load. Never resume them.
"""
import argparse
from collections import defaultdict
import csv
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False))


def command(argv):
    try:
        r = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        return dict(argv=argv, returncode=r.returncode, output=r.stdout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(argv=argv, error=str(e))


class Timers:
    """Main-thread call tree: inclusive and child-subtracted exclusive wall time.

    Paths keep setup, rollout, PPO and teardown distinct. Quantiles are per call,
    not across seeds. No synchronization is added to the CPU-only training path.
    """
    def __init__(self, enabled):
        self.enabled = enabled
        self.stack = []
        self.samples = defaultdict(list)

    def call(self, name, fn, *a, **kw):
        if not self.enabled or threading.current_thread() is not threading.main_thread():
            return fn(*a, **kw)
        path = '/'.join([f[0] for f in self.stack]+[name])
        frame = [name, 0.]
        self.stack.append(frame)
        start = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            dt = time.perf_counter()-start
            self.stack.pop()
            self.samples[path].append((dt, max(0., dt-frame[1])))
            if self.stack:
                self.stack[-1][1] += dt

    def patch(self, obj, name, label=None):
        original = getattr(obj, name)
        @functools.wraps(original)
        def timed(*a, **kw):
            return self.call(label or name, original, *a, **kw)
        setattr(obj, name, timed)

    def export(self, output):
        import numpy as np
        rows = []
        total = sum(sum(x[1] for x in s) for s in self.samples.values())
        for path, values in sorted(self.samples.items()):
            a = np.asarray(values)
            rows.append(dict(path=path, calls=len(a), inclusive_s=float(a[:, 0].sum()),
                             exclusive_s=float(a[:, 1].sum()), median_s=float(np.median(a[:, 0])),
                             p95_s=float(np.quantile(a[:, 0], .95)), exclusive_share=float(a[:, 1].sum()/total) if total else 0))
        write(output/'timings.json', dict(enabled=self.enabled, timed_root_s=total, rows=rows,
              semantics='Inclusive rows nest; sum exclusive only. Includes setup/close. Unwrapped native internals are unresolved.'))
        with (output/'timings.csv').open('w') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def install(t, output, detail=False):
    import torch
    import carb
    import sqlite3
    import flapping_bot.analysis.unified_ppo_training as r
    from flapping_bot.direct.flapping_bot.unified_trajectory_env import UnifiedTrajectoryEnv as Env
    from flapping_bot.physics.control_aero_3d import ControlAero3D
    from flapping_bot.path_tracking.trajectory_feedback import FeedbackStore, RolloutEvaluator
    import flapping_bot.path_tracking.trajectory_feedback as feedback
    from flapping_bot.path_tracking.unified_trajectory_cache import TrajectoryCache
    from isaaclab.sim import SimulationContext
    from isaaclab.scene import InteractiveScene
    from isaaclab.assets import Articulation
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    settings = carb.settings.get_settings()
    built_env = []
    original_build = r.build_env
    def build(*a, **kw):
        if getattr(BENCH_ARGS, 'evaluation_root', None):
            a = (a[0], BENCH_ARGS.evaluation_root, *a[2:])
        env = t.call('build_env', original_build, *a, **kw)
        built_env.append(env)
        versions = {}
        for name in ('torch', 'isaacsim', 'isaaclab', 'rsl-rl-lib', 'tensordict', 'numpy'):
            try: versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: versions[name] = None
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        extensions = {x['id']: x.get('package', {}).get('version') for x in manager.get_extensions()
                      if 'physx' in x['id'] or 'isaacsim.core' in x['id']}
        write(output/'runtime.json', dict(training_module=r.__file__, simulation_module=sys.modules[SimulationContext.__module__].__file__, versions=versions, extensions=extensions, torch_cuda=torch.version.cuda,
              torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
              affinity=sorted(os.sched_getaffinity(0)), settings={k: settings.get(k) for k in
              ('/physics/physxDispatcher', '/persistent/physics/numThreads', '/physics/numThreads', '/plugins/carb.tasking.plugin/threadCount')},
              physics_dt=env.physics_dt, control_dt=env.step_dt, decimation=env.cfg.decimation,
              device=env.device, plant=env.plant_identity, sqlite=str(env.rollout_evaluator.store.path.resolve())))
        assert env.cfg.decimation == 8 and abs(env.physics_dt-1/480) < 1e-12 and env.device == 'cpu'
        return env
    r.build_env = build
    if getattr(BENCH_ARGS, 'export_rollout', False):
        original_process = PPO.process_env_step
        original_returns = PPO.compute_returns
        captured = []
        exported = False
        def process(self, obs, rewards, dones, extras):
            if not exported:
                captured.append((rewards.clone(), dones.clone(), extras['time_outs'].clone()))
            return original_process(self, obs, rewards, dones, extras)
        def returns(self, obs):
            nonlocal exported
            result = original_returns(self, obs)
            if not exported:
                started = time.perf_counter()
                storage = self.storage
                payload = {name: getattr(storage, name).clone() for name in
                    ('actions','rewards','dones','values','actions_log_prob','mu','sigma','returns','advantages')}
                payload['observations'] = {k: v.clone() for k,v in storage.observations.items()}
                payload['last_observations'] = {k: v.clone() for k,v in obs.items()}
                payload['raw_rewards'] = torch.stack([x[0] for x in captured])
                payload['done_or_timeout'] = torch.stack([x[1] for x in captured])
                payload['timeouts'] = torch.stack([x[2] for x in captured])
                payload['terminated'] = payload['done_or_timeout'] & ~payload['timeouts']
                payload['policy'] = {k:v.clone() for k,v in self.policy.state_dict().items()}
                payload['version'] = built_env[0]._current_policy_version
                payload['last_values'] = self.policy.evaluate(obs).detach().clone().unsqueeze(0)
                torch.save(payload, output/'real_rollout.pt')
                write(output/'export_cost.json', {'seconds':time.perf_counter()-started})
                exported = True
            return result
        PPO.process_env_step = process
        PPO.compute_returns = returns
    if not t.enabled:
        return
    for name in ('train', 'evaluate', 'collect_logs', 'append_json', 'save_checkpoint', 'tensor_obs', 'make_policy', 'validate_evaluation_batch'):
        t.patch(r, name)
    for name in ('_pre_physics_step', '_apply_action', '_compute_control_3d_wrench', '_update_native_holonomic_load_diagnostics',
                 '_get_dones', '_interval_control', '_interval_controls', '_get_rewards', '_get_observations', '_update_reference', '_reset_idx', 'reset', 'close'):
        if hasattr(Env, name): t.patch(Env, name)
    old_step = Env.step
    def step(self, *a, **kw):
        padding = self.cfg.evaluation_tasks is not None and bool(self._evaluation_finished.all())
        return t.call('env_step_padding' if padding else 'env_step_active', old_step, self, *a, **kw)
    Env.step = step
    for cls, methods, prefix in (
        (SimulationContext, ('step', 'reset'), 'sim'),
        (InteractiveScene, ('write_data_to_sim', 'update'), 'scene'),
        (Articulation, ('set_external_force_and_torque', 'write_data_to_sim', 'update'), 'robot'),
        (ControlAero3D, ('evaluate',), 'aero'),
        (TrajectoryCache, ('project', 'query', 'positions', 'update_rows'), 'cache'),
        (FeedbackStore, ('_begin', '_step', '_append_evaluated', 'flush', 'close'), 'store'),
        (RolloutEvaluator, ('observe', 'observe_intervals', '_finish', 'begin', 'abort'), 'evidence'),
        (PPO, ('act', 'process_env_step', 'compute_returns', 'update'), 'ppo'),
        (ActorCritic, ('act', 'evaluate', 'act_inference'), 'policy'),
        (torch.optim.Adam, ('step', 'zero_grad'), 'optimizer'),
        (torch.Tensor, ('backward',), 'autograd')):
        for name in methods:
            if hasattr(cls, name): t.patch(cls, name, prefix+'.'+name)
    t.patch(feedback, 'canonical', 'evidence.json_encode')
    if detail:
        import flapping_bot.physics.control_aero_3d as aero
        import flapping_bot.direct.flapping_bot.unified_trajectory_env as env_module
        from flapping_bot.physics.control_aero_kinematics import LinkKinematics3D
        t.patch(LinkKinematics3D, 'sample', 'aero.kinematics')
        t.patch(aero, 'horseshoe_influence', 'aero.horseshoe_influence')
        t.patch(torch.linalg, 'cond', 'linalg.cond')
        t.patch(torch.linalg, 'solve', 'linalg.solve')
        t.patch(feedback, 'valid_interval', 'evidence.valid_interval')
        t.patch(env_module, 'valid_interval', 'env.valid_interval')
    original_connect = sqlite3.connect
    class Connection(sqlite3.Connection):
        def execute(self, *a, **kw):
            return t.call('sqlite.execute', super().execute, *a, **kw)
        def commit(self):
            return t.call('sqlite.commit_including_sync', super().commit)
    def connect(*a, **kw):
        kw.setdefault('factory', Connection)
        return original_connect(*a, **kw)
    sqlite3.connect = connect


def child(args):
    global BENCH_ARGS
    BENCH_ARGS = args
    output = args.output
    t = Timers(not args.no_profile)
    started = time.perf_counter()
    finalized = False
    def finish():
        nonlocal finalized
        if finalized: return
        finalized = True
        t.export(output)
        checkpoints = []
        for path in (output/'run').glob('model_*.pt'):
            checkpoints.append(dict(name=path.name, bytes=path.stat().st_size))
            path.unlink()
        latest = output/'run/latest_checkpoint.txt'
        if latest.exists(): latest.unlink()
        write(output/'benchmark_only.json', dict(purpose='performance only; never resume', checkpoints_deleted=checkpoints,
                                               child_wall_before_kit_close_s=time.perf_counter()-started))
    from isaaclab.app import AppLauncher
    original = AppLauncher.__init__
    def launch(self, *a, **kw):
        t.call('kit_start', original, self, *a, **kw)
        install(t, output, args.detail)
        old_close = self.app.close
        def close(*a, **kw):
            finish()  # Kit fast shutdown can terminate without Python finally/atexit.
            return old_close(*a, **kw)
        self.app.close = close
    AppLauncher.__init__ = launch
    argv = ['--backend', args.backend, '--sampling', args.sampling, '--seed', str(args.seed),
            '--num-envs', str(args.num_envs), '--rollout-steps', str(args.rollout_steps),
            '--iterations', str(args.iterations), '--threads', str(args.threads), '--output', str(output/'run'),
            '--plant-assets', str(args.plant_assets), '--rules', str(args.rules)]
    if args.evaluate_checkpoint:
        argv += ['--evaluate', '--smoke-evaluation', '--checkpoint', str(args.evaluate_checkpoint), '--eval-step-budget', str(args.eval_step_budget)]
    entry_root = ROOT
    sys.argv = [str(entry_root/'scripts/flapping_px4/train_unified_trajectory.py')]+argv
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        finish()


def hardware(output):
    commands = [['lscpu'], ['lscpu', '-e=CPU,NODE,SOCKET,CORE,MHZ,MAXMHZ,MINMHZ'], ['numactl', '--hardware'],
                ['free', '-b'], ['lsblk', '-o', 'NAME,TYPE,ROTA,SIZE,FSTYPE,MOUNTPOINTS,MODEL'],
                ['df', '-T', str(output)], ['findmnt', '-T', str(output)], ['nvidia-smi', '-q'],
                ['nvidia-smi', 'topo', '-m'], ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                ['git', '-C', str(ROOT), 'status', '--short']]
    write(output/'hardware.json', [command(c) for c in commands])
    write(output/'numa_sysfs.json', {str(p): p.read_text() for node in Path('/sys/devices/system/node').glob('node[0-9]*')
          for p in (node/'cpulist', node/'meminfo', node/'distance') if p.exists()})


def parent(args):
    import psutil
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'parameters.json', vars(args))
    hardware(args.output)
    shutil.copyfile(Path(__file__), args.output/'benchmark_source.py')
    env = os.environ.copy()
    entry_root = ROOT
    env['PYTHONPATH'] = str((args.implementation_root or entry_root)/'source/flapping_bot')+os.pathsep+env.get('PYTHONPATH', '')
    argv = ['bash', str(ROOT/'isaaclab.sh'), '-p', str(Path(__file__).resolve()), *sys.argv[1:], '--child']
    if args.affinity: argv = ['taskset', '-c', args.affinity]+argv
    started = time.perf_counter()
    tracked = {}
    psutil.cpu_percent(percpu=True)
    with (args.output/'console.log').open('w') as log, (args.output/'resources.jsonl').open('w') as trace:
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        while proc.poll() is None:
            now = time.perf_counter()
            if now-started > args.timeout:
                import signal
                os.killpg(proc.pid, signal.SIGTERM); timed_out = True
                try: proc.wait(10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                break
            processes = []
            try:
                for p in [psutil.Process(proc.pid), *psutil.Process(proc.pid).children(recursive=True)]:
                    try:
                        p = tracked.setdefault(p.pid, p)
                        processes.append(dict(pid=p.pid, name=p.name(), cpu_percent=p.cpu_percent(),
                            cpu_times=p.cpu_times()._asdict(), rss=p.memory_info().rss, threads=p.num_threads(),
                            affinity=p.cpu_affinity(), io=p.io_counters()._asdict(),
                            thread_cpu=[x._asdict() for x in p.threads()]))
                    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
            except psutil.NoSuchProcess: pass
            gpu = command(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'])
            db = {str(p.relative_to(args.output)): p.stat().st_size for p in (args.output/'run/evidence').glob('*.sqlite3*')}
            trace.write(json.dumps(dict(elapsed_s=now-started, processes=processes, cpu_per_core=psutil.cpu_percent(percpu=True),
                  memory=psutil.virtual_memory()._asdict(), disk=psutil.disk_io_counters()._asdict(), gpu=gpu, sqlite_bytes=db, frequency=[x._asdict() for x in psutil.cpu_freq(percpu=True)], temperatures={k:[x._asdict() for x in v] for k,v in psutil.sensors_temperatures().items()}))+'\n'); trace.flush()
            time.sleep(1)
    status = args.output/'run/status.json'
    passed = status.exists() and json.loads(status.read_text()).get('status') == 'completed' and not timed_out and (args.output/'timings.json').exists() and (args.output/'benchmark_only.json').exists()
    write(args.output/'completion.json', dict(passed=passed, timed_out=timed_out, returncode=proc.returncode,
                                            end_to_end_s=time.perf_counter()-started))
    print(json.dumps(dict(output=str(args.output), passed=passed)), flush=True)
    if not passed: raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['L3D', 'I3D'], default='L3D')
    p.add_argument('--sampling', choices=['fixed', 'adaptive'], default='fixed')
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--rollout-steps', type=int, default=48)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iterations', type=int, choices=(1, 2), default=1)
    p.add_argument('--export-rollout', action='store_true', help='Export first real storage for offline IPC benchmark only')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--affinity', help='Linux taskset CPU list; inherited by PhysX and torch')
    p.add_argument('--no-profile', action='store_true')
    p.add_argument('--detail', action='store_true', help='Additional I3D/validity attribution; compare separately')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=600, help='Hard process-group budget in seconds')
    p.add_argument('--plant-assets', type=Path, default=ROOT.parent/'control-oriented-3d-wing-aero')
    p.add_argument('--rules', type=Path, default=ROOT.parent/'control-oriented-3d-wing-aero/logs/unified_trajectory_phase2_v1/screening_rules_v2.json')
    p.add_argument('--eval-step-budget', type=int, default=256)
    p.add_argument('--evaluate-checkpoint', type=Path, help='Existing smoke snapshot; frozen development evaluation only, 256 steps')
    p.add_argument('--implementation-root', type=Path, help='Read-only baseline implementation for matched evaluation')
    p.add_argument('--evaluation-root', type=Path, help='Read-only source checkout matching absolute URDF paths of existing frozen snapshot')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.evaluation_root: args.evaluation_root = args.evaluation_root.resolve()
    if args.implementation_root: args.implementation_root = args.implementation_root.resolve()
    if args.evaluation_root and not args.evaluate_checkpoint: p.error('evaluation-root requires a frozen evaluation checkpoint')
    if min(args.num_envs, args.rollout_steps, args.threads, args.timeout) <= 0: p.error('positive budgets required')
    (child if args.child else parent)(args)


if __name__ == '__main__': main()

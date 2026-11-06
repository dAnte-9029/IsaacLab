"""Bounded phase-three validation: three PPO iterations total, no formal training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
old=ROOT.parent/'unified-rl-performance-phase2'
def run(script,args,destination):
    with (out/(destination.name+'.log')).open('w') as log:
        subprocess.run(['bash',str(ROOT/'isaaclab.sh'),'-p',str(ROOT/'scripts/flapping_px4'/script),*map(str,args)],
            cwd=ROOT,env=dict(os.environ,TERM='xterm'),stdout=log,stderr=subprocess.STDOUT,timeout=240,check=True)
    if not json.loads((destination/'completion.json').read_text())['passed']:raise RuntimeError(str(destination))
# The wrapper does not propagate Python's exit status; persist pytest's status.
test_status=out/'test_exit_status'
code="import pytest; from pathlib import Path; c=pytest.main(['-q','tests/test_unified_performance_phase3.py','tests/test_control_aero_3d.py']); Path("+repr(str(test_status))+").write_text(str(c))"
with (out/'tests.log').open('w') as log:
    subprocess.run(['bash',str(ROOT/'isaaclab.sh'),'-p','-c',code],cwd=ROOT,
        env=dict(os.environ,TERM='xterm',PYTHONPATH=str(ROOT/'source/flapping_bot')),stdout=log,stderr=subprocess.STDOUT,timeout=120)
if test_status.read_text()!='0':raise RuntimeError('Focused tests failed')
for backend in ('I3D','L3D'):
    for sampling in (('fixed','adaptive') if backend=='I3D' else ('fixed',)):
        dest=out/f'{backend}_{sampling}'
        run('benchmark_unified_performance.py',['--backend',backend,'--sampling',sampling,'--detail','--output',dest],dest)
    for kind in ('baseline','optimized'):
        dest=out/f'{backend}_replay_{kind}'
        args=['--backend',backend,'--output',dest]
        if kind=='baseline':args+=['--implementation-root',old,'--physical-root',old]
        run('replay_unified_phase3.py',args,dest)
subprocess.run(['bash',str(ROOT/'isaaclab.sh'),'-p',str(ROOT/'scripts/flapping_px4/benchmark_i3d_phase3_kernels.py'),'--output',str(out/'kernels.json')],check=True)
subprocess.run(['bash',str(ROOT/'isaaclab.sh'),'-p',str(ROOT/'scripts/flapping_px4/summarize_unified_phase3.py'),str(out)],check=True)
if not json.loads((out/'summary.json').read_text())['passed']:raise RuntimeError('Parity failed')

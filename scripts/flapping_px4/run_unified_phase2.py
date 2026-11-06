"""Reproduce phase-two bounded comparisons in a fresh directory; never formal training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
old=ROOT.parent/'unified-rl-performance-phase1';physical=ROOT.parent/'unified-trajectory-ppo-v1'
bench=ROOT/'scripts/flapping_px4/benchmark_unified_performance.py'
replay=ROOT/'scripts/flapping_px4/replay_unified_phase2.py'
def run(script,argv,destination,timeout=1100):
    with (out/(destination.name+'.launcher.log')).open('w') as log:
        subprocess.run(['bash',str(ROOT/'isaaclab.sh'),'-p',str(script),*map(str,argv)],cwd=ROOT,
                       env=dict(os.environ,TERM='xterm'),stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
    if not json.loads((destination/'completion.json').read_text())['passed']:
        raise RuntimeError(f'Failed bounded run: {destination}')
test_files=['tests/test_unified_performance_phase2.py','tests/test_unified_pretraining.py',
    'tests/test_unified_final_repairs.py','tests/test_unified_ppo_contract.py','tests/test_unified_physx_acceptance.py']
test_code = ('import pytest; from pathlib import Path; '
    + 'code=pytest.main('+repr(['-q',*test_files])+'); '
    + 'Path('+repr(str(out/'test_exit_status'))+').write_text(str(code))')
with (out/'tests.log').open('w') as log:
    subprocess.run(['bash',str(ROOT/'isaaclab.sh'),'-p','-c',test_code],cwd=ROOT,
        env=dict(os.environ,TERM='xterm',PYTHONPATH=str(ROOT/'source/flapping_bot')),stdout=log,stderr=subprocess.STDOUT,timeout=120)
if (out/'test_exit_status').read_text() != '0': raise RuntimeError('Focused tests failed')
for backend in ('L3D','I3D'):
    for sampling in ('fixed','adaptive'):
        destination=out/f'{backend}_{sampling}'
        run(bench,['--backend',backend,'--sampling',sampling,'--detail','--output',destination],destination)
    original=physical/f'logs/unified_physx_acceptance_20261106_v1/{backend}_development/evaluation_smoke_only.pt'
    payload=torch.load(original,map_location='cpu',weights_only=False)
    payload['contract']['eval_step_budget']=7200
    checkpoint=out/f'{backend}_budget7200.pt';torch.save(payload,checkpoint)
    for kind in ('baseline','optimized'):
        destination=out/f'{backend}_eval_{kind}7200'
        argv=['--backend',backend,'--evaluation-root',physical,'--evaluate-checkpoint',checkpoint,
              '--eval-step-budget','7200','--timeout','1000','--output',destination]
        if kind=='baseline': argv+=['--implementation-root',old]
        run(bench,argv,destination)
    for kind in ('baseline','optimized','profile'):
        destination=out/f'{backend}_replay_{kind}'
        argv=['--backend',backend,'--physical-root',physical,'--output',destination]
        if kind=='baseline': argv+=['--implementation-root',old]
        if kind=='profile': argv+=['--profile']
        run(replay,argv,destination,timeout=180)
subprocess.run([sys.executable,str(ROOT/'scripts/flapping_px4/summarize_unified_phase2.py'),str(out)],check=True)
print(out/'summary.json')

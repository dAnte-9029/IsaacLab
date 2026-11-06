"""Matched alternating CPU kernel attribution; not a PhysX throughput result."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/flapping_bot'))
from flapping_bot.physics import control_aero_3d as new
from flapping_bot.physics.control_aero_kinematics import LinkKinematics3D
from flapping_bot.analysis.crosswind_aero import CSV, URDF, CG
oldroot=ROOT.parent/'unified-rl-performance-phase2/source/flapping_bot/flapping_bot/physics'
def module(name,file):
    spec=importlib.util.spec_from_file_location('flapping_bot.physics.'+name,oldroot/file)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod
old=module('_old_aero','control_aero_3d.py');oldkin=module('_old_kin','control_aero_kinematics.py')
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(0)
results={}
with torch.inference_mode():
 for dtype in (torch.float32,torch.float64):
    cfg=new.ControlAero3DConfig(backend='I3D',twist_amplitude_deg=35)
    kin=LinkKinematics3D(CSV,URDF,cfg,dtype=dtype)
    v=torch.tensor([[8.,.5,-.2]],dtype=dtype).expand(256,-1);om=torch.tensor([[.1,.2,-.1]],dtype=dtype).expand_as(v)
    phase=torch.linspace(0,6.28,256,dtype=dtype);ref=CG.to(dtype)[None].expand_as(v)
    sample=kin.sample;captured=[]
    def capture(*args):captured.append(args);return sample(*args)
    kin.sample=capture
    g,m,rel=kin.prescribed(v,om,ref,phase,3.)
    kin.sample=sample;args=captured[0]
    mean=m.flow_three_quarter_w_mps.mean(1,keepdim=True);direction=(mean/mean.norm(dim=-1,keepdim=True)).expand_as(m.flow_three_quarter_w_mps)
    cases={
      'horseshoe':(lambda:old.horseshoe_influence(g,g.quarter_chord_w_m,direction,omit_self_bound=True),lambda:new.horseshoe_influence(g,g.quarter_chord_w_m,direction,omit_self_bound=True)),
      'kinematics':(lambda:oldkin.LinkKinematics3D.sample(kin,*args),lambda:sample(*args))}
    for name,functions in cases.items():
        first=[];outputs=[]
        for fn in functions:
            t=perf_counter();outputs.append(fn());first.append(perf_counter()-t)
        pairs=zip(*outputs) if name=='horseshoe' else ((getattr(x,k),getattr(y,k)) for x,y in zip(*outputs) for k in vars(x))
        assert all(torch.equal(x,y) for x,y in pairs)
        times=[[],[]]
        for block in range(6):
            for i in ((0,1) if block%2==0 else (1,0)):
                t=perf_counter()
                for _ in range(30):functions[i]()
                times[i].append((perf_counter()-t)/30)
        med=[float(torch.tensor(x,dtype=torch.float64).median()) for x in times]
        results[str(dtype)+'.'+name]=dict(exact_equal=True,first_call_s=first,blocks_s_per_call=times,median_s=med,speedup=med[0]/med[1])
a.output.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

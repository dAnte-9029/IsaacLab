"""Frozen Sobol screening, parameter sensitivities and engineering acceptance contract."""
import json,csv,math
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from scipy.stats import qmc,spearmanr
from flapping_bot.physics.tail_v2 import TailV2Polar,load_tail_config
from flapping_bot.physics.tail_aero import TailAeroModel
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/results/tail_v2_control_v2'
def dump(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False))
def csvwrite(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def query(model,alpha,delta=0,q=0,diff=0,rudder=0,speed=8):
    a=torch.as_tensor(alpha,dtype=torch.float64).reshape(-1);z=a*0
    v=torch.stack([speed*a.cos(),z,-speed*a.sin()],-1)
    return model.compute_surface_results(root_lin_vel_b=v,root_ang_vel_b=torch.stack([z,z+q,z],-1),left_elevon_rad=z+delta+diff,right_elevon_rad=z+delta-diff,rudder_rad=z+rudder,base_com_pos_b=v.new_tensor([[-.12154,.00541,-.01298]]).expand_as(v))
def moment(model,alpha,**kw):return sum(s.moment_b_about_com for s in query(model,alpha,**kw)).numpy()
def main():
    global OUT
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--evidence-dir",type=Path,default=OUT)
    OUT=parser.parse_args().evidence_dir.resolve()
    OUT.mkdir(exist_ok=True,parents=True);cand=OUT/'candidates';cand.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Frozen protocol already exists')
    bounds=dict(a2d=[5,2*math.pi],cl_limit=[.7,1.2],alpha_s_deg=[8,16],alpha_f_deg=[25,40],kappa=[0,.4],cd_min=[.015,.05],efficiency=[.33,.81],cd_delta=[0,.2],cl_md_delta=[0,2])
    dump(OUT/'parameter_bounds.json',{k:dict(range=v,source='v1 engineering interval; constrained by trends, not measured aircraft bounds',group='A' if k in list(bounds)[:5] else 'B',unit='deg' if k.endswith('_deg') else 'rad^-1' if k in ('a2d','cl_md_delta') else 'rad^-2' if k=='cd_delta' else '1') for k,v in bounds.items()})
    protocol=dict(seed=0,dt=1/480,control_hz=60,backend='L3D',static_candidates=96,physx_candidates=12,trim_rounds=2,trim_candidates_per_round=13,trim_cycles=5,confirmation_cycles=10,controller_candidates=4,controller_gains=9,controller_seconds=4,final_candidates=2,final_seconds=10,final_speeds=[8,10],final_cases=['hold','q+','q-','roll+','yaw+','pitch+','pitch-','roll_cmd+','roll_cmd-','yaw_cmd'],trim_bounds=[[7,1,3,-20,-8,-8],[10,18,4,15,8,8]],trim_order=['V_mps','nose_up_deg','frequency_Hz','symmetric_deg','rudder_deg','differential_deg'],filter_tau_s=.12,lambda_control=1,selection='physical gate; diverse static features; trim rank; control hard constraints before score',task_gates=dict(pitch_roll_rms_deg=5,heading_rms_deg=10,speed_min_mps=6,speed_max_mps=12,height_drift_max_m=3,saturation_fraction_max=.05,large_tail_fraction_max=.2,extrapolation_fraction_max=.1,attitude_max_deg=45),gate_type='engineering task tolerances, not measured aerodynamic limits',physical_gates='finite, CD>=0, sign/symmetry, bounded coefficients; paper trends qualitative',normal_region='abs(alpha_eff)<=alpha_s; extrapolation abs(alpha_eff)>=alpha_f',C_fixed=dict(cn90=1.8,cd_parallel=.025,source='inherited bounded flat-plate engineering closure; unvalidated at 90deg'),reference=dict(frame='body FLU',moment='base COM',horizontal_area_m2=.09468,vertical_area_m2=.0227904,nu_m2_s=1.5e-5),controller_score='mean normalized causal filtered 3-axis attitude error + vertical/speed drift + tail use; raw wingbeat rates descriptive')
    dump(OUT/'protocol.json',protocol)
    import shutil
    shutil.copyfile(ROOT/'docs/results/tail_v2_v1/literature_evidence.csv',OUT/'literature_constraints.csv')
    samples=qmc.Sobol(len(bounds),scramble=True,seed=0).random_base2(7)[:96]
    rows=[];features=[]
    for i,u in enumerate(samples):
        pol=asdict(TailV2Polar());pol.update({k:float(lo+(hi-lo)*u[j]) for j,(k,(lo,hi)) in enumerate(bounds.items())})
        conf=dict(id=f'c{i:03}',backend='v2',polar_horizontal=pol,polar_vertical=asdict(TailV2Polar()))
        path=cand/(conf['id']+'.json');dump(path,conf);model=TailAeroModel(load_tail_config(path),'cpu')
        alpha=np.deg2rad([0,6,12,18]);h=1e-4
        g=(moment(model,alpha,delta=h)-moment(model,alpha,delta=-h))/(2*h)
        ka=(moment(model,alpha+h)-moment(model,alpha-h))/(2*h)
        damp=(moment(model,alpha,q=h)-moment(model,alpha,q=-h))/(2*h)
        ds=(moment(model,alpha,diff=h)-moment(model,alpha,diff=-h))/(2*h)
        rud=(moment(model,alpha,rudder=h)-moment(model,alpha,rudder=-h))/(2*h)
        surf=query(model,np.linspace(-math.pi,math.pi,721),delta=.2)
        physical=all(torch.isfinite(s.force_b).all() and (s.drag_coefficient>=0).all() for s in surf) and g[0,1]>0 and damp[0,1]<0
        feat=np.r_[g[:,1],ka[:,1],damp[:,1],float(surf[0].drag_coefficient.mean())];features.append(feat)
        rows.append(dict(id=conf['id'],**pol,physical=bool(physical),reason='finite nonnegative drag and correct local sign' if physical else 'physical gate failed',G0=g[0,1],G12=g[2,1],K0=ka[0,1],K12=ka[2,1],Dq12=damp[2,1],roll_gain=ds[0,0],diff_yaw_cross=ds[0,2],rudder_yaw_gain=rud[0,2],rudder_roll_cross=rud[0,0],selected=False))
    f=np.array(features);f=(f-f.mean(0))/(f.std(0)+1e-9)
    valid=[i for i,r in enumerate(rows) if r['physical']];selected=[min(valid,key=lambda i:np.linalg.norm(f[i]))]
    while len(selected)<12:selected.append(max([i for i in valid if i not in selected],key=lambda i:min(np.linalg.norm(f[i]-f[j]) for j in selected)))
    for i in selected:rows[i]['selected']=True
    csvwrite(OUT/'candidate_results.csv',rows)
    sens=[]
    for k in bounds:
        for m in ['G0','G12','K12','Dq12']:
            sens.append(dict(parameter=k,metric=m,spearman=float(spearmanr([r[k] for r in rows],[r[m] for r in rows]).statistic),optimized=True,interpretation='global rank association; not independent identification'))
    csvwrite(OUT/'parameter_sensitivity.csv',sens)
    dump(cand/'legacy.json',dict(id='legacy',backend='legacy'))
    old=asdict(TailV2Polar());dump(cand/'original_v2.json',dict(id='original_v2',backend='v2',polar_horizontal=old,polar_vertical=old))
    dump(OUT/'selected.json',[rows[i]['id'] for i in selected]+['legacy','original_v2'])
    print('Frozen: 96 static, 12 diverse V2 + 2 references',flush=True)
if __name__=='__main__':main()

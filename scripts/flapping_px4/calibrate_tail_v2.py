"""Evidence-constrained candidate construction and static authority; no flight-score fitting."""
import argparse,csv,json,math
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from flapping_bot.physics.tail_aero import TailAeroCfg,TailAeroModel
from flapping_bot.physics.tail_v2 import TailV2Polar,geometry_eta
from flapping_bot.physics.tail_aero import _RIGHT_ELEVON_POLYGON_MM,_VERTICAL_TAIL_POLYGON_MM
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
def write(name,rows):
    with (a.output/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
polar=asdict(TailV2Polar())
(a.output/'tail_v2_nominal.json').write_text(json.dumps(dict(version='tail-v2-candidate-1',status='engineering candidate; not aircraft identified',polar=polar,reference_area_m2=dict(horizontal=.09468,vertical=.0227904),eta_horizontal=geometry_eta(_RIGHT_ELEVON_POLYGON_MM,0),eta_vertical=geometry_eta(_VERTICAL_TAIL_POLYGON_MM,1)),indent=2))
ranges=dict(a2d=[5.,2*math.pi],cl_limit=[.7,1.2],kappa=[0,.4],alpha_s_deg=[8,16],alpha_f_deg=[25,40],cd_min=[.015,.05],cd_delta=[0,.2],cl_md_delta=[0,2],efficiency=[.33,.81],cn90=[1.2,2.],cd_parallel=[.015,.05])
(a.output/'tail_v2_parameter_bounds.json').write_text(json.dumps({k:dict(nominal=v,bounds=ranges[k],unit=('rad^-1' if k in ('a2d','cl_md_delta') else 'rad^-2' if k=='cd_delta' else 'deg' if k.endswith('_deg') else '1'),status='engineering uncertainty interval, not confidence interval',reference='Ananda Sec 3.2-3.5; Traub Sec III.D; high-angle closure unvalidated') for k,v in polar.items()},indent=2))
write('literature_evidence.csv',[
 dict(paper='Anyoji 2014',source='pp 3-4 Sec 2.2-3.1; pp 6-8 Sec 3.2 Fig 7-10',Re='20000-80000',AR='endplates quasi-2D',flap_ratio=.3,reference_area='chord*span',evidence='thin plate small-angle near-linear; control effectiveness decreases with incidence',transfer='trend only; no digitized points'),
 dict(paper='Ananda 2015',source='pp 398-405 Sec 3.2-3.5',Re='60000-160000',AR='2-5',flap_ratio=0,reference_area='full wing planform',evidence='stall criterion moment break; corresponding CL 0.55-0.7; e0 approximately 0.33-0.81',transfer='Helmbold slope; drag efficiency interval; not CL hard cap'),
 dict(paper='Traub Coffman 2019',source='pp 2-4 Fig 3; Sec III.D Eq 2,5-10',Re='40000-80000',AR='wall-spanning quasi-2D',flap_ratio=.2,reference_area='full airfoil chord*span',evidence='flap lift offset and shifted drag polar; maximum lift increase depends on flap geometry',transfer='theoretical geometric eta; drag-shift magnitudes unresolved')])
rows=[]; curves=[]
for backend in ('legacy','v2'):
    model=TailAeroModel(TailAeroCfg(backend=backend,fixed_horizontal_effectiveness=.5,elevon_effectiveness=1.2),'cpu')
    def result(speed,alpha=0,delta=0,q=0,diff=0,rudder=0):
        v=torch.tensor([[speed*math.cos(alpha),0,-speed*math.sin(alpha)]],dtype=torch.float64)
        z=torch.tensor([0.],dtype=torch.float64)
        return model.compute_surface_results(root_lin_vel_b=v,root_ang_vel_b=torch.tensor([[0.,q,0.]],dtype=torch.float64),left_elevon_rad=z+delta+diff,right_elevon_rad=z+delta-diff,rudder_rad=z+rudder,base_com_pos_b=v.new_tensor([[-.12154,.00541,-.01298]]))
    def moment(*args,**kw):return sum(r.moment_b_about_com for r in result(*args,**kw))[0].numpy()
    for speed in (6,8,10):
        for deg in (0,6,12,20,45,90):
            alpha=math.radians(deg);h=1e-5
            rows.append(dict(backend=backend,speed_mps=speed,alpha_deg=deg,G_delta_Nm_per_rad=(moment(speed,alpha,h)[1]-moment(speed,alpha,-h)[1])/(2*h),K_alpha_Nm_per_rad=(moment(speed,alpha+h)[1]-moment(speed,alpha-h)[1])/(2*h),D_q_Nm_per_rad_s=(moment(speed,alpha,q=h)[1]-moment(speed,alpha,q=-h)[1])/(2*h),G_diff_Nm_per_rad=(moment(speed,alpha,diff=h)[0]-moment(speed,alpha,diff=-h)[0])/(2*h),G_rudder_Nm_per_rad=(moment(speed,alpha,rudder=h)[2]-moment(speed,alpha,rudder=-h)[2])/(2*h)))
    for deg in np.linspace(-180,180,721):
        for r in result(8,math.radians(deg)):
            assert torch.isfinite(r.force_b).all() and (r.drag_coefficient>=0).all()
            curves.append(dict(backend=backend,surface=r.name,alpha_deg=deg,CL=float(r.lift_coefficient[0]),CD=float(r.drag_coefficient[0])))
    assert all(torch.equal(r.force_b,torch.zeros_like(r.force_b)) for r in result(0))
write('authority_comparison.csv',rows);write('polar_curves.csv',curves)
write('calibration_results.csv',[dict(candidate='nominal-1',method='theory plus bounded engineering prior',cl_limit=.9,alpha_s_deg=12,alpha_f_deg=30,cd_min=.025,efficiency=.6,fit_rmse='not fitted: no reliable digitization',flight_objective_used=False)])
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4))
for backend,surface in [('legacy','left_elevon'),('v2','left_horizontal')]:
    rr=[r for r in curves if r['backend']==backend and r['surface']==surface and abs(r['alpha_deg'])<=90]
    for ax,col in zip(axes,['CL','CD']):ax.plot([r['alpha_deg'] for r in rr],[r[col] for r in rr],label=backend);ax.set(xlabel='alpha (deg)',ylabel=col);ax.legend();ax.grid()
fig.tight_layout();fig.savefig(a.output/'polars.png')
print('Static finite/drag/zero-speed checks passed; authority saved',flush=True)

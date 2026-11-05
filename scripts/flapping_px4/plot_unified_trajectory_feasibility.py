"""Compact final success map and representative reference/response plots."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
folders=[a.root/(b+'_final') for b in ['L3D','I3D']]
results=[json.loads((f/'results.json').read_text())+json.loads((a.root/(b+'_transitions')/'results.json').read_text()) for f,b in zip(folders,['L3D','I3D'])]
fig,axs=plt.subplots(1,3,figsize=(15,8),gridspec_kw={'width_ratios':[1.25,1,1]})
status=np.array([[int(r['status']=='VERIFIED FEASIBLE') for r in rows] for rows in results]).T
axs[0].imshow(status,cmap=ListedColormap(['#e6a15a','#4eac91']),vmin=0,vmax=1,aspect='auto')
axs[0].set_xticks([0,1],['L3D','I3D']);axs[0].set_yticks(range(len(results[0])),[r['name'] for r in results[0]])
for j in range(len(results[0])):
 for i in range(2):axs[0].text(i,j,'verified' if status[j,i] else 'not found',ha='center',va='center',fontsize=8)
axs[0].set_title('Fixed gates; one deterministic initial state')
for i,(rows,b) in enumerate(zip(results,['L3D','I3D'])):
 for r in rows:
  if r['name'].startswith('turn_') and 'reversal' not in r['name']:
   axs[1].scatter(r['geometry']['kappa'],r['horizontal_max_m'],marker='o' if i==0 else '^',color='#277da8' if i==0 else '#c45155')
  if r['name'].startswith('climb_'):
   axs[2].scatter(r['geometry']['gamma'],r['vertical_max_m'],marker='o' if i==0 else '^',color='#277da8' if i==0 else '#c45155')
 axs[1].scatter([],[],label=b,marker='o' if i==0 else '^',color='#277da8' if i==0 else '#c45155')
for ax,xlabel,ylabel in [(axs[1],'Heading curvature peak (rad/m)','Max horizontal error (m)'),(axs[2],'Slope peak (rad)','Max vertical error (m)')]:
 ax.axhline(2,color='black',linestyle='--',label='2 m gate');ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.3);ax.set_yscale('symlog',linthresh=2.)
axs[1].legend();fig.tight_layout();fig.savefig(a.root/'boundary.png',dpi=160);plt.close(fig)

selected=['turn_intermediate','climb_strong','turn_reversal_15m','vertical_reversal_15m','coupled']
fig,axs=plt.subplots(3,len(selected),figsize=(18,9))
for col,name in enumerate(selected):
 for folder,b,color in zip(folders,['L3D','I3D'],['#277da8','#c45155']):
  protocol=json.loads((folder/'protocol.json').read_text());tr=np.load(folder/'trace.npz')
  i=next(i for i,c in enumerate(protocol['cases']) if c['name']==name)
  path=Trajectory(TrajectorySpec(**protocol['cases'][i]['trajectory']))
  end=next(r for r in json.loads((folder/'results.json').read_text()) if r['name']==name)['completion_time_s']
  mask=tr['t'][:,i]<=end;t=tr['t'][mask,i];s=tr['progress'][mask,i];q=path.query(s)
  heading=np.unwrap(np.arctan2(tr['velocity'][mask,i,1],tr['velocity'][mask,i,0]))
  axs[0,col].plot(t,np.rad2deg(heading),color=color,label=b)
  axs[0,col].plot(t,np.rad2deg(q['heading']),color=color,ls='--',alpha=.6)
  axs[1,col].plot(t,tr['position'][mask,i,2]-20,color=color)
  axs[1,col].plot(t,q['position'][:,2]-20,color=color,ls='--',alpha=.6)
  axs[2,col].plot(t,tr['normal_accel_demand'][mask,i],color=color)
 axs[0,col].set_title(name)
 for ax in axs[:,col]:ax.grid(alpha=.3);ax.set_xlabel('Time (s)')
axs[0,0].set_ylabel('Course / reference heading (deg)');axs[1,0].set_ylabel('Height / reference change (m)');axs[2,0].set_ylabel('Reference v(actual)^2 curvature (m/s2)')
axs[0,0].legend();fig.suptitle('Solid: actual; dashed: geometric reference. Bottom: demand only, not a dynamics proof.')
fig.tight_layout();fig.savefig(a.root/'demand_response.png',dpi=150);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4))
for backend,color in [('L3D','#277da8'),('I3D','#c45155')]:
    folder=a.root/(backend+'_transitions');tr=np.load(folder/'trace.npz')
    protocol=json.loads((folder/'protocol.json').read_text());results=json.loads((folder/'results.json').read_text())
    for i,c in enumerate(protocol['cases'][:4]):
        axis=0 if 'turn' in c['name'] else 1
        d=15 if '15m' in c['name'] else 30
        mask=tr['t'][:,i]<=results[i]['completion_time_s']
        error=tr['position'][mask,i]-tr['reference'][mask,i]
        value=np.linalg.norm(error[:,:2],axis=1) if axis==0 else np.abs(error[:,2])
        axes[axis].plot(tr['progress'][mask,i]-70,value,color=color,ls='-' if d==15 else '--',label=f'{backend} reversal {d} m')
for ax,title in zip(axes,['Turn reversal: horizontal error','Climb/descent reversal: vertical error']):
    ax.set_title(title);ax.set_xlabel('Arc length from reversal onset (m)');ax.set_ylabel('Error (m)');ax.axhline(2,color='black',ls=':');ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(a.root/'transition_response.png',dpi=150);plt.close(fig)

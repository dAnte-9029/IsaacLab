"""Publication-style frozen endpoint and path-error traces, observed intervals only."""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
base=Path(os.environ.get('GOAL_OUTPUT_ROOT',root/'logs/unified_goal_consistent_reward_v1'))
a=np.load(base/'frozen_traces.npz')
from flapping_bot.analysis.unified_ppo_contract import evaluation_tasks,evaluation_origin
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec
from dataclasses import replace
fig,axes=plt.subplots(3,5,figsize=(17,8),sharex='col',constrained_layout=True)
names=['simple','sustained','reverse right first','reverse left first','coupled 3D']
for i in range(5):
 for g,color,style in [('B','0.6','--'),('G','#2878b5','-'),('R','#d95f02','-')]:
  x=a[f'{g}_25_{i}'];t=x[:,0]
  for row,y in enumerate([x[:,7],np.linalg.norm(x[:,4:6],axis=1),abs(x[:,6])]):
   axes[row,i].plot(t,y,label=g,color=color,ls=style,lw=1.5)
   axes[row,i].scatter(t[-1],y[-1],color=color,s=13)
 for row in range(3):axes[row,i].grid(alpha=.2)
 axes[0,i].axhline(.5,color='black',ls=':',lw=.8)
 axes[0,i].set_title(names[i]);axes[2,i].set_xlabel('Observed time (s)')
for row,label in enumerate(['True endpoint distance (m)','Horizontal path error (m)','Vertical path error (m)']):axes[row,0].set_ylabel(label)
axes[0,0].legend();fig.suptitle('Update 25 frozen policies | dots: observed termination | no extrapolation')
fig.savefig(base/'frozen_update25.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True)
for i in range(5):
 spec=TrajectorySpec(**evaluation_tasks()[i]['trajectory']);spec=replace(spec,origin=tuple(np.array(spec.origin)+evaluation_origin(5,i)))
 ref=Trajectory(spec).query(np.linspace(0,60,301))['position']
 axes[i].plot(ref[:,0],ref[:,1],color='black',ls=':',lw=1,label='path')
 axes[i].scatter(ref[-1,0],ref[-1,1],marker='*',color='black',s=40)
 for g,color in [('B','0.6'),('G','#2878b5'),('R','#d95f02')]:
  x=a[f'{g}_25_{i}'];axes[i].plot(x[:,1],x[:,2],label=g,color=color);axes[i].scatter(x[-1,1],x[-1,2],color=color,s=15)
 axes[i].set_title(names[i]);axes[i].set_xlabel('World x (m)');axes[i].grid(alpha=.2);axes[i].set_aspect('equal',adjustable='datalim')
axes[0].set_ylabel('World y (m)');axes[0].legend()
fig.savefig(base/'frozen_update25_xy.png',dpi=160)

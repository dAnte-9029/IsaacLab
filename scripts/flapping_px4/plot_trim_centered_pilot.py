"""Export scientific comparison figure from completed evidence only."""
import os,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[2];base=Path(os.environ.get('TRIM_OUTPUT_ROOT',root/'logs/unified_trim_centered_pilot_v1'))
c=json.loads((base/'comparison.json').read_text());u=[0,5,10,15,20,25]
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for g,color in zip('AMB',['#555555','#0072B2','#D55E00']):
 s=c['summary'][g];detail=c['details'][g]
 for ax,key in zip(axes[0],['seconds','distance']):ax.plot(u,[s[str(v)][key] for v in u],'o-',label=g,color=color)
 for ax,key in zip(axes[1],['common_h','common_v']):ax.plot(u,[next(t for t in detail[str(v)] if t['slot']==0)[key] for v in u],'o-',label=g,color=color)
for ax,title,ylabel in zip(axes.flat,['Frozen survival (5-task mean)','Projected effective distance (5-task mean)',f'Straight task: first {c["common_steps"]} steps',f'Straight task: first {c["common_steps"]} steps'],['seconds','m','horizontal RMSE (m)','vertical RMSE (m)']):
 ax.set(title=title,xlabel='PPO update',ylabel=ylabel,xticks=u);ax.grid(alpha=.25);ax.legend()
fig.savefig(base/'comparison.png',dpi=180)

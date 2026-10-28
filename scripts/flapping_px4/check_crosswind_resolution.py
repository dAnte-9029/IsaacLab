"""Compare already-generated targeted temporal/spatial refinements; no reruns."""
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--temporal',type=Path,required=True)
p.add_argument('--spatial',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
axes=['Fx','Fy','Fz','Mx','My','Mz']
base=pd.read_csv(a.baseline/'phase.csv.gz')
rows=[]
for label,path in [('time',a.temporal),('space',a.spatial)]:
    summary=json.loads((path/'summary.json').read_text())
    fine=pd.read_csv(path/'phase.csv.gz')
    for keys,g in fine[fine.component.eq('wings')].groupby(['model','frequency','case']):
        b=base[(base.model==keys[0])&(base.frequency==keys[1])&(base.case==keys[2])&(base.component=='wings')]
        joined=b.merge(g,on='phase_deg',suffixes=('_b','_f'))
        if len(joined)!=len(b): raise ValueError('Unmatched phases')
        for axis in axes:
            x,y=joined[axis+'_b'].to_numpy(),joined[axis+'_f'].to_numpy()
            rows.append(dict(refinement=label,model=keys[0],frequency=keys[1],case=keys[2],axis=axis,
                mean_baseline=x.mean(),mean_refined=g[axis].mean(),mean_difference=g[axis].mean()-x.mean(),
                matched_phase_mean_difference=y.mean()-x.mean(),
                rms_difference=np.sqrt(np.mean((y-x)**2)),rms_refined=np.sqrt(np.mean(y**2)),
                max_difference=np.max(abs(y-x)),maximum_phase_step_refined=np.max(abs(np.diff(np.r_[g[axis].to_numpy(),g[axis].iloc[0]])))))
    if any('failure' in v for v in summary['runs'].values()):
        print(label,'contains explicitly failed runs; see summary.json')
a.output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(a.output,index=False)
print(a.output)

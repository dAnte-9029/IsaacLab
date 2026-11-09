"""CPU-only fixed protocol boundary and channel tests."""
import torch
from run_sustained_tail_pulses import pulse_actions, GROUPS
expected={'O':[0,0,0,0],'R+':[0,.1,0,0],'R-':[0,-.1,0,0],'S+':[0,0,.1,.1],'S-':[0,0,-.1,-.1],'D+':[0,0,.1,-.1],'D-':[0,0,-.1,.1]}
for start in (120,144):
    for g in GROUPS:
        base=torch.arange(20,dtype=torch.float32).reshape(5,4)/20
        original=base.clone()
        for step in (start,start+1,start+18,start+19,start+60):
            y=pulse_actions(base,g,step,start)
            want=base.clone()
            if start<step<=start+18:want[1]+=torch.tensor(expected[g])
            assert torch.equal(y,want) and torch.equal(base,original)
        saturated=torch.ones(5,4)
        y=pulse_actions(saturated,g,start+1,start)
        assert torch.equal(y[0],saturated[0])
        assert (y.clamp(-1,1)<=1).all()
print('TAIL_PULSE_CPU_CHECKS_PASSED')
# A rotating reference frame must not be differentiated as inertial acceleration.
import numpy as np
t=np.arange(121)/60
v=np.tile([2.,0.,0.],(len(t),1));n=np.c_[np.cos(t),np.sin(t),np.zeros_like(t)]
a=(np.gradient(v,1/60,axis=0)*n).sum(1)
assert abs(a).max()==0 and abs(np.gradient((v*n).sum(1),1/60)).max()>1
# Common reference is identical for both perturbation signs.
yplus=(v+np.array([0.,.1,0.]))*n
yminus=(v-np.array([0.,.1,0.]))*n
np.testing.assert_allclose((yplus.sum(1)-yminus.sum(1))/.2,np.sin(t),atol=3e-15)
print('COMMON_FRAME_NUMERICAL_CHECKS_PASSED')

import math
import torch
from flapping_bot.physics.tail_aero import TailAeroCfg,TailAeroModel
from flapping_bot.physics.tail_v2 import TailV2Polar

def evaluate(alpha,delta=0.,q=0.):
    v=torch.tensor([[8*math.cos(alpha),0.,-8*math.sin(alpha)]],dtype=torch.float64)
    z=v[:,0]*0
    return TailAeroModel(TailAeroCfg(backend='v2'),'cpu').compute_surface_results(root_lin_vel_b=v,root_ang_vel_b=v.new_tensor([[0,q,0]]),elevator_rad=z+delta,rudder_rad=z)

def test_slope_symmetry_and_damping():
    h=1e-6
    a=evaluate(h)[0].effective_lift_slope_per_rad
    assert abs(float(evaluate(h)[0].lift_coefficient)/h-a)<1e-6
    pos=evaluate(.04,.03);neg=evaluate(-.04,-.03)
    for p,n in zip(pos,neg):
        torch.testing.assert_close(p.lift_coefficient,-n.lift_coefficient)
        torch.testing.assert_close(p.drag_coefficient,n.drag_coefficient)
    assert sum(float(r.moment_b_about_com[0,1]) for r in evaluate(0,q=h))-sum(float(r.moment_b_about_com[0,1]) for r in evaluate(0,q=-h))<0
    assert sum(float(r.moment_b_about_com[0,1]) for r in evaluate(0,delta=h))-sum(float(r.moment_b_about_com[0,1]) for r in evaluate(0,delta=-h))>0

def test_transition_c1():
    h=1e-6
    for angle in (12,30,-12,-30):
        x=math.radians(angle)
        for field in ('lift_coefficient','drag_coefficient'):
            f=lambda y:float(getattr(evaluate(y)[0],field))
            assert abs((f(x+h)-f(x))/h-(f(x)-f(x-h))/h)<1e-3

def test_invalid_parameters():
    import pytest
    with pytest.raises(ValueError):TailV2Polar(alpha_f_deg=10)

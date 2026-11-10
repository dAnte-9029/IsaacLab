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

def test_mutating_configuration_rebuilds_cached_polar():
    cfg=TailAeroCfg(backend='v2')
    model=TailAeroModel(cfg,'cpu')
    v=torch.tensor([[8.,0.,-.8]],dtype=torch.float64);z=v[:,0]*0
    kw=dict(root_lin_vel_b=v,root_ang_vel_b=v*0,elevator_rad=z,rudder_rad=z)
    first=model.compute_wrench(**kw)[0].clone()
    cfg.v2_parameters['a2d']=5.
    changed=model.compute_wrench(**kw)[0]
    assert not torch.allclose(first,changed)
    cfg.v2_parameters.clear()
    torch.testing.assert_close(model.compute_wrench(**kw)[0],first)

def test_json_separate_polars(tmp_path):
    import json
    from dataclasses import asdict
    from flapping_bot.physics.tail_v2 import load_tail_config
    data=dict(backend='v2',polar_horizontal=asdict(TailV2Polar()),polar_vertical=asdict(TailV2Polar()))
    path=tmp_path/'candidate.json';path.write_text(json.dumps(data))
    cfg=load_tail_config(path);model=TailAeroModel(cfg,'cpu')
    v=torch.tensor([[8.,1.,-.8]],dtype=torch.float64);z=v[:,0]*0
    kw=dict(root_lin_vel_b=v,root_ang_vel_b=v*0,elevator_rad=z,rudder_rad=z)
    first=model.compute_surface_results(**kw)
    cfg.polar_vertical['a2d']=5.
    second=model.compute_surface_results(**kw)
    torch.testing.assert_close(first[0].force_b,second[0].force_b)
    assert not torch.allclose(first[2].force_b,second[2].force_b)
    data['polar_horizontal'].pop('a2d');path.write_text(json.dumps(data))
    import pytest
    with pytest.raises(ValueError):load_tail_config(path)

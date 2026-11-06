"""Strict phase-two/three parity, including original rejection boundaries."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import torch
import pytest
import flapping_bot.physics.control_aero_3d as current
from flapping_bot.physics.ullt_geometry import elliptic_geometry, uniform_translation

BASE = Path(__file__).resolve().parents[2] / 'unified-rl-performance-phase2/source/flapping_bot/flapping_bot/physics/control_aero_3d.py'
spec = importlib.util.spec_from_file_location('flapping_bot.physics._phase2_reference', BASE)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('speed', [0., .001, .2499, .2501, 8., -8.])
def test_loads_exact(dtype, speed):
    g = elliptic_geometry(num_sections=20, batch_size=7, dtype=dtype)
    m = uniform_translation(g, speed)
    flow = m.flow_three_quarter_w_mps.clone(); flow[..., 2] = abs(speed)*.05
    m = replace(m, flow_three_quarter_w_mps=flow)
    cfg = current.ControlAero3DConfig(backend='I3D', include_added_mass=False)
    outputs = [mod.ControlAero3D(cfg).evaluate(g, m, flow[:,0]*0) for mod in (reference,current)]
    for name in vars(outputs[0]):
        assert torch.equal(getattr(outputs[0],name),getattr(outputs[1],name)), name

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('case', ['regular','self','bound_singular','wake_singular'])
def test_influence_and_geometric_rejections(dtype, case):
    g = elliptic_geometry(num_sections=20, batch_size=3, dtype=dtype)
    target = g.quarter_chord_w_m.clone()
    direction = torch.zeros_like(target); direction[...,0] = -1
    omit = case != 'bound_singular'
    if case == 'regular': target[...,0] += .1
    if case == 'wake_singular': target = g.edge_end_w_m + direction
    outputs=[]; errors=[]
    for mod in (reference,current):
        try: outputs.append(mod.horseshoe_influence(g,target,direction,omit_self_bound=omit)); errors.append(None)
        except RuntimeError as exc: errors.append(str(exc))
    assert errors[0] == errors[1]
    if 'singular' in case: assert errors[0] is not None
    else:
        for a,b in zip(*outputs): assert torch.equal(a,b)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('condition',[999999.,1000000.,1000001.,float('inf')])
def test_original_condition_threshold(monkeypatch,dtype,condition):
    # Feed a controlled diagonal circulation system through the original solve
    # path. No condition check is skipped or replaced: both call linalg.cond.
    g=elliptic_geometry(num_sections=2,dtype=dtype)
    m=uniform_translation(g,8.)
    cfg=current.ControlAero3DConfig(backend='I3D',include_added_mass=False)
    desired=torch.diag(torch.tensor([condition,1.],dtype=dtype))[None]
    a=torch.pi*g.chord_m
    kn=(desired-torch.eye(2,dtype=dtype)[None])/a[...,None]
    k=kn[...,None]*g.normal_w[:,:,None]
    outcomes=[]
    for mod in (reference,current):
        monkeypatch.setattr(mod,'horseshoe_influence',lambda *args,**kwargs:(k,k*0))
        try:
            out=mod.ControlAero3D(cfg).evaluate(g,m,torch.zeros(1,3,dtype=dtype))
            outcomes.append(('accepted',out.condition.tolist(),out.residual.tolist()))
        except RuntimeError as exc: outcomes.append(('rejected',str(exc)))
    assert outcomes[0]==outcomes[1]

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_residual_rejection_preserved(monkeypatch,dtype):
    g=elliptic_geometry(num_sections=20,dtype=dtype)
    m=uniform_translation(g,8.)
    monkeypatch.setattr(torch.linalg,'solve',lambda matrix,rhs: torch.ones_like(rhs))
    for mod in (reference,current):
        with pytest.raises(RuntimeError,match='circulation residual exceeded tolerance'):
            mod.ControlAero3D(current.ControlAero3DConfig(backend='I3D',include_added_mass=False)).evaluate(g,m,torch.zeros(1,3,dtype=dtype))

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_undefined_wake_rejection_preserved(dtype):
    g=elliptic_geometry(num_sections=2,dtype=dtype)
    m=uniform_translation(g,8.)
    f=m.flow_three_quarter_w_mps.clone();f[:,1]*=-1
    m=replace(m,flow_three_quarter_w_mps=f)
    for mod in (reference,current):
        with pytest.raises(RuntimeError,match='undefined wake direction'):
            mod.ControlAero3D(current.ControlAero3DConfig(backend='I3D',include_added_mass=False)).evaluate(g,m,torch.zeros(1,3,dtype=dtype))

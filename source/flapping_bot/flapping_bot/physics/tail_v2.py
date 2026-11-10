"""Three-unit quasi-steady tail; SI, body FLU, moments about supplied base COM.

Nominal closure parameters are engineering priors, not aircraft identification.
No legacy effectiveness multipliers are used. Geometry remains unchanged.
"""
from dataclasses import dataclass
import math
from functools import lru_cache
import torch
from .tail_aero import (TailAeroModel, TailSurfaceResult, _RIGHT_ELEVON_POLYGON_MM,
    _VERTICAL_TAIL_POLYGON_MM, _polyline_intersections, _horizontal_surface_center_mm,
    _vertical_surface_center_mm)

@dataclass(frozen=True)
class TailV2Polar:
    a2d: float = 2 * math.pi
    cl_limit: float = 0.9
    kappa: float = 0.0  # unresolved: no transferable flap plateau calibration
    alpha_s_deg: float = 12.0
    alpha_f_deg: float = 30.0
    cd_min: float = 0.025
    cd_delta: float = 0.0  # unresolved lower-bound candidate
    cl_md_delta: float = 0.0  # unresolved lower-bound candidate
    efficiency: float = 0.6
    cn90: float = 1.8  # numerical extrapolation only
    cd_parallel: float = 0.025

    def __post_init__(self):
        vals = vars(self)
        if not all(math.isfinite(v) for v in vals.values()):
            raise ValueError("Nonfinite polar")
        if not (0 < self.alpha_s_deg < self.alpha_f_deg < 90):
            raise ValueError("Invalid transition angles")
        if min(self.a2d,self.cl_limit,self.efficiency,self.cn90) <= 0 or self.kappa < 0:
            raise ValueError("Invalid lift parameters")
        if min(self.cd_min,self.cd_delta,self.cd_parallel) < 0:
            raise ValueError("Negative drag parameters")


def smoothstep(x):
    x = x.clamp(0, 1)
    return x*x*(3-2*x)


@lru_cache(maxsize=8)
def geometry_eta(poly, sweep_axis, samples=1024):
    lo=min(p[sweep_axis] for p in poly); hi=max(p[sweep_axis] for p in poly)
    weighted=area=flap=0.0
    for i in range(samples):
        cuts=_polyline_intersections(poly,lo+(i+0.5)*(hi-lo)/samples,sweep_axis=sweep_axis)
        if len(cuts)!=2: continue
        c=cuts[1]-cuts[0]
        f=max(0.,-cuts[0] if sweep_axis==0 else cuts[1])/c
        f=min(1.,f)
        theta=math.acos(2*f-1)
        weighted+=c*(math.pi+math.sin(theta)-theta)/math.pi
        area+=c; flap+=c*f
    return weighted/area, flap/area


class TailV2Model(TailAeroModel):
    def __init__(self,cfg,device,polar=None):
        super().__init__(cfg,device)
        self.polar=polar or TailV2Polar(**cfg.v2_parameters)
        self.horizontal=TailV2Polar(**cfg.polar_horizontal) if cfg.polar_horizontal is not None else self.polar
        self.vertical=TailV2Polar(**cfg.polar_vertical) if cfg.polar_vertical is not None else self.polar

    def compute_surface_results(self, *, root_lin_vel_b, root_ang_vel_b,
            rudder_rad, left_elevon_rad=None, right_elevon_rad=None,
            elevator_rad=None, base_com_pos_b=None):
        if elevator_rad is not None:
            if left_elevon_rad is not None or right_elevon_rad is not None:
                raise ValueError("Ambiguous elevator inputs")
            left_elevon_rad=right_elevon_rad=elevator_rad
        n=root_lin_vel_b.shape[0]
        for d in (left_elevon_rad,right_elevon_rad,rudder_rad):
            if d is None or d.shape!=(n,): raise ValueError("Expected actual joint angles (batch,)")
        cfg=self.cfg
        sh=cfg.fixed_horizontal.area+cfg.left_elevon.area+cfg.right_elevon.area
        sv=cfg.fixed_vertical.area+cfg.rudder.area
        sy,cx=horizontal_center()
        hz,ax=vertical_center()
        eh,_=geometry_eta(_RIGHT_ELEVON_POLYGON_MM,0)
        ev,_=geometry_eta(_VERTICAL_TAIL_POLYGON_MM,1)
        units=[('left_horizontal',sh/2,cfg.fixed_horizontal.span_m**2/sh,(-.5365+cx/1000,sy/1000,-.00335),(0,1,0),eh,left_elevon_rad),
               ('right_horizontal',sh/2,cfg.fixed_horizontal.span_m**2/sh,(-.5365+cx/1000,-sy/1000,-.00335),(0,1,0),eh,right_elevon_rad),
               ('vertical',sv,cfg.fixed_vertical.span_m**2/sv,(-.5045-ax/1000,0,-.00085+hz/1000),(0,0,1),ev,rudder_rad)]
        results=[]
        for name,area,ar,pos,axis,eta,delta in units:
            p=self.vertical if name == "vertical" else self.horizontal
            r=root_lin_vel_b.new_tensor(pos).expand(n,3)
            if base_com_pos_b is not None: r=r-base_com_pos_b
            span=root_lin_vel_b.new_tensor(axis).expand(n,3)
            chord=root_lin_vel_b.new_tensor((-1,0,0)).expand(n,3)
            v=root_lin_vel_b+torch.linalg.cross(root_ang_vel_b,r)
            vp=v-(v*span).sum(-1,keepdim=True)*span
            speed=vp.norm(dim=-1); vd=vp/speed.clamp_min(1e-9)[:,None]
            alpha=torch.atan2((torch.linalg.cross(chord,-vd)*span).sum(-1),(-vd*chord).sum(-1))
            # Legacy deflection_sign=-1 rotates chord; alpha increases with actual joint angle.
            ae=alpha+eta*delta
            a=p.a2d*ar/(p.a2d/math.pi+math.sqrt((p.a2d/math.pi)**2+ar**2))
            limit=p.cl_limit*(1+p.kappa*smoothstep(delta.abs()/math.radians(20)))
            low=limit*torch.tanh(a*ae/limit)
            w=smoothstep((ae.abs()-math.radians(p.alpha_s_deg))/math.radians(p.alpha_f_deg-p.alpha_s_deg))
            cl=(1-w)*low+w*p.cn90*ae.sin()*ae.cos()
            cdlo=p.cd_min+p.cd_delta*delta.square()+(low-p.cl_md_delta*delta).square()/(math.pi*p.efficiency*ar)
            cd=(1-w)*cdlo+w*(p.cd_parallel*ae.cos().square()+p.cn90*ae.sin().square())
            q=.5*cfg.air_density*speed.square()
            lf=(q*area*cl)[:,None]*torch.linalg.cross(vd,span)
            df=-(q*area*cd)[:,None]*vd
            force=lf+df
            results.append(TailSurfaceResult(name,force,torch.linalg.cross(r,force),lf,df,v,vp,speed,alpha,ae,q,cl,cd,w>0,delta,r,chord,span,1.,a,1.))
        return tuple(results)

@lru_cache(maxsize=1)
def horizontal_center():
    return _horizontal_surface_center_mm(_RIGHT_ELEVON_POLYGON_MM)

@lru_cache(maxsize=1)
def vertical_center():
    return _vertical_surface_center_mm(_VERTICAL_TAIL_POLYGON_MM)


def load_tail_config(path):
    """Load an explicit candidate; reject unknown/missing polar fields."""
    import json
    from dataclasses import fields
    from .tail_aero import TailAeroCfg
    data=json.loads(__import__('pathlib').Path(path).read_text())
    if data['backend']=='legacy':
        return TailAeroCfg(fixed_horizontal_effectiveness=.5,elevon_effectiveness=1.2)
    expected={f.name for f in fields(TailV2Polar)}
    for key in ('polar_horizontal','polar_vertical'):
        if set(data[key]) != expected: raise ValueError('Incomplete or unknown polar fields: '+key)
        TailV2Polar(**data[key])
    return TailAeroCfg(backend='v2',polar_horizontal=data['polar_horizontal'],polar_vertical=data['polar_vertical'])

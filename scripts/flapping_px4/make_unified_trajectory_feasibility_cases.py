"""Deterministic B-spline study recipes, independent of legacy curricula."""
import argparse
import json
from pathlib import Path
from flapping_bot.path_tracking.unified_trajectory import Trajectory,TrajectorySpec
from flapping_bot.path_tracking.trajectory_feasibility import features


def recipe(kappa=0.,gamma=0.,spacing=10.,reverse=False,hold_count=5):
    levels=[0.]*4+[1.]*hold_count+([-1.]*hold_count if reverse else [])+[0.]*4
    interior=[20.+spacing*j for j in range(len(levels)-4)]
    length=interior[-1]+30.
    return TrajectorySpec(knots=tuple([0.]*4+interior+[length]*4),
        kappa_coeff=tuple(kappa*v for v in levels),gamma_coeff=tuple(gamma*v for v in levels),length_m=length)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--controller',type=Path,required=True);a=p.parse_args()
    gains=json.loads(a.controller.read_text());cases=[]
    for name,kwargs in [('turn_mild',dict(kappa=.005)),('turn_moderate',dict(kappa=.02)),
        ('climb_mild',dict(gamma=.04)),('climb_moderate',dict(gamma=.12)),
        ('turn_reversal_30m',dict(kappa=.01,reverse=True,hold_count=4)),
        ('turn_reversal_15m',dict(kappa=.01,reverse=True,hold_count=4,spacing=5.)),
        ('vertical_reversal_30m',dict(gamma=.08,reverse=True,hold_count=4)),
        ('vertical_reversal_15m',dict(gamma=.08,reverse=True,hold_count=4,spacing=5.)),
        ('coupled',dict(kappa=.01,gamma=.08))]:
        spec=recipe(**kwargs);path=Trajectory(spec)
        cases.append(dict(name=name,trajectory=spec.to_dict(),gains=gains,features=features(path)))
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(cases,indent=2))
if __name__=='__main__':main()


def isolated_reversal(kappa=0.,gamma=0.,transition_m=15.):
    """Fixed 20 m signed plateaus and 30 m entry/exit ramps; vary reversal only."""
    d=transition_m
    interior=[20.,30.,40.,50.,70.,70.+d/3,70.+2*d/3,70.+d,90.+d,100.+d,110.+d,120.+d]
    levels=[0.]*4+[1.]*4+[-1.]*4+[0.]*4
    return TrajectorySpec(tuple([0.]*4+interior+[150.+d]*4),
        tuple(kappa*v for v in levels),tuple(gamma*v for v in levels),150.+d)

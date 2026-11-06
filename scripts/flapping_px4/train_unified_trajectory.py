"""Unified L/I PPO: same experiment contract, fixed or adaptive task sampling."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import traceback


def main():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['L3D','I3D'],required=True)
    p.add_argument('--sampling',choices=['fixed','adaptive'],required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--num-envs',type=int,default=256)
    p.add_argument('--rollout-steps',type=int,default=48)
    p.add_argument('--iterations',type=int,default=1000,help='Additional PPO iterations; no implicit training launch')
    p.add_argument('--eval-interval',type=int,default=25,help='Same frozen development evaluation budget for both modes; 0 disables')
    p.add_argument('--eval-step-budget',type=int,default=7200,help='Fixed steps per evaluation env, including padding; unfinished tasks fail the evaluation run')
    p.add_argument('--save-interval',type=int,default=25)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--resume',action='store_true',help='Restore optimizer/RNG/sampler/evidence snapshot; fresh physical episodes')
    p.add_argument('--eval-suite',choices=['development','final'],default='development')
    p.add_argument('--evaluate',action='store_true',help='Frozen 5-task complete-episode evaluation only')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--smoke-check',action='store_true',help='Additionally check isolated partial reset, without changing task contract')
    p.add_argument('--rules',type=Path,default=root.parent/'control-oriented-3d-wing-aero/logs/unified_trajectory_phase2_v1/screening_rules_v2.json')
    p.add_argument('--plant-assets',type=Path,default=root.parent/'control-oriented-3d-wing-aero')
    args=p.parse_args()
    if args.eval_suite=='final' and not args.evaluate: p.error('final suite is evaluation-only')
    if (min(args.num_envs,args.rollout_steps,args.iterations,args.save_interval,args.threads)<1
            or args.eval_interval<0 or args.eval_step_budget<1 or args.num_envs*args.rollout_steps<4): p.error('Invalid training sizes')
    if (args.resume or args.evaluate) and not args.checkpoint: p.error('checkpoint required')
    if args.resume and args.evaluate: p.error('Use either resume or evaluate')
    args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    if args.checkpoint: args.checkpoint=args.checkpoint.resolve()
    (args.output/'launch.json').write_text(json.dumps(dict(command=sys.argv,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}),indent=2))
    # Snapshot the integrated code; parent commit and immutable physical-source identity are in contract.json.
    snapshot=args.output/'source_snapshot'; snapshot.mkdir()
    for folder,pattern in [('analysis','unified_*.py'),('path_tracking','*trajectory*.py'),('direct/flapping_bot','unified_*.py')]:
        for path in (root/'source/flapping_bot/flapping_bot'/folder).glob(pattern):
            target=snapshot/folder/path.name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
    app=None
    try:
        (args.output/'status.json').write_text('{"status":"running"}\n')
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,device='cpu',kit_args=(
            f'--ext-folder {args.plant_assets.resolve()}/source/flapping_bot/native_extensions '
            '--enable omni.flapping_bot.holonomic_constraint')).app
        from flapping_bot.analysis.unified_ppo_training import train,evaluate
        result=(evaluate if args.evaluate else train)(args,root)
        result['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        (args.output/'status.json').write_text('{"status":"completed"}\n')
        print('UNIFIED_PPO_PASSED',args.output/'summary.json',flush=True)
    except BaseException:
        (args.output/'status.json').write_text(json.dumps(dict(status='failed',error=traceback.format_exc()),indent=2))
        raise
    finally:
        if app is not None: app.close()

if __name__=='__main__': main()

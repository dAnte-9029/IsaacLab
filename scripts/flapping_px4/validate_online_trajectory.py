"""Actual continuous sampling and bounded feedback scaling; synthetic only."""
import argparse
from pathlib import Path
from flapping_bot.analysis.unified_online_validation import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rules',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budget',type=int,default=1024)
    parser.add_argument('--sizes',type=int,nargs='+',default=[0,256,4096])
    args=parser.parse_args()
    run(args.rules,args.output,args.budget,tuple(args.sizes))

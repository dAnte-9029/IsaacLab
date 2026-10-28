"""CPU prescribed-motion lateral aerodynamics; no Isaac scene or training."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'source/flapping_bot'))
from flapping_bot.analysis.crosswind_aero import run

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--sections', type=int, default=10)
    p.add_argument('--max-cycles', type=int, default=32)
    p.add_argument('--models', nargs='+', choices=['delaurier', 'ullt4', 'ullt4_A'], default=['delaurier', 'ullt4', 'ullt4_A'])
    p.add_argument('--frequencies', nargs='+', type=float, default=[3., 4.])
    p.add_argument('--cases', nargs='+', default=None)
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--report-only', action='store_true')
    a = p.parse_args()
    if a.report_only:
        import pandas as pd
        from flapping_bot.analysis.crosswind_aero import analyze
        analyze(a.output_dir, pd.read_csv(a.output_dir/'means.csv'), pd.read_csv(a.output_dir/'phase.csv.gz'))
        raise SystemExit(0)
    result = run(a.output_dir, a.steps, a.sections, a.max_cycles, a.models, a.frequencies, a.cases, a.resume_from)

    raise SystemExit(int(any('failure' in v for v in result['runs'].values())))

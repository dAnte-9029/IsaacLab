#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir="${1:-logs/tail_v2_control_reproduction}"
if [ -e "$run_dir" ]; then
  echo "Refusing to overwrite existing results: $run_dir" >&2
  exit 1
fi
export TERM=xterm
export CONDA_PREFIX=/home/zn/anaconda3/envs/env_isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$PWD/source/flapping_bot${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$run_dir"
./isaaclab.sh -p scripts/flapping_px4/tail_v2_search.py --evidence-dir "$run_dir/evidence"
test -f "$run_dir/evidence/selected.json"
./isaaclab.sh -p scripts/flapping_px4/optimize_tail_v2_physx.py --headless --evidence-dir "$run_dir/evidence" --output "$run_dir/physx"
test -f "$run_dir/physx/completion.json"
./isaaclab.sh -p scripts/flapping_px4/tail_v2_control_report.py --logs "$run_dir/physx" --evidence-dir "$run_dir/evidence"
test -f "$run_dir/evidence/report_zh.md"

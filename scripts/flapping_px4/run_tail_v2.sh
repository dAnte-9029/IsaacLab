#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TERM=xterm
export CONDA_PREFIX=/home/zn/anaconda3/envs/env_isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="$PWD/source/flapping_bot${PYTHONPATH:+:$PYTHONPATH}"
./isaaclab.sh -p scripts/flapping_px4/calibrate_tail_v2.py --output docs/results/tail_v2_v1
for backend in legacy v2; do
  ./isaaclab.sh -p scripts/flapping_px4/validate_tail_v2_physx.py --backend L3D --tail-backend "$backend" --rounds 2 --headless --output "logs/tail_v2_v1/$backend"
  test -f "logs/tail_v2_v1/$backend/control_results.json"
done
./isaaclab.sh -p scripts/flapping_px4/summarize_tail_v2.py

#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/source/flapping_bot:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH:-}"
base=${TRIM_OUTPUT_ROOT:-$PWD/logs/unified_trim_centered_pilot_v1}
python -c 'import json,sys; g=json.load(open(sys.argv[1])); assert all(g.values()) and g["steps"]==12288; assert json.load(open(sys.argv[2]))["passed"]' "$base/A_gate.json" "$base/paired_initialization_check.json"
for pair in M:trim_mean B:trim_local; do
 group=${pair%%:*}; mode=${pair#*:}
 ./isaaclab.sh -p scripts/flapping_px4/train_unified_trajectory.py --initialization "$mode" --backend L3D --sampling fixed --seed 0 --num-envs 256 --rollout-steps 48 --iterations 25 --eval-interval 25 --save-interval 5 --eval-step-budget 7200 --threads 1 --output "$base/$group" > "$base/${group}_console.log" 2>&1
 python -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="completed"' "$base/$group/status.json"
done

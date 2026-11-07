#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/source/flapping_bot:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH:-}"
base=${GOAL_OUTPUT_ROOT:-$PWD/logs/unified_goal_consistent_reward_v1}
python -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]; assert json.load(open(sys.argv[2]))["passed"]' "$base/pretraining_gate.json" "$base/historical_replay_check.json"
mkdir -p "$base"
if [ -e "$base/G" ] || [ -e "$base/R" ]; then
 echo "Training output already exists: $base" >&2; exit 1
fi
for pair in G:strict_goal R:recovery_goal; do
 group=${pair%%:*}; variant=${pair#*:}
 ./isaaclab.sh -p scripts/flapping_px4/train_unified_trajectory.py --reward-variant "$variant" --initialization trim_local --backend L3D --sampling fixed --seed 0 --num-envs 256 --rollout-steps 48 --iterations 25 --eval-interval 25 --save-interval 5 --eval-step-budget 7200 --threads 1 --output "$base/$group" > "$base/${group}_console.log" 2>&1
 python -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="completed"' "$base/$group/status.json"
 for update in 0 5 10 15 20; do
  tag=$(printf '%06d' "$update")
  dest="$base/evaluation_${group}_${tag}"
  ./isaaclab.sh -p scripts/flapping_px4/train_unified_trajectory.py --evaluate --reward-variant "$variant" --backend L3D --sampling fixed --seed 0 --checkpoint "$base/$group/model_${tag}.pt" --output "$dest" --eval-step-budget 7200 --threads 1 > "${dest}.log" 2>&1
  python -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="completed"; assert json.load(open(sys.argv[2]))["passed"]' "$dest/status.json" "$dest/summary.json"
 done
done

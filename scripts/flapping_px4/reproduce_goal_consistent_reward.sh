#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/source/flapping_bot:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH:-}"
export GOAL_OUTPUT_ROOT=${GOAL_OUTPUT_ROOT:-$PWD/logs/unified_goal_consistent_reward_v1_reproduction}
# Refuse to overwrite any prior experiment.
if [ -e "$GOAL_OUTPUT_ROOT" ]; then echo "Output already exists: $GOAL_OUTPUT_ROOT" >&2; exit 1; fi
mkdir -p "$GOAL_OUTPUT_ROOT"
./isaaclab.sh -p -m pytest tests/test_unified_goal_reward.py tests/test_unified_ppo_contract.py tests/test_online_trajectory.py tests/test_unified_trajectory_cache.py -q > "$GOAL_OUTPUT_ROOT/tests.log" 2>&1
# isaaclab.sh may mask Python exit status.
rg -q '44 passed' "$GOAL_OUTPUT_ROOT/tests.log"
./isaaclab.sh -p scripts/flapping_px4/check_goal_reward_scale.py > "$GOAL_OUTPUT_ROOT/scale.log" 2>&1
./isaaclab.sh -p scripts/flapping_px4/check_goal_history.py > "$GOAL_OUTPUT_ROOT/historical_check.log" 2>&1
./isaaclab.sh -p scripts/flapping_px4/check_goal_physx.py > "$GOAL_OUTPUT_ROOT/physx_check.log" 2>&1
bash scripts/flapping_px4/run_goal_consistent_reward.sh
./isaaclab.sh -p scripts/flapping_px4/analyze_goal_endpoints.py > "$GOAL_OUTPUT_ROOT/endpoint_analysis.log" 2>&1
for group in G R; do
 ./isaaclab.sh -p scripts/flapping_px4/analyze_goal_training.py "$group" > "$GOAL_OUTPUT_ROOT/${group}_analysis.log" 2>&1
done
./isaaclab.sh -p scripts/flapping_px4/check_goal_witnesses.py > "$GOAL_OUTPUT_ROOT/witness_check.log" 2>&1
./isaaclab.sh -p scripts/flapping_px4/check_goal_recovery.py > "$GOAL_OUTPUT_ROOT/recovery.log" 2>&1
./isaaclab.sh -p scripts/flapping_px4/plot_goal_consistent_reward.py
./isaaclab.sh -p scripts/flapping_px4/report_goal_consistent_reward.py
python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["GOAL_OUTPUT_ROOT"]); assert json.load(open(p/"endpoint_comparison.json"))["passed"]; assert json.load(open(p/"recovery_check.json"))["actual_G_R_shared_initialization"]'

#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/source/flapping_bot:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH:-}"
root=$PWD
base=${TRIM_OUTPUT_ROOT:-$PWD/logs/unified_trim_centered_pilot_v1}
for group in A M B; do
 updates='0 5 10 15 20'
 ckdir="$base/$group"
 if [ "$group" = A ]; then
  updates='5 15 20'
  ckdir=$root/../unified-rl-performance-phase4/logs/unified_pilot_l3d_fixed_seed0_25u
 fi
 for update in $updates; do
  tag=$(printf '%06d' "$update")
  while [ ! -f "$ckdir/model_${tag}.pt" ]; do
   if [ -f "$ckdir/status.json" ] && rg -q '"failed"' "$ckdir/status.json"; then exit 1; fi
   sleep 5
  done
  dest="$base/evaluation_${group}_${tag}"
  runroot=$root
  if [ "$group" = A ]; then
   runroot=$(realpath "$root/../unified-rl-performance-phase4")
   export PYTHONPATH="$runroot/source/flapping_bot"
   if [ "$update" = 5 ]; then dest="${dest}_original_entry"; fi
  else
   export PYTHONPATH="$root/source/flapping_bot:$root/source/isaaclab:$root/source/isaaclab_assets:$root/source/isaaclab_tasks:$root/source/isaaclab_rl"
  fi
  "$runroot/isaaclab.sh" -p "$runroot/scripts/flapping_px4/train_unified_trajectory.py" --evaluate --backend L3D --sampling fixed --seed 0 --checkpoint "$ckdir/model_${tag}.pt" --output "$dest" --eval-step-budget 7200 --threads 1 > "${dest}.log" 2>&1
  python -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="completed"; assert json.load(open(sys.argv[2]))["passed"]' "$dest/status.json" "$dest/summary.json"
 done
done

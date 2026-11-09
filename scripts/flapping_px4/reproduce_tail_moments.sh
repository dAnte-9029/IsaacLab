#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
export MPLCONFIGDIR=/tmp/tail-moments-mpl
export PYTHONPATH="$PWD/source/flapping_bot:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH:-}"
result_dir="${1:-logs/tail_moment_attribution_v1}"
mkdir "$result_dir"
history_dir=/home/zn/IsaacLab/.worktrees/unified-goal-consistent-reward-v1/logs/unified_longrun_l3d_G_seed0_u026_150
for start in 120 144; do
 for group in O S+ S-; do
  ./isaaclab.sh -p scripts/flapping_px4/diagnose_tail_moments.py --g150 "$history_dir/model_000150.pt" --history "$history_dir/evaluation_000150" --output "$result_dir/T${start}_${group}" --start "$start" --group "$group" > "$result_dir/T${start}_${group}.log" 2>&1
  python - "$result_dir/T${start}_${group}" <<'PY_CHECK'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); assert json.loads((p/'summary.json').read_text())['passed']; assert (p/'physics.npz').exists()
PY_CHECK
 done
done

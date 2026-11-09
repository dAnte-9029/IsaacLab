#!/usr/bin/env bash
# Exactly eight independent processes; output must not exist. No retries.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
export PYTHONPATH="$PWD/source/flapping_bot:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH:-}"
intervention_output="${1:-logs/unified_sustained_cross_policy_intervention_v1}"
if [[ -e "$intervention_output" ]]; then
    echo "Output exists; refusing overwrite: $intervention_output" >&2
    exit 1
fi
historical_logs=/home/zn/IsaacLab/.worktrees/unified-goal-consistent-reward-v1/logs
./isaaclab.sh -p scripts/flapping_px4/check_sustained_cross_policy.py
./isaaclab.sh -p scripts/flapping_px4/run_sustained_cross_policy.py \
    --g25 "$historical_logs/unified_goal_consistent_reward_v1/G/model_000025.pt" \
    --g150 "$historical_logs/unified_longrun_l3d_G_seed0_u026_150/model_000150.pt" \
    --history "$historical_logs/unified_longrun_l3d_G_seed0_u026_150/evaluation_000150" \
    --output "$intervention_output"
# isaaclab.sh can mask a child exit code: require the actual completed receipt.
python - "$intervention_output" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);assert json.loads((p/'status.json').read_text())['status']=='completed'
assert len(list(p.glob('T*/summary.json')))==8
PY
./isaaclab.sh -p scripts/flapping_px4/analyze_sustained_cross_policy.py --input "$intervention_output"
python - "$intervention_output" <<'PY'
import json,sys
from pathlib import Path
r=json.loads((Path(sys.argv[1])/'analysis/results.json').read_text())
assert len(r['audit'])==8 and len(r['metrics'])==16
print('INTERVENTION_ANALYSIS_PASSED')
PY

./isaaclab.sh -p scripts/flapping_px4/verify_sustained_cross_policy.py --input "$intervention_output" --history "$historical_logs/unified_longrun_l3d_G_seed0_u026_150/evaluation_000150"

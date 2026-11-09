#!/usr/bin/env bash
# Exactly fourteen independent processes; output must not exist. No retries.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tail-authority-mpl"
export PYTHONPATH="$PWD/source/flapping_bot:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH:-}"
intervention_output="${1:-logs/unified_sustained_tail_authority_v1}"
if [[ -e "$intervention_output" ]]; then
    echo "Output exists; refusing overwrite: $intervention_output" >&2
    exit 1
fi
historical_logs=/home/zn/IsaacLab/.worktrees/unified-goal-consistent-reward-v1/logs
./isaaclab.sh -p scripts/flapping_px4/check_sustained_tail_pulses.py
./isaaclab.sh -p scripts/flapping_px4/run_sustained_tail_pulses.py \
    --g150 "$historical_logs/unified_longrun_l3d_G_seed0_u026_150/model_000150.pt" \
    --history "$historical_logs/unified_longrun_l3d_G_seed0_u026_150/evaluation_000150" \
    --output "$intervention_output"
# isaaclab.sh can mask a child exit code: require the actual completed receipt.
python - "$intervention_output" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);assert json.loads((p/'status.json').read_text())['status']=='completed'
assert len(list(p.glob('T*/summary.json')))==14
PY
./isaaclab.sh -p scripts/flapping_px4/analyze_sustained_tail_pulses.py --input "$intervention_output"
python - "$intervention_output" <<'PY'
import json,sys
from pathlib import Path
r=json.loads((Path(sys.argv[1])/'analysis/results.json').read_text())
assert len(r['audit'])==14
print('INTERVENTION_ANALYSIS_PASSED')
PY

./isaaclab.sh -p scripts/flapping_px4/verify_sustained_tail_pulses.py --input "$intervention_output" --history "$historical_logs/unified_longrun_l3d_G_seed0_u026_150/evaluation_000150"
python - "$intervention_output" <<'PY_CHECK'
import json,sys
from pathlib import Path
assert json.loads((Path(sys.argv[1])/'offline_acceptance.json').read_text())['passed']
PY_CHECK

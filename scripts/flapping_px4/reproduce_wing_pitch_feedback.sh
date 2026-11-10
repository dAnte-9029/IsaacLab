#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm MPLCONFIGDIR=/tmp/wing-pitch-mpl
export PYTHONPATH="$PWD/source/flapping_bot:${PYTHONPATH:-}"
out="${1:-/tmp/wing-pitch-feedback-reproduction}"
./isaaclab.sh -p scripts/flapping_px4/diagnose_wing_pitch_feedback.py --output "$out"
# Wrapper may hide Python exit status: verify actual complete artifacts separately.
python scripts/flapping_px4/check_wing_pitch_feedback.py "$out"

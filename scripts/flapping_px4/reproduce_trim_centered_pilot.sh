#!/usr/bin/env bash
set -euo pipefail
export TERM=xterm
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd "$(dirname "$0")/../.."
export TRIM_OUTPUT_ROOT=${TRIM_OUTPUT_ROOT:-$PWD/logs/unified_trim_centered_pilot_v1_reproduction}
mkdir "$TRIM_OUTPUT_ROOT"
export PYTHONPATH="$PWD/source/flapping_bot:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH:-}"
./isaaclab.sh -p scripts/flapping_px4/train_unified_trajectory.py --backend L3D --sampling fixed --seed 0 --num-envs 256 --rollout-steps 48 --iterations 1 --eval-interval 25 --save-interval 5 --eval-step-budget 7200 --threads 1 --output "$TRIM_OUTPUT_ROOT/A_reproduction" > "$TRIM_OUTPUT_ROOT/A_reproduction_launch.log" 2>&1
./isaaclab.sh -p scripts/flapping_px4/check_trim_original_reproduction.py > "$TRIM_OUTPUT_ROOT/A_gate.log" 2>&1
./isaaclab.sh -p scripts/flapping_px4/check_trim_initialization.py > "$TRIM_OUTPUT_ROOT/paired_check.log" 2>&1
python -c 'import json,sys; g=json.load(open(sys.argv[1])); assert all(g.values()) and g["steps"]==12288; assert json.load(open(sys.argv[2]))["passed"]' "$TRIM_OUTPUT_ROOT/A_gate.json" "$TRIM_OUTPUT_ROOT/paired_initialization_check.json"
bash scripts/flapping_px4/run_trim_centered_pilot.sh
bash scripts/flapping_px4/evaluate_trim_centered_pilot.sh
./isaaclab.sh -p scripts/flapping_px4/verify_trim_results.py > "$TRIM_OUTPUT_ROOT/verification.log" 2>&1
./isaaclab.sh -p scripts/flapping_px4/check_trim_recovery.py > "$TRIM_OUTPUT_ROOT/recovery_check.log" 2>&1
for group in A M B; do
 ./isaaclab.sh -p scripts/flapping_px4/analyze_trim_centered_pilot.py "$group" > "$TRIM_OUTPUT_ROOT/${group}_analysis.log" 2>&1
 test -f "$TRIM_OUTPUT_ROOT/${group}_diagnostics.json"
done
./isaaclab.sh -p scripts/flapping_px4/audit_trim_endpoints.py > "$TRIM_OUTPUT_ROOT/endpoint_audit.log" 2>&1
./isaaclab.sh -p scripts/flapping_px4/report_trim_centered_pilot.py > "$TRIM_OUTPUT_ROOT/comparison.log" 2>&1

./isaaclab.sh -p scripts/flapping_px4/plot_trim_centered_pilot.py

python -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); assert json.load(open(p/"verified_evaluations.json"))["passed"]; assert json.load(open(p/"recovery_check.json"))["actual_M_B_shared_initialization"]; assert len(json.load(open(p/"endpoint_audit.json"))["results"])==90; assert (p/"report_tables.md").exists() and (p/"comparison.png").exists()' "$TRIM_OUTPUT_ROOT"

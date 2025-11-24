#!/usr/bin/env bash
set -euo pipefail

# Ubuntu dual-GPU training launcher (torchrun + headless)
# Usage example:
#   chmod +x scripts/reinforcement_learning/rsl_rl/train_flapping_dual_gpu.sh
#   ./scripts/reinforcement_learning/rsl_rl/train_flapping_dual_gpu.sh --num_envs 1024 --max_iterations 2000

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT_DIR"

# Default args (can be overridden by CLI)
TASK="Isaac-FlappingBot-Direct-v0"
NPROC=${NPROC:-2}
EXTRA_ARGS=("--headless" "--distributed" "--task" "$TASK")

# Pass-through CLI args
EXTRA_ARGS+=("$@")

# Launch with torch.distributed.run (torchrun equivalent) across NPROC GPUs on this node
./isaaclab.sh -p -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node "$NPROC" \
  scripts/reinforcement_learning/rsl_rl/train.py "${EXTRA_ARGS[@]}"

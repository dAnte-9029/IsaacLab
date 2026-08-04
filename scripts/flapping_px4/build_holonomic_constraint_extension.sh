#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
extension_root="${repo_root}/source/flapping_bot/native_extensions/omni.flapping_bot.holonomic_constraint"
git_common_dir=$(git -C "${repo_root}" rev-parse --path-format=absolute --git-common-dir)
primary_checkout_root=$(dirname -- "${git_common_dir}")
default_physx_omni_root="$(dirname -- "${primary_checkout_root}")/PhysX-107.3/omni"
physx_omni_root="${PHYSX_OMNI_ROOT:-${default_physx_omni_root}}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "env_isaaclab" ]]; then
    echo "Activate env_isaaclab before running this build." >&2
    exit 2
fi

if [[ ! -f "${physx_omni_root}/include/omni/physx/IPhysxCustomJoint.h" ]]; then
    echo "Matching IPhysxCustomJoint.h not found below ${physx_omni_root}." >&2
    exit 3
fi

cmake -S "${extension_root}" -B "${extension_root}/build" \
    -DPHYSX_OMNI_ROOT="${physx_omni_root}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${extension_root}/build" --parallel "${BUILD_JOBS:-24}"

# New Tail Robot Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Switch the entire flapping stack to the latest five-joint robot and replace estimated tail geometry with direct file reads plus explicit placeholders for unknown parameters.

**Architecture:** The asset layer will point to a normalized repository-managed copy of the latest URDF/USD. The straight-flight and legacy direct environments will command a real rudder joint while preserving the existing four-channel action API. Tail aerodynamic metadata will be split into directly readable geometry fields and explicit placeholder fields for values that cannot be read from the available files.

**Tech Stack:** Python 3.11, IsaacLab articulation configs, URDF/XML parsing, STL geometry processing, pytest

---

### Task 1: Add normalized latest-robot asset files

**Files:**
- Create: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf`
- Create: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/config.yaml`
- Create: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/.asset_hash`
- Create: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/meshes/base_link.STL`
- Create: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/meshes/left_wing.STL`
- Create: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/meshes/right_wing.STL`
- Create: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/meshes/rudder.STL`
- Create: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/meshes/left_tail.STL`
- Create: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/meshes/right_tail.STL`

**Step 1: Copy the latest URDF and meshes into the repo with normalized names**

- Normalize link and joint names to:
  - `left_wing`
  - `right_wing`
  - `rudder`
  - `left_tail`
  - `right_tail`
- Rewrite mesh paths to repository-local relative paths.

**Step 2: Preserve directly readable inertial and joint data**

- Keep original masses, inertias, joint origins, joint axes, and limits.
- Do not estimate or tune anything during the copy.

**Step 3: Add converter metadata files**

- Mirror the structure used by the existing `flap_robot_v50` asset directory.

**Step 4: Verify the asset files are present**

Run: `find source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552 -maxdepth 3 -type f | sort`

Expected: the normalized URDF, metadata files, and five mesh files are listed.

### Task 2: Point the shared asset config to the new robot

**Files:**
- Modify: `source/isaaclab_assets/isaaclab_assets/robots/flapping_bot.py`

**Step 1: Update URDF/USD paths**

- Change the asset directory from `flap_robot_v50` to `flap_robot_552`.

**Step 2: Update initial joint state**

- Add `rudder` with a neutral default.
- Keep `left_wing`, `right_wing`, `left_tail`, and `right_tail`.

**Step 3: Update actuators**

- Keep wing actuator group for both wings.
- Keep tail actuator group for `left_tail` and `right_tail`.
- Add a rudder actuator group for `rudder`.

**Step 4: Verify import-level config integrity**

Run: `python - <<'PY'\nfrom isaaclab_assets.robots.flapping_bot import FLAPPING_BOT_CFG\nprint(FLAPPING_BOT_CFG.spawn.asset_path)\nprint(sorted(FLAPPING_BOT_CFG.init_state.joint_pos.keys()))\nprint(sorted(FLAPPING_BOT_CFG.actuators.keys()))\nPY`

Expected: asset path points to `flap_robot_552`, joint keys include `rudder`, and actuator keys include a rudder group.

### Task 3: Update the straight-flight environment for the five-joint robot

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`

**Step 1: Extend controlled-joint resolution**

- Add `rudder` to `controlled_joints`.
- Resolve `_IDX_RUDDER`.
- Include `rudder` in joint target buffers and reset logic.

**Step 2: Keep the four-channel public action contract**

- Preserve `action_space = 4`.
- Continue to interpret actions as `[flap_freq, rudder, elevon_pitch, elevon_roll]`.

**Step 3: Write the rudder command to simulation**

- Convert action 1 into a real rudder joint command.
- Continue mapping pitch/roll into the two elevon joints.

**Step 4: Update mass override body selection**

- Include the rudder body among appendages whose mass/inertia may be scaled.

**Step 5: Keep the existing teacher/PX4 interfaces unchanged**

- Do not change the controller output action format.

**Step 6: Verify joint resolution and command writing**

Run: `pytest tests/test_flapping_task_registration.py -q`

Expected: task registration still passes after the robot swap.

### Task 4: Update the legacy direct environment to accept the same robot

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/flapping_env.py`

**Step 1: Extend controlled-joint resolution**

- Add `rudder` to `controlled_joints`.
- Resolve `_IDX_RUDDER`.

**Step 2: Preserve the current three-action legacy interface**

- Leave the legacy action contract unchanged for now.
- Drive `rudder` to its neutral/default position in this environment unless a later interface change is requested.

**Step 3: Ensure writes include all resolved joints**

- Include `rudder` in joint target writes and reset paths so the latest robot loads cleanly.

**Step 4: Keep optional QSM filtering robust**

- Continue to filter QSM surfaces by actually available joint names.

**Step 5: Verify import/runtime safety**

Run: `pytest tests/test_flapping_reset_contract.py -q`

Expected: reset-contract tests still pass.

### Task 5: Add direct-read tail geometry loading with explicit placeholders

**Files:**
- Create: `source/flapping_bot/flapping_bot/physics/tail_geometry.py`
- Modify: `source/flapping_bot/flapping_bot/physics/__init__.py`
- Modify: `source/flapping_bot/flapping_bot/physics/tail_aero.py`

**Step 1: Add a direct-read metadata loader**

- Parse URDF joint origin, joint axis, joint limits, COM, mass, inertia, and mesh path.
- Compute projected area only for control surfaces that have dedicated meshes.
- Do not infer fixed-tail areas from `base_link`.

**Step 2: Add explicit placeholder fields for unknown parameters**

- Represent unreadable parameters as nullable fields or placeholder records.
- Mark them as required-for-tuning rather than silently estimating them.

**Step 3: Upgrade tail aero config structure**

- Replace the old single `elevator` plus single `rudder` config shape with a structure that can represent:
  - fixed horizontal tail
  - left elevon
  - right elevon
  - fixed vertical tail
  - rudder

**Step 4: Preserve a stable compute API**

- Keep the computation path usable by the straight-flight environment.
- Allow fixed surfaces to have zero deflection inputs.

**Step 5: Verify direct-read metadata in pure Python**

Run: `pytest tests/test_tail_geometry.py -q`

Expected: direct-readable parameters are populated and non-readable parameters remain placeholders.

### Task 6: Add focused tests for the new robot contract

**Files:**
- Create: `tests/test_tail_geometry.py`
- Modify: `tests/test_tail_aero.py`
- Create: `tests/test_flapping_asset_cfg.py`

**Step 1: Test asset config contract**

- Assert the shared asset config references the new robot.
- Assert all five expected joints are declared in init/actuator configuration.

**Step 2: Test tail geometry loader**

- Assert joint origin, axis, limits, mass, inertia, COM, and mesh paths are read from files.
- Assert fixed-tail quantities that cannot be read are placeholders rather than estimates.

**Step 3: Test tail aero config wiring**

- Assert the upgraded config can represent movable and fixed surfaces explicitly.

**Step 4: Run the focused test set**

Run: `pytest tests/test_flapping_asset_cfg.py tests/test_tail_geometry.py tests/test_tail_aero.py tests/test_flapping_task_registration.py tests/test_flapping_reset_contract.py -q`

Expected: all targeted tests pass.

### Task 7: Run final verification commands

**Files:**
- Modify: none

**Step 1: Run the targeted verification suite**

Run: `pytest tests/test_flapping_asset_cfg.py tests/test_tail_geometry.py tests/test_tail_aero.py tests/test_flapping_task_registration.py tests/test_flapping_reset_contract.py -q`

Expected: PASS

**Step 2: Sanity-check the normalized URDF contents**

Run: `python - <<'PY'\nfrom pathlib import Path\nimport xml.etree.ElementTree as ET\np = Path('source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf')\nroot = ET.parse(p).getroot()\nprint(sorted(j.attrib['name'] for j in root.findall('joint')))\nprint(sorted(l.attrib['name'] for l in root.findall('link')))\nPY`

Expected: normalized joint and link names include `left_wing`, `right_wing`, `rudder`, `left_tail`, and `right_tail`.

**Step 3: Review git diff for touched files**

Run: `git diff -- source/isaaclab_assets/isaaclab_assets/robots/flapping_bot.py source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py source/flapping_bot/flapping_bot/direct/flapping_bot/flapping_env.py source/flapping_bot/flapping_bot/physics/tail_aero.py source/flapping_bot/flapping_bot/physics/tail_geometry.py tests/test_flapping_asset_cfg.py tests/test_tail_geometry.py tests/test_tail_aero.py`

Expected: diff is limited to the intended asset, environment, aero, and test changes.

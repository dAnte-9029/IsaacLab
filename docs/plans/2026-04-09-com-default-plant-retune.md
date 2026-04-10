# COM-Default Plant Retune Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the DeLaurier + non-RL flapping-bot plant self-consistent around `base_body_com_x = -0.10 m`, then retune trim and downstream aero so `straight / turn / loiter` no longer depend on compensating for a bad COM or bad wing/tail moment reference.

**Architecture:** Keep the existing controller-only `fly_path_mission.py` verification path, but change the plant in layers: first promote the new COM as the default simulated rigid-body reference, then retune reset/trim for that plant, then improve the wing moment application point, then re-run tail balance search on the corrected plant, and only after that revisit controller tuning. Do not re-open RL, reward, simple QSM, or teacher-student work in this plan.

**Tech Stack:** Python 3.11, IsaacLab/Isaac Sim articulation APIs, Gymnasium env configs, PyTorch, pytest, headless `fly_path_mission.py` regression scripts.

---

### Task 1: Promote `base_body_com_x = -0.10 m` as the New Plant Default

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `scripts/flapping_px4/fly_path_mission.py`
- Test: `tests/test_path_tracking_env_contract.py`
- Test: `tests/test_fly_path_mission.py`

**Step 1: Write the failing tests**

- Change the existing config-default contract test so it expects:

```python
assert cfg.base_body_com_override_x_m == -0.10
```

- Add one parser/configure-env test that verifies:
  - default `args.base_body_com_override_x_m is None`
  - `_configure_env()` leaves the new default in place when the CLI flag is omitted
  - `_configure_env()` still overrides it when the CLI flag is passed

**Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_path_tracking_env_contract.py -k base_body_com_override_x_m -q
python -m pytest tests/test_fly_path_mission.py -k base_body_com_override_x -q
```

Expected: FAIL because the current default is `None`.

**Step 3: Write the minimal implementation**

- In `FlappingBotStraightFlightEnvCfg`, set:

```python
base_body_com_override_x_m: float | None = -0.10
```

- Mirror the same default into the path-tracking config shims.
- Do not remove the CLI flag. Keep the runtime override path intact for future sweeps.

**Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_path_tracking_env_contract.py -k base_body_com_override_x_m -q
python -m pytest tests/test_fly_path_mission.py -k base_body_com_override_x -q
python -m py_compile source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py scripts/flapping_px4/fly_path_mission.py
```

Expected: PASS.

**Step 5: Run one smoke rollout with no CLI override**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_straight \
  --steps 600 \
  --headless \
  --print_every 0
```

Expected:
- No immediate `terminated` at ~1.0 s
- Summary shows `base_body_com_override_x_m = -0.10`

**Step 6: Commit**

```bash
git add \
  source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  scripts/flapping_px4/fly_path_mission.py \
  tests/test_path_tracking_env_contract.py \
  tests/test_fly_path_mission.py
git commit -m "fix: default flapping bot base COM to -0.10 m"
```

### Task 2: Add Runtime COM and Early-Failure Audit to the Canonical Rollout Output

**Files:**
- Modify: `scripts/flapping_px4/fly_path_mission.py`
- Test: `tests/test_fly_path_mission.py`

**Step 1: Write the failing test**

- Extend the summary/metadata test to require these fields in `summary.json`:

```python
{
    "base_body_com_override_x_m": ...,
    "runtime_base_body_com_x_m": ...,
    "failure_kind": ...,
    "steps_completed": ...,
}
```

- If a helper function formats summary rows, test that it preserves `failure_kind` and `steps_completed`.

**Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_fly_path_mission.py -k summary -q
```

Expected: FAIL because `runtime_base_body_com_x_m` is not yet recorded.

**Step 3: Write the minimal implementation**

- In the rollout script, after reset and before the loop, read:

```python
env.unwrapped._robot.data.body_com_pos_b[:, base_id, 0]
```

- Record the runtime COM x value in the summary.
- Keep all existing summary keys unchanged.

**Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_fly_path_mission.py -k summary -q
```

Expected: PASS.

**Step 5: Run one canonical rollout and inspect the summary**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py \
  --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 \
  --phase level_turn \
  --steps 900 \
  --headless \
  --print_every 0 \
  --out_dir /tmp/flapping_com_audit
```

Expected:
- `summary.json` includes both configured and runtime COM x values
- `failure_kind` and `steps_completed` are populated if the rollout ends early

**Step 6: Commit**

```bash
git add scripts/flapping_px4/fly_path_mission.py tests/test_fly_path_mission.py
git commit -m "feat: log runtime COM and failure audit in mission summaries"
```

### Task 3: Retune Reset and Trim for the New COM Before Touching the Controller

**Files:**
- Create: `scripts/flapping_px4/run_reset_trim_sweep.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Test: `tests/test_fly_path_mission.py`
- Test: `tests/test_path_tracking_env_contract.py`

**Step 1: Write the failing test**

- Add one parser/config test for the sweep script that verifies it accepts and forwards:
  - `--reset_pitch_deg`
  - `--reset_flap_hz`
  - `--reset_elevon_pitch_deg`
- Add one config-default contract test covering the eventual new reset defaults after selection.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_fly_path_mission.py -k reset -q
python -m pytest tests/test_path_tracking_env_contract.py -k reset -q
```

Expected: FAIL because the sweep script does not exist and defaults still reflect the old plant.

**Step 3: Write the minimal implementation**

- Add `run_reset_trim_sweep.py` that runs only the non-RL canonical `level_straight` mission.
- Sweep a small grid over:
  - `reset_pitch_deg`
  - `reset_flap_hz`
  - `reset_elevon_pitch_deg`
- Score candidates using:
  - no `terminated` before `3.0 s`
  - highest `done_t_s`
  - smallest `abs(min_height_error_first3s)`
  - lowest `max_abs_pitch_first3s`
- Keep it simple. Reuse `fly_path_mission.py`; do not create another env path.

**Step 4: Run the sweep and pick new defaults**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/run_reset_trim_sweep.py --headless --out_root /tmp/flapping_trim_sweep
```

Expected:
- One candidate clearly reduces the ~1 m early drop at `COM=-0.10`
- The sweep writes a small leaderboard CSV

**Step 5: Apply the winning trim to the env defaults**

- Update:
  - `reset_pitch_deg`
  - `reset_flap_hz`
  - `reset_elevon_pitch_deg`
- Keep `pitch_cmd_deg` unchanged in this task unless the sweep shows the reset itself is insufficient.

**Step 6: Re-run canonical verification**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_straight --steps 2200 --headless --print_every 0
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_turn --steps 2400 --headless --print_every 0
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_loiter --steps 2800 --headless --print_every 0
```

Expected:
- Smaller `min_height_error_first3s`
- No regression to the old ~1 s `terminated` failure

**Step 7: Commit**

```bash
git add \
  scripts/flapping_px4/run_reset_trim_sweep.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py \
  tests/test_fly_path_mission.py \
  tests/test_path_tracking_env_contract.py
git commit -m "feat: retune reset trim for corrected base COM"
```

### Task 4: Replace the Wing Root-Origin Torque Approximation with a Better Effective Wing Application Point

**Files:**
- Create: `source/flapping_bot/flapping_bot/physics/wing_equivalent_ac.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Test: `tests/test_wing_equivalent_ac.py`
- Test: `tests/test_path_tracking_env_contract.py`

**Step 1: Write the failing tests**

- Add a pure unit test for an area-weighted quarter-chord helper:

```python
def test_area_weighted_quarter_chord_returns_expected_link_frame_point():
    # simple rectangular wing strips -> known quarter-chord center
```

- Add one env contract test asserting the DeLaurier wing wrench path uses a dedicated wing application point helper instead of the raw `body_pos_w` root only.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_wing_equivalent_ac.py -q
python -m pytest tests/test_path_tracking_env_contract.py -k wing_equivalent_ac -q
```

Expected: FAIL because the helper does not exist and env still uses the wing link origin.

**Step 3: Write the minimal implementation**

- In `wing_equivalent_ac.py`, compute an effective application point from the geometry CSV:
  - spanwise area weighting
  - quarter-chord point on each strip
  - mirrored handling for left/right wings
- In `straight_flight_env.py`, replace:

```python
p_wing_w = self._robot.data.body_pos_w[:, self._wing_body_ids, :]
```

with:

```python
p_wing_app_w = body_origin_w + quat_apply(body_quat_w, wing_ac_link)
```

- Keep the reference point for torque as `root_com_pos_w`.

**Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_wing_equivalent_ac.py tests/test_path_tracking_env_contract.py -k wing -q
python -m py_compile source/flapping_bot/flapping_bot/physics/wing_equivalent_ac.py source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py
```

Expected: PASS.

**Step 5: Run canonical verification and compare wing torque**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_turn --steps 2400 --headless --print_every 0 --out_dir /tmp/flapping_wing_ac_turn
```

Expected:
- `tau_total_b_y` and early pitch response change materially
- The result is easier to interpret physically than the wing-root approximation

**Step 6: Commit**

```bash
git add \
  source/flapping_bot/flapping_bot/physics/wing_equivalent_ac.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
  tests/test_wing_equivalent_ac.py \
  tests/test_path_tracking_env_contract.py
git commit -m "feat: use effective wing application point for DeLaurier wrench"
```

### Task 5: Re-Search Tail Balance on the Corrected Plant and Penalize Early Failure Explicitly

**Files:**
- Modify: `scripts/flapping_px4/run_tail_balance_grid_search.py`
- Modify: `tests/test_tail_balance_grid_search.py`

**Step 1: Write the failing tests**

- Add tests covering:
  - a candidate with `failure_kind="terminated"` before warmup receives a large penalty or is filtered out
  - `done_t_s` and `min_height_error_first3s` are included in aggregated metrics

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_tail_balance_grid_search.py -k early_failure -q
```

Expected: FAIL because the current scorer only relies on post-warmup metrics.

**Step 3: Write the minimal implementation**

- Extend score aggregation to include:
  - `done_t_s`
  - `failure_kind`
  - `min_height_error_first3s`
- Add a hard filter for canonical runs that `terminated` before `metrics_warmup_s`.
- Keep the search dimensions the same first:
  - `tail_fixed_horizontal_effectiveness`
  - `tail_elevon_effectiveness`

**Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_tail_balance_grid_search.py -q
python -m py_compile scripts/flapping_px4/run_tail_balance_grid_search.py
```

Expected: PASS.

**Step 5: Run the search on the new COM + new trim + new wing application point**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/run_tail_balance_grid_search.py \
  --out_root logs/flapping_px4/tail_balance_grid_search_com_m0p10 \
  --headless
```

Expected:
- No top-ranked candidate wins by merely surviving longer with terrible lateral error
- New best tail parameters differ from the old `0.55 / 1.00` search result

**Step 6: Commit**

```bash
git add scripts/flapping_px4/run_tail_balance_grid_search.py tests/test_tail_balance_grid_search.py
git commit -m "feat: rescore tail search with early-failure penalties"
```

### Task 6: Revisit PX4-Like Controller Only After Plant, Trim, and Tail Are Re-Established

**Files:**
- Modify: `source/flapping_bot/flapping_bot/px4_like/tecs.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/loiter_controller.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py`
- Test: `tests/test_tecs.py`
- Test: `tests/test_px4_path_tracking_controller.py`
- Test: `tests/test_px4_loiter_controller.py`

**Step 1: Freeze the plant and capture a fresh baseline**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_straight --steps 2200 --headless --print_every 0 --out_dir /tmp/final_plant_baseline
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_turn --steps 2400 --headless --print_every 0 --out_dir /tmp/final_plant_baseline
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_loiter --steps 2800 --headless --print_every 0 --out_dir /tmp/final_plant_baseline
```

Expected:
- The remaining error now reflects the new plant, not the old COM/reference bug.

**Step 2: Write failing controller regression tests only for the residual issue**

- If the remaining issue is still bank entry altitude loss, add tests against:
  - TECS pitch-side load-factor bias
  - bank-aware min-airspeed behavior
- Do not reopen unrelated controller refactors.

**Step 3: Make the smallest controller change that matches the new plant**

- Only after the previous tasks are complete.
- Prefer existing knobs before inventing new ones.

**Step 4: Verify**

Run:

```bash
python -m pytest tests/test_tecs.py tests/test_px4_path_tracking_controller.py tests/test_px4_loiter_controller.py -q
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_straight --steps 2200 --headless --print_every 0
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_turn --steps 2400 --headless --print_every 0
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_loiter --steps 2800 --headless --print_every 0
```

Expected:
- Controller changes now improve a stable plant instead of compensating for plant inconsistencies.

**Step 5: Commit**

```bash
git add \
  source/flapping_bot/flapping_bot/px4_like/tecs.py \
  source/flapping_bot/flapping_bot/px4_like/path_tracking_controller.py \
  source/flapping_bot/flapping_bot/px4_like/loiter_controller.py \
  source/flapping_bot/flapping_bot/px4_like/straight_line_controller.py \
  tests/test_tecs.py \
  tests/test_px4_path_tracking_controller.py \
  tests/test_px4_loiter_controller.py
git commit -m "fix: retune controller on corrected flapping plant"
```

### Task 7: Final Acceptance Matrix and Deferred Items

**Files:**
- Modify: `docs/plans/2026-04-09-com-default-plant-retune.md`
- Optional future work: `source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf`

**Step 1: Run the final acceptance matrix**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_straight --steps 2200 --headless --print_every 0
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_turn --steps 2400 --headless --print_every 0
./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py --task Isaac-FlappingBot-PathTracking-DeLaurier-Direct-v0 --phase level_loiter --steps 2800 --headless --print_every 0
./isaaclab.sh -p scripts/flapping_px4/fly_loiter.py --headless
```

Expected:
- No immediate ~1 s pitch-up termination
- Smaller early-drop magnitude
- Better `turn / loiter` altitude retention
- Acceptable lateral tracking

**Step 2: Document deferred items, do not mix them into this implementation**

Deferred:
- Updating the URDF inertial origin to bake in the new COM physically
- Recomputing inertias to match the shifted COM
- Reconsidering `override_appendage_masses` for higher-fidelity dynamics
- RL/domain-randomization around COM/inertia uncertainty

**Step 3: Commit**

```bash
git add docs/plans/2026-04-09-com-default-plant-retune.md
git commit -m "docs: record corrected flapping plant retune plan"
```

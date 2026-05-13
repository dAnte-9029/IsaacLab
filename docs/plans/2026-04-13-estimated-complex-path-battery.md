# Estimated Complex Path Battery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Formalize the `estimated + synthetic + 0.95kg` complex path-tracking battery, diagnose the current worst seeds, and run the next two realism gates: synthetic wind stress and `isaacsim` live-IMU spot checks.

**Architecture:** Reuse the existing `fly_path_mission.py` rollout/logger as the single path-mission execution path. Add the missing runtime knobs there so both ad-hoc debugging and suite runners can configure teacher state source, IMU source, mass override, and controller tuning profile without monkeypatching. Build one dedicated complex-suite runner on top of that entrypoint, then use it for no-wind baseline, wind stress, and live-IMU spot checks. Save worst-seed diagnosis as an analysis artifact rather than burying it in chat.

**Tech Stack:** Python 3.11, IsaacLab CLI entrypoints, pytest, CSV/JSON log analysis, existing flapping PX4 path-tracking scripts.

---

### Task 1: Formalize `fly_path_mission.py` runtime contract

**Files:**
- Modify: `scripts/flapping_px4/fly_path_mission.py`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- Modify: `source/flapping_bot/flapping_bot/px4_like/__init__.py`
- Test: `tests/test_fly_path_mission.py`
- Test: `tests/test_path_tracking_env_contract.py`

**Step 1: Write the failing tests**

Add tests that require:
- `fly_path_mission.py` parser to accept `--teacher_state_source`, `--policy_state_source`, `--imu_source`, `--seed`, `--total_mass_kg_override`
- `fly_path_mission.py` parser to accept controller-tuning controls needed for the estimated baseline, including a profile selector and explicit overrides
- `_configure_env()` to apply the new runtime state-source, IMU, mass, and controller fields onto `env_cfg`
- `FlappingBotPathTrackingEnvCfg` to expose teacher inner-loop elevon rate-limit fields so the path env can construct the teacher controller directly from config

**Step 2: Run the targeted tests to verify they fail**

Run:
`./isaaclab.sh -p -m pytest tests/test_fly_path_mission.py tests/test_path_tracking_env_contract.py -q`

Expected:
At least the new parser/config tests fail because the runtime contract is not exposed yet.

**Step 3: Implement the minimal runtime contract**

Implement:
- new parser arguments and `_configure_env()` mapping in `fly_path_mission.py`
- path-env config fields for teacher inner elevon pitch/roll rate limits
- path-env teacher-controller construction using those config fields
- summary logging of the actual runtime contract and controller profile data so later suite summaries are self-describing

**Step 4: Re-run the targeted tests**

Run:
`./isaaclab.sh -p -m pytest tests/test_fly_path_mission.py tests/test_path_tracking_env_contract.py -q`

Expected:
All targeted tests pass.

### Task 2: Add a reusable estimated complex battery runner

**Files:**
- Create: `scripts/flapping_px4/run_estimated_path_tracking_complex_suite.py`
- Test: `tests/test_estimated_path_tracking_complex_suite.py`

**Step 1: Write the failing tests**

Add tests that require:
- deterministic case list with exactly 5 baseline cases: fixed `multi_segment`, seeds `25`, `30`, `33`, `36`
- command builder to pass the formalized `fly_path_mission.py` runtime knobs for `estimated + synthetic + 0.95kg`
- scenario selection for:
  - `baseline_nowind`
  - `steady_crosswind`
  - `ou_gust`
  - `live_imu_spotcheck`
- summary aggregation over per-case `summary.json`

**Step 2: Run the suite-runner tests to verify they fail**

Run:
`./isaaclab.sh -p -m pytest tests/test_estimated_path_tracking_complex_suite.py -q`

Expected:
The new runner tests fail because the runner does not exist yet.

**Step 3: Implement the new suite runner**

Implement a dedicated runner that:
- executes cases strictly serially
- emits `manifest.json`, `summary.json`, and `cases.csv`
- reuses `plot_path_mission.py` so each case gets `plots.png`
- supports the four scenarios above using one codepath

**Step 4: Re-run the suite-runner tests**

Run:
`./isaaclab.sh -p -m pytest tests/test_estimated_path_tracking_complex_suite.py -q`

Expected:
All runner tests pass.

### Task 3: Run the formal baseline battery and diagnose the worst seeds

**Files:**
- Create: `docs/analysis/flapping_px4/estimated_complex_path_battery_20260413/diagnosis.md`
- Create: `docs/analysis/flapping_px4/estimated_complex_path_battery_20260413/worst_seed_summary.json`
- Output: `logs/flapping_px4/estimated_path_tracking_complex_suite/...`

**Step 1: Run the formal no-wind baseline battery**

Run the new suite runner in `baseline_nowind` mode with:
- `teacher_state_source=estimated`
- `policy_state_source=estimated`
- `imu_source=synthetic`
- `total_mass_kg_override=0.95`
- estimated controller tuning profile enabled

**Step 2: Identify the two worst cases by evidence**

Use the generated `summary.json` and per-case `trajectory_env0.csv` to rank cases by:
- `mean_abs_lateral_error_m`
- `p95_abs_lateral_error_m`
- `mean_abs_height_error_m`

**Step 3: Write a short diagnosis artifact**

Save a diagnosis that records:
- which two seeds are worst
- which segment patterns they share
- when their lateral and height peaks occur
- whether the degradation is mainly lateral, altitude, or both

**Step 4: Verify the diagnosis artifacts exist and are populated**

Run:
`./isaaclab.sh -p - <<'PY'`
`from pathlib import Path`
`base = Path("docs/analysis/flapping_px4/estimated_complex_path_battery_20260413")`
`assert (base / "diagnosis.md").exists()`
`assert (base / "worst_seed_summary.json").exists()`
`PY`

Expected:
Exit code `0`.

### Task 4: Run wind stress and live-IMU spot checks

**Files:**
- Output: `logs/flapping_px4/estimated_path_tracking_complex_suite/...`
- Modify: `docs/analysis/flapping_px4/estimated_complex_path_battery_20260413/diagnosis.md`

**Step 1: Run the synthetic steady-crosswind battery**

Run the suite runner in `steady_crosswind` mode over the same 5 formal cases.

**Step 2: Run the synthetic OU-gust battery**

Run the suite runner in `ou_gust` mode over the same 5 formal cases.

**Step 3: Run `isaacsim` live-IMU spot checks on the two diagnosed worst seeds**

Run the suite runner in `live_imu_spotcheck` mode over the two worst seeds only.

**Step 4: Update the diagnosis artifact with the new realism gates**

Append:
- whether wind causes graceful degradation or exposes a new failure mode
- whether `isaacsim` live IMU mostly preserves behavior or introduces extra estimator/controller mismatch
- the next controller work item that is actually justified by the new evidence

**Step 5: Full verification**

Run:
- `./isaaclab.sh -p -m py_compile scripts/flapping_px4/fly_path_mission.py scripts/flapping_px4/run_estimated_path_tracking_complex_suite.py source/flapping_bot/flapping_bot/direct/flapping_bot/path_tracking_env.py`
- `./isaaclab.sh -p -m pytest tests/test_fly_path_mission.py tests/test_path_tracking_env_contract.py tests/test_estimated_path_tracking_complex_suite.py tests/test_random_path_tracking_teacher_suite.py -q`

Expected:
All listed files compile and all listed tests pass.

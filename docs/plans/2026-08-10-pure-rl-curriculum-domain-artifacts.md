# PureRL Curriculum Contract and Real-Data Artifacts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Freeze the PureRL C1/C2/C3 and CPU/GPU shared contract, then build versioned B/C/D real-flight evidence artifacts without reading the sealed canonical test partition.

**Architecture:** Keep task semantics independent from the physics backend. IsaacLab owns the compact curriculum/backend contract and an evidence-only domain profile; `flap-system-identification` owns reproducible extraction from the canonical train/validation tables and raw ULogs. The generator writes three separate artifact directories with manifests, and only train statistics may populate candidate training ranges.

**Tech Stack:** Python 3.11, dataclasses, JSON/YAML, pandas, NumPy, PyArrow, PyULog, pytest, Isaac Lab configuration contracts.

---

### Task 1: Close the approved governor and retention batch

**Files:**
- Existing governor, reward, evaluation, rehearsal, retention, ADR, and test files in the IsaacLab worktree

**Step 1: Run the focused pure-logic suite**

Run the governor, reward, evaluation, path-contract, launcher, and retention tests through `./isaaclab.sh -p -m pytest` in `env_isaaclab`.

**Step 2: Run the native CPU Isaac gate**

Run `tests/test_native_cpu_pure_rl_runtime_isaac.py` in a fresh Isaac process and require its assertions to pass.

**Step 3: Commit only the approved batch**

Exclude the untracked playback/GIF files. Commit with `feat(rl): govern frequency and gate curriculum retention`.

### Task 2: Add a pure, backend-independent curriculum contract

**Files:**
- Create: `source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_curriculum_contract.py`
- Create: `tests/test_pure_rl_curriculum_contract.py`
- Create: `docs/decisions/ADR-2026-08-10-pure-rl-curriculum-domain-contract.md`
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`

**Step 1: Write failing contract tests**

Test that the contract freezes:

- C1 as no-wind straight flight;
- C2 as no-wind level/climb/descent with C1 retention required;
- C3 as no-wind turn/loiter/composite paths with C1 and C2 retention required;
- `dt=1/480 s`, policy decimation 8, direct three-surface plus frequency action, 555 observations, 0--5 Hz, and 2 Hz/s rise/fall governor;
- CPU native as the authoritative backend and GPU implicit drive as a candidate requiring CPU promotion;
- stage and backend changes cannot alter the shared semantic signature.

**Step 2: Verify RED**

Run `tests/test_pure_rl_curriculum_contract.py` and require failure because the contract module does not yet exist.

**Step 3: Implement the minimal immutable dataclasses and validator**

Keep the module free of Isaac imports. Reference its constants from the measured C1 environment defaults, and fail if a future configuration drifts from the shared signature.

**Step 4: Verify GREEN**

Run the new test plus existing action, observation, reward, reset, and path-contract tests.

### Task 3: Build fail-closed canonical train/validation resolution

**Files:**
- Create: `/tmp/flap-system-identification-rl-domain-artifacts/src/system_identification/analysis/rl_domain_artifacts.py`
- Create: `/tmp/flap-system-identification-rl-domain-artifacts/tests/test_rl_domain_artifacts.py`

**Step 1: Write failing resolver tests**

Cover active-default resolution, manifest/sample hash checks, train/validation-only loading, explicit refusal of test, missing files, duplicate log paths, and split-name normalization.

**Step 2: Verify RED**

Run only `tests/test_rl_domain_artifacts.py` and require the import or missing-function failure.

**Step 3: Implement the minimal resolver**

Resolve `configs/data/canonical_dataset_registry.yaml`, load only `train_samples.parquet` and `val_samples.parquet`, and record dataset identity, ratio, phase, frequency, frame, units, hashes, and row/log counts.

### Task 4: Build artifact B, sensor and wind evidence

**Files:**
- Modify: `/tmp/flap-system-identification-rl-domain-artifacts/src/system_identification/analysis/rl_domain_artifacts.py`
- Modify: `/tmp/flap-system-identification-rl-domain-artifacts/tests/test_rl_domain_artifacts.py`

**Step 1: Write failing profile tests**

Test train-only candidate quantiles, validation coverage, wind-vector summaries, Pitot freshness, per-log raw update cadence, finite/valid masks, and explicit `noise_identified=false` and `delay_identified=false` limitations.

**Step 2: Verify RED, implement, and verify GREEN**

Use fixed statistics `min`, `p01`, `p05`, `p50`, `p95`, `p99`, `max`, `mean`, and `std`. Keep train, validation, and external-reference summaries separate.

### Task 5: Build artifact C, trajectory companion table

**Files:**
- Modify: `/tmp/flap-system-identification-rl-domain-artifacts/src/system_identification/analysis/rl_domain_artifacts.py`
- Modify: `/tmp/flap-system-identification-rl-domain-artifacts/tests/test_rl_domain_artifacts.py`

**Step 1: Write failing extraction tests**

Cover PX4 position and loiter type mapping, haversine straight length, altitude delta and climb/descent classification, waypoint course change, loiter radius/direction, contiguous time intervals, missing previous/next points, and train/validation/external-reference provenance.

**Step 2: Verify RED, implement, and verify GREEN**

Do not infer a generic turn radius. Export waypoint course change, acceptance radius, commanded loiter radius, actual ground-speed/vertical-speed summaries, and a machine-readable limitation for unidentified ordinary-turn radius.

### Task 6: Build artifact D, sim-to-real reference statistics

**Files:**
- Modify: `/tmp/flap-system-identification-rl-domain-artifacts/src/system_identification/analysis/rl_domain_artifacts.py`
- Modify: `/tmp/flap-system-identification-rl-domain-artifacts/tests/test_rl_domain_artifacts.py`

**Step 1: Write failing reference tests**

Cover attitude derived from quaternion rather than the degenerate filtered pitch, state/action distribution records, within-log and within-segment action rates, 0.1/0.25/0.5-second state increments, frames/units, and absence of canonical test rows.

**Step 2: Verify RED, implement, and verify GREEN**

Write a compact metric table rather than copying the canonical samples. Mark numerical pass/fail thresholds as not yet set because no simulation candidate is being evaluated in this task.

### Task 7: Add the generator and materialize B/C/D

**Files:**
- Create: `/tmp/flap-system-identification-rl-domain-artifacts/scripts/build_rl_domain_artifacts.py`
- Create: `/tmp/flap-system-identification-rl-domain-artifacts/docs/analysis/rl_domain_artifacts_v1.md`
- Generate, do not commit: `/tmp/flap-system-identification-rl-domain-artifacts/artifacts/20260810_rl_domain_v1/`

**Step 1: Write a failing CLI test**

Use synthetic canonical tables and synthetic raw-topic frames. Require three non-overwriting artifact directories, manifests, the C Parquet table, the D metric Parquet table, and a compact candidate domain profile.

**Step 2: Implement the thin CLI and verify GREEN**

Default to canonical train/validation plus `/home/zn/QgcLogs`. Include 4.17 only as `external_reference`; do not let it affect candidate ranges.

**Step 3: Run the real materialization**

Generate the artifacts, inspect row counts and exclusions, and confirm `test_loaded=false` in every manifest.

### Task 8: Snapshot the compact evidence profile into IsaacLab

**Files:**
- Create: `source/flapping_bot/flapping_bot/config/real_flight_domain_profile_v1.json`
- Create: `tests/test_real_flight_domain_profile.py`
- Modify: `docs/PROJECT_STATE.md`

**Step 1: Write the failing profile contract test**

Require matching schema/dataset identity, train-only parameter source, validation/external-reference diagnostics, explicit units, and `status=candidate_not_promoted`.

**Step 2: Copy only the compact profile**

Do not add large Parquet artifacts to IsaacLab. The profile is evidence for later C2/C3 parameter approval and does not change current environment defaults.

**Step 3: Verify both repositories**

Run focused tests, each repository's full relevant suite, `py_compile`, `git diff --check`, and final status/diff review. Do not start PPO training or GPU qualification in this batch.

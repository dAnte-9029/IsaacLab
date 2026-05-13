# Flapping Robot Capability Envelope Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Quantify the current robot's sustainable envelope on the non-RL, DeLaurier, PX4-like controller chain, so we can distinguish straight-climb limit, level-turn limit, and coupled climb-turn limit before touching more controller tuning.

**Architecture:** Reuse the existing `fly_path_mission.py` teacher-only rollout path as the execution backend, add a thin sweep runner plus an analysis layer that computes sustained-tracking metrics from rollout CSVs, and generate envelope plots/tables. The sweep must separate three capability families: straight-only climb, level-only turn/loiter, and climb-plus-turn coupled missions. All conclusions must come from the teacher controller path with RL guidance disabled and the DeLaurier backend fixed.

**Tech Stack:** Python 3.11, IsaacLab wrapper scripts, pytest, matplotlib, existing `scripts/flapping_px4/*` tooling, CSV/JSON rollout artifacts.

---

### Task 1: Lock The Measurement Contract

**Files:**
- Modify: `scripts/flapping_px4/plot_path_mission.py`
- Modify: `tests/test_plot_path_mission.py`
- Create: `docs/analysis/flapping_px4/capability_envelope/README.md`

**Step 1: Write the failing test**

Add a test in `tests/test_plot_path_mission.py` that feeds a synthetic mission with changing `z_ref` and asserts the altitude panel uses time-varying reference altitude instead of a constant `height_sp_m`.

```python
def test_plot_uses_time_varying_reference_altitude_when_available(tmp_path):
    run_dir = _write_fake_run_dir_with_reference_z(tmp_path)
    plotted = _collect_altitude_series(run_dir)
    assert plotted["reference_label"] == "reference_z"
    assert plotted["reference_values"] == [10.0, 10.0, 11.0, 12.0]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plot_path_mission.py -k reference_altitude -q`
Expected: FAIL because `plot_path_mission.py` still plots only constant `height_sp_m`.

**Step 3: Write minimal implementation**

Add a small helper in `scripts/flapping_px4/plot_path_mission.py` that prefers time-varying `reference_z` from `trajectory_env0.csv` or `reference_path.csv`, and only falls back to `height_sp_m` if no per-sample altitude reference exists.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_plot_path_mission.py -k reference_altitude -q`
Expected: PASS.

**Step 5: Document the measurement contract**

Write `docs/analysis/flapping_px4/capability_envelope/README.md` with:
- controller-only chain requirements
- required metrics
- success/fail criteria
- output directory convention

**Step 6: Commit**

```bash
git add tests/test_plot_path_mission.py scripts/flapping_px4/plot_path_mission.py docs/analysis/flapping_px4/capability_envelope/README.md
git commit -m "test: lock capability envelope measurement contract"
```

### Task 2: Add A Sweep Runner For Capability Cases

**Files:**
- Create: `scripts/flapping_px4/run_capability_envelope_suite.py`
- Modify: `scripts/flapping_px4/fly_path_mission.py`
- Create: `tests/test_run_capability_envelope_suite.py`

**Step 1: Write the failing test**

Add a test for a pure helper in `scripts/flapping_px4/run_capability_envelope_suite.py` that expands a compact sweep spec into explicit cases.

```python
def test_build_capability_cases_covers_three_envelope_families():
    cases = build_capability_cases(
        climb_deltas_m=[1.0, 2.0],
        turn_radii_m=[16.0, 20.0],
        loiter_turns=[1.0],
    )
    families = {case.family for case in cases}
    assert families == {"straight_climb", "level_turn", "climb_turn"}
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_capability_envelope_suite.py -q`
Expected: FAIL because the script/helper does not exist yet.

**Step 3: Write minimal implementation**

Create `scripts/flapping_px4/run_capability_envelope_suite.py` that:
- reuses `./isaaclab.sh -p scripts/flapping_px4/fly_path_mission.py`
- enforces teacher-only / no-RL arguments
- emits one run directory per case
- records a manifest with exact CLI and case metadata

Case families to generate:
- `straight_climb`: straight segment with altitude change only
- `level_turn`: turn/loiter with fixed altitude only
- `climb_turn`: turn or loiter preceded by climb, or multi-segment mission with climb then turn

**Step 4: Add minimal hooks to `fly_path_mission.py` if needed**

Only if the current CLI cannot express the three families cleanly, add the smallest possible mission presets or metadata export fields. Avoid controller changes here.

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_run_capability_envelope_suite.py -q`
Expected: PASS.

**Step 6: Syntax check**

Run:
- `python -m py_compile scripts/flapping_px4/run_capability_envelope_suite.py`
- `python -m py_compile scripts/flapping_px4/fly_path_mission.py`

Expected: both PASS.

**Step 7: Commit**

```bash
git add scripts/flapping_px4/run_capability_envelope_suite.py scripts/flapping_px4/fly_path_mission.py tests/test_run_capability_envelope_suite.py
git commit -m "feat: add capability envelope sweep runner"
```

### Task 3: Define Sustainable Metrics In Code

**Files:**
- Create: `scripts/flapping_px4/analyze_capability_envelope.py`
- Create: `tests/test_analyze_capability_envelope.py`

**Step 1: Write the failing test**

Add tests for pure metric helpers:

```python
def test_case_is_unsustainable_when_height_error_and_saturation_persist():
    stats = summarize_case(
        height_error_m=[-0.1, -0.4, -0.8, -1.0],
        freq_hz=[4.8, 5.0, 5.0, 5.0],
        pitch_sp_deg=[-18.0, -22.0, -25.0, -25.0],
        speed_mps=[7.5, 7.0, 6.3, 5.9],
        reference_z_m=[10.0, 11.0, 12.0, 13.0],
        z_m=[9.9, 10.7, 11.0, 11.6],
    )
    assert stats["sustainable"] is False
    assert stats["failure_mode"] == "persistent_energy_deficit"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analyze_capability_envelope.py -q`
Expected: FAIL because the analyzer does not exist yet.

**Step 3: Write minimal implementation**

Implement `summarize_case()` and related helpers that compute, at minimum:
- `mean_abs_height_error_m`
- `max_abs_height_error_m`
- `mean_abs_lateral_error_m`
- `progress_ratio_final`
- `speed_min_mps`
- `freq_sat_ratio`
- `pitch_sat_ratio`
- `vertical_support_ratio_mean`
- `vertical_support_gap_vs_load_factor`
- `sustainable` boolean
- `failure_mode`

Define sustainability using post-warmup windows, not the first transient:
- tracking window starts after warmup and after segment-entry transient
- fail if persistent negative height error coexists with hard saturation and speed bleed
- fail if path completion is lost due to inability rather than timeout/configuration

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analyze_capability_envelope.py -q`
Expected: PASS.

**Step 5: Syntax check**

Run: `python -m py_compile scripts/flapping_px4/analyze_capability_envelope.py`
Expected: PASS.

**Step 6: Commit**

```bash
git add scripts/flapping_px4/analyze_capability_envelope.py tests/test_analyze_capability_envelope.py
git commit -m "feat: add capability envelope metrics"
```

### Task 4: Run Straight-Climb Envelope Sweep

**Files:**
- Use: `scripts/flapping_px4/run_capability_envelope_suite.py`
- Use: `scripts/flapping_px4/analyze_capability_envelope.py`
- Output: `docs/analysis/flapping_px4/capability_envelope/<timestamp>/straight_climb/*`

**Step 1: Run a coarse straight-climb sweep**

Run a coarse grid that changes only climb demand while keeping straight flight:

```bash
./isaaclab.sh -p scripts/flapping_px4/run_capability_envelope_suite.py \
  --families straight_climb \
  --climb_delta_grid_m 0.0,1.0,2.0,3.0,4.0 \
  --straight_length_grid_m 60,90,120 \
  --height_sp 10.0 \
  --out_root docs/analysis/flapping_px4/capability_envelope
```

**Step 2: Analyze the sweep**

Run:

```bash
./isaaclab.sh -p scripts/flapping_px4/analyze_capability_envelope.py \
  --run_root docs/analysis/flapping_px4/capability_envelope/<timestamp>
```

Expected deliverables:
- envelope summary CSV
- one straight-climb heatmap
- one per-case ranking by `sustainable` then `max_abs_height_error_m`

**Step 3: Interpret only the first hard limit**

Record:
- maximum sustainable climb gradient / delta
- whether the first limit is `freq_hz` saturation, `pitch_sp` saturation, speed decay, or support-ratio deficit

**Step 4: Commit analysis artifacts only if the repo policy for generated analysis allows it**

If not committing generated artifacts, skip commit and keep outputs local.

### Task 5: Run Level-Turn Envelope Sweep

**Files:**
- Use: `scripts/flapping_px4/run_capability_envelope_suite.py`
- Use: `scripts/flapping_px4/analyze_capability_envelope.py`
- Output: `docs/analysis/flapping_px4/capability_envelope/<timestamp>/level_turn/*`

**Step 1: Run a coarse level-turn sweep**

Sweep only turn geometry with zero climb:

```bash
./isaaclab.sh -p scripts/flapping_px4/run_capability_envelope_suite.py \
  --families level_turn \
  --turn_radius_grid_m 14,16,18,20,24,28 \
  --turn_sweep_grid_deg 90,120,180 \
  --loiter_radius_grid_m 16,20,24 \
  --loiter_turns_grid 1.0,1.5,2.0 \
  --height_sp 10.0 \
  --out_root docs/analysis/flapping_px4/capability_envelope
```

**Step 2: Analyze the sweep**

Run the analyzer and produce:
- sustainable radius boundary
- load-factor versus support-gap scatter
- saturation ratio versus roll demand plot

**Step 3: Record the first unsustainable condition**

Answer with evidence:
- smallest sustainable turn radius
- largest sustainable loiter duration
- whether the first failure is bank-induced support deficit or lateral/path-tracking breakdown

### Task 6: Run Coupled Climb-Turn Sweep

**Files:**
- Use: `scripts/flapping_px4/run_capability_envelope_suite.py`
- Use: `scripts/flapping_px4/analyze_capability_envelope.py`
- Output: `docs/analysis/flapping_px4/capability_envelope/<timestamp>/climb_turn/*`

**Step 1: Run a coupled sweep around the current problematic regime**

Focus on the regime that already looked marginal:

```bash
./isaaclab.sh -p scripts/flapping_px4/run_capability_envelope_suite.py \
  --families climb_turn \
  --climb_delta_grid_m 1.0,2.0,3.0 \
  --straight_length_grid_m 40,60,80 \
  --turn_radius_grid_m 16,20,24 \
  --turn_sweep_grid_deg 90,120,180 \
  --loiter_radius_grid_m 18,20,24 \
  --loiter_turns_grid 1.0,1.5 \
  --height_sp 10.0 \
  --out_root docs/analysis/flapping_px4/capability_envelope
```

**Step 2: Analyze the coupling penalty**

Compute, per geometry pair:
- straight-climb sustainable margin
- level-turn sustainable margin
- coupled-case sustainable margin
- coupling penalty = coupled margin loss relative to the two isolated cases

**Step 3: Identify the real mission-design boundary**

State whether the current mission failures come from:
- climb demand alone
- turn demand alone
- or the coupled energy budget

### Task 7: Produce A Decision Memo

**Files:**
- Create: `docs/analysis/flapping_px4/capability_envelope/<timestamp>/decision_memo.md`

**Step 1: Write the memo**

The memo must answer:
- what is the current sustainable straight-climb limit
- what is the current sustainable level-turn / loiter limit
- what coupled regimes are unsustainable
- which actuator/controller variable saturates first
- whether next work should go to mission geometry, controller energy management, or airframe authority

**Step 2: Include exact evidence**

Embed:
- 1 straight-climb heatmap
- 1 level-turn heatmap
- 1 coupled-case heatmap
- 1 representative time-series plot for the first failing case in each family

**Step 3: No tuning yet**

Do not change controller gains in this phase. The deliverable is an envelope and a ranked bottleneck list.

### Task 8: Final Verification

**Files:**
- Verify: `scripts/flapping_px4/run_capability_envelope_suite.py`
- Verify: `scripts/flapping_px4/analyze_capability_envelope.py`
- Verify: `scripts/flapping_px4/plot_path_mission.py`
- Verify: `tests/test_run_capability_envelope_suite.py`
- Verify: `tests/test_analyze_capability_envelope.py`
- Verify: `tests/test_plot_path_mission.py`

**Step 1: Run targeted unit tests**

Run:
- `python -m pytest tests/test_plot_path_mission.py -q`
- `python -m pytest tests/test_run_capability_envelope_suite.py -q`
- `python -m pytest tests/test_analyze_capability_envelope.py -q`

Expected: all PASS.

**Step 2: Run syntax verification**

Run:
- `python -m py_compile scripts/flapping_px4/plot_path_mission.py`
- `python -m py_compile scripts/flapping_px4/run_capability_envelope_suite.py`
- `python -m py_compile scripts/flapping_px4/analyze_capability_envelope.py`

Expected: all PASS.

**Step 3: Run one smoke sweep per family**

Run one reduced-size sweep for:
- `straight_climb`
- `level_turn`
- `climb_turn`

Expected:
- artifacts generated successfully
- manifest complete
- analyzer emits summary without crashing

**Step 4: Commit**

```bash
git add scripts/flapping_px4/plot_path_mission.py scripts/flapping_px4/run_capability_envelope_suite.py scripts/flapping_px4/analyze_capability_envelope.py tests/test_plot_path_mission.py tests/test_run_capability_envelope_suite.py tests/test_analyze_capability_envelope.py
git commit -m "feat: add flapping robot capability envelope evaluation"
```


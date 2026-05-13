# DeLaurier Physical Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and run a physically calibrated DeLaurier baseline using eight bounded, interpretable parameters.

**Architecture:** Extend the existing offline DeLaurier analysis module with parameter mapping, phase-shift handling, normalized objective evaluation, and deterministic random-search calibration. Add a CLI script that optimizes on a training subset, evaluates full train/val/test splits, and writes paper-ready result artifacts.

**Tech Stack:** Python 3.11, pandas, NumPy, PyTorch, pytest, existing `flapping_bot.physics` modules.

---

### Task 1: Parameter API and Objective Tests

**Files:**
- Modify: `source/flapping_bot/flapping_bot/analysis/delaurier_gain_calibration.py`
- Modify: `tests/test_delaurier_gain_calibration.py`

**Step 1: Write failing tests**

Add tests for:

```python
def test_physical_parameters_map_to_offline_config():
    params = PhysicalCalibrationParameters(...)
    cfg = params.to_offline_config(base_cfg)
    assert cfg.theta_w_deg == ...
```

```python
def test_normalized_objective_uses_train_scale_and_regularization():
    objective = normalized_wrench_objective(...)
    assert objective == pytest.approx(...)
```

**Step 2: Run RED**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_delaurier_gain_calibration.py
```

Expected: fail because physical calibration symbols do not exist.

**Step 3: Implement minimal API**

Add:

```python
PHYSICAL_PARAMETER_BOUNDS
PHYSICAL_PARAMETER_NOMINALS
PhysicalCalibrationParameters
parameter_vector_to_config
normalized_wrench_objective
```

**Step 4: Run GREEN**

Run the same pytest command and confirm pass.

### Task 2: Phase Delay Support

**Files:**
- Modify: `source/flapping_bot/flapping_bot/analysis/delaurier_gain_calibration.py`
- Modify: `tests/test_delaurier_gain_calibration.py`

**Step 1: Write failing tests**

Add a test showing that `phase_delay_s` shifts `wing_stroke_angle_rad`, `qd`, and `qdd` within each `log_id` without crossing log boundaries.

**Step 2: Run RED**

Run targeted pytest. Expected: fail because phase delay is ignored.

**Step 3: Implement phase-shifted wing state**

Add grouped interpolation by `log_id`. Use edge-value filling at log boundaries.

**Step 4: Run GREEN**

Run targeted pytest and full file pytest.

### Task 3: Search Helper

**Files:**
- Modify: `source/flapping_bot/flapping_bot/analysis/delaurier_gain_calibration.py`
- Modify: `tests/test_delaurier_gain_calibration.py`

**Step 1: Write failing tests**

Add deterministic test for random candidate generation:

```python
result = random_search_physical_calibration(..., seed=3, n_candidates=3)
assert list(result.trace.columns) == [...]
assert result.best_parameters.name == ...
```

Use a fake predictor callback so the test does not run torch physics.

**Step 2: Run RED**

Expected: fail because search helper does not exist.

**Step 3: Implement helper**

Add a small dataclass result and deterministic random sampling inside bounded ranges.

**Step 4: Run GREEN**

Run pytest.

### Task 4: CLI Script

**Files:**
- Create: `scripts/flapping_px4/run_delaurier_physical_calibration.py`

**Step 1: Implement CLI**

Load splits, select deterministic train subset, run search, evaluate final parameters on all splits, and write:

```text
parameters.csv
optimization_trace.csv
metrics_by_split.csv
summary.json
README.md
```

**Step 2: Smoke Test**

Run:

```bash
conda run -n flap-train-gpu python scripts/flapping_px4/run_delaurier_physical_calibration.py \
  --max-rows-per-split 128 \
  --calibration-rows 96 \
  --n-candidates 3 \
  --batch-size 64 \
  --output-dir /tmp/delaurier_physical_smoke
```

Expected: result artifacts are written and metrics contain no NaNs.

### Task 5: Full Result and Commit

**Files:**
- Create: `docs/analysis/effective_wrench/delaurier_physical_calibration_v1/*`

**Step 1: Run full calibration**

Run with a practical first budget:

```bash
conda run -n flap-train-gpu python scripts/flapping_px4/run_delaurier_physical_calibration.py \
  --calibration-rows 12000 \
  --n-candidates 48 \
  --batch-size 8192 \
  --output-dir docs/analysis/effective_wrench/delaurier_physical_calibration_v1
```

**Step 2: Verify**

Run:

```bash
./isaaclab.sh -p -m pytest tests/test_delaurier_gain_calibration.py
conda run -n flap-train-gpu python -m py_compile source/flapping_bot/flapping_bot/analysis/delaurier_gain_calibration.py scripts/flapping_px4/run_delaurier_physical_calibration.py
conda run -n flap-train-gpu python -c "import pandas as pd; ..."
```

**Step 3: Commit**

Commit implementation and result artifacts:

```bash
git add ...
git commit -m "feat: add DeLaurier physical calibration baseline"
```

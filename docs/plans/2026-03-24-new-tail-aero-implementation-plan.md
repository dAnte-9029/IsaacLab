# New Tail Aero Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current lumped `elevator + rudder` tail aerodynamics with a five-surface model driven by the latest tail geometry and theory-based nominal coefficients.

**Architecture:** Keep the environment action interface unchanged, but update the tail aero backend to explicitly represent `fixed_horizontal`, `left_elevon`, `right_elevon`, `fixed_vertical`, and `rudder`. Use geometry-derived surface placement plus low-order finite-wing coefficient formulas for nominal lift slope, with explicit shared drag and angle-limit assumptions.

**Tech Stack:** Python 3.11, PyTorch, pytest, IsaacLab direct env configuration.

---

### Task 1: Lock the intended five-surface behavior in tests

**Files:**
- Modify: `tests/test_tail_aero.py`

**Step 1: Write the failing test**

```python
def test_tail_aero_config_exposes_five_surfaces():
    cfg = TailAeroCfg()
    assert cfg.fixed_horizontal.name == "fixed_horizontal"
    assert cfg.left_elevon.name == "left_elevon"
    assert cfg.right_elevon.name == "right_elevon"
    assert cfg.fixed_vertical.name == "fixed_vertical"
    assert cfg.rudder.name == "rudder"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tail_aero.py::test_tail_aero_config_exposes_five_surfaces -q`

Expected: FAIL because the current config only exposes `elevator` and `rudder`.

**Step 3: Write the minimal implementation**

- Replace the config structure in `tail_aero.py` so all five surfaces exist explicitly.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tail_aero.py::test_tail_aero_config_exposes_five_surfaces -q`

Expected: PASS

### Task 2: Lock the new force/moment contract in tests

**Files:**
- Modify: `tests/test_tail_aero.py`

**Step 1: Write the failing tests**

```python
def test_tail_aero_symmetric_elevons_generate_pitch_without_roll():
    ...

def test_tail_aero_differential_elevons_generate_roll():
    ...

def test_tail_aero_rudder_generates_yaw():
    ...
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tail_aero.py -q`

Expected: FAIL because the current model has no separate left/right elevon surfaces.

**Step 3: Write the minimal implementation**

- Update `TailAeroModel.compute_wrench` to accept `left_elevon_rad`, `right_elevon_rad`, and `rudder_rad`.
- Sum body-frame wrench contributions from the five explicit surfaces.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tail_aero.py -q`

Expected: PASS

### Task 3: Wire the straight-flight environment to the new tail model

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`

**Step 1: Write the failing regression test**

```python
def test_tail_aero_compute_wrench_accepts_split_elevons():
    model = TailAeroModel(TailAeroCfg(), device=torch.device("cpu"))
    model.compute_wrench(...)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tail_aero.py -q`

Expected: FAIL if the environment-facing API still requires lumped `elevator_rad`.

**Step 3: Write the minimal implementation**

- Pass left/right elevon commands directly into `compute_wrench`.
- Keep the action space and joint command mapping unchanged.
- Retain the virtual roll/pitch augmentation terms for now, but base pitch augmentation on average elevon command.

**Step 4: Run targeted tests**

Run: `pytest tests/test_tail_aero.py tests/test_straight_flight_env_reset_contract.py -q`

Expected: PASS

### Task 4: Final verification

**Files:**
- Modify: none

**Step 1: Run the focused verification suite**

Run: `pytest tests/test_flapping_asset_cfg.py tests/test_tail_geometry.py tests/test_tail_aero.py tests/test_flapping_joint_contracts.py tests/test_flapping_task_registration.py tests/test_straight_flight_env_reset_contract.py -q`

Expected: PASS

**Step 2: Review diff scope**

Run: `git diff -- docs/plans/2026-03-24-new-tail-aero-implementation-plan.md source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py source/flapping_bot/flapping_bot/physics/tail_aero.py tests/test_tail_aero.py`

Expected: diff is limited to the new plan, tail aero backend, env wiring, and tests.

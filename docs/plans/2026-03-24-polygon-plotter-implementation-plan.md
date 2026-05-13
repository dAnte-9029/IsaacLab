# Polygon Plotter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a tiny script that plots one 2D polygon from ordered points written directly in Python code.

**Architecture:** The implementation will be a standalone `matplotlib` script under `scripts/tools/` with a couple of pure helper functions so the core behavior can be tested without opening a GUI. The script will keep the input inline as a simple `POINTS` constant and expose only minimal CLI flags for save/title/label control.

**Tech Stack:** Python 3.11, matplotlib, pytest

---

### Task 1: Add a failing test for polygon closure and validation

**Files:**
- Create: `tests/test_plot_polygon_2d.py`

**Step 1: Write the failing test**

- Load the script module from `scripts/tools/plot_polygon_2d.py`.
- Assert that:
  - an open polygon is closed by the helper
  - an already closed polygon is unchanged
  - too few points raises `ValueError`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_plot_polygon_2d.py -q`

Expected: FAIL because the script module does not exist yet.

### Task 2: Implement the minimal plotting script

**Files:**
- Create: `scripts/tools/plot_polygon_2d.py`

**Step 1: Add pure helpers**

- Add a validator for ordered points.
- Add a helper that returns a closed polygon path.

**Step 2: Add plotting entrypoint**

- Define inline `POINTS`.
- Plot one polygon with equal aspect ratio.
- Add point markers and optional point-index labels.
- Support `--save`, `--title`, and `--hide-labels`.

**Step 3: Run test to verify it passes**

Run: `pytest tests/test_plot_polygon_2d.py -q`

Expected: PASS

### Task 3: Run final verification

**Files:**
- Modify: none

**Step 1: Run targeted verification**

Run: `pytest tests/test_plot_polygon_2d.py -q`

Expected: PASS

**Step 2: Check script syntax**

Run: `python -m py_compile scripts/tools/plot_polygon_2d.py`

Expected: PASS

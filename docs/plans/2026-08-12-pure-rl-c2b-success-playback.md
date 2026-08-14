# PureRL C2b Successful Playback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Record the promoted C2b checkpoint on a deterministic +6 degree climb and emit verified GIF evidence.

**Architecture:** Extend the existing PureRL playback entry point rather than adding a second recorder. Put the
testable task/case validation in `pure_rl_playback_common.py`; keep Isaac scene setup, policy replay, path markers,
success checks, media generation, and manifest writing in `play_pure_rl_checkpoint.py`.

**Tech Stack:** Python, Isaac Lab/Isaac Sim, RSL-RL, imageio, ffmpeg, pytest.

---

### Task 1: Add C2b playback and record the promoted policy

**Files:**
- Modify: `scripts/flapping_rl/pure_rl_playback_common.py`
- Modify: `scripts/flapping_rl/play_pure_rl_checkpoint.py`
- Modify: `tests/test_pure_rl_playback_common.py`

**Step 1: Write failing helper tests**

Add tests that require a C2b playback case to resolve to task ID 1, +6 degrees, 17.5 m entry, and 25 m slope,
and require the C2 success predicate to reject termination or missing recovery.

**Step 2: Verify RED**

Run:

```bash
TERM=xterm ./isaaclab.sh -p -m pytest tests/test_pure_rl_playback_common.py -q
```

Expected: new tests fail because the playback case helpers do not exist.

**Step 3: Implement the minimal extension**

Add typed pure helper output for the selected playback task/case. Update the recorder to parse `--task` and
longitudinal arguments, configure C2 schedules, render the environment path, collect longitudinal telemetry, and
write the case and recovery result into the manifest. Preserve the current C1 defaults.

**Step 4: Verify GREEN**

Run the helper tests and Python compilation. Expected: all pass.

**Step 5: Record and validate artifacts**

Run `model_475.pt` for 12 seconds using the C2b task, +6 degree climb, 0 degree heading and phase, 30 FPS MP4,
and 15 FPS GIF. Verify the manifest reports no termination, recovery reached, finite metrics, nonblank frames, and
the expected task/slope. Inspect multiple decoded GIF frames for visible scene content and motion.

No commit is authorized for this task; leave the focused changes uncommitted and report the artifact paths.

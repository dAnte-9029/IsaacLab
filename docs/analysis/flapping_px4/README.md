# Flapping PX4 Analysis Guide

This directory collects controller-tuning notes, rollout evidence, and targeted diagnosis for the DeLaurier-backed flapping PX4-like stack.

## Current Recommended Reading Order

If you only want the current estimated-teacher baseline and the latest realism checks, read in this order:

1. [estimated_teacher_tuning_handoff_20260412.md](./estimated_teacher_tuning_handoff_20260412.md)
2. [imu_dual_teacher_bringup.md](./imu_dual_teacher_bringup.md)
3. [px4like_hardening_20260415/README.md](./px4like_hardening_20260415/README.md)

## Current Practical Baseline

- Airframe backend: `DeLaurier`
- Control focus: `estimated teacher`, not truth-teacher retuning, not RL reward/actor changes
- Runtime baseline:
  - `teacher_state_source=estimated`
  - `policy_state_source=estimated`
  - `imu_source=synthetic`
- Mass baseline used by current controller evidence: `total_mass_kg_override=0.95`

## How To Interpret `synthetic` vs `isaacsim`

- `synthetic`
  - Uses truth-derived IMU signals inside the repo-local estimator path.
  - Primary tuning baseline because it is easier to reproduce and isolates controller and estimator logic without extra Isaac sensor realism effects.
- `isaacsim`
  - Uses the real Isaac Lab IMU sensor object.
  - Treated as the next realism gate after the `synthetic` path is already acceptable.

The current evidence says both paths are viable. `synthetic` remains the main controller-tuning baseline, while `isaacsim` is the transfer and realism check.

## Most Relevant Current Artifacts

- [estimated_teacher_tuning_handoff_20260412.md](./estimated_teacher_tuning_handoff_20260412.md)
  - Handoff doc for the estimated-teacher tuning phase.
- [estimated_teacher_tuning_prompt_20260412.md](./estimated_teacher_tuning_prompt_20260412.md)
  - Prompt/spec used to constrain the tuning work.
- [imu_dual_teacher_bringup.md](./imu_dual_teacher_bringup.md)
  - Runtime contract and live Isaac IMU bringup notes.
- [px4like_hardening_20260415/README.md](./px4like_hardening_20260415/README.md)
  - Summary of the current PX4-inspired hardening pass, no-wind evidence, and `2 m/s` wind checks.

## Directory Notes

- `px4like_hardening_20260415/`
  - Latest high-value evidence for the estimated-teacher baseline.
- Older dated folders
  - Historical tuning attempts, one-off diagnosis passes, and archived plots.
  - Useful for forensics, but not the first place to start if the goal is to understand the current baseline.

## Short Decision Summary

- `estimated + synthetic + 0.95kg` is the current main realism baseline for controller work.
- `estimated + isaacsim` is no longer treated as a broken path; it is now mainly a realism transfer check.
- Under the current `2 m/s` steady-crosswind checks, both `synthetic` and `isaacsim` are broadly acceptable, with residual issues concentrated in lateral recapture during more difficult path segments rather than gross altitude collapse.

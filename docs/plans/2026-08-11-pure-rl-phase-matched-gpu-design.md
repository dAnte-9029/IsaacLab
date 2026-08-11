# PureRL phase-matched GPU implicit drive design

## Goal

Reduce CPU-native versus GPU-implicit free-flight error without changing the frozen stage-3 thresholds, CPU
plant, reward, observations, tail actuators, or aerodynamic parameters.

## Root-cause hypothesis

The CPU-native drive advances its sinusoidal reference through `IdealFrequencyPhaseState`, while the existing GPU
`ideal_coupled_drive` advances phase directly from the frequency-governor output. In the authoritative stage-3
replay, requested and applied frequency commands matched exactly, but actual mechanism frequency differed by up
to 0.060 Hz and commanded wing position differed by 14.12--16.52 deg. This is the first isolated mismatch to
remove.

## Selected approach

Add a selectable `phase_matched_implicit_drive` variant. It reuses the exact CPU-native
`step_ideal_frequency_phase` reference generator, then sends its position and velocity reference to the existing
GPU-compatible implicit wing actuator. It keeps the hard opposed-wing mimic, actual joint state, existing tail
actuators, replicated GPU PhysX, and current DeLaurier coupling.

The existing `ideal_coupled_drive` remains available as the GPU baseline. New C1 and C2a environment configs and
task IDs select the phase-matched variant explicitly.

## Data flow

```text
raw frequency action
  -> existing 2 Hz/s governor
  -> normalized phase target frequency
  -> IdealFrequencyPhaseState / step_ideal_frequency_phase
  -> q_cmd, qd_cmd, qdd_cmd
  -> existing GPU implicit left-wing position/velocity targets
  -> hard opposed-wing mimic
  -> actual wing joint state
  -> existing actual-motion per-wing DeLaurier loads
```

## Scope boundary

- Do not change CPU-native behavior or task IDs.
- Do not change the existing GPU implicit task defaults.
- Do not tune implicit actuator gains, tail gains, physical properties, reward, observations, or termination.
- Do not change `wing_aero_acceleration_source` in the first experiment. GPU actual acceleration versus CPU
  prescribed acceleration remains a separately testable residual hypothesis.
- Do not relax any drive-gate or paired-plant threshold.

## Validation

1. Contract tests prove the new variant is selectable, initializes the shared phase state, uses the ideal phase
   step, and preserves the existing implicit actuator/configuration path.
2. New task-registration tests prove that C1 and C2a phase-matched task IDs are explicit and that old GPU task IDs
   remain unchanged.
3. The 4 Hz and 5 Hz fixed-root drive gate must continue to pass.
4. The unchanged five-case paired-plant stage-3 gate must pass every existing threshold.

If phase-command mismatch collapses but stage 3 still fails, stop and attribute the residual before changing the
aerodynamic acceleration source or actuator gains.

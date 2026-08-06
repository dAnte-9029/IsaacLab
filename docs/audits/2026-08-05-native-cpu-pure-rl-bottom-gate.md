# Native CPU PureRL Bottom-Layer Gate

Date: 2026-08-05

Status: passed for the frequency contract, native CPU runtime, repeated reset,
launcher return-code propagation and generated-asset isolation described below.
This audit does not claim that the policy converges or that all four action
channels have sufficient closed-loop authority.

Update: the four-channel action-authority gate identified below was completed
on 2026-08-06. The Measured PureRL action is now direct-surface; see
`docs/audits/2026-08-06-pure-rl-action-step-response.md`.

## 1. Scope and provenance

- Branch: `feat/native-multibody-rl`
- Baseline: `flapping_rl` at `1200ca8d`
- Task: `Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-Direct-v0`
- Plant: canonical native measured-wing multibody plant on CPU
- Policy interface: existing four-dimensional high-level action
- Frequency range approved for this PureRL task: 0--5 Hz

The canonical DeLaurier controller task retains its 2--5 Hz range. No plant,
aerodynamic, observation, reward, termination or PPO hyperparameter was changed
for this gate.

## 2. Frequency action contract

The PureRL frequency action is now mapped as follows:

| Normalized action | Frequency target |
| ---: | ---: |
| -1.0 | 0 Hz |
| 0.0 | 2.5 Hz |
| 1.0 | 5 Hz |

Inputs are clamped to `[-1, 1]`. The mapping and inverse mapping preserve the
input tensor device and dtype and reject non-finite actions or invalid bounds.
The existing 4 Hz reset frequency maps to normalized action 0.6.

The environment observation continues to expose the internal actual frequency,
not the requested target. The native drive therefore retains its approximately
0.15 s frequency settling behavior rather than pretending that the actuator is
instantaneous.

## 3. Launcher and asset isolation

Native CPU child processes now use the active Conda Python directly. This is
required because the repository shell wrapper does not propagate a failing
child process return code reliably. Non-native launch behavior continues to use
the repository wrapper.

Native training and watcher processes prepend the current worktree project and
asset source trees to `PYTHONPATH`. URDF conversion is directed to a portable
run-local generated-asset directory. The bounded smoke therefore did not
rewrite tracked converter metadata or create generated assets in the primary
checkout.

## 4. Pure Python validation

The combined focused suite passed 57 tests. It covered:

- the 0--5 Hz forward and inverse action mapping;
- task-specific frequency bounds and preservation of the 2--5 Hz baseline;
- ideal inverse and sinusoidal drive contracts;
- environment reset contracts;
- native and non-native launcher command construction;
- worktree source precedence and portable asset cache routing.

The changed Python files also passed bytecode compilation and `git diff
--check`.

## 5. Isaac runtime gate

A fresh CPU process created 64 PureRL environments and ran 180 policy steps.
The test requested a repeated pattern spanning 0, 1, 2, 2.5, 3, 4 and 5 Hz and
forced 93 partial environment resets at policy steps 45, 90 and 135.

Results:

- observations, rewards, root state, joint state, frequency state, target
  frequency and aerodynamic forces and moments remained finite;
- no termination or truncation occurred under the gate's deliberately
  permissive termination thresholds;
- final targets for the protected environments were approximately 0, 2.5 and
  5 Hz;
- final actual frequencies were approximately 0.0000033, 2.5000031 and
  4.9999948 Hz;
- maximum wing reference tracking error was 0.0520 deg;
- maximum left/right mechanism synchronization error was 0.0000034 deg.

An initial test revision compared the post-physics joint state against the
pre-physics command and reported 1.921 deg at 5 Hz. That value is exactly the
phase advanced by one 1/480 s physics step. Correcting the test to compare
states at the same time index reduced the bound to the value above; no drive or
plant parameter was changed to obtain the pass.

## 6. End-to-end launcher smoke

An eight-environment, one-iteration native CPU launch completed 384 environment
steps, wrote `model_0.pt`, returned code 0 and completed the final one-environment
watcher evaluation. Its run directory is:

`logs/rsl_rl/flapping_bot_straight_flight/2026-08-05_19-18-55_native_cpu_zero5_bottom_gate`

The resolved environment configuration records CPU simulation and policy
devices, eight environments, reset freeze disabled, 0--5 Hz bounds, the
current-worktree URDF and the portable generated-asset directory. The watcher
score is startup evidence only and is not a convergence or policy-quality
result.

## 7. Remaining gates and limitations

This closes the mechanics needed to start high-level PureRL experiments for the
frequency channel, repeated reset and launch pipeline. Before interpreting
learning curves, a separate four-channel action-authority gate should verify
the signs, magnitudes, saturation behavior and cross-coupling of frequency,
rudder, elevon pitch and elevon roll about a reproducible trim condition.

A 1024-environment throughput/soak benchmark remains separate from functional
correctness. The host used the `powersave` CPU governor, so current throughput
must not be treated as a formal performance result.

Known non-fatal startup and shutdown warnings remain: missing native-extension
`generatedSchema.usda`, headless GLFW messages, an existing-prim warning during
URDF conversion and Isaac stage-detach/recursive-unload shutdown messages. The
fresh validation processes nevertheless returned code 0.

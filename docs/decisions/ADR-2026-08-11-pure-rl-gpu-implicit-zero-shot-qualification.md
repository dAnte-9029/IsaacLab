# ADR-2026-08-11: PureRL GPU implicit zero-shot qualification

- Status: Superseded by `ADR-2026-08-11-reject-pure-rl-gpu-implicit-training.md`
- Date: 2026-08-11
- Scope: CPU-trained PureRL checkpoint screening on the direct-GPU ideal-coupled plant
- Depends on: `ADR-2026-08-10-pure-rl-curriculum-domain-contract.md`

## Context

The authoritative PureRL plant uses the CPU-only native holonomic wing mechanism. The explicit
`gpu_implicit_candidate` instead uses the measured multibody aircraft, a high-gain implicit left-wing drive,
the hard opposed-wing mimic, replicated direct-GPU PhysX, and actual-motion per-wing DeLaurier loads. A
corrected fixed-root one-second probe showed acceptable 4 and 5 Hz tracking for screening and about four times
the raw environment-step throughput of the native CPU probe. It did not establish free-flight or policy
equivalence.

The promoted C2a policy candidate is:

`logs/rsl_rl/flapping_bot_straight_flight/2026-08-10_19-45-14_pure_rl_c2a_seed0/model_1500.pt`

Its existing CPU-native evidence passes the 16-case C1 fixed heading/phase suite and the 80-case C2a promotion
grid. The next question is whether the same policy can run zero-shot on the GPU candidate, and, if it cannot,
whether the failure is primarily associated with wing tracking, aerodynamic-load shift, actuator limits, or
closed-loop policy amplification.

## Decision

### First qualification is inference-only

The first GPU qualification loads the CPU-trained `model_1500.pt` without PPO training, optimizer restore,
fine-tuning, reward changes, plant tuning, or controller tuning. It evaluates:

- C1: the existing 16-case fixed heading/initial-flap-phase grid;
- C2a: the existing 80-case fixed level/climb/descent, heading, and initial-flap-phase promotion grid.

The signed 10-degree C2a grid is outside this first gate. It may be run later as diagnostic evidence, but it
cannot change the zero-shot pass/fail decision.

### Backends remain explicit and unequal in authority

Add explicit GPU C1 and C2a environment configurations and task IDs. They preserve
`measured_pure_rl_shared_v1` exactly:

- physics step `1/480 s`;
- policy rate 60 Hz through decimation 8;
- four actions: flap frequency, rudder, left elevon, right elevon;
- 0--5 Hz frequency range and 2 Hz/s governor;
- actual tail-joint aerodynamic deflection;
- 555 actor observations;
- the same reward, termination, reset, and path contracts.

Only the backend changes:

- `wing_drive_variant=ideal_coupled_drive`;
- actual-motion per-wing-link DeLaurier coupling;
- `sim.device=cuda`;
- `scene.replicate_physics=true`;
- measured wing/body properties and `retain_accelerations=false`;
- the accepted implicit-drive solver and actuator settings remain unchanged.

The CPU-native task IDs and defaults remain unchanged and authoritative. A GPU pass permits a later GPU PPO
screening trial; it does not promote the GPU plant or a GPU-trained policy to CPU-native authority.

### Closed-loop comparison

Each backend runs in a fresh Isaac Sim process. CPU and GPU use the same checkpoint, fixed case IDs, headings,
initial flap phases, C2a task/slope schedules, episode horizon, observation source, and policy inference code.
The evaluator fails closed when any schedule, action dimension, observation dimension, checkpoint load, or
backend contract differs.

The existing C1 and C2a success gates are the hard capability gates. CPU-to-GPU metric deltas are diagnostic;
they do not relax or replace those gates.

### Conditional action replay

If GPU closed-loop evaluation produces any task failure, nonfinite state, or runtime failure, select a bounded
set of diagnostic cases and rerun them in fresh processes:

1. CPU-native closed loop records the policy action sequence.
2. GPU implicit replays the exact CPU action sequence without querying the policy.
3. CPU and GPU traces are aligned by case ID and policy-step time.

If the GPU also fails under CPU-action replay, the evidence points toward plant/backend mismatch. If replay is
bounded but GPU closed loop fails, the evidence points toward observation-policy feedback amplifying a smaller
backend difference. This is diagnostic attribution, not proof of physical causality.

Detailed traces are limited to at most eight cases per suite. If there are eight or fewer failures, record all
of them. Otherwise select deterministically to cover termination cause and C2a task type before filling the
remaining slots by earliest failure and largest path error. Add at most two worst successful cases when slots
remain. All 16 C1 and 80 C2a cases still receive aggregate per-case metrics.

## Diagnostic artifacts

One qualification root contains:

```text
manifest.json
backend_comparison_summary.json
backend_comparison_summary.csv
per_case_metrics.csv
failure_report.json
failure_report.md
traces/<backend>/<case_id>.npz
plots/<case_id>.png
actions/cpu_closed_loop/<case_id>.npz
```

`manifest.json` records the checkpoint path, task IDs, backend contracts, suite versions, case order, devices,
episode horizon, and whether each subprocess completed. Generated outputs remain untracked.

`per_case_metrics.csv` records success, termination cause, duration, progress/recovery, route errors, tilt and
angular rate, frequency and tail limit fractions, action slew/governor activity, wing tracking error, and finite
state status.

Selected 60 Hz traces record:

- root position, quaternion, linear velocity, and angular velocity;
- path errors, tangent/normal velocities, reward terms, and termination flags;
- raw policy action, executed action, requested/applied frequency, and actual tail angles;
- commanded and actual wing position/velocity plus tracking and mimic errors;
- per-wing link force/moment and total wing, tail, and vehicle aerodynamic wrench.

The report ranks evidence under these labels:

- `runtime_or_nonfinite`;
- `wing_drive_tracking`;
- `aero_load_shift`;
- `policy_feedback_shift`;
- `actuator_or_governor_limit`;
- `flight_state_instability`;
- `path_tracking_only`;
- `unattributed`.

Every label includes the raw supporting metrics, first observed event, and closed-loop versus replay outcome.
The report uses terms such as `primary evidence` and `consistent with`; it must not state an unverified causal
conclusion.

## Qualification outcomes

- **Zero-shot pass:** both GPU C1 and GPU C2a pass their existing hard gates with finite complete grids. GPU PPO
  screening may be designed next, while CPU-native promotion remains mandatory.
- **Task failure with usable diagnostics:** the experiment is a valid negative result. Use the closed-loop/replay
  evidence to decide whether one narrowly scoped backend correction is justified.
- **Runtime or nonfinite failure:** stop before PPO. Fix only the demonstrated startup, force-retention, reset, or
  numerical defect, then rerun this unchanged gate.
- **Unattributed failure:** do not tune rewards or plant parameters. Improve only the missing diagnostic signal and
  rerun the affected selected cases.

## Alternatives considered

- Train immediately on GPU and check only the final policy on CPU: rejected because curriculum learning could
  adapt to a materially different plant before backend compatibility is measured.
- Compare aggregate returns only: rejected because it cannot separate drive tracking, aerodynamic load, actuator,
  and feedback failures.
- Record full high-rate traces for every case: rejected because 60 Hz selected-case traces contain the required
  evidence with much lower output and evaluation overhead.
- Tune implicit-drive gains after observing zero-shot failures: rejected for this gate because it would mix
  qualification with backend fitting.

## Consequences

- The first result directly answers whether the promoted CPU checkpoint can be reused on the GPU plant.
- A failed result still produces evidence suitable for a continue/stop decision.
- CPU-native baseline behavior and promotion authority do not change.
- GPU training throughput is not claimed by this inference gate; a separate short PPO timing trial is required
  after zero-shot qualification.

## Reconsideration triggers

Reconsider the GPU candidate only after one of the following:

- zero-shot C1/C2a passes and a GPU PPO timing trial is requested;
- replay isolates a narrow, correctable implementation defect;
- measured actuator/transmission data supports replacing the ideal implicit drive;
- the policy-facing action or observation contract changes, which creates a new experiment family.

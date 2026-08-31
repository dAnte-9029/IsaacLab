# ADR-2026-08-27: C1-initialized joint multi-task training for PureRL C3a

- Status: Accepted
- Date: 2026-08-27
- Scope: alternative C3a training lineage on the measured CPU-native plant
- Supersedes: the C3a source-lineage requirement in the 2026-08-11 C3 handoff and the C2c-source-only
  consequence in `ADR-2026-08-13-pure-rl-c3a-exact-c2c-rehearsal.md`, only when the explicit joint route is used

## Context

Sequential C3a training starts from the promoted C2c policy and then optimizes the shared actor on the accepted
`15/35/50` C1/C2c/C3a mixture. Multiple seed-0 experiments learned the complete C3a turn grid but lost the
positive 12-degree C2c climb. Exact C2c rehearsal, actor distillation, adaptive sampling, bounded optimizer warm
start, task-aware PPO, and a wider actor did not produce a checkpoint that passed C3a, C1, and C2c together.

These results do not distinguish path-dependent overwrite of a specialized C2c policy from interference inherent
to joint C1/C2c/C3a optimization. The accepted C3a environment already contains the authoritative C2c sampler and
query, the accepted `15/35/50` task mixture, climb/descent weighting of `2:1`, and task-aware PPO diagnostics. A
clean comparison can therefore change only the initialization: use the selected stable C1 policy rather than the
specialized C2c policy, and learn C2c and C3a together from the start.

The selected C1 authority source is:

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-08-07_05-05-44_curriculum1_overnight_seed2/model_1300.pt
```

The existing seed-0 C2a-to-C2c lineage used this same C1 checkpoint, so the sequential and joint routes share the
same stable-flight starting policy.

## Decision

Accept an explicit `c3a_joint_from_c1_v1` route as the primary next C3a experiment. It uses the registered measured
CPU-native C3a task and initializes policy weights only from the selected C1 `model_1300.pt`. It creates a fresh
optimizer and iteration counter.

The controlled recipe is:

- seed 0, 256 environments, 48 steps per environment, 16 mini-batches;
- 201 PPO iterations and checkpoints every 25 iterations, yielding evidence through `model_200.pt`;
- C1/C2c/C3a task probabilities `0.15/0.35/0.50`;
- C2c internal probabilities level/climb/descent `0, 2/3, 1/3`;
- strong-climb probability `0.5` within C2c climb;
- task-aware PPO, bounded warm-start guard, and gradient probe enabled;
- actor and critic hidden dimensions remain `[256,128]`;
- actor distillation, adaptive sampling, recovery recycling, wider actor, PCGrad, GEM, EWC, and residual policy
  disabled;
- CPU-native, no wind, train-only execution.

The launcher exposes this route only through `--c3a-joint-from-c1`, records source stage `c1_straight` and route
`c3a_joint_from_c1_v1` in `curriculum_source.json`, and rejects conflicting overrides. The existing sequential
`c2c -> c3a` route remains available and retains its original provenance checks.

The joint route is promotion eligible only under the unchanged authority contract. Starting at iteration 50,
checkpoints `50/75/100/125/150/175/200` are evaluated in separate fresh CPU-native processes on the frozen C3a,
C1, and C2c suites. Promotion still requires two sample-equivalent adjacent checkpoints that pass all three
suites and acceptance by the existing promotion helper. The promoted C2c result remains the C2c capability
baseline even though its actor is not the initialization source.

## Alternatives considered

- Random initialization was rejected because it would spend substantial sampling effort rediscovering stable
  flight and would not isolate curriculum path dependence.
- Continuing from C2c with more rehearsal or regularization was rejected as the primary next experiment because
  several controlled variants have already failed the strong-climb gate.
- Distillation, adaptive sampling, wider networks, PCGrad, GEM, EWC, and residual actors were excluded from the
  first joint run so initialization lineage remains the single changed experimental factor.
- Changing task probabilities was rejected because the accepted `15/35/50` mixture provides the closest
  comparison with prior sequential C3a experiments.
- Treating the run as permanently diagnostic-only was rejected: unchanged frozen gates can judge the resulting
  actor regardless of whether C2c competence was acquired before or jointly with C3a.

## Consequences

- The experiment directly tests whether a stable C1 prior permits one actor to acquire C2c and C3a jointly without
  the path-dependent overwrite observed after C2c specialization.
- A passing seed-0 run establishes existence for this initialization and optimization trajectory; it does not
  establish cross-seed robustness or a general continual-learning result.
- A failing run does not by itself prove insufficient network capacity; it instead motivates examination of joint
  optimization interference before adding more complex algorithms.
- No plant, reward, task geometry, action, 555-value observation, policy rate, network default, evaluation grid, or
  promotion threshold changes.

## Assumptions

- C1 `model_1300.pt` supplies stable flight while retaining the same policy architecture and observation contract
  as the C3a actor.
- The three-rollout bounded warm-start burn-in is sufficient for the C1 policy to reach active C2c and C3a phases;
  startup telemetry must verify this before the launch is handed off.
- The promoted C2c frozen-suite baseline remains the appropriate capability target for joint acquisition.
- Two hundred trainable-iteration checkpoints provide enough evidence to observe whether C2c and C3a improve
  together; `201` launcher iterations are required to materialize `model_200.pt`.

## Validation requirements

- Unit tests must prove the preset freezes every controlled value, records the joint route, and rejects sequential
  Phase A, wider actor, adaptive sampling, distillation, resume, concurrent evaluation, and wrong source stage.
- Existing launcher tests and `git diff --check` must pass.
- Startup evidence must confirm the exact C1 checkpoint, weights-only loading, fresh iteration state, CPU device,
  registered `[256,128]` actor/critic, active task-aware PPO and warm-start diagnostics, and no traceback.
- Frozen evaluation must use the unchanged 16-case C1, 112-case C2c v2, and 96-case C3a v2 suites in separate
  processes. Training reward and online telemetry are not promotion evidence.

## Reconsideration triggers

Reconsider this route if the C1 source cannot survive the initial mixed-task rollouts, the first trainable update
violates the warm-start displacement limit, C2c and C3a remain mutually exclusive through iteration 200, or the
frozen suites expose a failure outside the intended C2c/C3a acquisition question. Do not start C3b without a
legal adjacent C3a/C1/C2c passing pair.

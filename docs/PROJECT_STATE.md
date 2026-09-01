# Project State

## Historical stable baseline

- Branch: `flapping_rl`
- Frozen tag: `delaurier-strip-wrench-v1`
- Current implemented plant: attached-flow DeLaurier strip resultant with strip-integrated wing moment about base COM; five-surface tail `r x F` moment; PX4-like controller/task stack. `delaurier-strip-wrench-v1` records the pre-switch always-active `qd`-scaled virtual twist path, with separation disabled, `d_hat=0`, `c_mac=0`, `dM_a` enabled, `strip_integrated` moment mode, and induced drag disabled. Default near-single-rigid-body mass properties are `0.90415 kg`, CG `(-0.12154, 0.00541, -0.01298) m` in `base_link`, and diagonal inertia `(0.02329, 0.02573, 0.04270) kg m^2` about that CG.

## Current completed stage

2026-07-29 the formal environment engineering phase changed to `q=Gamma*sin(phase)`, with phase zero neutral and starting upstroke. The DeLaurier boundary now uses `phi_D=phase-pi/2`; the legacy cosine stroke remains explicitly selectable. The measured body/right-wing workbook values are also frozen as link-local FLU properties, with an explicit mirrored left wing and provenance hash. Main records: `docs/decisions/ADR-2026-07-29-engineering-flap-phase-sine.md` and `docs/decisions/ADR-2026-07-29-measured-multibody-mass-properties.md`.

## Active stage

The accepted mechanism decision is the project-local native `FlappingWingTrajectoryJoint`, not further effort-drive tuning. For the explicit measured-wing plant, PhysX solves one rheonomic left-wing trajectory row together with the hard opposed-wing mimic, measured wing inertia and per-wing DeLaurier loads. On CPU at 1/480 s, the corrected non-retained-wrench 16 position / 4 velocity iteration configuration passes the 8 m/s 2/3/4/5 Hz matrix with a worst trajectory error of 0.06249 degree and synchronization error below 2e-6 degree. Free-body no-load momentum, 2/5 Hz aerodynamic impulse-momentum closure and a one-second uncontrolled full-plant smoke all pass their numerical gates. The native custom constraint is incompatible with direct-GPU PhysX and fails closed on CUDA. A narrow fixed-root runtime probe measured about 12,006 env-steps/s for 1024 CPU-native environments, versus about 47,654 env-steps/s for a 1024-environment direct-GPU implicit-drive candidate. The latter retained measured wings, hard mimic and per-wing loads but had load-dependent steady tracking error up to 0.433 degree and is not equivalent to the native reference.

As accepted in `docs/decisions/ADR-2026-08-04-promote-native-multibody-default.md`, the canonical `FlappingBotStraightFlightDeLaurierEnvCfg` now selects the CPU native measured-wing plant at 1/480 s, decimation 4 and disabled physics replication. The previous near-single-rigid-body commanded-kinematics DeLaurier behavior remains available as `FlappingBotStraightFlightCommandedKinematicsDeLaurierEnvCfg`. Canonical DeLaurier processes must build and enable `omni.flapping_bot.holonomic_constraint` at Kit startup; controller and learned-policy compatibility is not assumed.

Opt-in native diagnostics now expose a common-coordinate multibody inverse-dynamics estimate of ideal mechanism torque/power and a non-applied actual-joint-acceleration DeLaurier shadow wrench. The 2--5 Hz continuous-frequency fixed/free-root matrix confirms the 0.1 degree mechanism gate at 1/480 and 1/1000 s but not 1/240 s. At 1/480 s, inverse-dynamics RMS metrics differed by at most 2.45 percent from 1/1000 s. The shadow-wrench difference decreased approximately linearly with time step, so prescribed analytical acceleration remains the applied aerodynamic input. See `docs/decisions/ADR-2026-08-05-native-holonomic-load-and-transient-diagnostics.md` and `docs/audits/2026-08-05-native-holonomic-transient-diagnostics.md`.

The measured PureRL C1 engineering loop is now closed on the CPU-native plant: direct flap-frequency/rudder/left-elevon/right-elevon actions, 60 Hz policy rate over 480 Hz physics, actual tail-joint deflection for tail aerodynamics, normalized 555-dimensional observations, term-level reward telemetry, 0--5 Hz frequency command with a 2 Hz/s governor and physical Hz/s penalty, randomized reset heading, rehearsal sampling, and retention evaluation. The shared C1/C2/C3 and CPU/GPU boundary is frozen by `docs/decisions/ADR-2026-08-10-pure-rl-curriculum-domain-contract.md`.

The direct-GPU implicit training route is rejected by `docs/decisions/ADR-2026-08-11-reject-pure-rl-gpu-implicit-training.md`. Phase matching made the fixed-root drive accurate and the isolated tail gate passed, but exact CPU-action replay still failed the frozen free-flight paired-plant gate in moment, position, or attitude metrics. No further implicit-drive tuning or reaction diagnostic is planned. The implementation remains screening-only for reproducibility; CPU-native is the sole training and promotion authority.

PureRL C2 longitudinal engineering is implemented. C2a `model_1575.pt` seeded the completed 500-iteration C2b run `2026-08-12_10-45-45_pure_rl_c2b_seed0`. C2b `model_450.pt` and `model_475.pt` each achieved 100 percent survival, level/climb/descent success, and recovery on the unchanged 112-case v1 fixed grid, then achieved 100 percent C1 retention success with zero termination. The promotion helper accepted the adjacent pair, so `model_475.pt` is the required weights-only source for C2c. C2c now uses the v2 maneuver envelope: 4--12-degree training, a 112-case `0, +/-4, +/-8, +/-12` promotion grid, and a separate +/-15-degree diagnostic. Reward progress remains velocity along the three-dimensional path tangent; lateral and vertical normal velocity are penalized without a target-speed term. The actor observation remains the C1 555-value contract. Promotion requires two adjacent passing checkpoints plus matching C1 retention evidence under the sample-equivalent cadence; missing or old-contract evidence fails closed. See `docs/audits/2026-08-12-pure-rl-c2b-promotion.md` and `docs/decisions/ADR-2026-08-12-pure-rl-maneuver-envelope-v2.md`.

PureRL C3 spatial engineering is implemented under project-local task IDs `Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C3a-Direct-v0`, `C3b-Direct-v0`, and `C3c-Direct-v0`. The first C3a run passed its turn grid but forgot strong C2c climb. Diagnostics localized failure to positive 12 degree tilt terminations and exposed that the original C3 centerline rehearsal did not reproduce the C2c path contract. C3 now retains a separate authoritative longitudinal path for C2c rehearsal. C3a uses 15/35/50 percent C1/C2c/turn sampling and weights C2c climb/descent 2:1. A bounded 50-iteration continuation preserved C1 and 96/96 C3a success but did not recover C2c, so C3a remains unpromoted. See `docs/decisions/ADR-2026-08-13-pure-rl-c3a-exact-c2c-rehearsal.md` and `docs/audits/2026-08-13-pure-rl-c3a-c2c-rehearsal.md`.

A bounded actor-only policy-distillation experiment now preserves the loaded C2c source actor on C1/C2c rehearsal observations while leaving C3a rows, critic learning, and action noise unconstrained. With coefficient 0.05 and 100 iterations, checkpoints 50/75/99 retained 100 percent C3a success. At iteration 99, C2c survival improved to 95.54 percent, climb success to 89.58 percent, and the `+12` degree slice to 11/16, compared with 89.29 percent, 75 percent, and 4/16 in the matched no-distillation experiment. Five `+12` degree tilt terminations remain, and the C1 tail-limit fraction is 11.64 percent, so no checkpoint passes the complete retention gates and C3a remains unpromoted. See `docs/decisions/ADR-2026-08-13-pure-rl-c3a-actor-distillation-experiment.md` and `docs/audits/2026-08-13-pure-rl-c3a-actor-distillation.md`.

A default-disabled retention-aware C3a scheduler now reallocates future on-policy reset rows among C1, exact C2c, and turn families from completed-episode signals. A fresh 200-iteration seed-0 run with distillation coefficient 0.05 preserved 100 percent C3a success and recovered the C1 tail-limit gate by iteration 150. C2c improved to 93.75 percent survival, 85.42 percent climb success, and 9/16 success at `+12` degrees at iteration 175, then regressed at iteration 199. No checkpoint passed C2c, so C3a remains unpromoted and `model_175.pt` is diagnostic only. See `docs/decisions/ADR-2026-08-13-pure-rl-c3a-retention-aware-sampling.md` and `docs/audits/2026-08-14-pure-rl-c3a-retention-aware-sampling.md`.

A weights-only C3a gradient probe localized a separate immediate regression to the first PPO iteration: the first rollout contained no active C2c strong-climb or C3a turn transitions, while a fresh Adam optimizer and adaptive learning-rate schedule produced actor displacement norm `0.38454`. The measured C3a configuration now enables a project-local bounded warm start only on weights-only loading: three rollout-only iterations, ten one-epoch updates ramping from fixed `1e-5` to `5e-5`, then the original epoch count at fixed `5e-5`, with a `0.10` actor-displacement fail-closed gate during those ten warm-up updates. A 16-environment CPU-native smoke preserved source actor weights and zero optimizer steps through checkpoints 0--2; checkpoint 3 used one epoch at `1e-5`, reached active C2c/turn states, and had actor displacement norm `0.00808`. This is startup validation, not retention or promotion evidence. See `docs/decisions/ADR-2026-08-14-pure-rl-c3a-bounded-warm-start.md`.

The controlled 500-iteration bounded-warm-start run learned and retained 96/96 C3a success through iterations 100, 200, 300, 400, 450, and 475, then regressed to 60/96 at iteration 499 with 36 roll-limit terminations. Adjacent C3a-passing checkpoints 450/475 both failed retention: C1 survived 16/16 but tail-limit fractions were 61.53/54.54 percent, while C2c survival was 85.71/87.50 percent, climb success was 66.67/70.83 percent, and positive 12-degree success was 0/16 and 2/16. The guard fixes the immediate cold-optimizer failure but the fixed rehearsal mixture does not prevent long-horizon C1/C2c forgetting or late C3a drift. No checkpoint is promotable. See `docs/audits/2026-08-18-pure-rl-c3a-bounded-warm-start.md`.

Task-aware PPO with actor distillation 0.05 and a separate `[512,256]` actor-capacity experiment both retained C3a but still failed the C2c strong-climb gate. The accepted next authority experiment therefore changes only initialization lineage: the explicit `c3a_joint_from_c1_v1` route starts weights-only from the selected stable C1 `model_1300.pt` and jointly learns the unchanged 15/35/50 C1/C2c/C3a mixture for 201 iterations. Distillation, adaptive sampling and the wider actor remain disabled. This route keeps the sequential C2c-to-C3a baseline available and retains the frozen C1, C2c and C3a promotion gates. See `docs/decisions/ADR-2026-08-27-pure-rl-c3a-joint-from-c1.md` and `docs/handoffs/2026-08-26-pure-rl-c3a-curriculum-forgetting.md`.

The completed `c3a_joint_from_c1_v1` run passed C3a and C1 from iteration 75 onward and reached 100 percent C2c survival/climb success at iteration 175, but every checkpoint failed the unchanged C2c path-accuracy limits. A fixed positive 12-degree comparison localized a deterministic requested-frequency oscillation: the joint actor reversed sign 27 times on the slope, with a 3.49 Hz oscillation matching the 3.43 Hz wingbeat. The promoted C2c actor kept a positive frequency request. A completed `0.05` squared request-delta experiment improved request amplitude, response delay and height error, but its best C2c checkpoint still narrowly failed mean and p95 height limits and sign flips remained essentially unchanged. The accepted next controlled route therefore applies `0.10` L1 total variation at the same clipped pre-governor request boundary. The baseline, squared route, plant, observation, governor, task mixture, network, PPO schedule and frozen gates remain unchanged. See `docs/decisions/ADR-2026-08-27-pure-rl-requested-frequency-smoothness.md` and `docs/decisions/ADR-2026-08-27-pure-rl-requested-frequency-total-variation.md`.

The completed L1 total-variation run still produced 24--27 deterministic request sign reversals on the fixed positive 12-degree trajectory. Current configuration evidence shows the generic action low-pass and rate limit are already disabled; only the accepted symmetric 2 Hz/s physical frequency governor is active. The next controlled route keeps that governor and replaces the hidden preceding-request objective with a `0.05 * abs(clipped request - governor-applied action)` penalty. The cubic flap-frequency proxy remains unchanged so request feasibility and mean-frequency bias remain separate questions. See `docs/decisions/ADR-2026-08-28-pure-rl-frequency-governor-gap.md`.

The governor-gap run produced one all-suite passing checkpoint at iteration 200, but a continuation to iteration 225 regressed to 95/96 C3a survival because one case reached the roll limit; C1 and C2c still passed. There is therefore no adjacent all-suite passing pair. Online and fixed-trajectory diagnostics also show that the governor-gap reward did not remove the wingbeat-synchronous requested-frequency route. The accepted controlled architecture experiment now uses two independent `[256,128]` actor trunks: the tail trunk retains the full 555-value phase-aware observation, while the frequency trunk receives a one-cycle averaged, phase-fixed transformation of the same observation. It warm-starts weights only from the iteration-200 checkpoint, retains the governor-gap recipe and fresh optimizer, and does not change the environment observation, plant, governor, reward weights or frozen grids. See `docs/decisions/ADR-2026-08-29-pure-rl-split-frequency-actor.md`.

The completed split-frequency-actor run was evaluated at iterations 50/75/100/125/150/175/200 in separate fresh CPU-native processes. `model_175.pt` and `model_200.pt` are an adjacent pair that both pass C3a plus frozen C1 and C2c retention; the standard promotion helper returned `promoted=true`, so the later `model_200.pt` is the formally promoted C3a checkpoint and the required weights-only source for the first C3b experiment. On the fixed deterministic positive 12-degree trajectory, the promoted split actor also removed the earlier high-amplitude active-climb request reversals. This does not promote C3b: the next step is zero-shot C3b evaluation, and every later C3b candidate must retain C1, C2c and C3a before promotion. See `docs/audits/2026-08-29-pure-rl-c3a-split-actor-promotion.md`.

The C3b v2 zero-shot grid exposed an invalid loiter horizon: its deterministic final event was 277.5 m, requiring 13.875 m/s in the fixed 20 s episode. C3b v3 shortens only that loiter profile to 110 m after entry, placing the final event at 127.5 m while preserving every template, heading, phase, slope and turn-sign slice. On the corrected 176-case grid, promoted C3a `model_200.pt` achieved 79.55 percent zero-shot success; all loiter, S-turn and descent-containing cases passed, while climb-containing templates 5 and 7 each passed 16/32. The first controlled C3b route therefore keeps the split actor and existing governor-gap recipe, uses task-aware PPO weights 15/20/15/50 for C1/C2c/C3a/C3b, and starts weights-only with a fresh optimizer from promoted C3a. See `docs/decisions/ADR-2026-08-29-pure-rl-c3b-v3-joint-training.md`.

The first C3b v1 run stopped near iteration 59 because a diagnostic per-rollout strong-C2c phase check observed 4 samples against a minimum of 16. Task-level C2c normalization remained valid, and the phase count is not an input to the PPO loss. The corrected `c3b_split_frequency_actor_v2` route therefore keeps the C3b task minimum at 32 and continues logging phase coverage, but disables only the diagnostic phase abort for C3b; C3a retains its minimum of 16. V2 restarts cleanly from promoted C3a instead of resuming the partial v1 optimizer state.

The completed C3b v2 run was evaluated at iterations 50 through 250 on C3b v3 plus frozen C3a, C2c and C1. No checkpoint passed every suite. `model_100.pt` retained all old suites and was the best C3b candidate, with remaining failures concentrated in templates 5/7 under strong positive climb. C2c retention regressed from iteration 175 onward. The bounded C3b adaptive-ability experiment was launched weights-only from `model_100.pt` on 2026-08-30, keeps the fixed `15/20/15/50` task-aware PPO objective and 50 percent C3b share, and adapts only simple-task reset allocation plus template-5/7 and strong-climb coverage. Its iteration-0 telemetry and source metadata match the frozen route; this is startup evidence, not performance or promotion evidence. See `docs/decisions/ADR-2026-08-30-pure-rl-c3b-adaptive-ability-sampling.md`.

The completed paired global-yaw consistency experiment recovered three C3b cases at iteration 100 but did not reduce matched heading-0/180 action divergence, did not pass C3b, and slightly regressed C2c. The next controlled route therefore changes only the policy attitude representation: every history quaternion is expressed relative to the fixed episode route heading with a unique sign, while world-frame physics and body-frame preview remain unchanged. A 16-environment CPU-native smoke passed, and the 101-iteration seed-0 run was launched weights-only from the same fixed-mixture C3b `model_100.pt` with adaptive sampling and yaw-consistency loss zero. See `docs/decisions/ADR-2026-08-31-pure-rl-heading-canonical-observation.md` and `docs/handoffs/2026-08-26-pure-rl-c3a-curriculum-forgetting.md`.

The route-heading-canonical C3b run subsequently promoted `model_75.pt`, but zero-shot C3c and a
101-iteration C3c continuation exposed that the four-group C3c sampler was not a complete C3 task
union. Its best continuation checkpoint retained only 168/176 C3b cases without improving the
88/96 zero-shot C3c result; later checkpoints degraded C3b further and iteration 100 also failed
C1. The accepted replacement experiment starts from promoted C2c `model_550.pt` and jointly trains
C1/C2c/C3a/all-C3b/C3c at fixed `15/20/15/25/25` weights using the final split actor and
route-heading-canonical observation. Existing staged routes remain available. See
`docs/decisions/ADR-2026-09-01-pure-rl-full-c3-joint-training.md`.

CPU-native end-to-end PPO scaling was benchmarked at equal work after separating training from checkpoint evaluation. For 245,760 transitions and 1,280 optimizer steps, 64/128/256 environments achieved median 838.0/1462.0/2185.5 steps/s and fixed-work iteration sums of 292.81/167.87/112.20 seconds. The 256-environment, 16-minibatch configuration gives 2.61 times the 64-environment throughput and is validated as the measured PureRL training default. An 80-iteration resumed C2a run processed 983,040 transitions in 7 minutes 37 seconds; adjacent `model_1550.pt` and `model_1575.pt` checkpoints passed both C2a and C1 retention. Measured PureRL launcher defaults are now 256 environments, 16 minibatches, 500 iterations, and a 25-iteration save interval; explicit overrides and non-measured defaults remain available. Sample-equivalent promotion cadence is frozen at 614,400 transitions minimum and 307,200 transitions between evidence. See `docs/decisions/ADR-2026-08-11-pure-rl-accelerated-training-defaults.md` and `docs/audits/2026-08-11-pure-rl-cpu-native-training-acceleration.md`.

Measured PureRL C1/C2/C3 launcher runs now also default to train-only execution so they reproduce the benchmarked process isolation. Checkpoint evaluation runs sequentially in fresh processes; `--concurrent-eval` is an explicit non-authority opt-in, while `--train-only` remains the visible authority flag. Non-measured task defaults are unchanged. See `docs/decisions/ADR-2026-08-12-pure-rl-sequential-evaluation-default.md`.

Flapping-room ground creation no longer downloads the upstream `default_environment.usd`. A project-local 500 by 500 m static cuboid keeps the collision surface at `z=0` and removes the evaluation startup dependency on the remote asset server. Measured PureRL evaluation loads the current registered configuration and writes converted robot assets to an evaluation-owned writable cache. The repaired fresh-process path completed the C2b fixed and C1 retention suites, and the final worktree contains no protected-source diff. See `docs/decisions/ADR-2026-08-12-project-local-offline-ground.md`.

Validation on 2026-08-12 passed 219 focused PureRL C1/C2/C3 unit and contract tests. Fresh-process runtime gates include the existing 64-environment C1 gate (`1 passed in 13.63 s`), six-environment C2a gate (`1 passed in 7.10 s`), and six-environment C3 gate (`1 passed in 7.34 s`). The C3 runtime gate exercised deterministic coupled paths, C3a/C3b Tensor path generation, finite 555-value observations and reward telemetry, orthonormal route frames, isolated partial reset, eight policy steps, and the 25/35 degree reward/termination boundaries. No PPO training, wind injection, direct-GPU qualification, reward-weight tuning, convergence claim, or real-flight validation was performed for C3. The compact real-flight profile remains evidence-only with `candidate_not_promoted`; it does not change environment defaults.

## Required reading

- `docs/handoffs/2026-08-11-pure-rl-curriculum3.md`
- `docs/handoffs/2026-08-10-pure-rl-curriculum2.md`
- `docs/handoffs/2026-08-05-native-multibody-plant.md`
- `docs/handoffs/2026-07-13-current-simulation-audit.md`
- `docs/architecture/current_simulation_call_chain.md`
- `docs/architecture/coordinate_frames_and_units.md`
- `docs/architecture/wing_tail_controller_interfaces.md`
- `docs/decisions/ADR-2026-07-14-delaurier-prescribed-dynamic-twist.md`
- `docs/decisions/ADR-2026-07-14-delaurier-airflow-frame-convention.md`
- `docs/decisions/ADR-2026-07-29-engineering-flap-phase-sine.md`
- `docs/decisions/ADR-2026-07-29-measured-multibody-mass-properties.md`
- `docs/decisions/ADR-2026-07-29-measured-wing-multibody-plant.md`
- `docs/decisions/ADR-2026-07-29-ideal-coupled-wing-drive.md`
- `docs/decisions/ADR-2026-07-30-prescribed-coupled-wing-mechanism.md`
- `docs/decisions/ADR-2026-07-31-sinusoidal-phase-speed-drive.md`
- `docs/decisions/ADR-2026-07-31-sinusoidal-phase-physx-coupling.md`
- `docs/decisions/ADR-2026-08-01-ideal-inverse-dynamics-phase-drive.md`
- `docs/decisions/ADR-2026-08-03-native-holonomic-wing-mechanism.md`
- `docs/decisions/ADR-2026-08-04-promote-native-multibody-default.md`
- `docs/decisions/ADR-2026-08-05-native-holonomic-load-and-transient-diagnostics.md`
- `docs/decisions/ADR-2026-08-10-pure-rl-curriculum-domain-contract.md`
- `docs/decisions/ADR-2026-08-11-reject-pure-rl-gpu-implicit-training.md`
- `docs/decisions/ADR-2026-08-11-pure-rl-sample-equivalent-promotion-cadence.md`
- `docs/decisions/ADR-2026-08-11-pure-rl-accelerated-training-defaults.md`
- `docs/decisions/ADR-2026-08-12-pure-rl-sequential-evaluation-default.md`
- `docs/decisions/ADR-2026-08-12-project-local-offline-ground.md`
- `docs/decisions/ADR-2026-08-13-pure-rl-c3a-exact-c2c-rehearsal.md`
- `docs/decisions/ADR-2026-08-13-pure-rl-c3a-actor-distillation-experiment.md`
- `docs/decisions/ADR-2026-08-13-pure-rl-c3a-retention-aware-sampling.md`
- `docs/decisions/ADR-2026-08-14-pure-rl-c3a-bounded-warm-start.md`
- `docs/decisions/ADR-2026-08-27-pure-rl-c3a-joint-from-c1.md`
- `docs/decisions/ADR-2026-08-27-pure-rl-requested-frequency-smoothness.md`
- `docs/decisions/ADR-2026-08-27-pure-rl-requested-frequency-total-variation.md`
- `docs/decisions/ADR-2026-08-28-pure-rl-frequency-governor-gap.md`
- `docs/decisions/ADR-2026-08-29-pure-rl-split-frequency-actor.md`
- `docs/decisions/ADR-2026-08-29-pure-rl-c3b-v3-joint-training.md`
- `docs/decisions/ADR-2026-08-30-pure-rl-c3b-adaptive-ability-sampling.md`
- `docs/audits/2026-08-29-pure-rl-c3a-split-actor-promotion.md`
- `docs/audits/2026-08-18-pure-rl-c3a-bounded-warm-start.md`
- `docs/audits/2026-07-29-ideal-coupling-feasibility.md`
- `docs/audits/2026-07-29-physx-sfwm-inertial-validation.md`
- `docs/audits/2026-07-29-multibody-wing-aero-coupling.md`
- `docs/audits/2026-07-30-multibody-aerodynamic-validation.md`
- `docs/audits/2026-08-01-sinusoidal-phase-aerodynamic-gate.md`
- `docs/audits/2026-08-01-ideal-inverse-dynamics-phase-gate.md`
- `docs/audits/2026-08-03-native-holonomic-validation.md`
- `docs/audits/2026-08-04-multibody-training-runtime-feasibility.md`
- `docs/audits/2026-08-05-native-holonomic-transient-diagnostics.md`
- `docs/audits/2026-08-11-pure-rl-gpu-implicit-paired-plant-gate.md`
- `docs/audits/2026-08-11-pure-rl-cpu-native-training-acceleration.md`
- `docs/audits/2026-08-12-pure-rl-c2b-promotion.md`
- `docs/audits/2026-08-13-pure-rl-c3a-c2c-rehearsal.md`
- `docs/audits/2026-08-13-pure-rl-c3a-actor-distillation.md`
- `docs/audits/2026-08-14-pure-rl-c3a-retention-aware-sampling.md`
- `docs/plans/closed_loop_model_integration_plan.md`

## Known Issues / Deferred Work

- **Simple-QSM right-wing mirror convention:** `straight_flight_env.py` now maps the physical flap coordinate into URDF joint space as `left=+q` and `right=-q`, while the simple-QSM backend may also apply opposite left/right hinge-axis signs. Passing raw joint velocity into that backend can therefore mirror the right-wing flapping direction twice. This does not affect the current DeLaurier wing-only main path or its offline comparison. Before simple QSM is reused as an RL baseline or comparison model, the joint/hinge convention must be corrected and covered by an explicit left/right symmetry test. This issue is recorded as unresolved; it is not claimed to be fixed.
- **Legacy `FlappingBotEnv`:** `flapping_env.py::FlappingBotEnv` is an early legacy/prototype environment. Formal development uses `straight_flight_env.py`; the legacy environment has not adopted the current mirrored left/right joint-space convention and is not used for the DeLaurier model, real-data offline comparison, or subsequent formal control experiments. It remains available for historical reproduction and is considered deprecated. Before any reactivation, its joint mapping and QSM conventions must be audited and updated. This deferred work does not block the current DeLaurier wing-only offline comparison.

## Known blockers

- No corrected-force adapter or corrected strip-distribution contract.
- Corrected `Fx/Fz` ownership and moment-closure assumption require human approval.
- The corrected mirrored joint-space mapping and `theta_a` frame fix change the pre-existing plant convention；closed-loop mission baselines have not yet been rerun.
- The sine phase switch changes reset pose and all phase-indexed online results. The native headless phase and mechanism gates now pass, but the closed-loop mission baseline still needs to be rebuilt after promotion.
- The rounded measured body diagonal is positive definite but has a `-3.10e-4 kg m^2` inertia triangle margin; the frozen release records rather than hides this metrology limitation.
- Historical actual-motion, sinusoidal-phase and inverse-dynamics aerodynamic gates used `retain_accelerations=True` while resubmitting a new wrench every step. Their reported load growth, failure times, drive effort and power are superseded and cannot be used as plant evidence. Those experimental drive variants have not been requalified because the native holonomic mechanism is the accepted direction.
- The prescribed mechanism uses moving zero-width PhysX joint limits. The 2/3/4/5 Hz fixed-root matrix is bounded and synchronized, but the imposed position sequence is not represented by a consistent PhysX `joint_vel` state. Nonzero base reaction may include projection impulses; pitch-moment phase differs by up to `14.45 deg`, and whole-articulation momentum closure has not been demonstrated.
- The native holonomic mechanism is CPU-only. PhysX rejects its custom constraint in a direct-GPU scene, and `scene.replicate_physics=False` reduces large-batch throughput.
- The corrected fixed-root aerodynamic matrix passes at 1/480 s with 16/4 solver iterations. At 1/240 s, 4 and 5 Hz fail the 0.1 degree trajectory gate with 0.15992 and 0.24970 degree errors.
- Same-process full stage reload remains unresolved: a second environment construction after detach/close hung during scene creation. Fresh-process startup and shutdown pass.
- The free-body aerodynamic gate closes linear momentum to below 4e-6 relative error and angular momentum to below 2.9 percent at 2/5 Hz. This is numerical integration evidence, not aerodynamic-model or real-flight validation.
- The repository `isaaclab.sh -p` wrapper discards the child Python exit status. Automated validation must inspect `all_cases_accepted` in the JSON output or invoke the worker with the activated Conda Python directly.
- The direct-GPU implicit-drive training route is rejected after failing the frozen free-flight paired-plant gate despite passing its phase-matched fixed-root drive and isolated tail gates. It remains screening-only for reproducibility and must not provide curriculum checkpoints or be treated as the authoritative multibody plant.
- The real-flight profile provides candidate support and descriptive statistics only. Pitot truth-noise and end-to-end delay, generic waypoint-turn radius, climb/descent waypoint distributions, wind randomization bounds and sim-real acceptance thresholds remain unidentified or unapproved.
- Native mechanism torque and power are available only as explicitly labeled multibody inverse-dynamics estimates. The exact PhysX constraint multiplier and motor-shaft quantities are not exposed or claimed.

## Exact next task

Let the `c3_full_joint_heading_canonical_from_promoted_c2c_v1` seed-0 run finish without concurrent
evaluation. Then evaluate candidate checkpoints in fresh CPU-native processes on C3c v2, C3b v3,
C3a v2, C2c v2 and C1 v2. Promote only the later member of an adjacent all-suite passing pair;
training telemetry or any single-suite pass remains insufficient.

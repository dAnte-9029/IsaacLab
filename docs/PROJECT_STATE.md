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

## Required reading

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
- `docs/audits/2026-07-29-ideal-coupling-feasibility.md`
- `docs/audits/2026-07-29-physx-sfwm-inertial-validation.md`
- `docs/audits/2026-07-29-multibody-wing-aero-coupling.md`
- `docs/audits/2026-07-30-multibody-aerodynamic-validation.md`
- `docs/audits/2026-08-01-sinusoidal-phase-aerodynamic-gate.md`
- `docs/audits/2026-08-01-ideal-inverse-dynamics-phase-gate.md`
- `docs/audits/2026-08-03-native-holonomic-validation.md`
- `docs/audits/2026-08-04-multibody-training-runtime-feasibility.md`
- `docs/audits/2026-08-05-native-holonomic-transient-diagnostics.md`
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
- The direct-GPU implicit-drive probe is only a fixed-root, one-second capability and throughput result. It has not passed the native plant's full frequency, momentum, free-flight, reset or controller gates and must not be treated as the latest authoritative multibody plant.
- Native mechanism torque and power are available only as explicitly labeled multibody inverse-dynamics estimates. The exact PhysX constraint multiplier and motor-shaft quantities are not exposed or claimed.

## Exact next task

Run a fresh-process canonical-task smoke through the normal training launcher, then rebuild the straight-flight controller baseline against the promoted plant. Separately implement a reproducible 1024-environment CPU-native end-to-end RL throughput smoke with the free root, complete wing/tail plant, observations, rewards, terminations, repeated batched resets, policy inference and separated physics/transfer/optimizer timings. Do not retune the wing mechanism or substitute the GPU implicit-drive approximation during these gates.

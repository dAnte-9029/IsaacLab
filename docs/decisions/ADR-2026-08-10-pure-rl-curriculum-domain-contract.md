# ADR-2026-08-10: PureRL curriculum, backend, and real-data domain contract

- Status: Accepted
- Date: 2026-08-10
- Scope: PureRL curriculum stages C1--C3, CPU/GPU backend ownership, and real-flight evidence intake

## Context

The measured multibody plant is authoritative on CPU because the native holonomic wing constraint is not supported by direct-GPU PhysX. A direct-GPU ideal coupled-drive plant is faster, but its fixed-root loaded response already differs from the native plant and it has not passed the native free-flight, momentum, reset, or task gates. Curriculum and policy design must therefore remain separable from backend qualification.

The first PureRL environment now has a four-channel direct action contract, 60 Hz policy rate, actual tail-joint aerodynamic deflection, normalized observations, term-level reward telemetry, a 0--5 Hz flap-frequency channel, a 2 Hz/s frequency governor, physical Hz/s reward, randomized reset heading, rehearsal sampling, and retention evaluation. C2/C3 task geometry and later wind randomization have not yet been implemented.

Real flight logs can inform sensor cadence, observed wind/airspeed support, path geometry, and descriptive state/action distributions. They cannot by themselves identify a general Pitot truth-noise model, end-to-end Pitot delay, generic waypoint-turn radius, or simulation acceptance thresholds.

## Decision

### Shared policy-facing contract

All curricula and backends use `measured_pure_rl_shared_v1`:

- physics step: 1/480 s;
- policy decimation: 8, yielding 60 Hz actions;
- action interface: flap frequency, rudder, left elevon, right elevon;
- flap-frequency range: 0--5 Hz;
- flap-frequency governor: 2 Hz/s rise and fall;
- tail aerodynamic deflection source: actual joint position;
- observation dimension: 555.

The immutable contract and graph validator live in `pure_rl_curriculum_contract.py`. The measured C1 environment references the shared constants instead of duplicating these values. A curriculum or backend with a different policy-facing signature is a new experiment family, not another stage/backend of this one.

### Curriculum boundary

- C1 `c1_straight`: straight-level flight, no wind.
- C2 `c2_longitudinal`: straight-level, climb, and descent; C1 retention is required.
- C3 `c3_lateral_composite`: straight-level, climb, descent, turn, loiter, and composite paths; C1 and C2 retention are required.

Task addition does not authorize changing action, observation, physics-step, policy-rate, governor, or tail-deflection semantics. New-stage training must include rehearsal of required earlier stages and report a retention matrix separately from current-stage success.

Wind is disabled in C1--C3. Wind randomization is a later domain stage so that task-complexity failure and robustness failure remain distinguishable.

### Backend ownership

`cpu_native_authority` is the only authoritative backend. It uses the native holonomic drive on CPU with physics replication disabled.

`gpu_implicit_candidate` is screening-only. It uses the ideal coupled drive on CUDA with physics replication enabled and must be promoted against the CPU authority. GPU throughput alone cannot promote it, and a GPU checkpoint is not accepted as a CPU-native result.

Before promotion, the GPU candidate must be compared against CPU authority with the same shared contract and matched seeds/tasks. At minimum, promotion evidence must cover fixed-root loaded tracking, free-flight reset/finite-state behavior, short-horizon state increments, task returns/success, and C1/C2 retention. Numerical tolerances and the final speed/accuracy decision remain a separate approved experiment.

### Real-data evidence boundary

The reproducible generator is owned by `/home/zn/flap-system-identification`. It resolves the active canonical registry, verifies manifest and sample hashes, and loads only train and validation Parquets.

- B: train-derived candidate airspeed/wind ranges, validation coverage, and raw Pitot cadence. The 2026-04-17 logs are external-reference cadence only.
- C: PX4 mission segment geometry from train, validation, and 2026-04-17 external-reference ULogs. Ordinary turn radius remains unidentified; explicit loiter radius is retained.
- D: train/validation descriptive state, action, action-rate, and 0.1/0.25/0.5 s within-segment increment statistics. No pass/fail threshold is set.

The compact snapshot `real_flight_domain_profile_v1.json` has status `candidate_not_promoted`. It is evidence for later parameter decisions and is not imported by the current environment configuration.

## Materialized evidence

The generated artifact root is:

`/home/zn/flap-system-identification/artifacts/20260810_rl_domain_v1`

It records 308,702 train rows from 20 logs, 79,587 validation rows from 5 logs, and 15 ULogs from 2026-04-17 as external reference. The train P05--P95 true-airspeed candidate interval is 5.118--10.229 m/s; north/east wind component intervals are -1.957--1.675 m/s and -1.527--1.798 m/s. Median raw Pitot update interval is approximately 0.100 s.

The trajectory artifact contains 138 setpoint segments: 68 straight-level, 21 waypoint-turn, and 49 loiter. Explicit loiter radii span 45--80 m. These logs contain no distinct climb/descent waypoint-altitude changes, so C2 climb/descent task ranges are not claimed from artifact C.

## Consequences

- C2/C3 can be implemented without redesigning action/observation/backend semantics.
- Earlier-stage forgetting becomes a promotion failure visible in the retention matrix rather than an informal observation.
- GPU work can proceed as a separate screening lane without contaminating CPU-native authority.
- Real-data support is available now, but any change to simulator defaults requires a separate review and promotion decision.
- Pitot noise/delay, climb/descent task bounds, wind randomization bounds, and sim-real acceptance thresholds remain open decisions.

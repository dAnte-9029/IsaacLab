# Effective Force Model Freeze and Isaac Integration Design

## Decision

Freeze one final deployable version of the paper's longitudinal gain-bias model, integrate it into IsaacLab as a replacement for the final body-frame longitudinal force components, and leave the simulator-computed moment unchanged. Reinforcement learning is explicitly out of scope until the numerical and dynamic validation gates in this design have been reviewed.

The integration is a **longitudinal effective-force replacement**, not a complete effective-wrench model. The final applied wrench is

```text
F_final = [F_effective_x, F_sim_y, F_effective_z]
M_final = M_sim
```

after the required FRD/FLU conversion. The effective-force prediction replaces the simulator's final `Fx/Fz`; it is not added as another residual on top of wing, tail, and parasite-drag force.

## Scope

In scope:

- select and refit one final full-data `phase_freq_q_gain_bias` model;
- export coefficients, feature schema, prior parameters, conventions, lineage, envelope, and reference vectors;
- vendor the immutable model artifact into IsaacLab;
- implement batched Torch inference;
- add `disabled`, `shadow`, and `replace_fx_fz` modes;
- prove that replacement changes only `Fx/Fz` and leaves `Fy/Mx/My/Mz` unchanged;
- run numerical, force-surface, trim, short-horizon, and controller-in-the-loop validation;
- issue a bounded credibility verdict.

Out of scope:

- changing or learning the moment model;
- reinforcement-learning training;
- claiming complete six-axis aerodynamic fidelity;
- changing existing task IDs;
- silently extrapolating beyond the retained flight envelope.

## Frozen Model Contract

The frozen artifact must contain enough information for standalone inference without importing `/home/zn/flap-system-identification`:

- schema and model version;
- target order: `fx_b`, `fz_b` in body FRD and SI units;
- model family: `phase_freq_q_gain_bias`;
- ordered correction features and ordered per-target gain/bias design columns;
- fill, mean, scale, coefficients, and intercept for each target;
- final ridge parameter and the rule that selected it;
- exact shaped DeLaurier prior parameters and geometry identifier;
- phase convention, frequency definition, body-rate convention, and FRD/FLU mapping;
- training-envelope statistics;
- dataset, prior, source commit, dirty-state, and command lineage;
- deterministic reference inputs and expected outputs;
- SHA-256 digest for the numerical model payload.

The exporter must run from a traceable source state. Prefer clean commits for the system-identification code and nested-CV selection artifacts. If a required paper artifact remains uncommitted, record its content hashes and dirty-state manifest before selection; a bare Git `HEAD` is not sufficient provenance.

Nested whole-log evaluation remains the evidence for generalization. Once the model family and hyperparameter-selection rule are frozen, the deployment coefficients are refit on all 29 retained logs. Full-data fit metrics are diagnostics and must not replace the paper's held-out metrics.

## Runtime Data Flow

At every physics step:

1. Isaac computes wing force/moment, tail force/moment, and parasite drag exactly as it does now.
2. The DeLaurier wing force used as the correction prior is converted from Isaac FLU to the artifact's FRD convention.
3. Isaac phase is converted to the canonical model phase. The current cosine stroke convention requires an explicit offset to the sine-based logged convention.
4. Flapping frequency and FRD pitch rate are assembled with the phase harmonics into the frozen feature vector.
5. The model predicts complete-vehicle effective `Fx/Fz` in FRD.
6. The prediction is converted back to Isaac FLU.
7. `Fx/Fz` in the final force are overwritten; `Fy` and the complete simulator moment are copied unchanged.
8. Raw force, predicted force, final force, raw moment, envelope status, and fallback status are cached for validation.

The runtime must reject an artifact whose prior parameters or conventions do not match the active environment configuration. A corrected-force configuration may use the paper's shaped DeLaurier parameters, but enabling force replacement must not itself alter the already-computed moment tensor.

## Operating Modes

- `disabled`: do not load or evaluate the model; preserve existing behavior.
- `shadow`: evaluate and log the model, but apply the original simulator wrench.
- `replace_fx_fz`: apply the effective `Fx/Fz` replacement and preserve `Fy/Mx/My/Mz`.

The default remains `disabled`. Validation starts in `shadow` and promotes to `replace_fx_fz` only after reference-vector and frame/phase tests pass.

## Error and Envelope Handling

- Missing or malformed artifacts fail at environment initialization, not during rollout.
- NaN or infinite model inputs/predictions are counted and use the configured safe fallback.
- Envelope membership is reported per environment and per episode.
- Out-of-envelope behavior must be explicit. The first implementation supports warning plus fallback to the raw simulator `Fx/Fz`; smooth blending may be added only if discontinuity testing shows it is needed.
- No feature is silently reordered, filled with an undocumented default, or inferred from a different convention.

## Validation Ladder

### Gate 1: Artifact and numerical parity

- frozen inference reproduces the exporter reference vectors;
- NumPy and Torch inference agree within declared float tolerances;
- model checksum and schema validation pass;
- paper phase, frequency, and body-rate conventions are explicit.

### Gate 2: Wrench composition contract

- `disabled` and `shadow` apply the original wrench;
- `replace_fx_fz` changes only body `Fx/Fz`;
- `Fy/Mx/My/Mz` remain unchanged within numerical tolerance;
- FRD/FLU sign tests pass at hand-computable states.

### Gate 3: Force-surface and runtime stability

- phase/frequency/airspeed/pitch-rate sweeps are finite and continuous inside the retained envelope;
- batched GPU inference has no device transfers in the physics loop;
- long shadow and replacement smoke runs have no NaNs or force spikes;
- out-of-envelope and fallback counts are recorded.

### Gate 4: Local dynamic credibility

- trim and open-loop responses are compared between raw and corrected plants;
- short-horizon replay starts from logged states and uses matching controls where available;
- longitudinal velocity/acceleration errors are reported separately from attitude/rate errors;
- the correction must not be credited with moment improvements because moment is unchanged.

### Gate 5: Controller-in-the-loop credibility

- run the existing PX4-like controller on straight-flight and recovery cases;
- compare completion, termination, speed/height error, attitude/rate behavior, control effort, and envelope occupancy;
- report raw-versus-corrected results without starting RL.

## Credibility Verdict

Validation produces separate conclusions rather than a single "credible/not credible" label:

1. numerical integration correct;
2. longitudinal force behavior supported inside the retained envelope;
3. short-horizon longitudinal dynamics improved or not improved;
4. controller-in-the-loop behavior acceptable or not acceptable;
5. rotational fidelity remains inherited from the simulator moment model and is not validated by the force correction.

RL work may begin only after these conclusions and their artifacts are reviewed.

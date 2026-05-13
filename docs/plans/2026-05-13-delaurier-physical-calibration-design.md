# Physically Calibrated DeLaurier Design

Date: 2026-05-13

## Goal

Build a physically calibrated DeLaurier baseline for the effective-wrench paper comparison. This baseline must remain interpretable: it may adapt a small number of platform-level physical parameters, but it must not become a high-capacity learned residual model.

## Recommended Approach

Use an eight-parameter bounded calibration on a deterministic training subset, then evaluate the selected parameters on the full train, validation, and test splits. This keeps optimization cost manageable while preserving strict train/test separation.

The calibrated parameter vector is:

```text
wing_normal_force_scale
wing_chordwise_force_scale
delaurier_theta_w_deg
twist_eta_max_deg
delaurier_induced_drag_efficiency
fuselage_drag_cda
tail_lift_scale
phase_delay_s
```

The first six parameters modify wing force magnitude, incidence/twist, and missing drag. `tail_lift_scale` modifies tail aerodynamic effectiveness. `phase_delay_s` shifts the wing kinematic time base within each log before DeLaurier prediction.

## Data Flow

1. Load `train_samples.parquet`, `val_samples.parquet`, and `test_samples.parquet`.
2. Select a deterministic training subset for optimization. The subset should be large enough to cover multiple logs but small enough for repeated DeLaurier evaluations.
3. Compute training-set wrench standard deviations for objective normalization.
4. Optimize the bounded eight-parameter vector on the training subset only.
5. Run DeLaurier prediction with the selected parameters on full train, validation, and test splits.
6. Write calibrated parameters, optimization trace, per-split metrics, summary JSON, and a README.

## Objective

Minimize mean squared normalized wrench error:

```text
mean_t mean_j ((w_Del,j(t; theta) - w_label,j(t)) / sigma_j)^2 + lambda * R(theta)
```

`sigma_j` is the training-set standard deviation of each wrench channel. `R(theta)` is a weak normalized distance from nominal values. The regularization is only a stabilizer; the final report must include both the parameter values and whether any parameter hit its bound.

## Optimization

Use random search followed by optional coordinate refinement. This is more robust than a gradient method because the DeLaurier implementation contains clamps, phase interpolation, and possible non-smooth effects.

The first implementation should use deterministic random sampling with a fixed seed. It should expose search budget flags so the smoke test can run quickly and the full paper run can use a larger budget.

## Outputs

The formal run writes:

```text
docs/analysis/effective_wrench/delaurier_physical_calibration_v1/
  parameters.csv
  optimization_trace.csv
  metrics_by_split.csv
  summary.json
  README.md
```

## Reporting Interpretation

This baseline tests how far a compact, physically interpretable adaptation of the analytical model can go. If a parameter hits a bound, that is a diagnostic result: the model class may be missing a relevant effect or the selected parameter range may be too narrow. Test-set performance is the paper result; training performance is only diagnostic.

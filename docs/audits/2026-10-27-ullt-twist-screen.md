# ULLT prescribed-twist small flight screen

## Scope and reproducibility

User-authorized exploratory comparison on three logs and six fixed parameter
combinations; no controller changes, default promotion, training or test access.
Decision: [prescribed-twist ADR](../decisions/ADR-2026-10-27-ullt-prescribed-twist-screen.md).
Branch: `feat/ullt-state-space-wing-aero`, base `a7b3e08e0c1e`;
implementation is uncommitted and exact source/geometry hashes are in the run manifest.

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/ullt-mpl-cache \
python scripts/flapping_px4/screen_ullt_twist.py \
  --identification-root /home/zn/flap-system-identification \
  --wing-csv outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv \
  --wing-urdf source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf \
  --output-dir logs/ullt_twist/reproduction_new
```

The script rejects existing output directories. Generated data and figures stay
under ignored `logs/ullt_twist/`; no external repository or raw log is modified.

`analysis/ullt_twist_screen.py::select_screen_flights` selects low/median/high
median logged frequency among eligible train flights, without load residuals:

| Log | Median logged frequency (Hz) |
|---|---:|
| log_1_2026-4-14-12-01-42 | 3.678 |
| log_3_2026-4-14-12-19-22 | 4.102 |
| log_1_2026-4-12-16-33-22 | 4.252 |

Each has one stable-frequency and one transition-selected 12 s window, each
with 2 s wake warmup. Original complete-cycle membership is retained when
computing cycle mean, peak-to-peak and harmonic phase errors. The three logs
are exploratory training data, not independent validation. The existing loader
verifies train/validation registry hashes but only these three train logs are
replayed/scored; sealed test remains unopened.

## Implementation and checks

- `physics/ullt_twist.py::twist_wing`: continuous root-local Quadratic-2 deformation
  around the leading-edge tube, projected section geometry and analytic motion
  derivatives. No independent uniform front-tube pitch or structural ODE.
- `analysis/ullt_flight_replay.py::flight_kinematics`: combines deformation with
  flapping and body motion; `FlightReplayConfig.twist` defaults to amplitude zero.
- `analysis/ullt_twist_screen.py`: fixed candidates, paired mask, per-log/cycle/
  transition metrics, phase and time plots. Frozen DeLaurier is unchanged.
- `tests/test_ullt_prescribed_twist.py`: exact zero-amplitude compatibility,
  independent scipy-rotation geometry, shared edges, time finite differences,
  mirrored loads, moving-body transport and batching, invalid-parameter rejection.

Twist kinematics and load reference remain SI/FRD at output; the moment is about
the same registered neutral whole-aircraft CG. The moving geometry updates both
force lever arms and the added-mass free couple. Metrics compare wing-only
predictions with whole-aircraft reconstructed loads, not measured wing force.

The zero-amplitude branch is bitwise equal in deterministic adapter tests.
Against the earlier 25-log replay, the new three-log zero-twist run matches all
6006 scored keys; maximum Fx/Fz/My differences are respectively
`2.17e-7 N / 1.12e-6 N / 7.34e-8 N m`. Batch composition changes the global
pole-rate integration subdivision in `ULLTBackend.advance`; these differences
are negligible integration effects, not a changed aerodynamic baseline.

Targeted validation: **98 passed, 2 skipped** (CUDA-dependent checks); existing
ULLT, flight replay, DeLaurier strip/dynamic/legacy-twist and multibody coupling
tests are included. Five affected Python files pass formatting and flake8;
`git diff --check` passes. The configured Conda environment is used throughout;
format/lint modules come from the existing `/tmp/ullt-stage1-format-tools`
directory, without installing or changing dependencies. No Isaac Sim or RL run.

`logs/ullt_twist/20261027_span_sensitivity_v1/` checks the historical
`(35 deg,-0.5,-25 deg)` candidate on `log_1_2026-4-12-16-33-22`,
36.14--40.14 s, with only the last 2 s scored. Increasing each wing from 10 to
20 panels changes Fx/Fz/My by RMS `0.432 N / 0.531 N / 0.0993 N m`,
or `4.54% / 2.73% / 6.00%` of the 10-panel prediction RMS. This is one
candidate/window, not a convergence guarantee for all candidates. Nearby score
differences smaller than this scale should not establish a preferred physical
parameter value.

The screen directory contains a verified `source_snapshot/` matching its
original manifest. Subsequent formatting is recorded in
`format_verification.json`; formatting preserves the executed Python ASTs.
The final report-only v2 separates the two time windows into individual plot
axes; its own manifest records the new report source. All numerical CSV tables
are byte-identical to v1. No aerodynamic replay was repeated for this layout change.

## Completed results

- Replay: `logs/ullt_twist/20261027_screen_v1/`.
- Final figures/report: `logs/ullt_twist/20261027_screen_report_v2/report.html`.
- All seven ULLT variants retain the same **6006** scored rows, **246** complete
  cycles and **3** transition events, with no reverse-flow or integration failures.
- Independent RMSE calculation agrees within `3.56e-15`; input keys, labels,
  frozen DeLaurier values, score masks and replay artifact hashes were verified.

Flight-equal mean RMSE (forces N, pitch moment N m):

| Model / (tip amplitude, kappa, phase offset) | Fx | Fz | My |
|---|---:|---:|---:|
| Frozen DeLaurier, no twist | 3.623 | 8.265 | 1.474 |
| ULLT zero twist | 8.668 | 14.041 | 2.239 |
| ULLT (20 deg, 0, -25 deg) | 7.630 | 10.000 | 2.029 |
| ULLT (35 deg, -0.5, -25 deg) | 6.330 | 9.820 | 1.876 |
| ULLT (40 deg, 0, -25 deg) | 6.246 | 9.460 | 1.873 |
| **ULLT (35 deg, -0.5, 0 deg)** | **4.873** | **6.953** | **1.445** |
| ULLT (35 deg, -0.5, -45 deg) | 8.063 | 13.941 | 2.317 |
| ULLT (35 deg, 0, -25 deg) | 6.619 | 9.177 | 1.909 |

The best candidate among these six lowers Fx/Fz/My RMSE by approximately
44%/50%/35% relative to zero twist, with improvement in all three axes on each
of the three flights. Against DeLaurier it improves Fz on all three flights,
Fx on none, and My on only one. The small macro My gain (about 2%) is not robust
evidence of superiority. At -45 deg phase offset My is worse than zero twist:
adding twist is not automatically beneficial. This screen does not identify
the globally best amplitude, curvature or phase.

Complete-cycle diagnostics, flight-equal aggregates:

| Model | Axis | Mean RMSE | Peak-to-peak MAE | H1 phase MAE (deg) |
|---|---|---:|---:|---:|
| DeLaurier | Fx | 1.150 | 2.652 | 23.09 |
| Zero twist | Fx | 5.440 | 10.785 | 23.92 |
| Best screened twist | Fx | 2.763 | 3.080 | 18.61 |
| DeLaurier | Fz | 4.252 | 14.095 | 12.67 |
| Zero twist | Fz | 8.436 | 28.650 | 8.51 |
| Best screened twist | Fz | 4.805 | 8.491 | 27.68 |
| DeLaurier | My | 0.773 | 1.244 | 59.68 |
| Zero twist | My | 1.106 | 3.415 | 65.30 |
| Best screened twist | My | 0.822 | 1.209 | 90.03 |

The best twist reduces cycle mean/amplitude errors relative to zero twist, but
Fz and My first-harmonic phase errors increase. Its Fz/My cycle-mean errors also
remain above DeLaurier. Lower waveform RMSE does not establish uniformly better
phase, moment prediction or underlying wing physics.

Across the three input-selected transition events, Fx/Fz/My RMSE is
`9.277/15.501/2.312` for zero twist,
`5.204/8.135/1.494` for the best screened twist, and
`3.511/8.952/1.528` for DeLaurier. These short segments support the same limited
trend, not full-flight or held-out robustness.

The pooled per-time maximum local incidence P50/P95 decreases from
`46.38/65.70 deg` (zero) to `27.19/39.28 deg` (best candidate). These are maxima
over the wing at each scored time, not the distribution of all individual
sections. Incidence remains high for attached-flow assumptions. Neither the
real structural twist nor a full-aircraft model has been validated.

## Interpretation limits

The twist parameters are prescribed candidates, not measured/identified fabric
deformation. A reduction in whole-aircraft residual cannot uniquely distinguish
wing flexibility, wind/airflow errors, other surfaces, and label/inertia errors.
The projected-section closure and attached-flow assumptions remain approximations;
angle statistics and numerical sensitivity must accompany the error table.

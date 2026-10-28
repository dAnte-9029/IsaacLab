# ULLT phase/span incidence and load attribution

## Scope and evidence

This is diagnostic attribution of **modeled wing loads versus reconstructed
whole-aircraft non-gravity loads**, not measured section forces or causal error
percentages. No physical coefficient, wake equation, default backend or training
configuration changed. Base commit: `b36b672e`; decision:
[section-attribution ADR](../decisions/ADR-2026-10-27-ullt-section-load-attribution.md).

Reuse the three train logs and six input-selected windows from the prior twist
screen, without additional fitting: zero twist and `(A_tip=35 deg,kappa=-0.5,
psi=0 deg)`. Each case has **6006 scored times, 20 sections, 120120 section rows**.
All six resultant force/moment components reproduce the archived replay exactly
(maximum difference **0.0**). Component-to-section and section-to-resultant closure
is below `2.14e-14` in independent checks. Original inputs/labels and raw logs remain unchanged.

Artifacts:

- `logs/ullt_sections/20261027_attribution_v1/report.html`: incidence and six-axis
  phase/span maps, threshold/area/load tables, original component values.
- `*_sections.parquet`: every scored section, signed incidence, side/span/phase,
  reference and projected area, pressure, circulation, six-component wrenches.
- `*_per_flight.csv` and `*_macro.csv`: retain flight variability; macro values
  are means of per-flight metrics, including quantiles (not pooled quantiles).
- `span_regions_*`, `phase_peak_summary.csv`, `added_mass_moment_parts_*`,
  `residual_reference_*`: supplementary localization and reference-load evidence.
- `independent_verification.json`, `verification_source.py`: independent checks
  and exact supplementary analysis, source/artifact hashes included.
- `logs/ullt_sections/20261027_resolution_v1/`: bounded 10/20-panel check.

## Definitions that matter

Incidence is the signed geometric angle at the moving 3/4-chord point relative
to prescribed incident flow, **before induced flow**. Thresholds use its absolute
value and are descriptive, not universal stall thresholds. Center gap contributes
no area. Reference area is CSV planform area represented by the neutral discretized
sections (`0.3572 m^2` for both wings at 10 panels per wing); deformed projected
model area is saved separately. An area-time fraction is the time average of
the fraction of this reference area above the threshold.

Absolute load share is `sum_time,sum_sections |section component| above threshold /
sum_time,sum_sections |section component| over all sections`, then averaged over
flights. It is neither a signed resultant percentage nor a fraction of measured
force error. This avoids spuriously huge ratios when opposite wings or phases
cancel. Signed means and resultant RMS are also retained.

Only **circulatory + added_mass + profile_drag** form an additive decomposition.
Added mass includes force lever arms about the registered neutral CG AND the
3/4-chord free couple. The circulatory flow-parallel projection is a nested
diagnostic, not another term to add or a uniquely identified induced-drag model.
High-angle regions and added mass overlap and must not be added as independent
"causes".

## 1. High incidence: extreme tips versus the broader loaded region

| Case / threshold | Area-time fraction | Absolute Fx share | Absolute Fz share | Absolute My share |
|---|---:|---:|---:|---:|
| Zero twist, >25 deg | 43.34% | 88.77% | 75.43% | 79.46% |
| Zero twist, >35 deg | 25.98% | 73.76% | 57.64% | 62.06% |
| Zero twist, >45 deg | 12.71% | 50.74% | 36.19% | 40.16% |
| Zero twist, >60 deg | 1.29% | 9.94% | 5.54% | 7.21% |
| Prescribed twist, >25 deg | 22.49% | 51.60% | 38.96% | 53.25% |
| Prescribed twist, >35 deg | 2.92% | 9.00% | 6.22% | 13.24% |
| Prescribed twist, >45 deg | 0.04% | 0.17% | 0.11% | 0.32% |
| Prescribed twist, >60 deg | 0 | 0 | 0 | 0 |

The remaining three axes are also covered: at >35 deg, zero-twist absolute
Fy/Mx/Mz shares are **44.95% / 67.30% / 83.90%**; with prescribed twist they
are **7.84% / 8.67% / 9.95%**. These refer to unsigned section burdens, even if
the corresponding net whole-aircraft axis is small.

The maximum-angle statistic alone was insufficient. The >60 deg region is small
and cannot account for most absolute Fz burden. The >35 deg region, however, is
not restricted to tiny tips: its Fz burden is approximately **20.9% in s<0.5,
49.7% in 0.5<=s<0.8, and 29.4% in s>=0.8** (root-local s on each wing).
About **71%** comes from outside the final 20% of span. In contrast, about 84%
of the >60 deg Fz burden lies in that outermost span region.

At the 160--170 deg phase bin (center 165 deg), the zero-twist >35 deg area
fraction averages **72.1%**, and that region contributes **-42.30 N Fz**.
Its Fx contribution peaks at the 170--180 deg bin at **23.22 N**. Thus the
important high-incidence loading occurs broadly around the downstroke, not just
at isolated small-area extrema. With twist, the >35 deg area peaks later,
at 240--250 deg, and reaches only 13.5% in the equal-flight phase average.

Area-weighted absolute-incidence P50/P95 are **21.91/52.73 deg** without twist
and **17.50/32.26 deg** with twist. These equal-flight quantiles must not be
compared as if identical to the earlier pooled per-time maximum-angle quantiles.

## 2. Does the region align with the load error?

For a region/component load C, evaluate `RMSE(prediction-log-C)` with the original
wake and kinematics fixed. This is an algebraic removal diagnostic, **not** a
valid prediction after deleting sections, applying stall, or changing circulation.
Error differences contain cross terms and are not additive attribution shares.

| Case | Diagnostic | Fx RMSE (N) | Fz RMSE (N) | My RMSE (N m) |
|---|---|---:|---:|---:|
| Zero twist | Original | 8.668 | 14.041 | 2.239 |
| Zero twist | Subtract all modeled loads at >35 deg | 3.997 | 8.600 | 0.878 |
| Zero twist | Subtract all modeled loads at >60 deg | 7.127 | 12.585 | 2.049 |
| Prescribed twist | Original | 4.873 | 6.953 | 1.445 |
| Prescribed twist | Subtract all modeled loads at >35 deg | 4.927 | 6.766 | 1.286 |

Zero-twist >35 deg Fz shares range from **47.4% to 64.7%** across the three logs,
and subtracting that region reduces Fz residual RMSE in all three. This supports
diagnostic concentration in a broad high-incidence region, but does not prove
which nonlinear aerodynamic correction is right.

With prescribed twist, >35 deg Fz shares range from **0.18% to 9.89%**. The
subtraction reduces Fz RMSE on two flights and increases it on one. The remaining
error is therefore not consistently concentrated in this extreme-incidence group.
However, **25--35 deg sections still contribute about 32.7% of absolute Fz**;
the evidence does not establish attached flow merely because >60 deg disappears.

## 3. Circulatory versus added-mass load

| Case | Component | Mean Fz (N) | Fz RMS (N) | Mean Fx (N) | Mean My (N m) |
|---|---|---:|---:|---:|---:|
| Zero twist | Circulatory | -16.125 | 25.612 | 8.498 | 1.067 |
| Zero twist | Added mass | -0.084 | 6.225 | -0.067 | 0.008 |
| Prescribed twist | Circulatory | -12.420 | 14.935 | 5.703 | 0.915 |
| Prescribed twist | Added mass | -0.056 | 2.613 | -0.011 | -0.118 |
| Both | Profile drag | 0 | 0 | 0 | 0 |

The reconstructed whole-aircraft mean Fz is **-8.606 N**. The zero-twist
predicted mean is **-16.209 N**, a **-7.603 N** residual bias. Only -0.084 N of
the predicted mean is the added-mass term; the model's excess mean load is
overwhelmingly in its circulatory component. Added mass nonetheless has appreciable
within-cycle RMS and must not be dismissed based only on its near-zero mean.

Subtracting the **entire** added-mass term changes zero-twist Fx/Fz/My RMSE from
`8.668/14.041/2.239` to `8.796/13.993/2.182`: it does not resolve the overload.
With prescribed twist the same diagnostic changes `4.873/6.953/1.445` to
`4.420/8.382/1.608`: Fz and My get worse, while Fx improves. This argues against
uniformly scaling down/removing added mass as the next correction.

Moment terms cannot be reduced to force lever arms alone. For zero-twist My,
added-mass lever-arm RMS is 0.476 N m and free-couple RMS is 0.411 N m, yet the
combined added-mass My RMS is only 0.077 N m because of cancellation. With
twist, corresponding RMS values are 0.365/0.377/0.366 N m. The saved six-axis
decomposition retains this cancellation and uses the same moment reference.

## 4. Missing drag and real deformation remain unidentifiable fractions

The present profile Cd is exactly zero. Existing circulatory force has a
flow-parallel projection through its orientation and induced flow, so this is
not evidence of "no drag of any kind".

The exact current profile-formula derivative gives the following illustrative
mean changes **per +0.01 Cd**, with the recorded flow held fixed:

| Case | Delta Fx (N) | Delta Fz (N) | Delta My (N m) |
|---|---:|---:|---:|
| Zero twist | -0.1223 | -0.0418 | +0.000077 |
| Prescribed twist | -0.1213 | -0.0413 | -0.000140 |

The 0.01 factor is just a derivative display scale, not an assumed or fitted
coefficient. Within this existing profile-drag direction, positive Cd reduces
mean Fx but makes mean Fz slightly more negative in these logs; it cannot
simultaneously remove the excessive mean Fx and upward Fz. Actual separated-flow
drag, altered force directions and deformation-dependent polars are not represented
by this single derivative. No fraction of physical error is assigned to missing drag.

Changing from zero to prescribed twist reduces modeled mean upward Fz magnitude
by **3.734 N**, mean Fx by **2.739 N**, and mean My by **0.277 N m**. Almost all
of the mean Fz change is circulatory (3.705 N); the added-mass mean changes by
only 0.029 N. This quantifies sensitivity to the assumed kinematics, not how much
the actual wing deforms. Twist was selected on these logs previously, so this is
not independent evidence for its true structural parameters.

Tail/body load, wind/airspeed, inertia/derivative reconstruction, unmeasured pitch,
twist/camber and nonlinear circulation remain confounded in the residual.

## Numerical limits and next decision

On `log_1_2026-4-12-16-33-22`, 36.14--40.14 s (last 2 s scored), refining from
10 to 20 panels per wing changes the zero-twist >35 deg area fraction from
**21.95% to 21.75%**, and its Fz absolute-load share from **53.11% to 52.92%**.
Fx/Fz/My waveform relative RMS differences are **5.24/2.19/5.13%** without twist
and **4.72/3.29/6.17%** with twist. The short twisted window has no >35 deg
sections at either resolution, so it does not establish convergence of the rare
high-incidence tails on the other flights. These are bounded diagnostics, not
full-envelope convergence or permission to ignore numerical error.

The supported next focus is **real deformation/incident-flow kinematics and the
circulatory response of the area carrying load**, including the 25--45 deg range.
The data do not currently prioritize changing added-mass coupling to cure the
mean overload, nor identify an LEV model uniquely. Confirm the kinematics before
using a nonlinear lift law to compensate for an incorrect rigid-wing assumption.
This is a diagnostic priority, not a conclusion that real LEVs, separation or
added-mass coupling are absent or already validated.

Validation: **103 passed, 2 CUDA checks skipped**; formatting/flake8 and
`git diff --check` pass. No simulation training, free-body coupling, parameter
fitting, default change, automatic commit or publication was performed.

## Reproduction

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/ullt-mpl-cache \
python scripts/flapping_px4/diagnose_ullt_sections.py \
  --source-screen logs/ullt_twist/20261027_screen_v1 \
  --wing-csv outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv \
  --wing-urdf source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf \
  --output-dir logs/ullt_sections/reproduction_new
PYTHONDONTWRITEBYTECODE=1 python scripts/flapping_px4/check_ullt_section_resolution.py \
  --source-screen logs/ullt_twist/20261027_screen_v1 \
  --wing-csv outputs_DeLaurier/right_wing_te_fit_poly5_gap50.csv \
  --wing-urdf source/isaaclab_assets/data/flapping_bot/robots/flap_robot_552/urdf/flap_robot_552.urdf \
  --output-dir logs/ullt_sections/resolution_reproduction_new
```

The supplemental verification script contains its explicit original artifact
paths; adapt those constants to reproduce supplemental tables in a new directory.
Input replay hashes, source versions, command, Git state and environment are
recorded in the main manifest; all output directories refuse silent replacement.

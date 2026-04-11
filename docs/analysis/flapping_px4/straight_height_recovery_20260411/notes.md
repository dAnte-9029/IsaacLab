# Straight Height Recovery Retune Notes

Date: 2026-04-11

Scope:

- Non-RL DeLaurier path-mission controller path.
- Plant kept fixed: COM override, tail effectiveness, elevon limit, DeLaurier backend, and wrench aggregation were not changed in this pass.
- Objective was to improve startup height recovery in `level_straight` without raising `max_flap_hz` above `5.0 Hz`.

## Selected Controller Defaults

The selected candidate was `small_band_gain1p8`:

- `tecs_altitude_hold_error_band_m = 0.05`
- `tecs_altitude_capture_time_const_s = 0.65`
- `tecs_altitude_error_gain = 1.8`
- `tecs_pitch_speed_weight = 0.5`
- `tecs_pitch_speed_weight_capture = 0.25`
- `tecs_capture_extra_climb_rate_mps = 1.0`
- `tecs_pitch_damping_gain = 0.16`

This candidate was chosen because it improved the long-straight height recovery distance materially while keeping overshoot small and avoiding elevon pitch saturation.

## Long Straight Evidence

Fresh long-straight, no path warmup:

| Metric | Baseline | Retuned |
|---|---:|---:|
| min height error (m) | -0.5365 | -0.3253 |
| time to sustained 0.05 m band after min (s) | 7.4417 | 4.7583 |
| distance to sustained 0.05 m band after min (m) | 58.6223 | 36.2750 |
| time to zero crossing after min (s) | 8.8833 | 6.6167 |
| distance to zero crossing after min (m) | 70.2425 | 51.0002 |
| mean abs height error (m) | 0.1267 | 0.0514 |
| p95 abs height error post-warmup (m) | 0.4580 | 0.1871 |
| max positive height error (m) | 0.0701 | 0.0697 |
| max abs pitch (deg) | 21.3212 | 23.5022 |
| max abs roll (deg) | 2.6125 | 2.5940 |
| freq saturation fraction | 0.0562 | 0.0821 |
| elevon pitch saturation fraction | 0.0000 | 0.0000 |

Interpretation:

- The minimum startup drop decreased by about 39%.
- Recovery into the 0.05 m height-error band shortened by about 38% in distance and about 36% in time.
- Zero crossing shortened by about 27% in distance and about 26% in time.
- Elevon pitch did not saturate, so this pass mainly improved TECS outer-loop demand rather than relying on actuator clipping.
- Frequency near-saturation increased modestly but stayed bounded by the existing 5 Hz limit.

## Canonical Regression

Canonical default runs after retune:

| Phase | mean abs height error post-warmup (m) | p95 abs height error post-warmup (m) | mean abs lateral error post-warmup (m) | max abs pitch (deg) | max abs roll (deg) | freq saturation fraction | elevon pitch saturation fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| level_straight | 0.0823 | 0.2243 | 0.4095 | 23.5022 | 2.5940 | 0.1629 | 0.0000 |
| level_turn | 0.1558 | 0.2323 | 0.2805 | 24.6918 | 19.3696 | 0.4798 | 0.0000 |
| level_loiter | 0.0755 | 0.1992 | 0.6041 | 24.6918 | 19.3696 | 0.7803 | 0.0000 |

All three canonical runs remained attitude-stable in the recorded trajectory:

- Pitch stayed below the configured pitch limit region.
- Roll stayed bounded during turn/loiter.
- Elevon pitch did not hit normalized action saturation.

The path-mission summaries mark these runs as `truncated` because the fixed `--steps` budgets ended near, but before, full path completion. Final progress ratios were about `0.986` for straight, `0.988` for turn, and `0.983` for loiter.

## Residual Risk

- Turn and loiter still spend substantial time near the 5 Hz frequency limit, especially loiter. That suggests banked-flight energy margin remains limited even though the straight startup recovery is better.
- This pass retuned TECS capture defaults globally for the straight-line teacher used by path tracking; a later pass should check whether loiter-specific entry geometry and bank-aware TECS behavior need a separate tuning layer.
- The response still has little positive overshoot. If faster height correction is required, the next diagnostic should compare requested pitch/elevon authority versus achieved vertical acceleration during the first 5 seconds, rather than increasing frequency above the sim2real 5 Hz cap.

## Artifacts

- `long_straight_height_recovery_compare.png`: baseline versus retuned long-straight height error.
- `final_long_straight_no_warmup.png`: final no-warmup long-straight trajectory plot.
- `final_canonical_straight.png`: final canonical straight plot.
- `final_canonical_turn.png`: final canonical turn plot.
- `final_canonical_loiter.png`: final canonical loiter plot.
- `metrics_summary.csv`: tabular metrics summary.
- `metrics_summary.json`: machine-readable metrics summary.

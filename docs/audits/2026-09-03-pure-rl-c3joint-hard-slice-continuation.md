# PureRL full-C3 hard-slice continuation audit

- Date: 2026-09-03
- Branch: `feat/native-multibody-rl`
- Worktree base commit: `4d757faf`
- Scope: bounded seed-0 continuation and fresh CPU-native frozen evaluation

## Observed facts

The diagnostic resumed the full-C3 joint seed-0 `model_175.pt` together with its optimizer state
and ran 25 additional PPO updates. The run produced `model_200.pt` and exited normally:

```text
logs/rsl_rl/flapping_bot_straight_flight/
2026-09-02_12-50-38_pure_rl_c3_full_joint_hardslice_resume175_to200_seed0_26iter/
```

The combined fresh-process result is recorded at:

```text
eval/c3_full_joint_hardslice_resume175_to200_seed0_frozen_v1/
model_200/checkpoint_summary.json
```

| Frozen suite | Result | Gate |
| --- | ---: | --- |
| C3c v2 | 96/96 | PASS |
| C3b v3 | 168/176 | FAIL |
| C3a v2 | 96/96 | PASS |
| C2c v2 | 112/112 | PASS |
| C1 v2 | 16/16 | PASS |

The original full-joint `model_200.pt` scored 88/96 on C3c and 168/176 on C3b. The bounded
continuation therefore recovered all eight C3c cases while retaining C1, C2c and C3a, but did not
change the C3b total. All eight remaining C3b failures are template 5, positive 12-degree climb,
turn sign `+1`, across four headings and two initial flap phases. Every failure is a roll-limit
termination.

The only checkpoint from the original full-joint run that passed all five suites was
`model_175.pt`. No adjacent checkpoint pair passed all five suites, so neither the original run nor
this one-checkpoint continuation satisfies the promotion rule.

## Sampling implementation discrepancy

`pure_rl_spatial_path.py::_sample_event_contracts` originally balanced hard-slice templates and
turn signs by computing a rank within each sampler call. Isaac Lab resets paths asynchronously, so
many calls contain one resetting path and restart that rank at zero. TensorBoard at iteration 200
reported C3b hard left/right fractions `0.0000/0.1136` and C3c high-climb left/right fractions
`0.0000/0.0999`. The completed run therefore sampled the targeted rows only with turn sign `+1`;
the same restart also selected only the first C3b hard template.

The committed correction samples C3b template 5/7 and turn sign independently with equal
probability. C3c hard-slice turn sign is likewise sampled independently. The regression test
`test_c3_joint_hard_slices_randomize_across_single_path_resets` repeatedly samples one path and
requires both templates and both signs to occur. This correction applies to future runs only; it
does not retroactively change the evidence above.

## Interpretation

The result supports a narrow conclusion: continued full-joint training with additional exposure to
the observed right-turn hard neighborhood can recover C3c while preserving the three simpler
abilities. It does not establish balanced hard-slice effectiveness, multi-seed robustness or a
promotable recipe. The unchanged 168/176 C3b score also shows that exposure alone did not remove
the remaining template-5 positive-climb roll-limit failure.

`model_175.pt` is consequently a reliable research initialization because it passed all five
frozen suites once. It must not be described as a formally promoted checkpoint.

## Next experiment boundary

The next task is to define a genuinely continuous three-dimensional path family rather than add
another event-sequence curriculum. Candidate paths should vary horizontal curvature and flight-path
slope smoothly along arc length, including climbing turns, descending turns, S-curves and helical
segments. Before training, freeze a separate continuous-curve evaluation grid and verify that the
policy observation exposes enough local tangent or lookahead information to anticipate changing
curvature.

Initialize the experiment from the original full-joint `model_175.pt`, retain the five existing
frozen suites as regression gates, and keep old-task rehearsal in the training distribution. Path
parameter ranges, observation changes, training proportions and the new frozen gate remain design
decisions; no continuous-curve implementation or training is claimed by this audit.

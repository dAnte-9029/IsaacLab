# Error Plot Notes

Current default is `inner_pitch_ki=0.9` on top of the frozen plant `COM=-0.10`, `tail=0.5/1.2`.

| Phase | height err 0.8 | height err 0.9 | lateral err 0.8 | lateral err 0.9 | progress 0.8 | progress 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| level_straight | 0.4277 | 0.4270 | 0.4574 | 0.4538 | 0.979230 | 0.979966 |
| level_turn | 0.5754 | 0.5756 | 0.3596 | 0.3572 | 0.979692 | 0.979030 |
| level_loiter | 0.5074 | 0.5138 | 0.3584 | 0.3357 | 0.979654 | 0.979908 |

Visual read:

- `straight`: error curves stay smooth; lateral bias is slightly smaller, height is essentially unchanged.
- `turn`: curves look almost the same; this retune does not materially change the turn residual.
- `loiter`: lateral error band is visibly tighter; height bias is slightly worse but still bounded and stable.
- None of the three tasks shows oscillatory growth or new instability after the controller retune.
# Loiter-First Primitive RL Repair Design

## Goal

Repair the current primitive-stage path-tracking RL pipeline so that `loiter` becomes a learnable skill without destroying the already improved `turn` behavior. The immediate target is still the no-wind primitive suite; wind and mixed missions stay out of scope until `straight`, `turn`, and `loiter` all become stable together.

## Current Evidence

The latest short resume run shows a clear asymmetry:

- `turn` improved to `completion_rate = 1.0` and `termination_rate = 0.0`
- `loiter` still fails with low final progress and full termination
- `straight` regressed while curve-task shaping was strengthened

This means the recent changes successfully taught “enter and hold a finite curve for a short segment”, but not “enter a circular orbit and keep it for long enough to complete a loop”.

## Root-Cause Hypothesis

The current training stack treats `loiter` too much like a longer `turn`, but the control problem is different.

`turn` only needs:

1. acquire the correct curve
2. stay on it briefly
3. finish before accumulated drift matters too much

`loiter` needs:

1. capture the target orbit
2. reject radial drift continuously
3. preserve tangential direction continuously
4. avoid accumulating phase / radius error for a full loop

The current reward and teacher shaping are strong enough to improve capture, but too weakly targeted for steady orbit maintenance. The current primitive curriculum also makes `loiter` appear as a full-loop task too early.

## Chosen Strategy

Use a **loiter-first second-stage repair**, not a global rewrite.

The repair keeps the same environment family, PPO stack, and teacher-guided structure. It changes only the pieces that matter for sustained circular motion:

1. add explicit `loiter` diagnostics
2. decompose `loiter` into easier sub-stages
3. add `loiter`-specific reward terms and milestones
4. make teacher guidance tighter only on `loiter`
5. keep a small `straight` rehearsal fraction to limit forgetting

## Alternatives Considered

### A. Just train longer

Rejected as the primary next step. Current evidence does not indicate a pure optimization-time bottleneck. `turn` already moves, `loiter` does not. That points to task formulation, not merely insufficient gradient steps.

### B. Move directly to mixed random missions

Rejected for now. It adds variance before the primitive circular-tracking skill is reliable. This would make debugging harder and slow down iteration.

### C. Move directly to wind robustness

Rejected for now. Wind would confound the current diagnosis. The missing capability is sustained orbit holding even in no wind.

## Design

### 1. Add explicit loiter diagnostics

The training/eval stack needs to distinguish:

- failed to capture the orbit
- captured orbit briefly but drifted out
- stayed on orbit but failed to make enough angular progress

The simplest way is to surface per-step or aggregated orbit-specific signals:

- radial error to target circle
- tangential alignment error around the circle
- angular progress around the loiter center
- milestone completion fractions such as quarter-turns

These diagnostics stay inside the same path-tracking environment and evaluator. No second environment is introduced.

### 2. Use staged loiter curriculum

Do not start with one full loop.

Instead, the primitive curriculum should progress like this:

1. `turn`-only
2. short `loiter` arcs such as quarter-turn
3. half-loop `loiter`
4. full-loop `loiter`
5. reintroduce the normal primitive mix with `straight` rehearsal

This keeps the geometry local early, then extends the time horizon only after the policy can already capture the orbit.

### 3. Make reward explicitly orbit-aware

The reward should stop relying mostly on generic path progress inside `loiter`.

For `loiter`, the reward should add:

- radial tracking bonus / penalty relative to target radius
- tangential alignment bonus / penalty
- incremental angular-progress reward
- milestone bonuses at `0.25`, `0.50`, `0.75`, and `1.00` loop progress

Generic path reward remains in place, but the extra shaping only activates on `loiter` segments.

### 4. Tighten teacher only where needed

Teacher tightening should be selective.

The recent curve-aware tightening helped `turn`, but it also contributed to `straight` degradation because the training distribution became too curve-dominant. The next version should:

- tighten more strongly on `loiter`
- keep `turn` at the current or slightly relaxed setting
- leave `straight` looser

This preserves exploration where the policy already behaves acceptably.

### 5. Add straight rehearsal to avoid forgetting

Once `loiter` shaping gets stronger, `straight` will keep drifting unless it is sampled intentionally.

The primitive-stage sampling should therefore include a small but persistent `straight` rehearsal fraction even during loiter-focused stages. This is not meant to optimize `straight`; it is there to prevent catastrophic forgetting.

## Validation Strategy

The validation order should remain narrow and diagnostic:

1. no-wind primitive suite only
2. inspect per-case metrics for `straight`, `turn`, `loiter`
3. inspect the new loiter-specific diagnostics
4. only after primitive suite is stable, return to mixed missions

The main acceptance signal for this stage is not “suite score improved a bit”. It is:

- `turn` stays passed
- `loiter completion_rate` rises materially
- `straight` stops collapsing

## Near-Term Success Criteria

This repair is successful when all of the following are true on the no-wind primitive suite:

- `turn` remains passed
- `loiter` reaches meaningful completion instead of immediate termination
- `straight` remains within an acceptable regression budget
- the suite no longer depends on a single primitive carrying the average

Only after that should the project move back to harder mixed missions, wind, or estimated-state RL.

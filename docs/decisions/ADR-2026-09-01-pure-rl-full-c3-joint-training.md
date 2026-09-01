# ADR: Train the complete PureRL C3 task union from promoted C2c

- Date: 2026-09-01
- Status: accepted experiment
- Scope: measured CPU-native PureRL spatial training

## Context

The route-heading-canonical C3b run formally promoted `model_75.pt`, but its zero-shot C3c
evaluation passed only 88 of 96 cases. A 101-iteration C3c continuation from that checkpoint did
not improve the best C3c result and progressively damaged older skills. At iterations 50, 75 and
100, C3c success was 88/96, 80/96 and 80/96, while C3b success fell from 168/176 to 152/176 and
116/176. Iteration 100 also failed the frozen C1 tail-limit gate.

The existing C3c sampler is not the complete C3 task union. Its 15 percent earlier-spatial bucket
contains isolated C3a turns and only the two basic multi-turn C3b templates. It does not rehearse
the complete C3b sequential set, including turn/vertical transitions and loiter. C3b and C3c are
also orthogonal task axes: C3b exercises sequential switching, while C3c exercises simultaneous
horizontal and vertical demand. Treating C3c as a superset of C3b is therefore invalid.

## Decision

Add one default-disabled `c3joint` sampling mode behind the explicit
`--c3-full-joint-from-c2c` launcher route. Keep the registered C3c task and all existing C3a,
C3b and C3c baseline behavior unchanged unless this route is selected.

The new reset distribution is fixed for the first controlled experiment:

```text
C1      0.15
C2c     0.20
C3a     0.15
C3b     0.25
C3c     0.25
```

C3a samples isolated turns. C3b samples all seven sequential templates with the existing uniform
template contract. C3c samples coupled turn and climb/descent paths with the approved elliptical
demand bound. Five-group task-aware PPO normalizes and weights these task families separately.

Training starts weights-only from promoted C2c `model_550.pt`, uses a fresh optimizer and maps the
shared `[256,128]` actor strictly into the existing independent frequency and tail trunks. The
final split actor, route-heading-canonical observation, governor-gap reward, bounded warm start,
seed 0, 256 environments, 16 mini-batches and 25-iteration save cadence are retained. The budget
is 301 iterations because C3a, C3b and C3c are learned together from a C2 source.

Adaptive sampling, actor distillation, yaw-consistency loss, EWC, an additional head and reward
changes remain disabled. This keeps the task union and initialization lineage as the experiment's
substantive changes.

## Consequences and validation

C3a, C3b and C3c remain separate frozen evaluation suites and diagnostic labels, but no longer
serve as mandatory sequential policy checkpoints in this experiment. Promotion requires two
adjacent checkpoints to pass fresh CPU-native C3c, C3b, C3a, C2c and C1 suites. Training reward,
online success and source-weight loading are not promotion evidence.

The shared-to-split and world-to-canonical migration can initially disturb C1/C2c behavior. The
bounded warm start limits the first updates, but only fresh frozen evaluation can establish
retention. If C3b or C3c remains under-covered, first inspect per-family/template/sign/severity
counts before introducing adaptive sampling or another optimization method.


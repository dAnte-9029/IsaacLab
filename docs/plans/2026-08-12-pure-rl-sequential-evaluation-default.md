# PureRL Sequential Evaluation Default Implementation Plan

**Goal:** Make the validated train-only acceleration path the measured PureRL launcher default while preserving
explicit concurrent evaluation and non-measured behavior.

## Implementation

1. Add launcher contract tests for measured defaults, explicit concurrency, compatibility mode, conflicting
   flags, and unchanged non-measured defaults.
2. Add `--concurrent-eval` and resolve watcher behavior from the task and explicit flags.
3. Record the orchestration decision in a supplemental ADR, update the acceleration audit, and make C2/C3
   authority commands explicitly train-only.
4. Run the focused launcher suite, watcher regression suite, static compilation, and diff checks.

## Completion criteria

- Measured PureRL C1/C2/C3 launches do not start a watcher unless `--concurrent-eval` is supplied.
- `--train-only` remains valid, conflicting flags fail closed, and non-measured defaults do not change.
- Documentation no longer instructs authority training and checkpoint evaluation to overlap.

# Continuous-curve engineering handoff

- Worktree: `.worktrees/native-multibody-rl`
- Branch/base: `feat/native-multibody-rl`, `41ea8565`
- Status: engineering steps 1--3 implemented and validated; changes uncommitted
- Decision: `docs/decisions/ADR-2026-09-09-continuous-curve-path-contract.md`

## Delivered behavior

The explicit `Isaac-FlappingBot-StraightFlight-DeLaurier-MeasuredPureRL-C4-Direct-v0`
task consumes continuously varying three-dimensional paths. The generator uses
3-D arc length, analytic curvature/slope derivatives, a smooth entry, harmonics
and optional chirp. It integrates a dense 300 m table, screens the entire table,
and exposes a 140 m completion location with continued tracking through 20 s.

`pure_rl_continuous_curve_file` loads validated versioned recipes. Initial
altitude, route heading and flap phase come from the file. Partial resets map
recipes by persistent environment ID. The environment retains 555 actor values,
CPU-native physics, 60 Hz policy rate and the existing observation/query/reward
pipeline; C4 defines its smooth turn-activity input and separate telemetry.

The registered C4 task deliberately has no PPO runner entry. Six-group PPO,
mixed training, checkpoint evaluation and promotion remain unimplemented.
The new 96-case grid is a frozen diagnostic set, not a generalization test.
Candidate generation and Torch CUDA tests do not change CPU PhysX authority.

## Files

Added:

- `source/flapping_bot/flapping_bot/path_tracking/continuous_curve.py`
- `scripts/flapping_px4/generate_continuous_curve.py`
- `tests/test_continuous_curve.py`
- `tests/test_native_cpu_continuous_curve_runtime_isaac.py`
- the decision and this handoff

Modified:

- `source/flapping_bot/flapping_bot/direct/flapping_bot/pure_rl_spatial_path.py`:
  additive C4 adapter/stage; C3 event generation preserved
- `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`:
  opt-in configuration, replay/reset integration and C4 telemetry
- `source/flapping_bot/flapping_bot/direct/__init__.py` and
  `source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py`: registration/export
- `tests/test_pure_rl_spatial_path.py`: retain the old altitude-floor assertion
  for the four old spatial configurations; C4 has its own stronger reference floor
- `docs/PROJECT_STATE.md`

Unrelated transition diagnostic edits in
`scripts/flapping_rl/compare_pure_rl_c3b_transition_cases.py` and its test remain
untouched. The root worktree's asset configuration edit was also untouched.
No upstream framework, aerodynamic parameters, actuator physics or old task
defaults were changed. No commit or push was performed.

## Validation evidence

All Python commands used `env_isaaclab` and `./isaaclab.sh -p`.
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` bounded test CPU threading.

| Check | Observed result |
| --- | --- |
| Continuous path, existing spatial path/env/eval/registration and split actor tests | 109 passed, 36.03 s |
| New fresh CPU-native C4 runtime test | 1 passed, 15.01 s |
| Existing fresh C3 geometry/reward/reset runtime test | 1 passed, 2 unrelated adaptive tests deselected, 17.37 s |
| Frozen 96-case JSON generation, NPZ export and PNG rendering | passed |
| Standalone CLI JSON reload and plotting | passed |
| Compilation of changed/new Python modules | passed |
| Tracked diff and new-file whitespace checks | passed |

The final pure run included both float32/64 on CPU and CUDA, with no skips.
Coverage includes arc-length/frame checks, analytic derivative comparison,
kinematic screening, bad metadata rejection, exact serialization replay,
single-reset mirror diversity, normalized preview distinction, yaw equivalence,
bounded sequential projection and completion boundaries. The CLI test runs from
outside the repository and checks non-overwrite behavior.

The C4 runtime test uses six active native constraints and checks the actual
reset height/phase, finite observations/rewards/root state, cached path queries,
eight policy steps, C4 telemetry and two repeated partial resets. It does not
load or evaluate a learned policy. Runtime logs contain Gym re-registration and
Isaac shutdown warnings, but pytest reports pass. The initial sandboxed Isaac
startup crashed before test execution; the successful fresh runs followed the
user's switch to unrestricted execution.

An initial legacy test failed because its all-stage loop assumed the old 0.05 m
reference floor. The assertion remains unchanged for C3; C4's 4 m floor is
tested separately. An initial standalone CLI import failed because the project
package was not on its path; the script now resolves its own checkout, covered
by the subprocess test. Neither initial failure is counted as a passing run.

Local generated evidence is ignored by Git:

```text
logs/continuous_curve/20260909_contract/
  diagnostic96.json
  diagnostic96.npz
  preview.png
  c4-pure-tests-final.log
  c4-native-runtime-final.log
  c4-c3-retention-smoke.log
```

## Reproduce

From the RL worktree:

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TERM=xterm

./isaaclab.sh -p -m pytest -q -p no:cacheprovider \
  tests/test_continuous_curve.py tests/test_pure_rl_spatial_path.py \
  tests/test_pure_rl_spatial_env_contract.py tests/test_pure_rl_spatial_eval.py \
  tests/test_pure_rl_spatial_task_registration.py tests/test_pure_rl_split_actor.py

./isaaclab.sh -p -m pytest -q -p no:cacheprovider \
  tests/test_native_cpu_continuous_curve_runtime_isaac.py

./isaaclab.sh -p -m pytest -q -p no:cacheprovider \
  tests/test_native_cpu_pure_rl_spatial_runtime_isaac.py \
  -k geometry_reward_and_reset_runtime_gate
```

Generate to new filenames (the script rejects overwriting):

```bash
./isaaclab.sh -p scripts/flapping_px4/generate_continuous_curve.py \
  --diagnostic-grid --output /tmp/c4-grid-new.json \
  --table-output /tmp/c4-grid-new.npz --plot /tmp/c4-grid-new.png

./isaaclab.sh -p scripts/flapping_px4/generate_continuous_curve.py \
  --count 6 --seed 0 --output /tmp/c4-candidates-new.json
```

For replay, construct `FlappingBotStraightFlightDeLaurierMeasuredPureRLC4EnvCfg`
after native Kit startup and assign `cfg.pure_rl_continuous_curve_file` to the
JSON path. The runtime test is a complete executable example. The JSON recipe
is authoritative; NPZ is an inspection/export format, not an alternate loader.

## Exact next task and limitations

Implement the C4 checkpoint evaluator and deterministic traces, then run
zero-shot from the original full-joint `model_175.pt`. Preserve its split actor
and heading-canonical semantics. Analyze requested/applied frequency, actual
tail motion, speed, tracking error and preview response before changing inputs
or training. Distinguish policy failure from infeasible demand.

The current screen is sampled kinematics under an assumed speed band. It does
not certify between-sample extrema, full tracking-tube clearance, aerodynamic
loads, motor power or dynamic tracking feasibility. The engineering thresholds
are not identified physical limits. A preview distinction test is not proof
that a trained actor exploits the available information.

Before PPO training, separately define development/held-out path sets, implement
six-group rehearsal and PPO, and freeze per-path/aggregate scoring and promotion.
The source checkpoint remains research-only without an adjacent all-suite
passing pair. No zero-shot, new policy training, retention-suite checkpoint
evaluation, multi-seed result or promotion is claimed here.

## Zero-shot evaluator continuation (2026-09-09)

The earlier implementation-status statements above describe the engineering gate.
Checkpoint evaluation is now implemented by
`source/flapping_bot/flapping_bot/analysis/continuous_curve_evaluation.py`, with
CLI `scripts/flapping_px4/evaluate_continuous_curve.py` and pure scoring tests in
`tests/test_continuous_curve_evaluation.py`. No PPO runner or promotion was added.

The evaluator loads the recorded split actor configuration, checks heading-canonical
provenance, and loads the full model state strictly with `weights_only=True`.
It never constructs or restores an optimizer. Native CPU physics, seed 0 and
555 observations are checked. Each row records only its first episode. A local
`_get_rewards` adapter snapshots metrics and actual state before DirectRLEnv's
automatic reset. NPZ traces include the actor input and raw action at t-dt,
and errors, progress, curvature/slope, actual frequency/phase, requested/applied
frequency, commanded/actual tail angles, world pose/velocity and termination
causes at t. There is no inferred causal response-lag claim.

Per-path accuracy uses the existing ADR thresholds, 140 m completion and the full
20 s timeout (one policy-step tolerance), rejecting nonfinite traces. Reverse
motion, roll, speed-band violations and frequency gap remain diagnostic. Summary
P95 values are explicitly **means of per-case P95**, not a pooled percentile.
Family and mirror slices report counts and success rates. `evaluation_complete`
means every expected first episode ended; `promotion_eligible` is always false.
`status.json: completed` means execution completed, not that tracking passed.
Missing, duplicate and partial cases cannot claim evaluation completion.

Validation: the first evaluator smoke caught a syntax error before evaluation;
it was corrected. A fresh `20260909_evaluator_smoke_v2` run loaded the original
checkpoint strictly and completed six native environments for eight steps.
The smoke's summary correctly reports incomplete evaluation. Pure tests cover
completion without survival, error thresholds, nonfinite values and missing,
duplicate or incomplete case sets. Existing curve-generator tests are also run.

Formal zero-shot is launched detached at the user's request, without subsequent
monitoring. Inputs are the original full-joint run's `model_175.pt` and
`logs/continuous_curve/20260909_contract/diagnostic96.json`. Output directory:
`logs/continuous_curve/20260909_zero_shot_175/`; launcher log/PID are sibling
`20260909_zero_shot_175.log` and `.pid` files. The directory is exclusive-create.
`launch.json` records checkpoint hash, command, Git state and seed; `curves.json`
is the replay copy; `summary.json` and `traces/*.npz` are written after evaluation.
`status.json` records caught failures. A hard process crash may leave running
status; no result should be inferred solely from launch or wrapper exit status.

Reproduction (use a new output directory):

```bash
source /home/zn/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TERM=xterm
./isaaclab.sh -p scripts/flapping_px4/evaluate_continuous_curve.py \
  --checkpoint logs/rsl_rl/flapping_bot_straight_flight/2026-09-01_10-13-43_pure_rl_c3_full_joint_from_c2c_seed0_301iter/model_175.pt \
  --curves logs/continuous_curve/20260909_contract/diagnostic96.json \
  --output logs/continuous_curve/new_zero_shot
```

Add `--smoke-steps 8` for six paths and eight steps. Formal run results, retention
and dynamic feasibility remain unverified until a later result-review task.

## Random-reset and production C4 preview continuation (2026-09-10)

See `docs/decisions/ADR-2026-09-10-c4-random-reset-and-preview.md` for the accepted
new defaults and exact validation. The random sampler now resamples only failed
rows up to a configurable bound. `pure_rl_continuous_curve_sampling` exposes
candidate ranges; file replay remains fixed. C4 defaults to configurable five
preview times up to 1.2 s. Old task defaults and evaluator CLI scale meanings
remain unchanged. New tests cover screening retries, accepted-row preservation,
seed reproduction, CPU/CUDA/dtypes and native random partial resets.

Independent seed-104729 random 96-case evaluation uses the original175 weights,
1.2 s preview and `logs/continuous_curve/20260910_random_generator/independent96.json`.
Output: `logs/continuous_curve/20260910_random_generator/evaluation/`; sibling
`evaluation.log`. Run results are pending; no training or promotion is claimed.

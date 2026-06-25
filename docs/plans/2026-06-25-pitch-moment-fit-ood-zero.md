# Pitch Moment Fit and OOD Zero Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fit and validate a low-dimensional pitch-moment correction so Isaac's effective-force runtime reaches zero OOD samples in the agreed pre-RL validation suite.

**Architecture:** `/home/zn/flap-system-identification` owns real-log prior export, coefficient fitting, model selection, and the canonical pitch-moment artifact. `/home/zn/IsaacLab` vendors that artifact, adds independent runtime inference, and validates that replacing `My` removes pitch-rate-driven force-model OOD without changing the established effective `Fx/Fz` contract.

**Tech Stack:** Python 3.11, NumPy, pandas, PyTorch, pytest, IsaacLab/Isaac Sim, JSON/CSV/parquet artifacts, whole-log train/validation/test splitting.

---

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-06-25
- Verification Status: UNVERIFIED
- Version Label: pitch_moment_ood_zero_plan_v1

## Experiment overview

- **Title:** Low-dimensional pitch-moment correction for effective-force OOD elimination.
- **Objective:** Test whether `M_y = k_w M_{y,wing}^{prior} + k_t M_{y,tail}^{prior} + M_0` can make the Isaac robot remain inside the real-flight envelope and produce zero effective-force OOD samples in validation rollouts.
- **Hypothesis:** The existing OOD is mainly caused by pitch-rate response mismatch from the uncorrected moment model, so a simple fitted pitch-moment correction should reduce or eliminate force-model OOD without changing the force model.
- **Type:** analysis + simulation validation.

## Acceptance gates

All gates must pass before this work is considered complete:

1. system-identification tests pass for prior export, fitting, artifact serialization, and split isolation;
2. Isaac unit tests pass for moment-correction inference, frame/sign convention, and wrench composition;
3. frozen effective `Fx/Fz` reference and force-sweep tests still pass;
4. `disabled` and `shadow` moment modes produce identical applied dynamics;
5. `replace_my` changes only body `My` and preserves `Fx/Fz/Fy/Mx/Mz` except for the already configured effective-force replacement;
6. validation-suite OOD count is `0`, fallback count is `0`, OOD termination count is `0`;
7. pitch rate, pitch attitude, speed, and angle-of-attack proxy stay inside the agreed real-flight envelope;
8. if the three-parameter model cannot satisfy gates 6-7, stop and report diagnostics instead of widening the force-model envelope.

## Task 0: Create or reuse isolated worktrees

**Files:**

- Read: `/home/zn/IsaacLab/.git/`
- Read: `/home/zn/flap-system-identification/.git/`
- Create or reuse: `/tmp/IsaacLab-pitch-moment-ood-zero/`
- Create or reuse: `/tmp/flap-system-identification-pitch-moment-ood-zero/`

**Step 1: Inspect current dirty state**

Run:

```bash
git -C /home/zn/IsaacLab status --short
git -C /home/zn/flap-system-identification status --short
```

Expected: existing user or prior-agent changes are visible before new work starts.

**Step 2: Create dedicated worktrees if needed**

If the `/tmp/...pitch-moment-ood-zero` worktrees do not already exist, create them from the current integration branches or from the reviewed effective-force branches:

```bash
git -C /home/zn/IsaacLab worktree add -b pitch-moment-ood-zero /tmp/IsaacLab-pitch-moment-ood-zero HEAD
git -C /home/zn/flap-system-identification worktree add -b pitch-moment-ood-zero /tmp/flap-system-identification-pitch-moment-ood-zero HEAD
```

Expected: clean isolated worktrees. If branches already exist, reuse the existing worktrees and record their `HEAD` commits.

**Step 3: Record baseline effective-force behavior**

In IsaacLab, run the current effective-force tests before touching moment code:

```bash
cd /tmp/IsaacLab-pitch-moment-ood-zero
TERM=xterm ./isaaclab.sh -p -m pytest \
  tests/test_effective_force_model_artifact.py \
  tests/test_effective_force_correction.py \
  tests/test_effective_force_wrench_composition.py \
  tests/test_validate_effective_force_model.py \
  -q
```

Expected: all tests pass. If they fail, stop and fix or rebase the worktree before adding pitch-moment work.

## Task 1: Build the real-flight envelope artifact

**Files:**

- Create: `/tmp/flap-system-identification-pitch-moment-ood-zero/scripts/build_real_flight_envelope.py`
- Create: `/tmp/flap-system-identification-pitch-moment-ood-zero/tests/test_build_real_flight_envelope.py`
- Output: `/tmp/flap-system-identification-pitch-moment-ood-zero/artifacts/20260625_pitch_moment_v1/real_flight_envelope.json`

**Step 1: Write failing tests**

Test that the envelope builder:

```python
def test_envelope_uses_min_max_without_padding():
    rows = make_rows(q=[-1.0, 2.0], pitch=[0.1, 0.3], speed=[4.0, 5.0])
    envelope = build_envelope(rows)
    assert envelope["body_rate_q_frd"]["min"] == -1.0
    assert envelope["body_rate_q_frd"]["max"] == 2.0


def test_envelope_records_source_counts_by_log():
    rows = make_rows(log_id=["a", "a", "b"])
    envelope = build_envelope(rows)
    assert envelope["source_counts_by_log"] == {"a": 2, "b": 1}
```

**Step 2: Run tests and confirm failure**

```bash
cd /tmp/flap-system-identification-pitch-moment-ood-zero
pytest tests/test_build_real_flight_envelope.py -q
```

Expected: missing script or missing function failure.

**Step 3: Implement the envelope builder**

Compute min, max, mean, standard deviation, and sample counts for:

- FRD pitch rate `q`;
- pitch attitude;
- airspeed or speed proxy used by current logs;
- angle-of-attack proxy if present;
- effective-force feature columns, including phase, frequency, `q`, `q*sin(phase)`, and `q*cos(phase)`.

Do not add padding in this task. The target is to test against the recorded envelope, not to create a permissive envelope.

**Step 4: Generate the artifact**

Run:

```bash
python scripts/build_real_flight_envelope.py \
  --input-root artifacts/20260623_final_effective_force_fx_fz_v1 \
  --output artifacts/20260625_pitch_moment_v1/real_flight_envelope.json
```

Expected: JSON exists, includes source files and per-log counts, and records no padded limits.

**Step 5: Commit**

```bash
git add scripts/build_real_flight_envelope.py tests/test_build_real_flight_envelope.py artifacts/20260625_pitch_moment_v1/real_flight_envelope.json
git commit -m "feat: export real flight envelope for moment validation"
```

## Task 2: Export fixed wing and tail pitch-moment priors from real logs

**Files:**

- Create: `/tmp/flap-system-identification-pitch-moment-ood-zero/scripts/export_pitch_moment_priors.py`
- Create: `/tmp/flap-system-identification-pitch-moment-ood-zero/tests/test_export_pitch_moment_priors.py`
- Read or mirror contract from: `/home/zn/IsaacLab/scripts/flapping_px4/export_delaurier_prior_predictions.py`
- Output: `/tmp/flap-system-identification-pitch-moment-ood-zero/artifacts/20260625_pitch_moment_v1/pitch_moment_priors.parquet`

**Step 1: Write failing tests**

Cover the critical convention and non-placeholder behavior:

```python
def test_export_contains_component_moments_not_zero_placeholder(tmp_path):
    output = export_one_synthetic_log(tmp_path)
    assert "my_wing_prior_frd" in output.columns
    assert "my_tail_prior_frd" in output.columns
    assert output["my_wing_prior_frd"].abs().max() > 0.0


def test_export_preserves_label_and_log_identity(tmp_path):
    output = export_one_synthetic_log(tmp_path)
    assert {"my_b", "log_id", "split"}.issubset(output.columns)
```

**Step 2: Run tests and confirm failure**

```bash
pytest tests/test_export_pitch_moment_priors.py -q
```

Expected: missing exporter failure.

**Step 3: Implement exporter**

For every retained log sample, export:

- `my_b`;
- `my_wing_prior_frd`;
- `my_tail_prior_frd`;
- `my_total_prior_frd`;
- `log_id`, split, timestamp or sample index;
- phase, cycle-average frequency, pitch rate, velocity, pitch attitude, elevator or tail command.

The exporter must compute priors from fixed wing and tail model components. Do not use `prior_my_b` from the current effective-force artifact because it is a zero placeholder.

**Step 4: Run exporter**

```bash
python scripts/export_pitch_moment_priors.py \
  --effective-force-artifact artifacts/20260623_final_effective_force_fx_fz_v1/model.json \
  --output artifacts/20260625_pitch_moment_v1/pitch_moment_priors.parquet
```

Expected: output has finite `my_b`, finite component priors, and no missing split/log identifiers.

**Step 5: Commit**

```bash
git add scripts/export_pitch_moment_priors.py tests/test_export_pitch_moment_priors.py artifacts/20260625_pitch_moment_v1/pitch_moment_priors.parquet
git commit -m "feat: export pitch moment component priors"
```

## Task 3: Fit the three-parameter pitch-moment model

**Files:**

- Create: `/tmp/flap-system-identification-pitch-moment-ood-zero/src/system_identification/pitch_moment_fit.py`
- Create: `/tmp/flap-system-identification-pitch-moment-ood-zero/scripts/fit_pitch_moment_correction.py`
- Create: `/tmp/flap-system-identification-pitch-moment-ood-zero/tests/test_pitch_moment_fit.py`
- Output: `/tmp/flap-system-identification-pitch-moment-ood-zero/artifacts/20260625_pitch_moment_v1/model.json`
- Output: `/tmp/flap-system-identification-pitch-moment-ood-zero/artifacts/20260625_pitch_moment_v1/metrics.json`

**Step 1: Write failing numerical tests**

Test closed-form recovery and prior-centered ridge:

```python
def test_weighted_fit_recovers_known_coefficients():
    data = synthetic_pitch_data(kw=1.2, kt=0.7, m0=0.03)
    fit = fit_pitch_moment_model(data, lambda_=0.0)
    assert fit.kw == pytest.approx(1.2)
    assert fit.kt == pytest.approx(0.7)
    assert fit.m0 == pytest.approx(0.03)


def test_ridge_pulls_gains_toward_one_without_penalizing_bias():
    data = underdetermined_pitch_data()
    fit = fit_pitch_moment_model(data, lambda_=100.0)
    assert fit.kw == pytest.approx(1.0, abs=0.1)
    assert fit.kt == pytest.approx(1.0, abs=0.1)
```

**Step 2: Run tests and confirm failure**

```bash
pytest tests/test_pitch_moment_fit.py -q
```

Expected: missing module failure.

**Step 3: Implement fitting code**

Implement:

```text
theta = (A^T W A + lambda P)^-1 (A^T W y + lambda P theta0)
P = diag(1, 1, 0)
theta0 = [1, 1, 0]^T
```

Use equal-log weighting: each log contributes the same total weight.

**Step 4: Implement model selection**

Fit on training logs for a small `lambda` grid, select by validation RMSE or a declared composite metric, and evaluate the selected model once on held-out logs.

Compare:

- raw prior;
- bias-only prior;
- fitted model.

Report RMSE, MAE, bias, correlation, `R^2`, prior-column correlation, and condition number.

**Step 5: Run fitting**

```bash
python scripts/fit_pitch_moment_correction.py \
  --priors artifacts/20260625_pitch_moment_v1/pitch_moment_priors.parquet \
  --output-dir artifacts/20260625_pitch_moment_v1
```

Expected: `model.json` contains `kw`, `kt`, `m0`, units, convention, split provenance, selected `lambda`, and metric tables.

**Step 6: Commit**

```bash
git add src/system_identification/pitch_moment_fit.py \
  scripts/fit_pitch_moment_correction.py \
  tests/test_pitch_moment_fit.py \
  artifacts/20260625_pitch_moment_v1/model.json \
  artifacts/20260625_pitch_moment_v1/metrics.json
git commit -m "feat: fit low dimensional pitch moment correction"
```

## Task 4: Add offline response validation

**Files:**

- Create: `/tmp/flap-system-identification-pitch-moment-ood-zero/scripts/evaluate_pitch_moment_response.py`
- Create: `/tmp/flap-system-identification-pitch-moment-ood-zero/tests/test_evaluate_pitch_moment_response.py`
- Output: `/tmp/flap-system-identification-pitch-moment-ood-zero/artifacts/20260625_pitch_moment_v1/response_metrics.json`

**Step 1: Write failing tests**

Test that the evaluator computes direction and peak-error metrics:

```python
def test_response_metrics_detect_wrong_sign():
    real = np.array([0.0, 1.0, 2.0])
    pred = np.array([0.0, -1.0, -2.0])
    metrics = response_metrics(real, pred)
    assert metrics["direction_match"] is False
```

**Step 2: Implement response evaluator**

For held-out or validation segments, compare raw prior, bias-only prior, and fitted model on:

- pointwise `My` error;
- sign agreement;
- peak response direction;
- time-aligned pitch-rate or pitch-response proxy where available.

If full state replay is not available in system-identification, explicitly label this as a log-aligned response proxy and leave Isaac rollout response validation to later tasks.

**Step 3: Run evaluator**

```bash
python scripts/evaluate_pitch_moment_response.py \
  --priors artifacts/20260625_pitch_moment_v1/pitch_moment_priors.parquet \
  --model artifacts/20260625_pitch_moment_v1/model.json \
  --output artifacts/20260625_pitch_moment_v1/response_metrics.json
```

Expected: fitted model has reasonable sign and peak behavior. If it is worse than raw prior, stop and report.

**Step 4: Commit**

```bash
git add scripts/evaluate_pitch_moment_response.py \
  tests/test_evaluate_pitch_moment_response.py \
  artifacts/20260625_pitch_moment_v1/response_metrics.json
git commit -m "test: evaluate pitch moment response metrics"
```

## Task 5: Vendor the pitch-moment artifact into IsaacLab

**Files:**

- Create: `/tmp/IsaacLab-pitch-moment-ood-zero/source/flapping_bot/flapping_bot/data/pitch_moment_correction/model.json`
- Create: `/tmp/IsaacLab-pitch-moment-ood-zero/source/flapping_bot/flapping_bot/physics/pitch_moment_correction.py`
- Create: `/tmp/IsaacLab-pitch-moment-ood-zero/tests/test_pitch_moment_correction.py`

**Step 1: Copy artifact with provenance**

Copy only the canonical system-identification `model.json` and any required manifest files. Record the source SHA-256 in the vendored manifest.

**Step 2: Write failing Isaac inference tests**

```python
def test_pitch_moment_model_predicts_scalar_my():
    model = PitchMomentCorrectionModel(kw=1.2, kt=0.7, m0=0.03)
    pred = model.predict(my_wing=torch.tensor([2.0]), my_tail=torch.tensor([1.0]))
    assert pred.item() == pytest.approx(3.13)
```

Also test artifact schema rejection for missing units or convention.

**Step 3: Implement Torch inference**

Keep inference small and allocation-free inside the physics loop. The model should accept batched tensors for `my_wing_prior` and `my_tail_prior` and return a batched scalar `My`.

**Step 4: Run tests**

```bash
cd /tmp/IsaacLab-pitch-moment-ood-zero
TERM=xterm ./isaaclab.sh -p -m pytest tests/test_pitch_moment_correction.py -q
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/data/pitch_moment_correction/model.json \
  source/flapping_bot/flapping_bot/physics/pitch_moment_correction.py \
  tests/test_pitch_moment_correction.py
git commit -m "feat: add pitch moment correction artifact"
```

## Task 6: Integrate `shadow` and `replace_my` modes in Isaac

**Files:**

- Modify: `/tmp/IsaacLab-pitch-moment-ood-zero/source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Modify: `/tmp/IsaacLab-pitch-moment-ood-zero/source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py`
- Modify: `/tmp/IsaacLab-pitch-moment-ood-zero/source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py`
- Create: `/tmp/IsaacLab-pitch-moment-ood-zero/tests/test_pitch_moment_wrench_composition.py`

**Step 1: Write failing composition tests**

Test the exact contract:

```python
def test_replace_my_changes_only_pitch_moment():
    raw_force = torch.tensor([[1.0, 2.0, 3.0]])
    raw_torque = torch.tensor([[0.1, 0.2, 0.3]])
    corrected_my = torch.tensor([0.9])
    force, torque = compose_pitch_moment_wrench(raw_force, raw_torque, corrected_my, mode="replace_my")
    assert torch.equal(force, raw_force)
    assert torque[0, 0] == raw_torque[0, 0]
    assert torque[0, 1] == pytest.approx(0.9)
    assert torque[0, 2] == raw_torque[0, 2]
```

Also test that `shadow` returns the raw applied torque but records the prediction.

**Step 2: Add config fields**

Add a pitch-moment correction config with:

- `pitch_moment_mode`: `disabled | shadow | replace_my`;
- `pitch_moment_artifact_path`;
- strict artifact validation at environment construction.

Default mode must remain `disabled`.

**Step 3: Integrate in `_apply_action` or the existing wrench assembly point**

Use the already computed wing and tail moment components:

- derive `my_wing_prior` and `my_tail_prior` in the artifact's body convention;
- evaluate the pitch-moment model;
- apply replacement only in `replace_my`;
- cache raw, predicted, final, and mode status for diagnostics.

Do not change the effective-force `Fx/Fz` logic in this task.

**Step 4: Register validation task IDs**

Add task IDs that combine effective-force replacement with pitch-moment shadow/replacement without breaking existing IDs. Example names:

- `Isaac-FlappingBot-StraightFlight-EffectiveForceReplace-PitchMomentShadow-Direct-v0`
- `Isaac-FlappingBot-StraightFlight-EffectiveForceReplace-PitchMomentReplace-Direct-v0`

**Step 5: Run targeted tests**

```bash
TERM=xterm ./isaaclab.sh -p -m pytest \
  tests/test_pitch_moment_wrench_composition.py \
  tests/test_effective_force_wrench_composition.py \
  tests/test_flapping_task_registration.py \
  -q
```

Expected: new and existing composition contracts pass.

**Step 6: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
  source/flapping_bot/flapping_bot/direct/flapping_bot/__init__.py \
  source/isaaclab_tasks/isaaclab_tasks/direct/flapping_bot/__init__.py \
  tests/test_pitch_moment_wrench_composition.py
git commit -m "feat: integrate pitch moment correction modes"
```

## Task 7: Add OOD-zero rollout validation

**Files:**

- Modify: `/tmp/IsaacLab-pitch-moment-ood-zero/scripts/flapping_rl/eval_effective_force_credibility.py`
- Create: `/tmp/IsaacLab-pitch-moment-ood-zero/tests/test_eval_pitch_moment_ood_zero.py`
- Output: `/tmp/effective_force_validation/pitch_moment_ood_zero/credibility_report.json`
- Output: `/tmp/effective_force_validation/pitch_moment_ood_zero/*.csv`

**Step 1: Write failing report tests**

Test that the credibility report rejects nonzero OOD:

```python
def test_ood_zero_gate_fails_on_any_ood():
    report = build_report(ood_count=1, fallback_count=0, termination_count=0)
    assert report["ood_zero_verdict"] == "fail"
```

**Step 2: Extend rollout logging**

Add columns for:

- `pitch_moment_mode`;
- `my_raw`, `my_wing_prior`, `my_tail_prior`, `my_predicted`, `my_final`;
- effective-force OOD and fallback flags;
- pitch rate, pitch attitude, speed, angle-of-attack proxy;
- real-flight-envelope membership.

**Step 3: Run nominal comparison**

Run at least these modes from identical seeds and initial conditions:

```bash
TERM=xterm ./isaaclab.sh -p scripts/flapping_rl/eval_effective_force_credibility.py \
  --task Isaac-FlappingBot-StraightFlight-EffectiveForceReplace-PitchMomentShadow-Direct-v0 \
  --output-dir /tmp/effective_force_validation/pitch_moment_ood_zero/shadow \
  --steps 240 --seed 0 --device cuda:1

TERM=xterm ./isaaclab.sh -p scripts/flapping_rl/eval_effective_force_credibility.py \
  --task Isaac-FlappingBot-StraightFlight-EffectiveForceReplace-PitchMomentReplace-Direct-v0 \
  --output-dir /tmp/effective_force_validation/pitch_moment_ood_zero/replace_my \
  --steps 240 --seed 0 --device cuda:1
```

Expected: `shadow` matches raw applied dynamics; `replace_my` changes pitch dynamics and reports OOD count.

**Step 4: Run scenario matrix**

Add a small scenario matrix before RL:

- nominal straight flight;
- small elevator or tail-command perturbation;
- small frequency perturbation within the training envelope;
- reset states sampled from held-out real-log states if available.

Expected: every scenario must have OOD count `0`, fallback count `0`, OOD termination count `0`, and real-flight-envelope violations `0`.

**Step 5: Stop on failure**

If any scenario has nonzero OOD, produce a diagnostic table with:

- first failing step;
- failing feature name;
- observed value;
- envelope bound;
- pitch rate and phase at failure;
- raw and corrected `My`.

Do not widen the force envelope. Do not add damping in this task.

**Step 6: Commit**

```bash
git add scripts/flapping_rl/eval_effective_force_credibility.py tests/test_eval_pitch_moment_ood_zero.py
git commit -m "test: add pitch moment ood zero validation"
```

## Task 8: Add bootstrap ranges for later domain randomization

**Files:**

- Create: `/tmp/flap-system-identification-pitch-moment-ood-zero/scripts/bootstrap_pitch_moment_coefficients.py`
- Create: `/tmp/flap-system-identification-pitch-moment-ood-zero/tests/test_bootstrap_pitch_moment_coefficients.py`
- Output: `/tmp/flap-system-identification-pitch-moment-ood-zero/artifacts/20260625_pitch_moment_v1/bootstrap_coefficients.csv`
- Output: `/tmp/flap-system-identification-pitch-moment-ood-zero/artifacts/20260625_pitch_moment_v1/domain_randomization.json`

**Step 1: Write failing bootstrap tests**

Test that sampling is by whole log:

```python
def test_bootstrap_samples_whole_logs_not_rows():
    samples = bootstrap_log_ids(["a", "b", "c"], seed=0)
    assert all(isinstance(sample, list) for sample in samples)
```

**Step 2: Implement bootstrap**

Resample logs with replacement, refit `[k_w, k_t, M_0]`, and save the joint coefficient samples. Derive domain-randomization bounds from joint samples, not independent arbitrary ranges.

**Step 3: Run bootstrap**

```bash
python scripts/bootstrap_pitch_moment_coefficients.py \
  --priors artifacts/20260625_pitch_moment_v1/pitch_moment_priors.parquet \
  --model artifacts/20260625_pitch_moment_v1/model.json \
  --output-dir artifacts/20260625_pitch_moment_v1 \
  --num-bootstrap 1000
```

Expected: coefficient distribution exists and can be used later for domain randomization. This does not start RL training.

**Step 4: Commit**

```bash
git add scripts/bootstrap_pitch_moment_coefficients.py \
  tests/test_bootstrap_pitch_moment_coefficients.py \
  artifacts/20260625_pitch_moment_v1/bootstrap_coefficients.csv \
  artifacts/20260625_pitch_moment_v1/domain_randomization.json
git commit -m "feat: export pitch moment randomization ranges"
```

## Task 9: Final verification and handoff

**Files:**

- Read: both worktree `git status --short`
- Output: `/tmp/effective_force_validation/pitch_moment_ood_zero/final_summary.md`

**Step 1: Run full targeted system-identification tests**

```bash
cd /tmp/flap-system-identification-pitch-moment-ood-zero
pytest \
  tests/test_build_real_flight_envelope.py \
  tests/test_export_pitch_moment_priors.py \
  tests/test_pitch_moment_fit.py \
  tests/test_evaluate_pitch_moment_response.py \
  tests/test_bootstrap_pitch_moment_coefficients.py \
  -q
```

Expected: all pass.

**Step 2: Run full targeted Isaac tests**

```bash
cd /tmp/IsaacLab-pitch-moment-ood-zero
TERM=xterm ./isaaclab.sh -p -m pytest \
  tests/test_effective_force_model_artifact.py \
  tests/test_effective_force_correction.py \
  tests/test_effective_force_wrench_composition.py \
  tests/test_validate_effective_force_model.py \
  tests/test_pitch_moment_correction.py \
  tests/test_pitch_moment_wrench_composition.py \
  tests/test_eval_pitch_moment_ood_zero.py \
  tests/test_flapping_task_registration.py \
  -q
```

Expected: all pass.

**Step 3: Run final rollouts**

Run the scenario matrix from Task 7 and produce a final summary table with:

- per-scenario OOD count;
- fallback count;
- termination count;
- real-flight-envelope violation count;
- `q` max/min;
- pitch max/min;
- speed max/min;
- first failure diagnostics if any gate fails.

Expected success criterion:

```text
OOD count = 0
fallback count = 0
OOD termination count = 0
real-flight-envelope violation count = 0
```

**Step 4: Verify repository status**

```bash
git -C /tmp/flap-system-identification-pitch-moment-ood-zero status --short
git -C /tmp/IsaacLab-pitch-moment-ood-zero status --short
```

Expected: only intentional generated reports remain untracked, or both worktrees are clean after commits.

**Step 5: Handoff report**

Write a concise report with:

- system-identification commit hash;
- IsaacLab commit hash;
- model coefficients;
- selected `lambda`;
- held-out metrics;
- rollout OOD/fallback/termination counts;
- whether the three-parameter model achieved the OOD-zero target;
- if failed, the minimal diagnostic evidence and the next candidate extension.


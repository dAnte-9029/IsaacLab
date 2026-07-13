# Effective Force Model Freeze and Isaac Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Freeze the paper's final longitudinal effective-force model, replace only IsaacLab's final body `Fx/Fz` with its prediction, preserve simulator `Fy/Mx/My/Mz`, and produce evidence for numerical and dynamic credibility before any RL work.

**Architecture:** The system-identification repository owns model selection, all-log refitting, lineage, and the canonical frozen artifact. IsaacLab vendors that artifact, implements independent batched Torch inference, and composes a hybrid wrench with effective longitudinal force and the existing simulator moment. Integration begins in shadow mode and advances through explicit numerical, force-surface, local-dynamics, and controller-in-the-loop gates.

**Tech Stack:** Python 3.11, NumPy, pandas, scikit-learn ridge artifacts, PyTorch, IsaacLab/Isaac Sim, pytest, JSON/NPZ, parquet, existing PX4-like validation scripts.

---

## Worktree and repository boundary

Implementation uses two code repositories plus uncommitted paper-selection artifacts. Complete Task 0 before creating worktrees. Execute system-identification tasks in `/home/zn/flap-system-identification`; execute simulator tasks in `/home/zn/IsaacLab`. Do not copy source files between repositories. Only the frozen model artifact crosses the boundary.

## Stage summary

| Stage | Work | Result |
|---|---|---|
| 1 | Lock model family, alpha rule, data, prior, and conventions | Reviewed freeze specification |
| 2 | Fit all-log model and export immutable artifact | Standalone model JSON/NPZ and reference vectors |
| 3 | Implement independent Isaac Torch inference | Batched inference matching the exporter |
| 4 | Add shadow and `Fx/Fz` replacement modes | Hybrid wrench with unchanged moment |
| 5 | Prove numerical and runtime correctness | Unit, contract, and smoke tests |
| 6 | Compare force surfaces and operating envelope | Static/sweep validation report |
| 7 | Compare local and closed-loop dynamics | Credibility report and go/no-go decision |

### Task 0: Freeze the source state before creating worktrees

**Files:**
- Read: `/home/zn/flap-system-identification/.git/`
- Read: `/home/zn/IsaacLab/.git/`
- Read: `/home/zn/paper/AeroConf_effective_aero/.git/`
- Read: `/home/zn/paper/AeroConf_effective_aero/research_notes/20260612_nested_outer_cv/`
- Create: `/tmp/20260623_effective_force_source_state/`

**Purpose:** Prevent a clean worktree from silently omitting the current nested-CV results or any model-defining uncommitted changes.

**Step 1: Record repository revisions and status**

Run:

```bash
git -C /home/zn/flap-system-identification rev-parse HEAD
git -C /home/zn/flap-system-identification status --short
git -C /home/zn/IsaacLab rev-parse HEAD
git -C /home/zn/IsaacLab status --short
git -C /home/zn/paper/AeroConf_effective_aero rev-parse HEAD
git -C /home/zn/paper/AeroConf_effective_aero status --short
```

Expected: revisions and dirty paths are captured without modifying any repository.

**Step 2: Hash the uncommitted nested-CV selection inputs**

Create a sorted SHA-256 manifest for the JSON/CSV files consumed by final alpha selection and store it under `/tmp/20260623_effective_force_source_state/`. Do not hash generated figures or unrelated LaTeX examples.

**Step 3: Decide how the required dirty state becomes reproducible**

Preferred order:

1. user-reviewed commits containing the required model and nested-CV artifacts;
2. a dedicated reviewed patch plus content-hash manifest;
3. stop if neither is available.

Do not commit unrelated user changes and do not assume the current paper `HEAD` contains the nested-CV results.

**Step 4: Create dedicated worktrees from the reviewed revisions**

Use `superpowers:using-git-worktrees` after the source revisions are confirmed. If a reviewed patch is required, apply only that patch to the new worktree and verify its hash.

**Step 5: Record baseline targeted tests**

Run the current model-selection/system-identification tests and current Isaac flapping contract tests. Save the commands and outputs in `/tmp/20260623_effective_force_source_state/` so later regressions are compared with a real baseline.

Expected: implementation starts from a reproducible source snapshot, not merely the visible dirty working directory.

### Task 1: Lock the deployment-model selection and lineage

**Files:**
- Create: `/home/zn/flap-system-identification/docs/plans/2026-06-23-final-effective-force-model-freeze.md`
- Create: `/home/zn/flap-system-identification/scripts/select_final_fx_fz_alpha.py`
- Create: `/home/zn/flap-system-identification/tests/test_select_final_fx_fz_alpha.py`
- Read: `/home/zn/paper/AeroConf_effective_aero/research_notes/20260612_nested_outer_cv/`
- Read: `/home/zn/paper/AeroConf_effective_aero/sections/03_method.tex`

**Purpose:** Turn the paper's per-fold selection evidence into one deterministic deployment rule without using outer-test metrics to select the final alpha.

**Step 1: Write failing alpha-selection tests**

Test that the selector:

```python
def test_select_alpha_uses_only_inner_validation_rows(tmp_path):
    rows = make_selection_rows_with_misleading_outer_test_winner()
    selected = select_final_alpha(rows)
    assert selected.alpha == expected_inner_validation_winner
    assert selected.used_outer_test is False
```

Also test deterministic tie-breaking and rejection of mixed model families or phase conventions.

**Step 2: Run the tests and confirm the expected failure**

Run:

```bash
cd /home/zn/flap-system-identification
pytest tests/test_select_final_fx_fz_alpha.py -q
```

Expected: import or missing-function failure.

**Step 3: Implement the minimal selector**

The selector must aggregate inner-validation `fx_fz_mean` RMSE for only `phase_freq_q_gain_bias`, use a deterministic tie policy, and write:

```json
{
  "model_family": "phase_freq_q_gain_bias",
  "selected_alpha": 1.0,
  "selection_metric": "mean inner-validation fx_fz_mean RMSE",
  "uses_outer_test": false,
  "source_files": []
}
```

The shown alpha is illustrative; the script output is authoritative.

**Step 4: Run the selector on the nested-CV artifacts**

Run:

```bash
python scripts/select_final_fx_fz_alpha.py \
  --nested-root /home/zn/paper/AeroConf_effective_aero/research_notes/20260612_nested_outer_cv \
  --output artifacts/20260623_final_effective_force_fx_fz_v1/selection.json
```

Expected: one selected alpha, six outer folds represented, and `uses_outer_test=false`.

**Step 5: Write and review the freeze note**

Record the exact dataset root, selected shaped-prior parameters, geometry CSV, phase/frequency source, model family, alpha rule, target units, coordinate frames, and repository revisions. Explicitly state that nested-CV metrics remain the generalization evidence and the new all-log fit is deployment-only.

**Step 6: Commit the selection work**

```bash
git add docs/plans/2026-06-23-final-effective-force-model-freeze.md \
  scripts/select_final_fx_fz_alpha.py tests/test_select_final_fx_fz_alpha.py
git commit -m "feat: lock final effective force model selection"
```

### Task 2: Define a standalone frozen-model schema and reference inference

**Files:**
- Create: `/home/zn/flap-system-identification/src/system_identification/frozen_effective_force.py`
- Create: `/home/zn/flap-system-identification/tests/test_frozen_effective_force.py`
- Modify: `/home/zn/flap-system-identification/src/system_identification/__init__.py`

**Purpose:** Create a small canonical artifact contract that can be implemented independently in IsaacLab.

**Step 1: Write failing schema and prediction tests**

Cover:

```python
def test_frozen_model_rejects_reordered_features(): ...
def test_frozen_model_predicts_gain_bias_channels(): ...
def test_frozen_model_round_trip_preserves_predictions(tmp_path): ...
def test_frozen_model_checksum_detects_numerical_payload_change(tmp_path): ...
```

Use a hand-computable model with two samples and known gain/bias results.

**Step 2: Run the tests and verify failure**

```bash
pytest tests/test_frozen_effective_force.py -q
```

Expected: missing module.

**Step 3: Implement the frozen schema**

Provide dataclasses or validated dictionaries for:

```python
@dataclass(frozen=True)
class FrozenRidgeChannel:
    target: str
    design_columns: tuple[str, ...]
    feature_fill: tuple[float, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

@dataclass(frozen=True)
class FrozenEffectiveForceModel:
    schema_version: int
    feature_columns: tuple[str, ...]
    channels: tuple[FrozenRidgeChannel, FrozenRidgeChannel]
    prior: dict[str, object]
    conventions: dict[str, object]
    envelope: dict[str, object]
    lineage: dict[str, object]
```

Implement JSON serialization, SHA-256 over canonical numerical payload, strict feature ordering, and NumPy reference inference.

**Step 4: Verify the schema tests pass**

```bash
pytest tests/test_frozen_effective_force.py -q
```

Expected: all pass.

**Step 5: Commit**

```bash
git add src/system_identification/frozen_effective_force.py \
  src/system_identification/__init__.py tests/test_frozen_effective_force.py
git commit -m "feat: define frozen effective force artifact"
```

### Task 3: Refit on all retained logs and export the immutable artifact

**Files:**
- Create: `/home/zn/flap-system-identification/scripts/finalize_fx_fz_effective_force_model.py`
- Create: `/home/zn/flap-system-identification/tests/test_finalize_fx_fz_effective_force_model.py`
- Reuse: `/home/zn/flap-system-identification/scripts/train_fx_fz_structured_correction.py`
- Reuse: `/home/zn/flap-system-identification/scripts/train_deployable_wrench_correction_v2.py`
- Output: `/home/zn/flap-system-identification/artifacts/20260623_final_effective_force_fx_fz_v1/`

**Purpose:** Produce the actual deployment coefficients and enough evidence to reproduce them.

**Step 1: Write failing all-log-fit tests**

Test that the finalizer:

- concatenates train/val/test only after keyed alignment;
- refuses duplicate or missing sample keys;
- fits exactly `phase_freq_q_gain_bias` with the selected alpha;
- exports two 22-column ridge designs, totaling 44 scalar coefficients;
- writes feature/prior/convention/envelope/lineage metadata;
- writes deterministic reference cases and predictions;
- never reports all-log fit metrics as held-out metrics.

**Step 2: Run tests and verify failure**

```bash
pytest tests/test_finalize_fx_fz_effective_force_model.py -q
```

Expected: missing finalizer.

**Step 3: Implement the finalizer using existing feature and design helpers**

Use `build_v2_feature_frame`, `_with_intercept`, `_gain_bias_design`, and the existing ridge fit path. Do not reimplement phase feature construction. Save:

```text
model.json
selection.json
manifest.json
reference_cases.json
reference_predictions.csv
all_log_fit_metrics.csv
README.md
```

**Step 4: Run the real finalization command**

```bash
python scripts/finalize_fx_fz_effective_force_model.py \
  --split-root dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1 \
  --prior-root artifacts/20260604_delaurier_other_parameter_sweep_fixed_twist10_ratio8_sg0p03_v1/priors/separation__twist_eta_max_deg_10p0__alpha0_deg_4p0__eta_s_0p65__cd_f_0p0__enable_separation_sep_on__alpha_stall_max_deg_18p0__cd_cf_1p2__xi_1p0 \
  --selection artifacts/20260623_final_effective_force_fx_fz_v1/selection.json \
  --output-root artifacts/20260623_final_effective_force_fx_fz_v1
```

Expected: 29 unique logs, 448960 retained samples, exact paper feature family, no stale phase source, and a valid checksum. If current regenerated artifacts no longer match these row counts or lineage, stop and update the freeze note instead of forcing the command through.

**Step 5: Re-load the artifact and verify reference parity**

```bash
pytest tests/test_frozen_effective_force.py \
  tests/test_finalize_fx_fz_effective_force_model.py -q
```

Expected: max absolute reference prediction error at float64 tolerance and no schema warnings.

**Step 6: Commit code, not generated bulk predictions**

```bash
git add scripts/finalize_fx_fz_effective_force_model.py \
  tests/test_finalize_fx_fz_effective_force_model.py
git commit -m "feat: export final effective force model"
```

### Task 4: Vendor and validate the model artifact in IsaacLab

**Files:**
- Create: `source/flapping_bot/flapping_bot/config/models/effective_force_fx_fz_v1.json`
- Create: `source/flapping_bot/flapping_bot/config/models/effective_force_fx_fz_v1.references.json`
- Create: `source/flapping_bot/flapping_bot/config/models/README.md`
- Create: `tests/test_effective_force_model_artifact.py`

**Purpose:** Make IsaacLab inference independent of the training checkout while retaining exact provenance.

**Step 1: Write failing artifact-contract tests**

Test that the vendored files exist, parse, expose exactly the two targets, use the expected feature order and conventions, and match the source checksum recorded in `README.md`.

**Step 2: Run the targeted test and verify failure**

```bash
./isaaclab.sh -p -m pytest tests/test_effective_force_model_artifact.py -q
```

Expected: missing artifact.

**Step 3: Copy only the immutable inference payload and reference cases**

Copy from the reviewed system-identification output. Do not vendor training parquets, metrics tables, plots, or absolute-path-dependent files. The README records the source artifact path, checksum, command, model scope, and limitation that moments are not corrected.

**Step 4: Run contract tests**

```bash
./isaaclab.sh -p -m pytest tests/test_effective_force_model_artifact.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/config/models tests/test_effective_force_model_artifact.py
git commit -m "data: vendor frozen effective force model"
```

### Task 5: Implement independent batched Torch inference

**Files:**
- Create: `source/flapping_bot/flapping_bot/physics/effective_force_correction.py`
- Modify: `source/flapping_bot/flapping_bot/physics/__init__.py`
- Create: `tests/test_effective_force_correction.py`

**Purpose:** Evaluate the artifact on the Isaac device without pandas, scikit-learn, CPU transfers, or training-repository imports.

**Step 1: Write failing pure-Torch tests**

Cover:

```python
def test_phase_mapping_matches_sine_based_logged_convention(): ...
def test_flu_to_frd_pitch_rate_sign(): ...
def test_torch_inference_matches_vendored_reference_cases(): ...
def test_inference_preserves_batch_device_and_dtype(): ...
def test_nonfinite_input_returns_fallback_mask(): ...
def test_envelope_status_is_reported_per_environment(): ...
```

The phase test must encode the current Isaac cosine stroke and the logged sine-based zero convention explicitly.

**Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tests/test_effective_force_correction.py -q
```

Expected: missing module.

**Step 3: Implement the minimal inference module**

Provide:

```python
class EffectiveLongitudinalForceModel:
    @classmethod
    def from_json(cls, path: Path, device: torch.device) -> "EffectiveLongitudinalForceModel": ...

    def predict_frd(
        self,
        prior_fx_fz_frd: torch.Tensor,
        phase_isaac_rad: torch.Tensor,
        flap_frequency_hz: torch.Tensor,
        pitch_rate_flu_rad_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
```

The second returned tensor is the validity/envelope mask. Register coefficients as immutable Torch tensors and perform no file I/O after initialization.

**Step 4: Run reference tests**

```bash
./isaaclab.sh -p -m pytest \
  tests/test_effective_force_model_artifact.py \
  tests/test_effective_force_correction.py -q
```

Expected: Torch/reference max absolute error within the recorded float32 tolerance.

**Step 5: Commit**

```bash
git add source/flapping_bot/flapping_bot/physics/effective_force_correction.py \
  source/flapping_bot/flapping_bot/physics/__init__.py \
  tests/test_effective_force_correction.py
git commit -m "feat: add effective force inference"
```

### Task 6: Add shadow and `Fx/Fz` replacement composition

**Files:**
- Modify: `source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py`
- Create: `tests/test_effective_force_wrench_composition.py`
- Modify: `tests/test_path_tracking_env_contract.py`

**Purpose:** Integrate the model after all component forces and moments are computed, while making the unchanged-moment contract directly testable.

**Step 1: Write failing pure composition tests**

Extract a pure helper and test:

```python
def test_replace_fx_fz_preserves_fy_and_all_moments():
    force_final, torque_final = compose_effective_longitudinal_wrench(
        force_sim_flu, torque_sim_flu, force_effective_frd, mode="replace_fx_fz"
    )
    assert torch.equal(force_final[:, 1], force_sim_flu[:, 1])
    assert torch.equal(torque_final, torque_sim_flu)

def test_shadow_applies_original_wrench(): ...
def test_disabled_does_not_require_artifact(): ...
def test_invalid_prediction_falls_back_to_raw_force(): ...
```

**Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tests/test_effective_force_wrench_composition.py -q
```

Expected: missing helper/config.

**Step 3: Add configuration fields and runtime caches**

Add disabled-by-default fields:

```python
effective_force_mode: str = "disabled"
effective_force_model_path: Path | None = None
effective_force_out_of_envelope_policy: str = "fallback_raw"
```

Add debug caches for raw total force, prior `Fx/Fz`, predicted effective force, applied force, raw/applied moment, valid mask, and envelope mask.

**Step 4: Compose the final wrench after existing component calculations**

Keep:

```python
force_sim = f_w_sum + f_tail + f_drag
torque_sim = tau_w_sum + tau_tail
```

Then predict from the shaped wing prior, convert FRD/FLU, and call the pure composition helper. In `shadow`, log but apply `force_sim/torque_sim`. In `replace_fx_fz`, overwrite only force columns 0 and 2 and pass `torque_sim` unchanged.

**Step 5: Add artifact/prior compatibility checks at initialization**

Reject mismatched DeLaurier parameters, geometry identity, density assumptions, phase convention, or unsupported mode values before simulation begins.

**Step 6: Run targeted tests**

```bash
./isaaclab.sh -p -m pytest \
  tests/test_effective_force_wrench_composition.py \
  tests/test_effective_force_correction.py \
  tests/test_path_tracking_env_contract.py -q
```

Expected: pass, including exact unchanged-moment assertions.

**Step 7: Commit**

```bash
git add source/flapping_bot/flapping_bot/direct/flapping_bot/straight_flight_env.py \
  tests/test_effective_force_wrench_composition.py tests/test_path_tracking_env_contract.py
git commit -m "feat: replace longitudinal force with frozen model"
```

### Task 7: Add shadow, sweep, and runtime-validation tooling

**Files:**
- Create: `scripts/flapping_px4/validate_effective_force_model.py`
- Create: `tests/test_validate_effective_force_model.py`
- Create: `docs/analysis/flapping_px4/effective_force_validation/README.md`

**Purpose:** Produce reviewable evidence before allowing the model to affect long controller rollouts.

**Step 1: Write failing validation-helper tests**

Test deterministic grids, metrics aggregation, NaN/spike detection, envelope counts, output naming, and raw/shadow/replacement comparison tables.

**Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tests/test_validate_effective_force_model.py -q
```

Expected: missing script/helpers.

**Step 3: Implement three bounded validation modes**

- `reference`: replay vendored reference cases through Torch inference;
- `sweep`: phase/frequency/airspeed/pitch-rate grid within the retained envelope;
- `rollout`: run `disabled`, `shadow`, and `replace_fx_fz` with identical seeds and commands.

Write timestamped outputs under:

```text
artifacts/YYYYMMDD_HHMMSS_effective_force_validation/
```

Include CSV/JSON metrics and force/torque traces. The report must show that shadow and disabled applied wrenches match and that replacement moments match raw moments.

**Step 4: Run reference and sweep validation**

```bash
./isaaclab.sh -p scripts/flapping_px4/validate_effective_force_model.py \
  --mode reference --headless
./isaaclab.sh -p scripts/flapping_px4/validate_effective_force_model.py \
  --mode sweep --headless
```

Expected: reference tolerance passes, no in-envelope nonfinite predictions, and plots/tables exist.

**Step 5: Run identical-seed rollout comparison**

```bash
./isaaclab.sh -p scripts/flapping_px4/validate_effective_force_model.py \
  --mode rollout --seed 0 --steps 2400 --headless
```

Expected: three mode rows, no NaNs, explicit envelope/fallback counts, and unchanged applied moments for shadow/replacement composition.

**Step 6: Commit tooling and documentation**

```bash
git add scripts/flapping_px4/validate_effective_force_model.py \
  tests/test_validate_effective_force_model.py \
  docs/analysis/flapping_px4/effective_force_validation/README.md
git commit -m "test: add effective force validation workflow"
```

### Task 8: Run local-dynamics and controller-in-the-loop credibility gates

**Files:**
- Create: `scripts/flapping_px4/eval_effective_force_credibility.py`
- Create: `tests/test_eval_effective_force_credibility.py`
- Output: `artifacts/YYYYMMDD_HHMMSS_effective_force_credibility/`
- Update: `docs/analysis/flapping_px4/effective_force_validation/README.md`

**Purpose:** Determine what the corrected robot model is demonstrably useful for, without converting force-model evidence into an unsupported six-axis claim.

**Step 1: Write failing metric/gate tests**

Test aggregation of:

- trim existence and equilibrium error;
- speed/height/attitude/rate transient metrics;
- completion and termination;
- control effort and saturation;
- envelope occupancy and fallback rate;
- raw-versus-corrected paired comparison;
- separate longitudinal and rotational verdicts.

**Step 2: Run tests and verify failure**

```bash
./isaaclab.sh -p -m pytest tests/test_eval_effective_force_credibility.py -q
```

Expected: missing evaluator.

**Step 3: Implement the evaluator without RL dependencies**

Support:

- trim/open-loop cases over central retained-envelope conditions;
- short-horizon replay hooks for logged initial state and controls;
- existing PX4-like straight-flight and recovery controller cases;
- identical seeds and initial conditions for raw and corrected plants.

The output verdict has five explicit fields:

```text
numerical_integration
longitudinal_force_support
short_horizon_longitudinal_dynamics
controller_in_loop_behavior
rotational_fidelity_scope
```

`rotational_fidelity_scope` must state that moment remains inherited from the simulator and is not validated by the effective-force replacement.

**Step 4: Run the nominal credibility battery**

```bash
./isaaclab.sh -p scripts/flapping_px4/eval_effective_force_credibility.py \
  --seed 0 --headless
```

Expected: raw and corrected reports, trace artifacts, envelope counts, and no RL training.

**Step 5: Review results before defining numerical go/no-go thresholds**

Do not tune thresholds after seeing only the corrected result. Record the paired raw/corrected evidence, identify missing real-log replay inputs, and agree on the next validation battery. A model that merely flies is not automatically credible.

**Step 6: Commit the evaluator**

```bash
git add scripts/flapping_px4/eval_effective_force_credibility.py \
  tests/test_eval_effective_force_credibility.py \
  docs/analysis/flapping_px4/effective_force_validation/README.md
git commit -m "test: evaluate corrected plant credibility"
```

### Task 9: Run final regression and publish the bounded handoff

**Files:**
- Update: `docs/analysis/flapping_px4/effective_force_validation/README.md`
- Create: `docs/analysis/flapping_px4/effective_force_validation/credibility_report.md`

**Purpose:** Verify repository health and document exactly what is ready before discussing RL.

**Step 1: Run system-identification regression**

```bash
cd /home/zn/flap-system-identification
pytest tests/test_select_final_fx_fz_alpha.py \
  tests/test_frozen_effective_force.py \
  tests/test_finalize_fx_fz_effective_force_model.py \
  tests/test_deployable_wrench_correction_v2.py \
  tests/test_fx_fz_structured_correction.py -q
```

Expected: all pass.

**Step 2: Run Isaac targeted regression**

```bash
cd /home/zn/IsaacLab
./isaaclab.sh -p -m pytest \
  tests/test_effective_force_model_artifact.py \
  tests/test_effective_force_correction.py \
  tests/test_effective_force_wrench_composition.py \
  tests/test_validate_effective_force_model.py \
  tests/test_eval_effective_force_credibility.py \
  tests/test_path_tracking_env_contract.py -q
```

Expected: all pass.

**Step 3: Run the broader flapping test suite**

```bash
./isaaclab.sh -p -m pytest tests -k "flapping or delaurier or effective_force or path_tracking" -q
```

Expected: no new failures relative to the recorded pre-change baseline.

**Step 4: Write the credibility report**

The report includes:

- source artifact and checksum;
- model scope and exact replacement semantics;
- parity and regression commands/results;
- force-surface and runtime findings;
- trim, replay, and controller findings;
- envelope and fallback rates;
- unresolved moment-model limitations;
- explicit RL go/no-go recommendation.

**Step 5: Commit the bounded handoff**

```bash
git add docs/analysis/flapping_px4/effective_force_validation
git commit -m "docs: report effective force credibility"
```

## Stop conditions

Stop and report instead of continuing when any of the following occurs:

- the final all-log dataset or prior cannot be proven to match the paper conventions;
- alpha selection reads outer-test metrics;
- the vendored artifact fails reference parity;
- active DeLaurier parameters differ from the frozen prior contract;
- phase or FRD/FLU sign tests are ambiguous;
- `replace_fx_fz` changes `Fy` or any moment component;
- NaN, unbounded extrapolation, or hidden fallback appears;
- validation requires changing the moment model or starting RL.

## Execution handoff

Execute Tasks 1-3 in the system-identification worktree, review the generated artifact, then execute Tasks 4-9 in the IsaacLab worktree. Do not begin Isaac integration before the artifact review checkpoint, and do not begin RL after Task 9 without a separate approved design.

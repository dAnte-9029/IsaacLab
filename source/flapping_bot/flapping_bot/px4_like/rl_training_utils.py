"""Utilities for teacher-guided RL training schedules and action transforms."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import torch
import torch.nn as nn

Tensor = torch.Tensor
_SUPPORTED_TEACHER_GUIDANCE_MODES = ("envelope", "residual")
ACTOR_DISTILLATION_MASK_KEY = "actor_distillation_mask"


class ActorPolicyDistillationAugmentor:
    """Provide frozen teacher targets through RSL-RL's auxiliary actor-loss hook."""

    def __init__(self, policy: object, *, mask_key: str = ACTOR_DISTILLATION_MASK_KEY) -> None:
        if bool(getattr(policy, "is_recurrent", False)):
            raise ValueError("Actor policy distillation does not support recurrent policies.")
        if bool(getattr(policy, "state_dependent_std", False)):
            raise ValueError("Actor policy distillation requires a state-independent action standard deviation.")

        actor = getattr(policy, "actor", None)
        actor_normalizer = getattr(policy, "actor_obs_normalizer", None)
        obs_groups = getattr(policy, "obs_groups", None)
        if actor is None or actor_normalizer is None or not isinstance(obs_groups, dict):
            raise ValueError("policy does not expose the actor observation path required for distillation.")

        self.teacher_actor = copy.deepcopy(actor).eval()
        self.teacher_actor_normalizer = copy.deepcopy(actor_normalizer).eval()
        self.policy_obs_groups = tuple(obs_groups["policy"])
        self.mask_key = str(mask_key)
        self._teacher_actions: Tensor | None = None
        self._mask: Tensor | None = None
        for parameter in self.teacher_actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.teacher_actor_normalizer.parameters():
            parameter.requires_grad_(False)

    def __call__(
        self,
        *,
        obs: object | None,
        actions: Tensor | None,
        env: object,
    ) -> tuple[object | None, Tensor | None]:
        """Duplicate observations and replace old-task targets with teacher means."""
        del env
        if (obs is None) == (actions is None):
            raise ValueError("exactly one of obs or actions must be provided.")

        if obs is not None:
            if self.mask_key not in obs:
                raise ValueError(f"distillation mask observation {self.mask_key!r} is missing.")
            mask = obs[self.mask_key]
            if mask.ndim != 2 or mask.shape[1] != 1:
                raise ValueError("actor distillation mask must have shape (batch, 1).")
            actor_obs = torch.cat([obs[group] for group in self.policy_obs_groups], dim=-1)
            with torch.inference_mode():
                normalized_obs = self.teacher_actor_normalizer(actor_obs)
                self._teacher_actions = self.teacher_actor(normalized_obs)
            self._mask = mask.to(dtype=torch.bool)
            return torch.cat((obs, obs), dim=0), None

        if self._teacher_actions is None or self._mask is None:
            raise RuntimeError("distillation observations must be processed before action targets.")
        assert actions is not None
        if self._teacher_actions.shape != actions.shape or self._mask.shape[0] != actions.shape[0]:
            raise ValueError("cached teacher targets do not match the student action batch.")
        target_actions = torch.where(self._mask, self._teacher_actions, actions.detach())
        augmented_actions = torch.cat((actions, target_actions), dim=0)
        self._teacher_actions = None
        self._mask = None
        return None, augmented_actions


def maybe_enable_actor_policy_distillation(*, runner: object) -> bool:
    """Attach a frozen actor teacher after weights-only policy loading."""
    env = getattr(runner, "env", None)
    algorithm = getattr(runner, "alg", None)
    cfg = getattr(env, "cfg", None)
    coefficient = float(getattr(cfg, "pure_rl_actor_distillation_coefficient", 0.0))
    if coefficient == 0.0:
        return False
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("actor distillation coefficient must be finite and non-negative.")
    if algorithm is None or getattr(algorithm, "policy", None) is None:
        raise ValueError("runner does not expose a PPO policy for actor distillation.")
    if getattr(algorithm, "symmetry", None) is not None:
        raise ValueError("actor distillation cannot share PPO's auxiliary symmetry-loss slot.")

    storage = getattr(algorithm, "storage", None)
    stored_observations = getattr(storage, "observations", None)
    if stored_observations is None or ACTOR_DISTILLATION_MASK_KEY not in stored_observations:
        raise ValueError("rollout storage does not contain the actor distillation mask.")

    augmentor = ActorPolicyDistillationAugmentor(algorithm.policy)
    algorithm.symmetry = {
        "use_data_augmentation": False,
        "use_mirror_loss": True,
        "data_augmentation_func": augmentor,
        "mirror_loss_coeff": coefficient,
        "_env": env,
    }
    return True


def linear_anneal(step: int, *, start: float, end: float, duration_steps: int) -> float:
    """Linearly interpolate from ``start`` to ``end`` over ``duration_steps``."""
    if duration_steps <= 0:
        return float(end)
    alpha = min(max(float(step) / float(duration_steps), 0.0), 1.0)
    return float(start) + (float(end) - float(start)) * alpha


def piecewise_linear_anneal(step: int, *, steps: Sequence[int], values: Sequence[float]) -> float:
    """Linearly interpolate across a sequence of schedule knots."""
    if len(steps) != len(values):
        raise ValueError("steps and values must have the same length.")
    if len(steps) == 0:
        raise ValueError("steps and values must not be empty.")
    if any(int(steps[i]) > int(steps[i + 1]) for i in range(len(steps) - 1)):
        raise ValueError("steps must be monotonically non-decreasing.")

    step = int(step)
    if step <= int(steps[0]):
        return float(values[0])

    for idx in range(len(steps) - 1):
        left_step = int(steps[idx])
        right_step = int(steps[idx + 1])
        left_val = float(values[idx])
        right_val = float(values[idx + 1])

        if step <= right_step:
            duration = max(right_step - left_step, 1)
            alpha = float(step - left_step) / float(duration)
            return left_val + (right_val - left_val) * alpha

    return float(values[-1])


def teacher_guidance_is_active(step: int, *, enabled: bool, disable_after_steps: int) -> bool:
    """Return whether teacher guidance should still be applied at ``step``."""
    if not enabled:
        return False
    if int(disable_after_steps) >= 0 and int(step) >= int(disable_after_steps):
        return False
    return True


def resolve_teacher_guidance_mode(mode: str) -> str:
    """Validate and normalize the teacher-guidance action semantics."""
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in _SUPPORTED_TEACHER_GUIDANCE_MODES:
        raise ValueError(
            f"unsupported teacher guidance mode: {mode!r}. Expected one of {_SUPPORTED_TEACHER_GUIDANCE_MODES}."
        )
    return resolved_mode


def compute_teacher_guidance_delta(
    step: int,
    *,
    enabled: bool,
    delta_init: float,
    delta_final: float,
    anneal_steps: int,
    schedule_steps: Sequence[int],
    schedule_deltas: Sequence[float],
    disable_after_steps: int,
) -> float:
    """Compute the current teacher-action envelope width."""
    if not teacher_guidance_is_active(step, enabled=enabled, disable_after_steps=disable_after_steps):
        return float(delta_final)

    if len(schedule_steps) > 0 or len(schedule_deltas) > 0:
        if len(schedule_steps) != len(schedule_deltas):
            raise ValueError("schedule_steps and schedule_deltas must have the same length.")
        return piecewise_linear_anneal(step, steps=schedule_steps, values=schedule_deltas)

    return linear_anneal(step, start=delta_init, end=delta_final, duration_steps=anneal_steps)


def apply_teacher_action_envelope(teacher_actions: Tensor, rl_actions: Tensor, delta: float | Tensor) -> Tensor:
    """Limit executed actions to a bounded deviation around teacher actions."""
    if teacher_actions.shape != rl_actions.shape:
        raise ValueError("teacher_actions and rl_actions must have the same shape.")

    if not torch.is_tensor(delta):
        delta = torch.tensor(delta, dtype=teacher_actions.dtype, device=teacher_actions.device)
    else:
        delta = delta.to(device=teacher_actions.device, dtype=teacher_actions.dtype)

    bounded_delta = torch.clamp(rl_actions - teacher_actions, min=-delta, max=delta)
    return torch.clamp(teacher_actions + bounded_delta, min=-1.0, max=1.0)


def apply_teacher_residual_action(teacher_actions: Tensor, rl_residual_actions: Tensor, delta: float | Tensor) -> Tensor:
    """Execute teacher action plus a bounded policy residual."""
    if teacher_actions.shape != rl_residual_actions.shape:
        raise ValueError("teacher_actions and rl_residual_actions must have the same shape.")

    if not torch.is_tensor(delta):
        delta = torch.tensor(delta, dtype=teacher_actions.dtype, device=teacher_actions.device)
    else:
        delta = delta.to(device=teacher_actions.device, dtype=teacher_actions.dtype)

    return torch.clamp(teacher_actions + rl_residual_actions * delta, min=-1.0, max=1.0)


def apply_teacher_guided_actions(
    teacher_actions: Tensor,
    rl_actions: Tensor,
    *,
    delta: float | Tensor,
    mode: str,
) -> Tensor:
    """Apply one supported teacher-guidance action transform."""
    resolved_mode = resolve_teacher_guidance_mode(mode)
    if resolved_mode == "envelope":
        return apply_teacher_action_envelope(teacher_actions, rl_actions, delta)
    if resolved_mode == "residual":
        return apply_teacher_residual_action(teacher_actions, rl_actions, delta)
    raise AssertionError(f"unreachable teacher guidance mode: {resolved_mode}")


def should_bootstrap_teacher_guided_policy(
    *,
    teacher_guidance_enabled: bool,
    teacher_guidance_mode: str,
    zero_actor_init: bool,
    is_resume: bool,
) -> bool:
    """Return whether a fresh residual teacher-guided run should start from zero residual output."""
    return bool(teacher_guidance_enabled) and resolve_teacher_guidance_mode(teacher_guidance_mode) == "residual" and bool(
        zero_actor_init
    ) and not bool(is_resume)


def _zero_last_linear_layer(module: nn.Module) -> None:
    """Zero the parameters of the last linear layer in a module."""
    for submodule in reversed(list(module.modules())):
        if isinstance(submodule, nn.Linear):
            nn.init.zeros_(submodule.weight)
            nn.init.zeros_(submodule.bias)
            return
    raise ValueError("module does not contain a linear layer to zero-initialize.")


def maybe_bootstrap_teacher_guided_policy(
    *,
    policy: object,
    teacher_guidance_enabled: bool,
    teacher_guidance_mode: str,
    zero_actor_init: bool,
    is_resume: bool,
) -> bool:
    """Zero-initialize the actor mean head for fresh residual teacher-guided runs."""
    if not should_bootstrap_teacher_guided_policy(
        teacher_guidance_enabled=teacher_guidance_enabled,
        teacher_guidance_mode=teacher_guidance_mode,
        zero_actor_init=zero_actor_init,
        is_resume=is_resume,
    ):
        return False

    actor = getattr(policy, "actor", None)
    if actor is None:
        raise ValueError("policy does not expose an actor module for bootstrap initialization.")
    _zero_last_linear_layer(actor)
    return True


def load_runner_checkpoint_for_warm_start(
    *,
    runner: object,
    checkpoint_path: str,
    map_location: str | None = None,
) -> dict | None:
    """Load checkpoint weights without restoring optimizer or learning iteration state."""
    load_fn = getattr(runner, "load", None)
    if load_fn is None:
        raise ValueError("runner does not expose a load() method.")
    if not hasattr(runner, "current_learning_iteration"):
        raise ValueError("runner does not expose current_learning_iteration.")

    infos = load_fn(checkpoint_path, load_optimizer=False, map_location=map_location)
    runner.current_learning_iteration = 0
    if maybe_enable_actor_policy_distillation(runner=runner):
        coefficient = float(runner.env.cfg.pure_rl_actor_distillation_coefficient)
        print(
            "[INFO]: Enabled frozen actor policy distillation after warm start "
            f"(coefficient={coefficient:g})."
        )
    return infos


def should_randomize_initial_episode_length(*, task: str | None, load_weights_only: bool) -> bool:
    """Return whether training should randomize vector-env episode offsets at startup."""
    task_name = "" if task is None else str(task)
    if bool(load_weights_only) and "PathTracking" in task_name:
        return False
    return True


def compute_recovery_teacher_mask(
    *,
    lateral_error: Tensor,
    height_error: Tensor,
    airspeed: Tensor,
    tilt_deg: Tensor,
    ang_rate_deg_s: Tensor,
    lateral_error_trigger_m: float,
    height_error_trigger_m: float,
    min_safe_airspeed_mps: float,
    tilt_trigger_deg: float,
    ang_rate_trigger_deg_s: float,
) -> Tensor:
    """Return a boolean mask for states that should use recovery teacher guidance."""
    return (
        (torch.abs(lateral_error) > lateral_error_trigger_m)
        | (torch.abs(height_error) > height_error_trigger_m)
        | (airspeed < min_safe_airspeed_mps)
        | (torch.abs(tilt_deg) > tilt_trigger_deg)
        | (torch.abs(ang_rate_deg_s) > ang_rate_trigger_deg_s)
    )

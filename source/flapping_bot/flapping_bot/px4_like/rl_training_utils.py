"""Utilities for teacher-guided RL training schedules and action transforms."""

from __future__ import annotations

import csv
import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn

Tensor = torch.Tensor
_SUPPORTED_TEACHER_GUIDANCE_MODES = ("envelope", "residual")
ACTOR_DISTILLATION_MASK_KEY = "actor_distillation_mask"
ACTOR_GRADIENT_PROBE_GROUP_KEY = "actor_gradient_probe_group"
ACTOR_GRADIENT_PROBE_C1_GROUP = 0
ACTOR_GRADIENT_PROBE_C2C_GROUP = 1
ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP = 2
ACTOR_GRADIENT_PROBE_C3A_GROUP = 3
ACTOR_GRADIENT_PROBE_ACTIVE_C3A_GROUP = 4
TASK_AWARE_C3B_GROUP = 5
_LARGE_ACTOR_HIDDEN_DIMS = (512, 256)
_NET2WIDER_PRIMARY_FRACTION = 0.55


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


class ActorGradientConflictProbe:
    """Measure task-specific PPO actor gradients without changing the update."""

    def __init__(
        self,
        algorithm: object,
        *,
        output_path: str | Path,
        interval: int,
        minimum_samples: int,
    ) -> None:
        policy = getattr(algorithm, "policy", None)
        storage = getattr(algorithm, "storage", None)
        if policy is None or storage is None:
            raise ValueError("gradient probe requires a PPO policy and rollout storage.")
        if bool(getattr(policy, "is_recurrent", False)):
            raise ValueError("gradient probe does not support recurrent policies.")
        if bool(getattr(policy, "state_dependent_std", False)):
            raise ValueError("gradient probe requires a state-independent action standard deviation.")
        actor = getattr(policy, "actor", None)
        actor_normalizer = getattr(policy, "actor_obs_normalizer", None)
        get_actor_obs = getattr(policy, "get_actor_obs", None)
        if actor is None or actor_normalizer is None or get_actor_obs is None:
            raise ValueError("policy does not expose the actor observation path required by the gradient probe.")
        if isinstance(interval, bool) or int(interval) != interval or interval <= 0:
            raise ValueError("gradient probe interval must be a positive integer.")
        if isinstance(minimum_samples, bool) or int(minimum_samples) != minimum_samples or minimum_samples <= 0:
            raise ValueError("gradient probe minimum samples must be a positive integer.")

        self.algorithm = algorithm
        self.policy = policy
        self.storage = storage
        self.actor_parameters = tuple(actor.parameters())
        if not self.actor_parameters:
            raise ValueError("gradient probe actor has no trainable parameters.")
        self.output_path = Path(output_path)
        self.interval = int(interval)
        self.minimum_samples = int(minimum_samples)
        self.iteration = 0
        self._original_update = algorithm.update

    @staticmethod
    def _dot(left: tuple[Tensor, ...], right: tuple[Tensor, ...]) -> Tensor:
        return sum((left_part * right_part).sum() for left_part, right_part in zip(left, right))

    @classmethod
    def _norm(cls, gradient: tuple[Tensor, ...]) -> Tensor:
        return torch.sqrt(torch.clamp(cls._dot(gradient, gradient), min=0.0))

    @classmethod
    def _cosine(cls, left: tuple[Tensor, ...], right: tuple[Tensor, ...]) -> Tensor:
        denominator = cls._norm(left) * cls._norm(right)
        return cls._dot(left, right) / torch.clamp(denominator, min=1.0e-12)

    def _distribution_log_prob(self, observations: object, actions: Tensor) -> Tensor:
        mean, std = self._distribution_parameters(observations)
        return torch.distributions.Normal(mean, std).log_prob(actions).sum(dim=-1)

    def _distribution_parameters(self, observations: object) -> tuple[Tensor, Tensor]:
        actor_obs = self.policy.get_actor_obs(observations)
        actor_obs = self.policy.actor_obs_normalizer(actor_obs)
        mean = self.policy.actor(actor_obs)
        noise_std_type = str(getattr(self.policy, "noise_std_type", ""))
        if noise_std_type == "scalar":
            std = self.policy.std.expand_as(mean)
        elif noise_std_type == "log":
            std = torch.exp(self.policy.log_std).expand_as(mean)
        else:
            raise ValueError(f"unsupported actor noise standard-deviation type: {noise_std_type!r}.")
        return mean, std

    def _gradient_for_mask(
        self,
        *,
        mask: Tensor,
        name: str,
    ) -> tuple[tuple[Tensor, ...], dict[str, float]]:
        count = int(mask.sum().item())
        if count < self.minimum_samples:
            raise RuntimeError(
                f"gradient probe group {name!r} has {count} samples; "
                f"at least {self.minimum_samples} are required."
            )
        observations = self.storage.observations.flatten(0, 1)[mask]
        actions = self.storage.actions.flatten(0, 1)[mask]
        advantages = self.storage.advantages.flatten(0, 1)[mask].squeeze(-1)
        old_log_prob = self.storage.actions_log_prob.flatten(0, 1)[mask].squeeze(-1)
        new_log_prob = self._distribution_log_prob(observations, actions)
        ratio = torch.exp(new_log_prob - old_log_prob)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
            ratio,
            1.0 - float(self.algorithm.clip_param),
            1.0 + float(self.algorithm.clip_param),
        )
        loss = torch.maximum(surrogate, surrogate_clipped).mean()
        gradient = tuple(
            part.detach()
            for part in torch.autograd.grad(loss, self.actor_parameters, allow_unused=False)
        )
        metrics = {
            f"gradient_probe/count_{name}": float(count),
            f"gradient_probe/loss_{name}": float(loss.detach().item()),
            f"gradient_probe/norm_{name}": float(self._norm(gradient).item()),
            f"gradient_probe/advantage_mean_{name}": float(advantages.mean().item()),
            f"gradient_probe/advantage_std_{name}": float(advantages.std(unbiased=False).item()),
            f"gradient_probe/ratio_mean_{name}": float(ratio.detach().mean().item()),
            f"gradient_probe/clipped_fraction_{name}": float(
                ((ratio < 1.0 - self.algorithm.clip_param) | (ratio > 1.0 + self.algorithm.clip_param))
                .to(dtype=torch.float32)
                .mean()
                .item()
            ),
        }
        return gradient, metrics

    def _measure(self) -> tuple[dict[str, float], dict[str, tuple[Tensor, ...]]]:
        group = self.storage.observations[ACTOR_GRADIENT_PROBE_GROUP_KEY].flatten(0, 1).squeeze(-1)
        masks = {
            "c1": group == ACTOR_GRADIENT_PROBE_C1_GROUP,
            "c2c_all": (group == ACTOR_GRADIENT_PROBE_C2C_GROUP)
            | (group == ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP),
            "c2c_non_strong": group == ACTOR_GRADIENT_PROBE_C2C_GROUP,
            "c2c_strong": group == ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP,
            "c3a": (group == ACTOR_GRADIENT_PROBE_C3A_GROUP)
            | (group == ACTOR_GRADIENT_PROBE_ACTIVE_C3A_GROUP),
        }
        metrics: dict[str, float] = {}
        gradients: dict[str, tuple[Tensor, ...]] = {}
        for name, mask in masks.items():
            gradients[name], group_metrics = self._gradient_for_mask(mask=mask, name=name)
            metrics.update(group_metrics)

        for left, right in (
            ("c2c_all", "c3a"),
            ("c2c_strong", "c3a"),
            ("c2c_non_strong", "c3a"),
            ("c1", "c3a"),
        ):
            pair = f"{left}_vs_{right}"
            metrics[f"gradient_probe/dot_{pair}"] = float(self._dot(gradients[left], gradients[right]).item())
            metrics[f"gradient_probe/cos_{pair}"] = float(
                self._cosine(gradients[left], gradients[right]).item()
            )
        return metrics, gradients

    def _measure_post_update(self) -> dict[str, float]:
        group = self.storage.observations[ACTOR_GRADIENT_PROBE_GROUP_KEY].flatten(0, 1).squeeze(-1)
        masks = {
            "c1": group == ACTOR_GRADIENT_PROBE_C1_GROUP,
            "c2c_all": (group == ACTOR_GRADIENT_PROBE_C2C_GROUP)
            | (group == ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP),
            "c2c_strong": group == ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP,
            "c3a": (group == ACTOR_GRADIENT_PROBE_C3A_GROUP)
            | (group == ACTOR_GRADIENT_PROBE_ACTIVE_C3A_GROUP),
        }
        observations = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        old_log_prob = self.storage.actions_log_prob.flatten(0, 1).squeeze(-1)
        old_mean = getattr(self.storage, "mu", None)
        old_std = getattr(self.storage, "sigma", None)
        metrics: dict[str, float] = {}
        for name, mask in masks.items():
            count = int(mask.sum().item())
            if count < self.minimum_samples:
                continue
            mean, std = self._distribution_parameters(observations[mask])
            new_log_prob = torch.distributions.Normal(mean, std).log_prob(actions[mask]).sum(dim=-1)
            ratio = torch.exp(new_log_prob - old_log_prob[mask])
            metrics[f"gradient_probe/post_ratio_mean_{name}"] = float(ratio.detach().mean().item())
            metrics[f"gradient_probe/post_clipped_fraction_{name}"] = float(
                ((ratio < 1.0 - self.algorithm.clip_param) | (ratio > 1.0 + self.algorithm.clip_param))
                .to(dtype=torch.float32)
                .mean()
                .item()
            )
            metrics[f"gradient_probe/post_entropy_{name}"] = float(
                torch.distributions.Normal(mean, std).entropy().sum(dim=-1).detach().mean().item()
            )
            if old_mean is not None and old_std is not None:
                old_mean_batch = old_mean.flatten(0, 1)[mask]
                old_std_batch = old_std.flatten(0, 1)[mask]
                kl = torch.sum(
                    torch.log(std / old_std_batch + 1.0e-5)
                    + (torch.square(old_std_batch) + torch.square(old_mean_batch - mean))
                    / (2.0 * torch.square(std))
                    - 0.5,
                    dim=-1,
                )
                metrics[f"gradient_probe/post_kl_{name}"] = float(kl.detach().mean().item())
        return metrics

    def _write_csv(self, metrics: dict[str, float]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"iteration": self.iteration, **metrics}
        write_header = not self.output_path.exists()
        with self.output_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def update(self) -> dict[str, float]:
        """Run the original PPO update and attach non-invasive probe metrics."""

        warm_start_guard = getattr(self.algorithm, "ppo_warm_start_guard", None)
        if warm_start_guard is not None and (
            int(warm_start_guard.iteration) < int(warm_start_guard.burn_in_iterations)
        ):
            result = self._original_update()
            self.iteration += 1
            return result

        if self.iteration % self.interval != 0:
            result = self._original_update()
            self.iteration += 1
            return result

        metrics, gradients = self._measure()
        actor_before = tuple(parameter.detach().clone() for parameter in self.actor_parameters)
        result = self._original_update()
        actor_delta = tuple(
            parameter.detach() - before
            for parameter, before in zip(self.actor_parameters, actor_before)
        )
        delta_norm = self._norm(actor_delta)
        metrics["gradient_probe/actor_update_norm"] = float(delta_norm.item())
        metrics.update(self._measure_post_update())
        for name in ("c2c_all", "c2c_strong"):
            gradient = gradients[name]
            directional_derivative = self._dot(gradient, actor_delta)
            desired_alignment = -directional_derivative / torch.clamp(
                self._norm(gradient) * delta_norm,
                min=1.0e-12,
            )
            metrics[f"gradient_probe/update_directional_derivative_{name}"] = float(
                directional_derivative.item()
            )
            metrics[f"gradient_probe/update_alignment_{name}"] = float(desired_alignment.item())
        result.update(metrics)
        self._write_csv(metrics)
        self.iteration += 1
        return result


class TaskAwarePpoAdapter:
    """Balance curriculum-task actor updates while leaving the critic loss unchanged."""

    def __init__(
        self,
        algorithm: object,
        *,
        output_path: str | Path,
        task_weights: Sequence[float],
        minimum_task_samples: int,
        minimum_phase_samples: int,
    ) -> None:
        policy = getattr(algorithm, "policy", None)
        storage = getattr(algorithm, "storage", None)
        if policy is None or storage is None:
            raise ValueError("task-aware PPO requires a policy and rollout storage.")
        if bool(getattr(policy, "is_recurrent", False)):
            raise ValueError("task-aware PPO does not support recurrent policies.")
        if len(task_weights) not in (3, 4):
            raise ValueError("task-aware PPO requires C1/C2c/C3a or C1/C2c/C3a/C3b weights.")
        weights = tuple(float(value) for value in task_weights)
        if any((not math.isfinite(value)) or value <= 0.0 for value in weights):
            raise ValueError("task-aware PPO weights must be finite and positive.")
        weight_sum = sum(weights)
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("task-aware PPO weights must sum to one.")
        if (
            isinstance(minimum_task_samples, bool)
            or int(minimum_task_samples) != minimum_task_samples
            or minimum_task_samples <= 0
        ):
            raise ValueError("minimum_task_samples must be a positive integer.")
        if (
            isinstance(minimum_phase_samples, bool)
            or int(minimum_phase_samples) != minimum_phase_samples
            or minimum_phase_samples < 0
        ):
            raise ValueError("minimum_phase_samples must be a non-negative integer.")

        self.algorithm = algorithm
        self.storage = storage
        self.output_path = Path(output_path)
        self.task_weights = weights
        self.minimum_task_samples = int(minimum_task_samples)
        self.minimum_phase_samples = int(minimum_phase_samples)
        self.iteration = 0
        self._original_update = algorithm.update
        self._original_generator = storage.mini_batch_generator

    def _task_masks(self, group: Tensor) -> dict[str, Tensor]:
        masks = {
            "c1": group == ACTOR_GRADIENT_PROBE_C1_GROUP,
            "c2c": (group == ACTOR_GRADIENT_PROBE_C2C_GROUP)
            | (group == ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP),
            "c3a": (group == ACTOR_GRADIENT_PROBE_C3A_GROUP)
            | (group == ACTOR_GRADIENT_PROBE_ACTIVE_C3A_GROUP),
        }
        if len(self.task_weights) == 4:
            masks["c3b"] = group == TASK_AWARE_C3B_GROUP
        return masks

    def _prepare_advantages(self) -> dict[str, float]:
        group = self.storage.observations[ACTOR_GRADIENT_PROBE_GROUP_KEY].flatten(0, 1).squeeze(-1)
        raw_advantages = (self.storage.returns - self.storage.values).flatten(0, 1).squeeze(-1)
        normalized = torch.zeros_like(raw_advantages)
        masks = self._task_masks(group)
        metrics: dict[str, float] = {}
        total_count = int(group.numel())
        for (name, mask), desired_weight in zip(masks.items(), self.task_weights, strict=True):
            count = int(mask.sum().item())
            if count < self.minimum_task_samples:
                raise RuntimeError(
                    f"task-aware PPO group {name!r} has {count} samples; "
                    f"at least {self.minimum_task_samples} are required."
                )
            task_advantages = raw_advantages[mask]
            mean = task_advantages.mean()
            std = task_advantages.std(unbiased=False)
            empirical_weight = float(count) / float(total_count)
            scale = float(desired_weight) / empirical_weight
            normalized[mask] = (task_advantages - mean) / torch.clamp(std, min=1.0e-8) * scale
            metrics[f"task_aware/count_{name}"] = float(count)
            metrics[f"task_aware/raw_advantage_mean_{name}"] = float(mean.item())
            metrics[f"task_aware/raw_advantage_std_{name}"] = float(std.item())
            metrics[f"task_aware/empirical_weight_{name}"] = empirical_weight
            metrics[f"task_aware/advantage_scale_{name}"] = scale

        known = torch.zeros_like(group, dtype=torch.bool)
        for mask in masks.values():
            known |= mask
        if not bool(torch.all(known)):
            unknown = torch.unique(group[~known]).detach().cpu().tolist()
            raise RuntimeError(f"task-aware PPO received unknown task/phase groups: {unknown}.")
        self.storage.advantages = normalized.reshape_as(self.storage.advantages).clone()
        metrics["task_aware/count_c2c_strong_phase"] = float(
            (group == ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP).sum().item()
        )
        metrics["task_aware/count_c3a_active_phase"] = float(
            (group == ACTOR_GRADIENT_PROBE_ACTIVE_C3A_GROUP).sum().item()
        )
        return metrics

    def _validate_phase_coverage(self, metrics: dict[str, float]) -> None:
        if self.minimum_phase_samples == 0:
            return
        warm_start_guard = getattr(self.algorithm, "ppo_warm_start_guard", None)
        if warm_start_guard is not None and (
            int(warm_start_guard.iteration) < int(warm_start_guard.burn_in_iterations)
        ):
            return
        for name in ("c2c_strong_phase", "c3a_active_phase"):
            count = int(metrics[f"task_aware/count_{name}"])
            if count < self.minimum_phase_samples:
                raise RuntimeError(
                    f"task-aware PPO phase {name!r} has {count} samples; "
                    f"at least {self.minimum_phase_samples} are required."
                )

    def _stratified_generator(self, num_mini_batches: int, num_epochs: int = 8):
        group = self.storage.observations[ACTOR_GRADIENT_PROBE_GROUP_KEY].flatten(0, 1).squeeze(-1)
        observations = self.storage.observations.flatten(0, 1)
        tensors = (
            self.storage.actions.flatten(0, 1),
            self.storage.values.flatten(0, 1),
            self.storage.advantages.flatten(0, 1),
            self.storage.returns.flatten(0, 1),
            self.storage.actions_log_prob.flatten(0, 1),
            self.storage.mu.flatten(0, 1),
            self.storage.sigma.flatten(0, 1),
        )
        group_ids = [
            ACTOR_GRADIENT_PROBE_C1_GROUP,
            ACTOR_GRADIENT_PROBE_C2C_GROUP,
            ACTOR_GRADIENT_PROBE_STRONG_C2C_GROUP,
            ACTOR_GRADIENT_PROBE_C3A_GROUP,
            ACTOR_GRADIENT_PROBE_ACTIVE_C3A_GROUP,
        ]
        if len(self.task_weights) == 4:
            group_ids.append(TASK_AWARE_C3B_GROUP)
        strata = tuple(
            torch.nonzero(group == group_id, as_tuple=False).squeeze(-1)
            for group_id in group_ids
        )
        for _epoch in range(num_epochs):
            chunks = []
            for indices in strata:
                permutation = indices[torch.randperm(indices.numel(), device=indices.device)]
                chunks.append(torch.tensor_split(permutation, num_mini_batches))
            for batch_id in range(num_mini_batches):
                batch_idx = torch.cat([parts[batch_id] for parts in chunks if parts[batch_id].numel() > 0])
                batch_idx = batch_idx[torch.randperm(batch_idx.numel(), device=batch_idx.device)]
                yield (
                    observations[batch_idx],
                    *(tensor[batch_idx] for tensor in tensors),
                    (None, None),
                    None,
                )

    def _write_csv(self, metrics: dict[str, float]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"iteration": self.iteration, **metrics}
        write_header = not self.output_path.exists()
        with self.output_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def update(self) -> dict[str, float]:
        """Apply task-wise normalization and phase-stratified feedforward mini-batches."""

        metrics = self._prepare_advantages()
        self._validate_phase_coverage(metrics)
        self.storage.mini_batch_generator = self._stratified_generator
        try:
            result = self._original_update()
        finally:
            self.storage.mini_batch_generator = self._original_generator
        result.update(metrics)
        self._write_csv(metrics)
        self.iteration += 1
        return result


class PpoWarmStartGuard:
    """Delay and soften the first PPO updates after a weights-only warm start."""

    def __init__(
        self,
        algorithm: object,
        *,
        burn_in_iterations: int,
        warmup_update_iterations: int,
        initial_learning_rate: float,
        target_learning_rate: float,
        warmup_num_learning_epochs: int,
        actor_update_norm_limit: float,
    ) -> None:
        policy = getattr(algorithm, "policy", None)
        actor = getattr(policy, "actor", None)
        storage = getattr(algorithm, "storage", None)
        optimizer = getattr(algorithm, "optimizer", None)
        if actor is None or storage is None or optimizer is None:
            raise ValueError("PPO warm-start guard requires an actor, rollout storage, and optimizer.")
        if isinstance(burn_in_iterations, bool) or int(burn_in_iterations) != burn_in_iterations:
            raise ValueError("warm-start burn-in iterations must be a non-negative integer.")
        if burn_in_iterations < 0:
            raise ValueError("warm-start burn-in iterations must be a non-negative integer.")
        if (
            isinstance(warmup_update_iterations, bool)
            or int(warmup_update_iterations) != warmup_update_iterations
            or warmup_update_iterations <= 0
        ):
            raise ValueError("warm-start update iterations must be a positive integer.")
        if (
            isinstance(warmup_num_learning_epochs, bool)
            or int(warmup_num_learning_epochs) != warmup_num_learning_epochs
            or warmup_num_learning_epochs <= 0
        ):
            raise ValueError("warm-start learning epochs must be a positive integer.")
        if not math.isfinite(initial_learning_rate) or initial_learning_rate <= 0.0:
            raise ValueError("warm-start initial learning rate must be finite and positive.")
        if not math.isfinite(target_learning_rate) or target_learning_rate < initial_learning_rate:
            raise ValueError("warm-start target learning rate must be finite and at least the initial rate.")
        if not math.isfinite(actor_update_norm_limit) or actor_update_norm_limit <= 0.0:
            raise ValueError("warm-start actor update norm limit must be finite and positive.")

        self.algorithm = algorithm
        self.storage = storage
        self.optimizer = optimizer
        self.actor_parameters = tuple(actor.parameters())
        if not self.actor_parameters:
            raise ValueError("PPO warm-start guard actor has no trainable parameters.")
        self.burn_in_iterations = int(burn_in_iterations)
        self.warmup_update_iterations = int(warmup_update_iterations)
        self.initial_learning_rate = float(initial_learning_rate)
        self.target_learning_rate = float(target_learning_rate)
        self.warmup_num_learning_epochs = int(warmup_num_learning_epochs)
        self.actor_update_norm_limit = float(actor_update_norm_limit)
        self.original_num_learning_epochs = int(getattr(algorithm, "num_learning_epochs"))
        self.iteration = 0
        self._original_update = algorithm.update

        # Keep the cross-stage run on a bounded fixed schedule. RSL-RL's adaptive
        # schedule otherwise raises the learning rate on the first zero-KL minibatch.
        self.algorithm.schedule = "fixed"
        self._set_learning_rate(self.initial_learning_rate)

    @staticmethod
    def _parameter_delta_norm(before: tuple[Tensor, ...], after: tuple[Tensor, ...]) -> float:
        squared_norm = sum(torch.square(current - previous).sum() for previous, current in zip(before, after))
        return float(torch.sqrt(torch.clamp(squared_norm, min=0.0)).item())

    def _set_learning_rate(self, learning_rate: float) -> None:
        self.algorithm.learning_rate = float(learning_rate)
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)

    def _empty_update_result(self) -> dict[str, float]:
        result = {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
        }
        if getattr(self.algorithm, "rnd", None) is not None:
            result["rnd"] = 0.0
        if getattr(self.algorithm, "symmetry", None) is not None:
            result["symmetry"] = 0.0
        return result

    def update(self) -> dict[str, float]:
        """Skip entry-only rollouts, then apply the bounded fixed-LR schedule."""

        if self.iteration < self.burn_in_iterations:
            self.storage.clear()
            result = self._empty_update_result()
            result.update(
                {
                    "warm_start/skipped_update": 1.0,
                    "warm_start/update_index": -1.0,
                    "warm_start/learning_rate": 0.0,
                    "warm_start/num_learning_epochs": 0.0,
                    "warm_start/actor_update_norm": 0.0,
                }
            )
            self.iteration += 1
            return result

        update_index = self.iteration - self.burn_in_iterations
        if update_index < self.warmup_update_iterations:
            learning_rate = linear_anneal(
                update_index,
                start=self.initial_learning_rate,
                end=self.target_learning_rate,
                duration_steps=max(self.warmup_update_iterations - 1, 1),
            )
            num_learning_epochs = self.warmup_num_learning_epochs
        else:
            learning_rate = self.target_learning_rate
            num_learning_epochs = self.original_num_learning_epochs

        self._set_learning_rate(learning_rate)
        self.algorithm.num_learning_epochs = num_learning_epochs
        actor_before = tuple(parameter.detach().clone() for parameter in self.actor_parameters)
        try:
            result = self._original_update()
        finally:
            self.algorithm.num_learning_epochs = self.original_num_learning_epochs
        actor_after = tuple(parameter.detach() for parameter in self.actor_parameters)
        actor_update_norm = self._parameter_delta_norm(actor_before, actor_after)
        result.update(
            {
                "warm_start/skipped_update": 0.0,
                "warm_start/update_index": float(update_index),
                "warm_start/learning_rate": float(learning_rate),
                "warm_start/num_learning_epochs": float(num_learning_epochs),
                "warm_start/actor_update_norm": actor_update_norm,
            }
        )
        self.iteration += 1
        if (
            update_index < self.warmup_update_iterations
            and actor_update_norm > self.actor_update_norm_limit
        ):
            raise RuntimeError(
                "PPO warm-start actor update exceeded the configured limit: "
                f"{actor_update_norm:.6f} > {self.actor_update_norm_limit:.6f}."
            )
        return result


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


def maybe_enable_ppo_warm_start_guard(*, runner: object) -> bool:
    """Attach the default-disabled PPO guard after weights-only loading."""

    env = getattr(runner, "env", None)
    algorithm = getattr(runner, "alg", None)
    cfg = getattr(env, "cfg", None)
    if not bool(getattr(cfg, "pure_rl_warm_start_guard_enabled", False)):
        return False
    if algorithm is None:
        raise ValueError("runner does not expose a PPO algorithm for the warm-start guard.")

    guard = PpoWarmStartGuard(
        algorithm,
        burn_in_iterations=getattr(cfg, "pure_rl_warm_start_burn_in_iterations", 3),
        warmup_update_iterations=getattr(cfg, "pure_rl_warm_start_update_iterations", 10),
        initial_learning_rate=float(getattr(cfg, "pure_rl_warm_start_initial_learning_rate", 1.0e-5)),
        target_learning_rate=float(getattr(cfg, "pure_rl_warm_start_target_learning_rate", 5.0e-5)),
        warmup_num_learning_epochs=getattr(cfg, "pure_rl_warm_start_num_learning_epochs", 1),
        actor_update_norm_limit=float(getattr(cfg, "pure_rl_warm_start_actor_update_norm_limit", 0.10)),
    )
    algorithm.update = guard.update
    algorithm.ppo_warm_start_guard = guard
    return True


def maybe_enable_actor_gradient_conflict_probe(*, runner: object) -> bool:
    """Attach the default-disabled actor gradient probe to a weights-only run."""

    env = getattr(runner, "env", None)
    algorithm = getattr(runner, "alg", None)
    cfg = getattr(env, "cfg", None)
    if not bool(getattr(cfg, "pure_rl_actor_gradient_probe_enabled", False)):
        return False
    if algorithm is None:
        raise ValueError("runner does not expose a PPO algorithm for the gradient probe.")
    storage = getattr(algorithm, "storage", None)
    stored_observations = getattr(storage, "observations", None)
    if stored_observations is None or ACTOR_GRADIENT_PROBE_GROUP_KEY not in stored_observations:
        raise ValueError("rollout storage does not contain the actor gradient probe group.")
    log_dir = getattr(runner, "log_dir", None)
    if log_dir is None:
        raise ValueError("gradient probe requires a runner log directory.")

    probe = ActorGradientConflictProbe(
        algorithm,
        output_path=Path(log_dir) / "actor_gradient_conflict.csv",
        interval=int(getattr(cfg, "pure_rl_actor_gradient_probe_interval", 1)),
        minimum_samples=int(getattr(cfg, "pure_rl_actor_gradient_probe_minimum_samples", 32)),
    )
    algorithm.update = probe.update
    algorithm.actor_gradient_conflict_probe = probe
    return True


def maybe_enable_task_aware_ppo(*, runner: object) -> bool:
    """Attach default-disabled task-aware normalization and stratification."""

    env = getattr(runner, "env", None)
    algorithm = getattr(runner, "alg", None)
    cfg = getattr(env, "cfg", None)
    if not bool(getattr(cfg, "pure_rl_task_aware_ppo_enabled", False)):
        return False
    if algorithm is None:
        raise ValueError("runner does not expose a PPO algorithm for task-aware training.")
    storage = getattr(algorithm, "storage", None)
    stored_observations = getattr(storage, "observations", None)
    if stored_observations is None or ACTOR_GRADIENT_PROBE_GROUP_KEY not in stored_observations:
        raise ValueError("rollout storage does not contain the task-aware task/phase group.")
    log_dir = getattr(runner, "log_dir", None)
    if log_dir is None:
        raise ValueError("task-aware PPO requires a runner log directory.")

    adapter = TaskAwarePpoAdapter(
        algorithm,
        output_path=Path(log_dir) / "task_aware_ppo.csv",
        task_weights=getattr(cfg, "pure_rl_task_aware_ppo_task_weights", (0.15, 0.35, 0.50)),
        minimum_task_samples=int(getattr(cfg, "pure_rl_task_aware_ppo_minimum_task_samples", 32)),
        minimum_phase_samples=int(getattr(cfg, "pure_rl_task_aware_ppo_minimum_phase_samples", 16)),
    )
    algorithm.update = adapter.update
    algorithm.task_aware_ppo = adapter
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


def expand_actor_state_dict_net2wider(
    *,
    source_state: Mapping[str, Tensor],
    target_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Double a two-layer actor width while preserving its action mean exactly.

    The critic and action-noise state must retain identical shapes. The two
    actor hidden layers must each be exactly twice the source width; all other
    architecture changes fail closed.
    """

    if set(source_state) != set(target_state):
        raise ValueError("Net2Wider requires identical policy state-dict keys.")
    actor_keys = {
        "actor.0.weight",
        "actor.0.bias",
        "actor.2.weight",
        "actor.2.bias",
        "actor.4.weight",
        "actor.4.bias",
    }
    if {key for key in source_state if key.startswith("actor.")} != actor_keys:
        raise ValueError("Net2Wider supports only the registered two-hidden-layer actor MLP.")

    source_w1 = source_state["actor.0.weight"]
    source_b1 = source_state["actor.0.bias"]
    source_w2 = source_state["actor.2.weight"]
    source_b2 = source_state["actor.2.bias"]
    source_head = source_state["actor.4.weight"]
    source_head_bias = source_state["actor.4.bias"]
    target_w1 = target_state["actor.0.weight"]
    target_b1 = target_state["actor.0.bias"]
    target_w2 = target_state["actor.2.weight"]
    target_b2 = target_state["actor.2.bias"]
    target_head = target_state["actor.4.weight"]
    target_head_bias = target_state["actor.4.bias"]

    source_h1, input_dim = source_w1.shape
    source_h2, source_w2_input = source_w2.shape
    action_dim, source_head_input = source_head.shape
    expected_shapes = {
        "actor.0.weight": (2 * source_h1, input_dim),
        "actor.0.bias": (2 * source_h1,),
        "actor.2.weight": (2 * source_h2, 2 * source_h1),
        "actor.2.bias": (2 * source_h2,),
        "actor.4.weight": (action_dim, 2 * source_h2),
        "actor.4.bias": (action_dim,),
    }
    actual_shapes = {
        "actor.0.weight": tuple(target_w1.shape),
        "actor.0.bias": tuple(target_b1.shape),
        "actor.2.weight": tuple(target_w2.shape),
        "actor.2.bias": tuple(target_b2.shape),
        "actor.4.weight": tuple(target_head.shape),
        "actor.4.bias": tuple(target_head_bias.shape),
    }
    if source_w2_input != source_h1 or source_head_input != source_h2:
        raise ValueError("Source actor state is not a connected two-hidden-layer MLP.")
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"Net2Wider requires exact twofold hidden widths; expected {expected_shapes}, "
            f"received {actual_shapes}."
        )
    for key, source_value in source_state.items():
        if key in actor_keys:
            continue
        if tuple(source_value.shape) != tuple(target_state[key].shape):
            raise ValueError(f"Net2Wider cannot change non-actor parameter shape: {key}.")

    def _as_target(source: Tensor, target: Tensor) -> Tensor:
        return source.to(device=target.device, dtype=target.dtype)

    result = {key: value.detach().clone() for key, value in target_state.items()}
    for key, source_value in source_state.items():
        if key not in actor_keys:
            result[key] = _as_target(source_value, target_state[key]).detach().clone()

    w1 = _as_target(source_w1, target_w1)
    b1 = _as_target(source_b1, target_b1)
    w2 = _as_target(source_w2, target_w2)
    b2 = _as_target(source_b2, target_b2)
    head = _as_target(source_head, target_head)
    head_bias = _as_target(source_head_bias, target_head_bias)
    secondary_fraction = 1.0 - _NET2WIDER_PRIMARY_FRACTION

    result["actor.0.weight"] = torch.cat((w1, w1), dim=0).clone()
    result["actor.0.bias"] = torch.cat((b1, b1), dim=0).clone()
    widened_w2 = torch.cat(
        (
            w2 * _NET2WIDER_PRIMARY_FRACTION,
            w2 * secondary_fraction,
        ),
        dim=1,
    )
    result["actor.2.weight"] = torch.cat((widened_w2, widened_w2), dim=0).clone()
    result["actor.2.bias"] = torch.cat((b2, b2), dim=0).clone()
    result["actor.4.weight"] = torch.cat(
        (
            head * _NET2WIDER_PRIMARY_FRACTION,
            head * secondary_fraction,
        ),
        dim=1,
    ).clone()
    result["actor.4.bias"] = head_bias.detach().clone()
    return result


def split_actor_state_dict_from_shared_actor(
    *,
    source_state: Mapping[str, Tensor],
    target_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Map a shared four-action actor into independent frequency and tail trunks.

    Both new trunks receive exact copies of the source hidden layers. The
    frequency output receives source action row zero and the tail output
    receives rows one through three. Critic and action-noise state are copied
    unchanged. Only the registered two-hidden-layer ``[256, 128]`` shape is
    accepted.
    """

    source_actor_keys = {
        "actor.0.weight",
        "actor.0.bias",
        "actor.2.weight",
        "actor.2.bias",
        "actor.4.weight",
        "actor.4.bias",
    }
    target_actor_keys = {
        f"actor.{branch}.{suffix}"
        for branch in ("frequency_actor", "tail_actor")
        for suffix in (
            "0.weight",
            "0.bias",
            "2.weight",
            "2.bias",
            "4.weight",
            "4.bias",
        )
    }
    if {key for key in source_state if key.startswith("actor.")} != source_actor_keys:
        raise ValueError("Split warm start requires the registered shared two-hidden-layer actor.")
    if {key for key in target_state if key.startswith("actor.")} != target_actor_keys:
        raise ValueError("Split warm start target is not the registered two-trunk actor.")

    source_non_actor = {key for key in source_state if not key.startswith("actor.")}
    target_non_actor = {key for key in target_state if not key.startswith("actor.")}
    if source_non_actor != target_non_actor:
        raise ValueError("Split warm start requires identical non-actor state-dict keys.")
    for key in source_non_actor:
        if tuple(source_state[key].shape) != tuple(target_state[key].shape):
            raise ValueError(f"Split warm start cannot change non-actor parameter shape: {key}.")

    source_shapes = {
        "actor.0.weight": (256, 555),
        "actor.0.bias": (256,),
        "actor.2.weight": (128, 256),
        "actor.2.bias": (128,),
        "actor.4.weight": (4, 128),
        "actor.4.bias": (4,),
    }
    if any(tuple(source_state[key].shape) != shape for key, shape in source_shapes.items()):
        raise ValueError("Split warm start requires the shared [256, 128] four-action actor.")

    expected_target_shapes = {
        "actor.frequency_actor.0.weight": (256, 555),
        "actor.frequency_actor.0.bias": (256,),
        "actor.frequency_actor.2.weight": (128, 256),
        "actor.frequency_actor.2.bias": (128,),
        "actor.frequency_actor.4.weight": (1, 128),
        "actor.frequency_actor.4.bias": (1,),
        "actor.tail_actor.0.weight": (256, 555),
        "actor.tail_actor.0.bias": (256,),
        "actor.tail_actor.2.weight": (128, 256),
        "actor.tail_actor.2.bias": (128,),
        "actor.tail_actor.4.weight": (3, 128),
        "actor.tail_actor.4.bias": (3,),
    }
    if any(
        tuple(target_state[key].shape) != shape
        for key, shape in expected_target_shapes.items()
    ):
        raise ValueError("Split warm start requires two [256, 128] actor trunks.")

    def _as_target(source: Tensor, target: Tensor) -> Tensor:
        return source.to(device=target.device, dtype=target.dtype).detach().clone()

    result = {key: value.detach().clone() for key, value in target_state.items()}
    for key in source_non_actor:
        result[key] = _as_target(source_state[key], target_state[key])
    for branch in ("frequency_actor", "tail_actor"):
        for layer in ("0", "2"):
            for parameter in ("weight", "bias"):
                source_key = f"actor.{layer}.{parameter}"
                target_key = f"actor.{branch}.{layer}.{parameter}"
                result[target_key] = _as_target(source_state[source_key], target_state[target_key])

    result["actor.frequency_actor.4.weight"] = _as_target(
        source_state["actor.4.weight"][0:1],
        target_state["actor.frequency_actor.4.weight"],
    )
    result["actor.frequency_actor.4.bias"] = _as_target(
        source_state["actor.4.bias"][0:1],
        target_state["actor.frequency_actor.4.bias"],
    )
    result["actor.tail_actor.4.weight"] = _as_target(
        source_state["actor.4.weight"][1:4],
        target_state["actor.tail_actor.4.weight"],
    )
    result["actor.tail_actor.4.bias"] = _as_target(
        source_state["actor.4.bias"][1:4],
        target_state["actor.tail_actor.4.bias"],
    )
    return result


def _actor_hidden_dims(policy: object) -> tuple[int, ...]:
    actor = getattr(policy, "actor", None)
    if actor is None:
        return ()
    linear_layers = [module for module in actor.modules() if isinstance(module, nn.Linear)]
    return tuple(layer.out_features for layer in linear_layers[:-1])


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

    policy = getattr(getattr(runner, "alg", None), "policy", None)
    actor = getattr(policy, "actor", None)
    split_frequency_actor = bool(
        getattr(actor, "is_pure_rl_split_frequency_actor", False)
    )
    if split_frequency_actor:
        loaded = torch.load(checkpoint_path, weights_only=False, map_location=map_location)
        if not isinstance(loaded, dict) or not isinstance(loaded.get("model_state_dict"), Mapping):
            raise ValueError("Warm-start checkpoint does not contain model_state_dict.")
        source_state = loaded["model_state_dict"]
        target_state = policy.state_dict()
        if set(source_state) == set(target_state) and all(
            tuple(value.shape) == tuple(target_state[key].shape)
            for key, value in source_state.items()
        ):
            policy.load_state_dict(source_state)
        else:
            split_state = split_actor_state_dict_from_shared_actor(
                source_state=source_state,
                target_state=target_state,
            )
            policy.load_state_dict(split_state)
            print(
                "[INFO]: Warm-started independent frequency/tail actor trunks from the "
                "shared [256, 128] actor."
            )
        infos = loaded.get("infos")
    elif _actor_hidden_dims(policy) == _LARGE_ACTOR_HIDDEN_DIMS:
        loaded = torch.load(checkpoint_path, weights_only=False, map_location=map_location)
        if not isinstance(loaded, dict) or not isinstance(loaded.get("model_state_dict"), Mapping):
            raise ValueError("Warm-start checkpoint does not contain model_state_dict.")
        source_state = loaded["model_state_dict"]
        target_state = policy.state_dict()
        if all(
            key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
            for key, value in source_state.items()
        ) and set(source_state) == set(target_state):
            policy.load_state_dict(source_state)
        else:
            expanded_state = expand_actor_state_dict_net2wider(
                source_state=source_state,
                target_state=target_state,
            )
            policy.load_state_dict(expanded_state)
            print(
                "[INFO]: Expanded warm-start actor with function-preserving Net2Wider "
                "([256, 128] -> [512, 256])."
            )
        infos = loaded.get("infos")
    else:
        infos = load_fn(checkpoint_path, load_optimizer=False, map_location=map_location)
    runner.current_learning_iteration = 0
    if maybe_enable_actor_policy_distillation(runner=runner):
        coefficient = float(runner.env.cfg.pure_rl_actor_distillation_coefficient)
        print(
            "[INFO]: Enabled frozen actor policy distillation after warm start "
            f"(coefficient={coefficient:g})."
        )
    if maybe_enable_ppo_warm_start_guard(runner=runner):
        print("[INFO]: Enabled bounded PPO warm-start guard after weights-only loading.")
    if maybe_enable_actor_gradient_conflict_probe(runner=runner):
        print("[INFO]: Enabled actor gradient conflict probe after warm start.")
    if maybe_enable_task_aware_ppo(runner=runner):
        print("[INFO]: Enabled task-aware PPO normalization and stratified mini-batches.")
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

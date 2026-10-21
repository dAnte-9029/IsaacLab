"""Training-only behavior memory and non-invasive six-task PPO diagnostics."""

import json
import math
from pathlib import Path

import torch

from flapping_bot.px4_like.rl_training_utils import ActorGradientConflictProbe
from flapping_bot.path_tracking.c4_curriculum import C4_CURRICULUM_BIN_OFFSET


MEMORY_SCHEMA = "c4_successful_behavior_memory_v1"
GROUP_KEY = "actor_gradient_probe_group"


def family_masks(groups: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map existing PPO IDs to physical task families without actor inputs."""
    return {"c1": groups == 0, "c2c": (groups == 1) | (groups == 2),
            "c3a": (groups == 3) | (groups == 4), "c3b": groups == 5,
            "c3c": groups == 6, "c4": groups == 7}


class BehaviorMemory:
    """Frozen balanced observation/teacher-mean bank; not a PPO replay buffer."""

    def __init__(self, payload: dict, *, seed: int) -> None:
        if payload.get("schema") != MEMORY_SCHEMA or payload.get("role") != "training_only":
            raise ValueError("Only versioned training-only behavior memory is accepted")
        self.observations = payload["observations"].detach().clone()
        self.actions = payload["actions"].detach().clone()
        self.families = payload["families"].detach().clone()
        n = len(self.observations)
        if (self.observations.shape != (n, 555) or self.actions.shape != (n, 4)
                or self.families.shape != (n,) or n == 0
                or self.observations.device.type != "cpu" or self.actions.device.type != "cpu"
                or self.families.device.type != "cpu"
                or self.observations.dtype != torch.float32 or self.actions.dtype != torch.float32
                or self.families.dtype != torch.long
                or not torch.isfinite(self.observations).all() or not torch.isfinite(self.actions).all()
                or (self.families < 0).any() or (self.families > 5).any()):
            raise ValueError("Invalid memory tensor shape, device, dtype or values")
        self.indices = [(self.families == i).nonzero().flatten() for i in range(6)]
        if any(len(indices) < 32 for indices in self.indices):
            raise ValueError("Memory must cover every task with at least 32 samples")
        self.generator = torch.Generator().manual_seed(seed)
        self.calls = 0
        self._targets: torch.Tensor | None = None

    def draw(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced family draws consume only the memory's private RNG."""
        selected = []
        for i, indices in enumerate(self.indices):
            size = count // 6 + int(i < count % 6)
            selected.append(indices[torch.randint(len(indices), (size,), generator=self.generator)])
        rows = torch.cat(selected)
        rows = rows[torch.randperm(count, generator=self.generator)]
        return self.observations[rows], self.actions[rows]

    def __call__(self, *, obs: object | None, actions: torch.Tensor | None, env: object):
        """Append memory only to auxiliary actor loss; PPO/value batches stay fresh."""
        del env
        if (obs is None) == (actions is None):
            raise ValueError("Exactly one of observations or actions is required")
        if obs is not None:
            if self._targets is not None:
                raise RuntimeError("Unconsumed memory targets")
            observations, self._targets = self.draw(obs.batch_size[0])
            memory_obs = obs.clone()
            memory_obs["policy"] = observations.to(device=obs["policy"].device, dtype=obs["policy"].dtype)
            self.calls += 1
            return torch.cat((obs, memory_obs), dim=0), None
        if self._targets is None or self._targets.shape != actions.shape:
            raise RuntimeError("Missing or mismatched memory targets")
        targets = self._targets.to(device=actions.device, dtype=actions.dtype)
        self._targets = None
        return None, torch.cat((actions.detach(), targets), dim=0)

    def state_dict(self) -> dict:
        return dict(rng=self.generator.get_state(), calls=self.calls)

    def load_state_dict(self, state: dict) -> None:
        self.generator.set_state(state["rng"])
        self.calls = int(state["calls"])


def attach_behavior_memory(algorithm: object, memory: BehaviorMemory, coefficient: float) -> None:
    """Use the existing auxiliary actor-loss slot without editing RSL-RL."""
    if not math.isfinite(coefficient) or coefficient <= 0 or algorithm.symmetry is not None:
        raise ValueError("Positive memory coefficient and unused auxiliary slot required")
    algorithm.symmetry = dict(use_data_augmentation=False, use_mirror_loss=True,
                              data_augmentation_func=memory, mirror_loss_coeff=coefficient, _env=None)


class C4UpdateDiagnostics(ActorGradientConflictProbe):
    """Report all-family and tail gradient conflicts plus actual post-update drift."""

    def update(self) -> dict[str, float]:
        guard = getattr(self.algorithm, "ppo_warm_start_guard", None)
        if self.iteration % self.interval or (guard is not None and guard.iteration < guard.burn_in_iterations):
            result = self._original_update()
            self.iteration += 1
            return result
        groups = self.storage.observations[GROUP_KEY].flatten(0, 1).squeeze(-1)
        masks = family_masks(groups)
        metrics, gradients, snapshots = {}, {}, {}
        tail_indices = [i for i, (name, _) in enumerate(self.policy.actor.named_parameters()) if "tail" in name]
        for name, full_mask in masks.items():
            # Deterministic subsampling avoids perturbing the training RNG.
            indices = full_mask.nonzero().flatten()
            if len(indices) < self.minimum_samples:
                metrics[f"count_{name}"] = len(indices)
                continue
            if len(indices) > 256:
                indices = indices[torch.linspace(0, len(indices) - 1, 256).long()]
            mask = torch.zeros_like(full_mask)
            mask[indices] = True
            gradient, values = self._gradient_for_mask(mask=mask, name=name)
            gradients[name] = gradient
            metrics.update(values)
            observations = self.storage.observations.flatten(0, 1)[mask].clone()
            returns = self.storage.returns.flatten(0, 1)[mask].detach().clone()
            with torch.no_grad():
                mean, std = self._distribution_parameters(observations)
                snapshots[name] = (observations, returns, mean.clone(), std.clone())
                metrics[f"critic_mse_before_{name}"] = float((self.policy.evaluate(observations) - returns).square().mean())
        if "c4" in gradients:
            for name, gradient in gradients.items():
                if name == "c4":
                    continue
                metrics[f"cos_{name}_c4"] = float(self._cosine(gradient, gradients["c4"]))
                if tail_indices:
                    metrics[f"tail_cos_{name}_c4"] = float(self._cosine(
                        tuple(gradient[i] for i in tail_indices),
                        tuple(gradients["c4"][i] for i in tail_indices)))
        before = tuple(p.detach().clone() for p in self.actor_parameters)
        result = self._original_update()
        delta = tuple(p.detach() - old for p, old in zip(self.actor_parameters, before))
        metrics["actor_update_norm"] = float(self._norm(delta))
        with torch.no_grad():
            for name, (observations, returns, old_mean, old_std) in snapshots.items():
                mean, std = self._distribution_parameters(observations)
                kl = torch.log(std / old_std) + (old_std.square() + (old_mean - mean).square()) / (2 * std.square()) - .5
                metrics[f"post_kl_{name}"] = float(kl.sum(-1).mean())
                metrics[f"critic_mse_after_{name}"] = float((self.policy.evaluate(observations) - returns).square().mean())
                metrics[f"gradient_dot_update_{name}"] = float(self._dot(gradients[name], delta))
                for channel in range(4):
                    metrics[f"mean_delta_{name}_{channel}"] = float((mean - old_mean)[:, channel].mean())
                    metrics[f"std_{name}_{channel}"] = float(std[:, channel].mean())
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise RuntimeError("Nonfinite training diagnostic")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a") as stream:
            stream.write(json.dumps(dict(iteration=self.iteration, **metrics), allow_nan=False) + "\n")
        self.iteration += 1
        return result


class TrainingEpisodeTracker:
    """Track pre-reset training errors, scores and optional memory candidates."""

    def __init__(self, num_envs: int, horizon: int, *, collect: bool = False) -> None:
        self.errors = torch.zeros(num_envs, horizon + 2, 2)
        self.counts = torch.zeros(num_envs, dtype=torch.long)
        self.collect = collect
        self.pending: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in range(num_envs)]
        self.memory: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in range(6)]
        self.accepted_episodes = [0] * 6
        self.rejected_episodes = [0] * 6
        self.generator = torch.Generator().manual_seed(32452843)

    def add_inputs(self, observations: torch.Tensor, actions: torch.Tensor) -> None:
        if self.collect:
            for row in (self.counts % 6 == 0).nonzero().flatten().tolist():
                self.pending[row].append((observations[row].detach().clone(), actions[row].detach().clone()))

    def observe(self, env: object, curriculum: object | None) -> list[dict]:
        """Consume terminal values before Isaac resets rows; incomplete episodes never count."""
        error = torch.stack((env._eval_pure_rl_cross_track_error_m.abs(),
                             env._eval_pure_rl_height_error_m.abs()), dim=1).detach()
        if not torch.isfinite(error).all():
            raise RuntimeError("Nonfinite training episode tracking error")
        rows = torch.arange(len(self.counts))
        if (self.counts >= self.errors.shape[1]).any():
            raise RuntimeError("Episode exceeded tracking buffer")
        self.errors[rows, self.counts] = error
        self.counts += 1
        result = []
        for row in (env.reset_terminated | env.reset_time_outs).nonzero().flatten().tolist():
            count = int(self.counts[row])
            family = int(env._pure_rl_spatial_path.task_family_id[row])
            values = self.errors[row, :count]
            mean, p95 = values.mean(0), torch.quantile(values, .95, dim=0)
            survived = bool(env.reset_time_outs[row] and not env.reset_terminated[row])
            completed = family != 5 or float(env._eval_pure_rl_along_track_progress_m[row]) >= 140
            success = bool(survived and completed and (mean <= .5).all() and (p95 <= 1.5).all())
            bin_id = int(env._pure_rl_spatial_path.hard_slice_id[row]) - C4_CURRICULUM_BIN_OFFSET if family == 5 else -1
            if curriculum is not None and family == 5:
                curriculum.observe(bin_id, float(success))
            result.append(dict(family=family, bin_id=bin_id, steps=count, success=success,
                               mean_error=mean.tolist(), p95_error=p95.tolist()))
            if self.collect:
                if success and self.pending[row]:
                    self.accepted_episodes[family] += 1
                    for sample in self.pending[row]:
                        # Bounded reservoir across successful episodes, not just the first.
                        seen = getattr(self, "seen", [0] * 6)
                        self.seen = seen
                        seen[family] += 1
                        bank = self.memory[family]
                        if len(bank) < 4096:
                            bank.append(sample)
                        else:
                            index = int(torch.randint(seen[family], (), generator=self.generator))
                            if index < len(bank):
                                bank[index] = sample
                else:
                    self.rejected_episodes[family] += 1
                self.pending[row].clear()
            self.counts[row] = 0
        return result

    def payload(self, provenance: dict) -> dict:
        """Freeze a bank only after every family has successful coverage."""
        observations, actions, families = [], [], []
        for family, samples in enumerate(self.memory):
            if len(samples) < 32:
                raise RuntimeError(f"Insufficient successful memory for family {family}")
            observations.extend(s[0] for s in samples)
            actions.extend(s[1] for s in samples)
            families.extend([family] * len(samples))
        return dict(schema=MEMORY_SCHEMA, role="training_only", observations=torch.stack(observations),
                    actions=torch.stack(actions), families=torch.tensor(families), provenance=provenance,
                    accepted_episodes=self.accepted_episodes, rejected_episodes=self.rejected_episodes)

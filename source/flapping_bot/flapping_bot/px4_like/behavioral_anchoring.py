"""Full-bank tail functional constraints for the bounded D-50 PPO comparison.

Only independent tail parameters are backtracked. For an accepted fraction alpha,
Adam moments/step are the candidate values: exactly a single Adam step at alpha*lr
for this block (no weight decay). Rejection restores its parameters AND optimizer
state. Frequency, critic and log_std retain their ordinary PPO steps.
"""

import copy
import time

import torch
from rsl_rl.algorithms import PPO

from flapping_bot.px4_like.c4_retention import BehaviorMemory

ANCHOR_SCHEMA = 'd50_tail_anchors_v1'
FAMILIES = ('c1', 'c2c', 'c3a', 'c3b', 'c3c', 'c4')
SOFT_COEFFICIENT = 1.0
EPSILON = 0.02 ** 2
MAX_BACKTRACKS = 10


class TailAnchors:
    """Immutable full training bank, equally weighted by task, unclipped means."""

    def __init__(self, payload: dict) -> None:
        if payload.get('anchor_schema') != ANCHOR_SCHEMA:
            raise ValueError('Missing D-50 anchor schema')
        memory = BehaviorMemory(payload, seed=0)
        self.payload = copy.deepcopy(payload)
        self.observations, self.targets = memory.observations, memory.actions
        self.indices = memory.indices
        self.epsilon = torch.tensor(payload['epsilon'], dtype=torch.float64)
        if self.epsilon.shape != (6,) or not torch.isfinite(self.epsilon).all() or (self.epsilon <= 0).any():
            raise ValueError('Six finite positive family constraints required')

    def distances(self, tail: torch.nn.Module) -> torch.Tensor:
        residual = tail(self.observations) - self.targets[:, 1:]
        return torch.stack([residual[index].square().mean() for index in self.indices])

    @torch.no_grad()
    def feasible(self, tail: torch.nn.Module) -> bool:
        distance = self.distances(tail).double()
        return bool(torch.isfinite(distance).all() and (distance <= self.epsilon).all())

    @torch.no_grad()
    def metrics(self, actor: torch.nn.Module) -> dict:
        delta = actor(self.observations) - self.targets
        result = {}
        for name, index in zip(FAMILIES, self.indices):
            d = delta[index]
            result[name] = dict(tail_rms=float(d[:, 1:].square().mean().sqrt()),
                                action_rms=d.square().mean(0).sqrt().tolist(),
                                symmetric_rms=float(((d[:, 2] + d[:, 3]) / 2).square().mean().sqrt()),
                                differential_rms=float(((d[:, 2] - d[:, 3]) / 2).square().mean().sqrt()))
        return result


class ConstrainedTailStep:
    """Backtrack each actual Adam step against all six full-bank inequalities."""

    def __init__(self, optimizer: torch.optim.Adam, tail: torch.nn.Module,
                 anchors: TailAnchors, max_backtracks: int = MAX_BACKTRACKS) -> None:
        if not isinstance(optimizer, torch.optim.Adam) or any(g['weight_decay'] for g in optimizer.param_groups):
            raise ValueError('Constraint semantics require Adam with zero weight decay')
        self.optimizer, self.tail, self.anchors = optimizer, tail, anchors
        self.parameters = tuple(tail.parameters())
        self.max_backtracks = max_backtracks

    def step(self) -> dict:
        started = time.perf_counter()
        before = [p.detach().clone() for p in self.parameters]
        states = [copy.deepcopy(self.optimizer.state.get(p)) for p in self.parameters]
        optimizer_started = time.perf_counter()
        self.optimizer.step()
        optimizer_seconds = time.perf_counter() - optimizer_started
        with torch.no_grad():
            delta = [p - old for p, old in zip(self.parameters, before)]
            alpha, evaluations = 1.0, 0
            for attempt in range(self.max_backtracks + 1):
                alpha = 2.0 ** -attempt
                if attempt:
                    for p, old, change in zip(self.parameters, before, delta):
                        p.copy_(old + alpha * change)
                evaluations += 1
                if self.anchors.feasible(self.tail):
                    break
            else:
                alpha = 0.0
                for p, old, state in zip(self.parameters, before, states):
                    p.copy_(old)
                    if state is None:
                        self.optimizer.state.pop(p, None)
                    else:
                        self.optimizer.state[p] = state
                if not self.anchors.feasible(self.tail):
                    raise RuntimeError('Hard constraint violated even after exact rollback')
                evaluations += 1
            candidate_norm = float(sum(d.square().sum() for d in delta).sqrt())
        return dict(alpha=alpha, checks=evaluations, candidate_tail_norm=candidate_norm,
                    effective_tail_norm=alpha * candidate_norm,
                    constrained_step_seconds=time.perf_counter() - started,
                    optimizer_seconds=optimizer_seconds,
                    constraint_overhead_seconds=time.perf_counter() - started - optimizer_seconds)


class AnchoredPPO(PPO):
    """Existing feedforward PPO equations with explicit full-bank S/H hooks.

    The supported CPU configuration is deliberately narrow. Task-aware advantage
    preparation and the existing fixed learning-rate guard remain outside update.
    No teacher, symmetry loss, normalization, RND or adaptive LR is added here.
    """

    def configure_anchors(self, payload: dict, mode: str) -> None:
        if mode not in ('S', 'H'):
            raise ValueError('Expected S or H')
        self.anchors = TailAnchors(payload)
        self.anchor_mode = mode
        self.anchor_updates = 0
        self.constraint = ConstrainedTailStep(self.optimizer, self.policy.actor.tail_actor, self.anchors)

    def update(self) -> dict[str, float]:
        if (self.policy.is_recurrent or self.symmetry or self.rnd or self.is_multi_gpu
                or self.schedule != 'fixed' or self.normalize_advantage_per_mini_batch):
            raise ValueError('Unsupported PPO configuration for controlled anchoring')
        started = time.perf_counter()
        before = [p.detach().clone() for p in self.policy.actor.parameters()]
        rollout_metrics = {}
        if 'actor_gradient_probe_group' in self.storage.observations:
            group = self.storage.observations['actor_gradient_probe_group'].squeeze(-1)
            c4 = group == 7
            rollout_metrics = dict(c4_transitions=int(c4.sum()),
                                   c4_mean_reward=float(self.storage.rewards.squeeze(-1)[c4].mean()))
        totals = dict(value_function=0., surrogate=0., entropy=0., anchoring=0.)
        steps = []
        anchor_seconds = 0.
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs, actions, values, advantages, returns, log_prob, old_mu, old_sigma, hidden, masks in generator:
            self.policy.act(obs, masks=masks, hidden_state=hidden[0])
            current_log_prob = self.policy.get_actions_log_prob(actions)
            current_value = self.policy.evaluate(obs, masks=masks, hidden_state=hidden[1])
            entropy = self.policy.entropy.mean()
            ratio = torch.exp(current_log_prob - log_prob.squeeze())
            surrogate = torch.maximum(-advantages.squeeze() * ratio,
                                      -advantages.squeeze() * ratio.clamp(1. - self.clip_param, 1. + self.clip_param)).mean()
            if self.use_clipped_value_loss:
                clipped = values + (current_value - values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((current_value - returns).square(), (clipped - returns).square()).mean()
            else:
                value_loss = (current_value - returns).square().mean()
            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            if self.anchor_mode == 'S':
                tick = time.perf_counter()
                auxiliary = self.anchors.distances(self.policy.actor.tail_actor).mean()
                (SOFT_COEFFICIENT * auxiliary).backward()
                anchor_seconds += time.perf_counter() - tick
                totals['anchoring'] += float(auxiliary.detach())
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            if self.anchor_mode == 'H':
                steps.append(self.constraint.step())
            else:
                self.optimizer.step()
            totals['value_function'] += float(value_loss.detach())
            totals['surrogate'] += float(surrogate.detach())
            totals['entropy'] += float(entropy.detach())
        count = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        tick = time.perf_counter()
        metrics = self.anchors.metrics(self.policy.actor)
        # Use the direct full-bank MSE for the decision, avoiding sqrt roundoff.
        violated = not self.anchors.feasible(self.policy.actor.tail_actor)
        anchor_seconds += time.perf_counter() - tick
        self.anchor_updates += 1
        self.anchor_report = dict(update=self.anchor_updates, mode=self.anchor_mode, families=metrics,
                                 constraint_violated=violated, optimizer_steps=count,
                                 accepted_tail_steps=sum(s['alpha'] > 0 for s in steps) if steps else count,
                                 backtracked_tail_steps=sum(0 < s['alpha'] < 1 for s in steps),
                                 rejected_tail_steps=sum(s['alpha'] == 0 for s in steps),
                                 alpha_sum=sum(s['alpha'] for s in steps) if steps else float(count),
                                 step_details=steps, anchor_seconds=anchor_seconds,
                                 constraint_overhead_seconds=sum(s['constraint_overhead_seconds'] for s in steps),
                                 update_seconds=time.perf_counter() - started,
                                 rollout=rollout_metrics,
                                 actor_delta_norm=float(sum((p - b).square().sum()
                                     for p, b in zip(self.policy.actor.parameters(), before)).sqrt()),
                                 action_std=self.policy.log_std.detach().exp().tolist())
        self.anchor_report['losses'] = {k: v / count for k, v in totals.items()}
        if self.anchor_mode == 'H' and violated:
            raise RuntimeError(f'Full-anchor constraint failure: {self.anchor_report}')
        return self.anchor_report['losses']


def register_anchored_ppo() -> None:
    """Register without modifying the installed RSL-RL implementation."""
    import rsl_rl.runners.on_policy_runner as runner_module
    runner_module.AnchoredPPO = AnchoredPPO

"""Predeclared phase-6 learning objectives; distances in metres, dt in seconds."""
import torch

REWARD_CONTRACT = dict(recovery_weight=.5, recovery_delta_m=2., recovery_scale_m=2.,
    goal_weight=1., goal_scale_m=5., gamma=.99,
    terminal_potential='zero for completed/failed/invalid_data; physical successor for timeout',
    timeout_boundary='existing PPO bootstraps pre-action value; no policy-invariance claim')


def goal_potential(distance):
    return -distance/(distance+5.)


def shaping_terms(error, previous_potential, goal_distance, terminated, dt, gamma):
    # Invalid body samples terminate; no unbounded distance penalty is meaningful there.
    error = torch.where(torch.isfinite(error), error, torch.zeros_like(error))
    recovery = -.5*dt*(torch.sqrt(error.square()+4.)-2.)/2.
    successor = torch.where(terminated, torch.zeros_like(goal_distance), goal_potential(goal_distance))
    goal = gamma*successor-previous_potential
    return recovery, goal

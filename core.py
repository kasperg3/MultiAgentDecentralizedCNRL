import numpy as np
import scipy.signal

import torch
import torch.nn as nn

# function to unpack observation from gym environment
def unpackObs(obs):
    return obs['achieved_goal'], obs['desired_goal'], np.concatenate((obs['observation'],obs['desired_goal'])), np.concatenate((obs['observation'], obs['achieved_goal']))


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        observation_dim = observation_space['observation'].shape[0]
        achieved_goal_dim = observation_space['achieved_goal'].shape[0]
        desired_goal_dim = observation_space['desired_goal'].shape[0]
        assert achieved_goal_dim == desired_goal_dim

        # state size = observation size + goal size
        state_dim = observation_dim + desired_goal_dim
        action_dim = action_space.shape[0]
        action_highbound = action_space.high[0]


        # build policy and value functions
        self.pi = MLPActor(state_dim, action_dim, hidden_sizes, activation, action_highbound)
        self.q = MLPQFunction(state_dim, action_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
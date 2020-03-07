# TODO: Add DQN with pythorch
import math

import torch
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
import utils

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class network(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim):
        """
        Arguments:
            state_dim: number of input.
            action_dim: number of action-value to output, one-to-one correspondence to action.
        """

        super(network, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return F.relu(self.l3(a))


class DQN(object):

    def __init__(self,
                 state_dim,
                 action_dim,
                 gamma=0.99,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=200,
                 target_update=10):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            state_dim: number of input.
            action_dim: number of action-value to output, one-to-one correspondence to action.
            batch_size: the number of batches to sample for training
            gamma: reward decay
            eps_start: epsilon greedy at the start of training
            eps_end: epsilon greedy at the end of training
            eps_decay: the rate of decay
            target_update:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = network(self.state_dim, self.action_dim).to(self.device)
        self.target_net = network(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.steps_done = 0

    def select_action(self, state):
        """
        select_action - will select an action accordingly to an epsilon greedy policy. Simply put,
        we’ll sometimes use our model for choosing the action, and sometimes we’ll just sample one uniformly.
        The probability of choosing a random action will start at EPS_START and will decay exponentially towards
        EPS_END. EPS_DECAY controls the rate of the decay.
        Arguments:
            state: The input of the network
        """
        sample = random.random()
        eps_threshold = self.eps_end+(self.eps_start - self.eps_end)*math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def train(self, replay_buffer, batch_size):
        pass

    def save(self, filename):
        torch.save(self.policy_net, filename + "policy")
        torch.save(self.target_net, filename + "target")

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename + "policy"))
        self.target_net.load_state_dict(torch.load(filename + "target"))


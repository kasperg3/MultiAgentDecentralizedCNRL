import gym
import gym_mergablerobots
import torch
import numpy as np
env = gym.make('Concept-v0')
# Set seeds
env.seed(1000)
torch.manual_seed(1000)
np.random.seed(1000)

env.reset()
env.reset()
while True:
    env.render()
    env.step([5, 5])
    env.step([0, 5])
    env.step([2, 5])
    env.step([4, 5])
    env.step([1, 5])
    while True:
        env.render()
    break
env.close()
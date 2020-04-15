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
while True:
    env.render()
    for _ in range(10):
        action = [2, 5]
        env.step(action)
    for _ in range(10):
        action = [4, 5]
        env.step(action)
    for _ in range(10):
        action = [1, 5]
        env.step(action)
    break
env.close()
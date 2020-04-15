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
for _ in range(1000):
    env.render()
    action = [env.action_space.sample(), env.action_space.sample()]
    print(action)
    env.step(action)
env.close()
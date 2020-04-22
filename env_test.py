import gym
import gym_mergablerobots
import torch
import numpy as np
env = gym.make('Concept-v0')
env.reset()
while True:
    env.render()
    env.step([5, 5])  # 5 = open gripper
    env.step([6, 0])
    env.step([6, 1])
    env.step([6, 4])  # close
    env.step([6, 2])
    env.step([6, 3])
    env.step([6, 3])
    env.step([6, 3])
    env.step([6, 3])
    env.step([0, 6])
    env.step([1, 6])
    env.step([4, 6])  # 4 = close gripper
    env.step([2, 6])
    env.step([3, 6])
    env.step([3, 6])
    env.step([3, 6])
    env.step([3, 6])
    env.reset()
env.close()

import gym_mergablerobots.envs.mergablerobots.ur_pick_and_place

import gym
env = gym.make('UrPickAndPlace-v0')
env.reset()
for _ in range(1000):
    env.render()
    # env.step(env.action_space.sample()) # take a random action
    env.reset()
env.close()
import gym_mergablerobots.envs.mergablerobots.ur_pick_and_place
import gym
env = gym.make('UrReach-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
import gym
import gym_mergablerobots
env = gym.make('UrBinPicking-v0', reward_type='reach')
env.reset()
for _ in range(1000):
    env.render()
env.close()
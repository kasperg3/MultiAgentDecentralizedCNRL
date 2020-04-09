import gym
import gym_mergablerobots
env = gym.make('Concept-v0')
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    env.step(action)

env.close()

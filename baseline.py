import gym
import numpy as np
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import TD3
from stable_baselines.td3 import policies
from stable_baselines.common import cmd_util
from gym.wrappers import FilterObservation, FlattenObservation

import gym_mergablerobots

reward_type = 'sparse'

env = gym.make('UrReach-v1', reward_type=reward_type)
env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(policies.MlpPolicy, env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=1000000, log_interval=10)
model.save("td3_UrReach")

while True:
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

env.close()

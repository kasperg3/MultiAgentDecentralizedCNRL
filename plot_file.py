import gym_mergablerobots.envs.mergablerobots.ur_pick_and_place
import pickle
import matplotlib.pyplot as plt
import numpy as np
# import gym
# env = gym.make('UrPickAndPlace-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     #env.step(env.action_space.sample()) # take a random action
# env.close()
x_axis = range(50)

# plot example from load data to plot
DDPG_HER_reach_dense = pickle.load(open("./plot_data/plot_data_DDPG_HER_reach_dense_reward_shaping_plot_test", "rb"))
plt.figure()
plt.ylabel('Reward')
plt.xlabel('Epoch')
DDPG_HER_reach_dense_plot, = plt.plot(np.mean(DDPG_HER_reach_dense, axis=1), )
plt.legend([x_axis, DDPG_HER_reach_dense_plot], ['DDPG_HER_reach_dense', 'Test score'])
plt.show()

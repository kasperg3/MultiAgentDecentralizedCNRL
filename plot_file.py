import matplotlib.pyplot as plt
import numpy as np

# #pnp_load = np.load('results/TD3_UrBinPickingPlace-v0_1234_train.npy')
# load1 = np.load('results/final_concepts/agent0/TD3_UrBinPickingPlace-v0_1000_train_success.npy')
# load2 = np.load('results/new_states/TD3_UrBinPickingPlace-v0_1000_train_success.npy')
#
# plt.plot(load2, label='Place (with rotation)')
# plt.plot(load1, label='Place (reduced rotation)')
# plt.ylabel('Success Rate')
# plt.xlabel('Samples * 5e3')
# plt.legend()
# plt.show()

N = 5000

load1 = np.load('results/DQN/lr_optimisation_04_05/agent0_scores_lr=0.005_history.npy')
load1 = np.convolve(load1, np.ones((N,))/N, mode='valid')
load2 = np.load('results/DQN/lr_optimisation_04_05/agent0_scores_lr=0.001_history.npy')
load2 = np.convolve(load2, np.ones((N,))/N, mode='valid')
load3 = np.load('results/DQN/lr_optimisation_04_05/agent0_scores_lr=0.0005_history.npy')
load3 = np.convolve(load3, np.ones((N,))/N, mode='valid')
load4 = np.load('results/DQN/lr_optimisation_04_05/agent0_scores_lr=0.0001_history.npy')
load4 = np.convolve(load4, np.ones((N,))/N, mode='valid')
load5 = np.load('results/DQN/lr_optimisation_04_05/agent0_scores_lr=1e-05_history.npy')
load5 = np.convolve(load5, np.ones((N,))/N, mode='valid')
load6 = np.load('results/DQN/lr_optimisation_04_05/agent0_scores_lr=5e-05_history.npy')
load6 = np.convolve(load6, np.ones((N,))/N, mode='valid')

plt.plot(load1, label='0.005')
plt.plot(load2, label='0.001')
plt.plot(load5, label='1e-05')
plt.plot(load6, label='5e-05')
plt.plot(load3, label='0.0005')
plt.plot(load4, label='0.0001')
plt.ylabel('Average Episode Return')
plt.xlabel('Samples * 5e3')
plt.legend()
plt.show()
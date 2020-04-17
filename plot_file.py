import matplotlib.pyplot as plt
import numpy as np

#pnp_load = np.load('results/TD3_UrBinPickingPlace-v0_1234_train.npy')
load1 = np.load('TD3/results/temp/quaternion_reward/TD3_UrBinPickingPlace-v0_1234_train_success.npy')
load2 = np.load('results/place_weight_test/TD3_UrBinPickingPlace-v0_12025_train_success.npy')
load3 = np.load('results/rot_optimal_concepts/TD3_UrBinPickingPlace-v0_1000_train_success.npy')
load4 = np.load('results/reach_tau_test/TD3_UrBinPickingReach-v0_1000_train_success.npy')
load5 = np.load('results/reach_tau_test/TD3_UrBinPickingReach-v0_50000_train_success.npy')
load6 = np.load('results/reach_tau_test/TD3_UrBinPickingReach-v0_10000_train_success.npy')

plt.plot(load1, label='Place')
#plt.plot(load2, label='Wd:Wr=1:0.25')
#plt.plot(load3, label='Wd:Wr=1:1')
# plt.plot(load4, label='tau=0.0001')
# plt.plot(load5, label='tau=0.00005')
# plt.plot(load6, label='tau=0.00001')
plt.ylim((0,1))
plt.ylabel('Success Rate')
plt.xlabel('Samples * 5e3')
plt.legend()
plt.show()
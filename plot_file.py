import matplotlib.pyplot as plt
import numpy as np

#pnp_load = np.load('results/TD3_UrBinPickingPlace-v0_1234_train.npy')
place_load = np.load('results/TD3_UrBinPickingPlace-v0_1234_test.npy')
orient_load = np.load('results/TD3_UrBinPickingOrient-v0_1234_test.npy')
lift_load = np.load('results/TD3_UrBinPickingLift-v0_1234_test.npy')
reach_load = np.load('results/TD3_UrBinPickingReach-v0_1234_test.npy')
plot_load = lift_load[:400]


positions = [0, 100, 200, 300, 400] #
labels = [0, 0.5, 1, 1.5, 2]
plt.xticks(positions, labels)
plt.plot(plot_load, label='Test score')
plt.ylabel('Reward')
plt.xlabel('Samples in millions')
plt.legend(["Lift"])
plt.show()
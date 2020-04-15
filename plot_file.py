import matplotlib.pyplot as plt
import numpy as np

#pnp_load = np.load('results/TD3_UrBinPickingPlace-v0_1234_train.npy')
reach_load = np.load('results/TD3_UrBinPickingReach-v0_1234_test.npy')
orient_load = np.load('results/TD3_UrBinPickingOrient-v0_1234_test.npy')
lift_load = np.load('results/TD3_UrBinPickingLift-v0_1234_test.npy')
place_load = np.load('results/TD3_UrBinPickingPlace-v0_1234_test.npy')

reach_load = reach_load[:200]
orient_load = orient_load[:200]
lift_load = lift_load[:200]
place_load = place_load[:200]

#for custom x-axis
#positions = [0, 100, 200] #
#labels = [0, 0.5, 1, 1.5, 2]
#plt.xticks(positions, labels)

plt.plot(reach_load, label='Test score')
plt.plot(orient_load, label='Test score')
plt.plot(lift_load, label='Test score')
plt.plot(place_load, label='Test score')
plt.ylabel('Success Rate')
plt.xlabel('Samples * 5e3')
plt.legend(["Reach", "Orient", "Lift", "Place"])
plt.show()
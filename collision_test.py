import os
import mujoco_py
import numpy as np
import gym
import argparse
import gym_mergablerobots
MODEL_XML_PATH = '/home/nikolaj/master/mergableindustrialrobots/gym-mergablerobots/gym_mergablerobots/envs/assets/bin_picking.xml'

# Load the model and make a simulator
# model = mujoco_py.load_model_from_path(MODEL_XML_PATH)  # model: class PyMjModel
# sim = mujoco_py.MjSim(model)
# viewer = mujoco_py.MjViewer(sim)

parser = argparse.ArgumentParser()
parser.add_argument("--reward_type", default='place')

env = gym.make('UrBinPickingPlace-v0', reward_type='place')

for _ in range(100):
    # Simulate 1000 steps so humanoid has fallen on the ground
    env.reset()
    for _ in range (50):
        env.render()
        #env.step(env.action_space.sample())  # take a random action
env.close()
sim = env.sim

#     print("TEST left \n" + str(sim.data.get_body_xmat('robot0:left_finger')))
#     print("TEST right \n" + str(sim.data.get_body_xmat('robot0:right_finger')))
#
#     print('number of contacts', sim.data.ncon)
#     for i in range(sim.data.ncon):
#         # Note that the contact array has more than `ncon` entries,
#         # so be careful to only read the valid entries.
#         contact = sim.data.contact[i]
#         print('contact:', i)
#         print('distance:', contact.dist)
#         print('geom1:', contact.geom1, sim.model.geom_id2name(contact.geom1))
#         print('geom2:', contact.geom2, sim.model.geom_id2name(contact.geom2))
#         print('contact position:', contact.pos)
#
#         # Use internal functions to read out mj_contactForce
#         c_array = np.zeros(6, dtype=np.float64)
#         mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, c_array)
#
#         # Convert the contact force from contact frame to world frame
#         ref = np.reshape(contact.frame, (3, 3))
#         c_force = np.dot(np.linalg.inv(ref), c_array[0:3])
#         c_torque = np.dot(np.linalg.inv(ref), c_array[3:6])
#         print('contact force in world frame:', c_force)
#         print('contact torque in world frame:', c_torque)
#         print()
#
#     print('Geom 1:', sim.model.geom_id2name(1))
#     print('Geom 2:', sim.model.geom_id2name(2))
#     print('Geom 3:', sim.model.geom_id2name(3))
#     print('Geom 4:', sim.model.geom_id2name(4))
#     print('Geom 5:', sim.model.geom_id2name(5))
#     print('Geom 6:', sim.model.geom_id2name(6))
#     print('Geom 7:', sim.model.geom_id2name(7))
#     print('Geom 8:', sim.model.geom_id2name(8))
#     print('Geom 9:', sim.model.geom_id2name(9))
#     print('Geom 10:', sim.model.geom_id2name(10))
#     print('Geom 11:', sim.model.geom_id2name(11))
#     print('Geom 12:', sim.model.geom_id2name(12))
#     print('Geom 13:', sim.model.geom_id2name(13))
#     print('Geom 14:', sim.model.geom_id2name(14))
#     print('Geom 15:', sim.model.geom_id2name(15))
#     print('Geom 16:', sim.model.geom_id2name(16))
#     print('Geom 17:', sim.model.geom_id2name(17))
#     print('Geom 17:', sim.model.geom_id2name(18))
#     print('Geom 17:', sim.model.geom_id2name(19))
#     print('Geom 17:', sim.model.geom_id2name(20))
#     print('Geom 17:', sim.model.geom_id2name(21))
# print('done')
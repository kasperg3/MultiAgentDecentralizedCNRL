import os
from gym import utils
from gym_mergablerobots.envs.ur_bin_picking_env import UrBinPickingEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = '/home/kasper/workspace/mergableindustrialrobots/gym-mergablerobots/gym_mergablerobots/envs/assets/bin_picking.xml'


class UrBinPicking(UrBinPickingEnv, utils.EzPickle):
    def __init__(self, reward_type='reach'):
        initial_qpos = {
            # TODO: Create a new initial qpos
            'robot0:joint1': -0,
            'robot0:joint2': -0.2,
            'robot0:joint3': 2,
            'robot0:joint4': 0.6,
            'robot0:joint5': 1.6,
            'robot0:joint6': 0
        }
        UrBinPickingEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            n_substeps=20,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            box_range=0.15,
            success_threshold=0.05)
        utils.EzPickle.__init__(self)

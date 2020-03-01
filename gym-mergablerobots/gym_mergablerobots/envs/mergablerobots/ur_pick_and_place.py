import os
from gym import utils
from gym_mergablerobots.envs.ur_env import UrEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = '/home/kasper/workspace/mergableindustrialrobots/gym-mergablerobots/gym_mergablerobots/envs/assets/single_robot.xml'


class URPickAndPlaceEnv(UrEnv, utils.EzPickle):
    def __init__(self, reward_type='dense_reward_shaping'):
        initial_qpos = {
            'robot0:joint1': -2,
            'robot0:joint2': -2.053,
            'robot0:joint3': 2.3,
            'robot0:joint4': -1.8,
            'robot0:joint5': -1.50,
            'robot0:joint6': -1.7588
        }
        UrEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

import os
from gym import utils
from gym_mergablerobots.envs.ur_env import UrEnv

# Ensure we get the path separator correct on windows
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
MODEL_XML_PATH = ROOT_DIR[:-15] + '/assets/single_robot.xml'


class URPickAndPlaceEnv(UrEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            'robot0:joint1': -0,
            'robot0:joint2': -0.2,
            'robot0:joint3': 2,
            'robot0:joint4': 0.6,
            'robot0:joint5': 1.6,
            'robot0:joint6': 0
        }
        UrEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0, target_in_the_air=True, target_offset=0.0,
            obj_range=0.10, target_range=0.15, distance_threshold=0.01,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

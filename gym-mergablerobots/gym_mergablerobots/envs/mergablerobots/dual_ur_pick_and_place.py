import os
from gym import utils
from gym_mergablerobots.envs.ur_env import UrEnv


# Ensure we get the path separator correct on windows
MODEL_XML_PATH ='/home/nikolaj/master/mergableindustrialrobots/gym-mergablerobots/gym_mergablerobots/envs/assets/dual_robot.xml'


class DualURPickAndPlaceEnv(UrEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
        }
        UrEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

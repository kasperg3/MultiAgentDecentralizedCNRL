import os
from gym import utils
from gym_mergablerobots.envs.concept_env import ConceptEnv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
MODEL_XML_PATH = ROOT_DIR[:-15] + '/assets/dual_robot.xml'


class Concept(ConceptEnv, utils.EzPickle):
    def __init__(self):
        initial_qpos = {
            'robot0:joint1': -0,
            'robot0:joint2': -0.2,
            'robot0:joint3': 2,
            'robot0:joint4': 0.6,
            'robot0:joint5': 1.6,
            'robot0:joint6': 0,
            'robot1:joint1': -1.3,
            'robot1:joint2': -0.2,
            'robot1:joint3': 2,
            'robot1:joint4': 0.6,
            'robot1:joint5': 1.6,
            'robot1:joint6': 0
        }
        ConceptEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            n_substeps=20,
            initial_qpos=initial_qpos,
            n_actions=7,
            n_agents=2,
            max_action_steps=35)
        utils.EzPickle.__init__(self)

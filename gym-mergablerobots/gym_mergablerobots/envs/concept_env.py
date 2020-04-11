import random
import os
import copy
from gym.envs.robotics import rotations, utils
from gym_mergablerobots.envs import robot_env
from scipy.spatial.transform import Rotation
import mujoco_py
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class ConceptEnv(gym.Env):
    """
    """

    def _is_collision(self):
        pass

    def __init__(self,
                 model_path,
                 n_substeps,
                 initial_qpos,
                 n_actions,
                 n_agents, ):

        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.n_actions = n_actions
        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

        # Multi-agent variables

        self.actions_available = {"NOOP": 0,
                                  "REACH": 1,
                                  "LIFT": 2,
                                "ORIENT": 3,
                                  "CLOSE_GRIPPER": 4,
                                  "OPEN_GRIPPER": 5,
                                  "PLACE": 6}

        self.n_agents = n_agents
        self.agent_actions = np.empty(n_agents)         # The current action for each agent
        for i in range(n_agents):
            self.agent_actions[i] = -1  # initial action to a invalid action

        # TODO load policies for each actions
        for i in range(self.n_actions):
            pass

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def compute_reward(self):
        # Compute distance between goal and the achieved goal.
        reward = 0
        return reward

    def _step_callback(self):
        pass

    def choose_action(self, action, agent):
        # get a movement from a policy dependent on the agent and the action chosen
        # Set agent_action_done to True if
        # No-Op Action
        if action == self.actions_available["NOOP"]:
            pass
        elif action == self.actions_available["REACH"]:
            # Reach Action
            pass
        elif action == self.actions_available["LIFT"]:
            # Lift Action
            pass
        elif action == self.actions_available["ORIENT"]:
            # Orient Action
            pass
        elif action == self.actions_available["CLOSE_GRIPPER"]:
            # Close Gripper Action
            pass
        elif action == self.actions_available["OPEN_GRIPPER"]:
            # Open Gripper Action
            pass
        elif action == self.actions_available["PLACE"]:
            # Place Action
            pass

        # TODO Implement this and replace self.sample_action()
        return self.sample_action()

    """ step:
        action: Contains an array of n discrete agent actions ranging from 0 to available action of the agent
    """
    def step(self, action):
        # TODO What will happen if two agents finish simultaneously

        # TODO Add logic to choose what concept to use:
        # This should be the same concept until the concept is done
        # Choose a new concept independently depending on the robot
        agent_movement = np.empty((2, 5))
        for agent in range(self.n_agents):
            if self.agent_actions[agent] == -1:
                agent_movement[agent] = self.choose_action(action[agent], agent)
            else:
                # TODO: If the action is not done, get a action from a concept/trained policy
                pass

        # Act in the environment
        self._set_action(agent_movement)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        done = False
        info = {}

        for i in range(self.agent_actions.size):
            # Check if an action is done.
            if self.agent_actions[i] == -1:
                info["agent_done"] = i
            else:
                # Set to -1 if the agent is not done
                info["agent_done"] = -1

        # TODO If action is done, compute reward for the agent
        reward = self.compute_reward()
        return obs, reward, done, info

    def sample_action(self):
        return np.array((
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1))))

    def _set_action(self, action):
        # Change action space if number of actions is changed
        assert action.shape == (2, 5)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        mocap_action = np.zeros((2, 7))
        actuator_action = np.zeros((2, 9))
        for i in range(action.shape[0]):
            pos_ctrl, rot_ctrl, gripper_ctrl = action[i][:3], action[i][3], action[i][4]
            pos_ctrl *= 0.02  # limit maximum change in position
            # Only do z rotation
            z_rot = rotations.euler2quat([0, np.pi, rot_ctrl * 2 * np.pi]) * 0.05
            gripper_ctrl_arr = np.array((gripper_ctrl, gripper_ctrl))
            actuator_action[i] = np.concatenate([pos_ctrl, z_rot, gripper_ctrl_arr])
            mocap_action[i] = np.concatenate([pos_ctrl, z_rot])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, actuator_action)
        utils.mocap_set_action(self.sim, mocap_action)

    def _get_obs(self):
        # positions
        # TODO Design observations space
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        achieved_goal = grip_pos.copy()
        return grip_pos.copy()

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:robotiq_base_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.episode_steps = 0
        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.sim.data.get_site_xpos('box')[:3]
        return goal.ravel().copy()

    def _is_success(self, achieved_goal, desired_goal):
        return False

    def _is_failed(self):
        return False

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper0_target = self.sim.data.get_site_xpos('robot0:grip')
        gripper0_rotation = [0, -1, 1, 0]
        gripper1_target = self.sim.data.get_site_xpos('robot1:grip')
        gripper1_rotation = [0, -1, 1, 0]

        self.sim.data.set_mocap_pos('robot0:mocap', gripper0_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper0_rotation)
        self.sim.data.set_mocap_pos('robot1:mocap', gripper1_target)
        self.sim.data.set_mocap_quat('robot1:mocap', gripper1_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper0_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.initial_gripper1_xpos = self.sim.data.get_site_xpos('robot1:grip').copy()
        self.initial_box_xpos = self.sim.data.get_site_xpos('box').copy()

    def render(self, mode='human', width=500, height=500):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

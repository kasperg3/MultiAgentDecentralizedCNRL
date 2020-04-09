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
    """Superclass for UR environments.
    """

    def _is_collision(self):
        pass

    def __init__(self,
                 model_path,
                 n_substeps,
                 initial_qpos,
                 n_actions,):

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

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32')

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def compute_reward(self):
        # Compute distance between goal and the achieved goal.
        reward = 0
        return reward

    def _step_callback(self):
        pass

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # TODO Add multiple actions
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {}

        reward = self.compute_reward()
        return obs, reward, done, info

    def _set_action(self, action):
        # Change action space if number of is changed
        assert action.shape == (5,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3], action[4]
        pos_ctrl *= 0.01  # limit maximum change in position

        # Only do z rotation
        z_rot = rotations.euler2quat([0, np.pi, rot_ctrl * 2 * np.pi]) * 0.05
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])

        action = np.concatenate([pos_ctrl, z_rot, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        achieved_goal = grip_pos.copy()

        obs = None
        return {
            'observation': grip_pos.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

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
            self.render()
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

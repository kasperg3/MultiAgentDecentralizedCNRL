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

import TD3


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
                 n_agents,
                 max_action_steps):

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
        obs = self._get_obs()
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

        # Multi-agent variables
        self.actions_available = {
                                    "REACH": 0,
                                    "LIFT": 1,
                                    "ORIENT": 2,
                                    "PLACE": 3,
                                    "CLOSE_GRIPPER": 4,
                                    "OPEN_GRIPPER": 5,
                                    "NOOP": 6}

        self.n_agents = n_agents
        self.current_action_steps = [0, 0]
        self.max_action_steps = max_action_steps

        # Policies and TD3 variables
        self.policies = {}
        for action in self.actions_available:
            action_dim = 7
            max_action = 1.0
            if action == "REACH":
                name = "Reach"
                state_dim = 29
            elif action == "LIFT":
                name = "Lift"
                state_dim = 31
            elif action == "ORIENT":
                name = "Orient"
                state_dim = 39
            elif action == "PLACE":
                name = "Place"
                action_dim = 4
                state_dim = 34
            else:  # If not a trained policy, then skip and dont load a policy
                continue
            kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
            }
            file_name = f"TD3_UrBinPicking{name}-v0_1000"
            self.policies[self.actions_available[action]] = TD3.TD3(**kwargs)
            self.policies[self.actions_available[action]].load(f"./models/new_states/{file_name}")

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def compute_reward(self):
        # Compute distance between goal and the achieved goal.
        reward = 0
        return reward

    def _step_callback(self):
        pass

    """
    get_concept_state: Returns the state space of the agent
    concept: integer corresponding to a specific concept
    agent: number of the agent used to get the right state space
    """

    def get_concept_state(self, concept, agent):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot' + str(agent) + ':grip')
        grip_rot_cart = Rotation.from_matrix(
            self.sim.data.get_site_xmat('robot' + str(agent) + ':grip'))  # The rotation of gripper(Cartesian)
        grip_rot = Rotation.as_quat(grip_rot_cart)
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot' + str(agent) + ':grip') * dt
        grip_velr = self.sim.data.get_site_xvelr('robot' + str(agent) + ':grip') * dt

        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt

        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        box_pos = self.sim.data.get_site_xpos('box')
        box_rel_pos = box_pos - grip_pos
        box_rot = Rotation.from_matrix(self.sim.data.get_site_xmat('box')).as_quat()

        object_pos = self.sim.data.get_site_xpos('object' + str(agent))
        object_rel_pos = object_pos - grip_pos
        object_rot = Rotation.from_matrix(self.sim.data.get_site_xmat('object' + str(agent))).as_quat()

        achieved_goal = grip_pos.copy()

        # Define a goal and concat at the front of the observation
        if concept == self.actions_available["REACH"]:
            target_height = 0.05  # The height of the box
            box_offset = [0, 0, target_height]
            goal = self.sim.data.get_site_xpos('box')[:3].copy() + box_offset
            # The relative position to the goal
            goal_rel_pos = goal - achieved_goal
            obs = np.concatenate([
                grip_pos,
                grip_rot,
                grip_velp,
                grip_velr,
                box_pos.ravel(),
                box_rel_pos.ravel(),
                box_rot.ravel(),
                goal_rel_pos,
            ])
        elif concept == self.actions_available["LIFT"]:
            # The relative position is between object and goal position
            object_height = self.sim.data.get_site_xpos('object' + str(agent))[2]
            goal = (self.sim.data.get_site_xpos('object' + str(agent))[2] + 0.15).ravel()
            goal_rel_height = goal - object_height
            obs = np.concatenate([
                grip_pos,
                grip_rot,
                grip_velp,
                grip_velr,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                box_pos.ravel(),
                box_rel_pos.ravel(),
                box_rot.ravel(),
                goal_rel_height,
            ])
        elif concept == self.actions_available["ORIENT"]:
            # Update the goal to follow the box
            goal = self.sim.data.get_site_xpos('object' + str(agent))
            # The relative position to the goal
            goal_rel_pos = goal - achieved_goal
            obs = np.concatenate([
                grip_pos,
                grip_rot,
                grip_velp,
                grip_velr,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                object_rot.ravel(),
                box_pos.ravel(),
                box_rel_pos.ravel(),
                box_rot.ravel(),
                goal_rel_pos,
            ])
        elif concept == self.actions_available["PLACE"]:
            # TODO: make the goal dynamic, so one can change it if one wants to move the position of place
            goal = np.concatenate((np.array([0.63, 0.5, 0.43]), rotations.mat2quat(self.sim.data.get_site_xmat('object' + str(agent)))))
            achieved_goal = np.concatenate((object_pos, object_rot.ravel()))
            goal_rel_pos = goal[:3] - achieved_goal[:3]
            grip_q = rotations.mat2quat(self.sim.data.get_site_xmat('robot0:grip'))
            goal_q = goal[3:]

            goal_rel_rot = np.array(2 * np.arccos(np.abs(np.inner(grip_q, rotations.quat_conjugate(goal_q)))))

            obs = np.concatenate([
                grip_pos,
                grip_rot,
                object_pos,
                object_rot.ravel(),
                object_velp,
                object_velr,
                object_rel_pos,
                goal_rel_pos,
                goal_rel_rot.ravel(),
            ])

        return np.concatenate((goal, obs))

    """
        choose_action: Takes a action for a single agent
        action: Discrete action from the DQN
        agent: used to determine what states to use for the different policies
        returns: agent_done , agent_movement
    """

    def choose_action(self, action, agent):
        # get a movement from a policy dependent on the agent and the action chosen
        # Set agent_action_done to True if
        self.move_allowed[agent] = True
        agent_movement = np.zeros(7)
        agent_done = False
        self.current_action_steps[agent] += 1  # increment the amount of
        if self.current_action_steps[agent] >= self.max_action_steps:
            agent_done = True
            self.current_action_steps[agent] = 0
        elif action == self.actions_available["NOOP"]:
            self.move_allowed[agent] = False
        elif action == self.actions_available["REACH"]:
            state = self.get_concept_state(action, agent)
            agent_movement = self.policies[action].select_action(state)
            d = np.linalg.norm(self.sim.data.get_site_xpos('robot' + str(agent) + ':grip') - (self.sim.data.get_site_xpos('box') + [0, 0, 0.05]), axis=-1)
            agent_done = (d < 0.05).astype(np.bool)
        elif action == self.actions_available["LIFT"]:
            state = self.get_concept_state(action, agent)
            agent_movement = self.policies[action].select_action(state)
            table_height = 0.414
            object_height = self.sim.data.get_site_xpos('object' + str(agent))[2]
            if np.abs(object_height - table_height) >= 0.15:
                agent_done = True
        elif action == self.actions_available["ORIENT"]:
            state = self.get_concept_state(action, agent)
            agent_movement = self.policies[action].select_action(state)
            d = np.linalg.norm(self.sim.data.get_site_xpos('robot' + str(agent) + ':grip') - self.sim.data.get_site_xpos('object' + str(agent)), axis=-1)
            agent_done = (d < 0.01).astype(np.bool)
        elif action == self.actions_available["CLOSE_GRIPPER"]:
            self.gripper_ctrl[agent] = 0.3
        elif action == self.actions_available["OPEN_GRIPPER"]:
            self.gripper_ctrl[agent] = -1
        elif action == self.actions_available["PLACE"]:
            state = self.get_concept_state(action, agent)
            policy_output = self.policies[action].select_action(state)
            rot_ctrl = (rotations.euler2quat([0, np.pi, policy_output[3] * 2 * np.pi]) * rotations.quat_conjugate(rotations.mat2quat(self.sim.data.get_site_xmat('robot0:grip')))) * 2
            pos_crtl = policy_output[:3] * 0.5
            agent_movement = np.concatenate((pos_crtl, rot_ctrl))

        return agent_done, agent_movement

    """ step:
            This method should only return when the agent has completed an action or timeouted
        action: Contains an array of n discrete agent actions ranging from 0 to available action of the agent
    """

    def step(self, action):
        # TODO What will happen if two agents finish simultaneously
        # This should be the same concept until the concept is done
        # Choose a new concept independently depending on the robot
        agent_movement = np.empty((2, 7))
        info = {"agent_done": [-1, -1]}

        while True:
            for agent in range(self.n_agents):
                agent_done, agent_movement[agent] = self.choose_action(action[agent], agent)
                if agent_done:
                    info["agent_done"][agent] = True

            # Act in the environment
            self._set_action(agent_movement)
            self.sim.step()
            self._step_callback()
            done = False
            self.render()

            # If any agents are done, then break the while
            for is_done in info["agent_done"]:
                if is_done == 1:
                    # TODO If action is done, compute reward for the agent
                    obs = self._get_obs()
                    reward = self.compute_reward()
                    return obs, reward, done, info

    def sample_action(self):
        return np.array((
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1)),
            float(self.np_random.uniform(-1, 1))))

    def _set_action(self, action):
        # Change action space if number of actions is changed
        assert action.shape == (2, 7)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        mocap_action = np.zeros((2, 7))
        actuator_action = np.zeros((2, 2))
        for i in range(action.shape[0]):
            pos_ctrl, rot_ctrl, gripper_ctrl = action[i][:3], action[i][3:7], self.gripper_ctrl[i]
            pos_ctrl *= 0.05  # limit maximum change in position
            rot_ctrl *= 0.01
            actuator_action[i] = np.array((gripper_ctrl, gripper_ctrl))

            # Disallow movement when closing or opening gripper
            if not self.move_allowed[i]:
                pos_ctrl = np.zeros(3)
                pos_ctrl[2] = 0.0002  # To avoid drifting in z
                rot_ctrl = np.zeros(4)

            mocap_action[i] = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, np.concatenate((np.zeros(14), actuator_action.ravel())))
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
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.forward()

    def sample_box_position(self):
        box_xpos = self.initial_box_xpos[:2] + self.np_random.uniform(-0.1, 0.1, size=2)
        self.box_qpos = self.sim.data.get_joint_qpos('box:joint')
        assert self.box_qpos.shape == (7,)
        self.box_qpos[:2] = box_xpos
        # Set box position
        self.initial_box_xpos = self.box_qpos[:3]
        self.sim.data.set_joint_qpos('box:joint', self.box_qpos)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.episode_steps = 0

        self.sample_box_position()

        #Start with the gripper open:
        self.gripper_ctrl = np.array((-0.3, -0.3))
        target_y_range = 0.05  # The length of the box
        target_x_range = 0.1  # The width of the box
        target_height = 0.05  # The height of the box

        # Allow movement for all other actions than close and open gripper
        self.move_allowed = np.array((True, True), np.bool)

        # random object spawn (z-rot and xyz)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_offset = [
            float(self.np_random.uniform(-target_x_range, target_x_range)),
            float(self.np_random.uniform(-target_y_range, target_y_range)),
            target_height]
        object_z_angle = float(self.np_random.uniform(-np.pi, np.pi))
        object_qpos_euler = rotations.quat2euler(object_qpos[3:])
        object_qpos_euler[2] = object_z_angle
        object_qpos[3:] = rotations.euler2quat(object_qpos_euler)
        object_qpos[:3] = self.sim.data.get_joint_qpos('box:joint')[:3] + object_offset

        # random object1 spawn (z-rot and xyz)
        object_qpos = self.sim.data.get_joint_qpos('object1:joint')
        object_offset = [
            float(self.np_random.uniform(-target_x_range, target_x_range)),
            float(self.np_random.uniform(-target_y_range, target_y_range)),
            target_height]
        object_z_angle = float(self.np_random.uniform(-np.pi, np.pi))
        object_qpos_euler = rotations.quat2euler(object_qpos[3:])
        object_qpos_euler[2] = object_z_angle
        object_qpos[3:] = rotations.euler2quat(object_qpos_euler)
        object_qpos[:3] = self.sim.data.get_joint_qpos('box:joint')[:3] + object_offset

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.sim.data.get_site_xpos('box')[:3]
        return goal.ravel().copy()

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
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
